#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

#include <thread>
#include <mutex>


// Currently the model runs with the simple draft and checker models
// It alternates serially, like so
// Generate n tokens
// Check the last n tokens
// The goal is to rewrite this code, such that it runs on seperate threads naively for this loop to run, | will seperate threads
// Generate n tokens |  (first run)
// Generate n more tokens | check the last n tokens (for the other runs)
// If the checker rejects a token, first we should just get it functional. It should regenerate all of it, until it is correct.
// However, regardless of whether it passes or not, it should be drafting new tokens, unless it recieves a fail, in which case it should get reset
// If the checker accepts n tokens, it will wait until the draft creates more tokens and continues as usual
// I think it is important to time both the drafting and checking stages, to see if there is a clear bottleneck, but more importantly if there is a way to remove said bottleneck
// After getting it to succesfully run on multithreads, we then implement KV cache regeneration and making partial runs work. We could also measure what the ideal "n" is here.

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    //TODO: START PARALLEL HERE!
    // TODO: simplify this logic
    {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;

        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.draft.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        auto cparams = common_context_params_to_llama(params_dft);
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    }

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt)       == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = (common_context_can_seq_rm(ctx_dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us(); // start prompt encode timer

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling)); // make target sampler, not draft

    // eval the prompt
    llama_decode(ctx_tgt,       llama_batch_get_one(inp.data(), inp.size() - 1));
    llama_decode(ctx_dft.get(), llama_batch_get_one(inp.data(), inp.size() - 1));
    //decode prompt minus last token into target kv for both target and draft

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt)); // allocates memory for ctx_tgt

    int n_past = inp.size() - 1; // KV position cursor

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1); // Speculator common prompt

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1); // alloc target branch

    size_t n_draft = 0;

    llama_tokens draft; // current draft tokens
    common_prompt_checkpoint ckpt; // cpkt state snapshot for rollback

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    std::mutex mtx;

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id)); // update checkpoint position with new token count, KV pos_min/max of target

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            } // checkpoint draft model

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ -1,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec); // run draft model, writes into draft

            // save the original draft size
            n_draft = draft.size(); // new draft size? or original?

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
            {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); // restore draft KV to pre-draft state

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1); // drops drafts scratch tokens
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }
        // Verify phase
        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt); // clear target batch
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true); // add id_last, n_past++, logits on

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
            } // append draft[i] at n_past + i and its logits

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt); // target decode, this is the part that saves time
        }

        // evaluate the same batch with the draft model
        {
            // TODO: extend to support MTP, Eagle, etc. See server code for reference
            llama_decode(ctx_dft.get(), batch_tgt);
        } // eval draft to align KV cache

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get())); //clone sampler if ckpt mode
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft); // target samples greedily walks draft while it agrees returns draft prefix+1 target token

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            draft = std::move(ids); // keep accepted part as next iterations draft

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); // restore target KV

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1); // trim past pos_max
            }

            {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); // same as previous but with draft

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens); // truncate history
            smpl = std::move(smpl_save); // restore sampler clone

            n_past = (int) prompt_tgt.size(); // reset cursor

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1); // tell speculator how many draft tokens accepted

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i]; // push id_last to prompt_tgt

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            } // check for end of generation token

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            } // check for color (drafted), plain (target)
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt),       seq_id, n_past, -1); // remove rejected tail from both KV
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, n_past, -1); // remove rejected tail from both KV
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        } // stop on EOS or n_predict
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
