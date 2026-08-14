#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include "draft_worker.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

#include <future>
#include <thread>


// The draft model now runs on its own thread. main() owns the target model only
// and never touches ctx_dft or the speculator; everything crosses draft_channel
// as plain data:
//
//   main:   prefill | verify draft | send committed tokens | wait next draft
//   worker: prefill | draft ahead  | replay verified batch | draft again
//
// Both prefills overlap, and while the target verifies round N the worker is
// already drafting round N+1.
//
// v1 discards a rejected draft wholesale and lets the worker regenerate from
// the committed state, so there is no checkpoint/rollback path here. Partial
// draft reuse and KV cache regeneration on rejection are v2 work, as is
// searching for the ideal n_draft now that both stages are timed separately.

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

    // start the draft thread before our own prefill so the two prefills overlap
    draft_worker  worker;
    draft_channel ch;

    std::promise<bool> ready_promise;
    std::future<bool>  ready = ready_promise.get_future();

    std::thread th_dft(draft_thread_main,
                       std::ref(worker), std::ref(ch), std::cref(params),
                       ctx_tgt, std::cref(inp), std::move(ready_promise));

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
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));
    //decode prompt minus last token into target kv, the worker prefills itself

    const auto t_enc_end = ggml_time_us();

    // measured separately: including it in the encode window would charge the
    // draft model's load against the target's prompt throughput
    const auto t_ready_start = ggml_time_us();

    if (!ready.get()) {
        LOG_ERR("%s: draft worker failed to initialise\n", __func__);

        ch.shutdown();
        th_dft.join();

        return 1;
    }

    const auto t_ready_us = ggml_time_us() - t_ready_start;

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt)); // allocates memory for ctx_tgt

    int n_past = inp.size() - 1; // KV position cursor

    const auto & params_spec = params.speculative;

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1); // alloc target branch

    size_t n_draft = 0;

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // the draft is produced concurrently by the worker thread
        llama_tokens draft;
        if (!ch.wait_draft(draft)) {
            break;
        }

        n_draft = draft.size();

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

        // batch_tgt is refilled every iteration, so copy it out for the worker
        draft_batch_view verified;
        verified.assign_from(batch_tgt);

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
        const size_t n_prompt_prev = prompt_tgt.size();

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i]; // push id_last to prompt_tgt

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            } // check for end of generation token

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("%c[%dm%s%c[37m", 0x1b, (36 - 0 % 6), token_str.c_str(), 0x1b);
            } else {
                LOG("%s", token_str.c_str());
            } // check for color (drafted), plain (target)
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1); // remove rejected tail, the worker trims its own
        }

        // hand the committed state to the worker so its mirror stays identical.
        // skipped on EOG: the post-processing loop broke early, so n_past counts
        // tokens that never reached prompt_tgt and the worker would trim to a
        // cursor past its own KV contents
        if (!has_eos) {
            draft_request req;

            req.verified   = std::move(verified);
            req.n_accepted = (uint16_t) (ids.size() - 1);
            req.accepted.assign(prompt_tgt.begin() + (ptrdiff_t) n_prompt_prev, prompt_tgt.end()); // exactly what we appended above
            req.id_last    = id_last;
            req.n_past     = n_past;

            ch.send_request(std::move(req));
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        } // stop on EOS or n_predict
    }

    ch.shutdown();
    th_dft.join();

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
    LOG_INF("  t_draft   = %8.3f s\n", worker.timing.t_draft_us   / 1e6f);
    LOG_INF("  t_process = %8.3f s\n", worker.timing.t_process_us / 1e6f);
    LOG_INF("  t_wait    = %8.3f s\n", worker.timing.t_wait_us    / 1e6f);
    LOG_INF("  n_calls   = %" PRId64 "\n", worker.timing.n_draft_calls);
    LOG_INF("  n_tokens  = %" PRId64 "\n", worker.timing.n_draft_toks);
    LOG_INF("  n_threads = %d\n", worker.timing.n_threads_dft);
    LOG_INF("  t_ready   = %8.3f s (main blocked on worker load + prefill)\n", t_ready_us / 1e6f);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
