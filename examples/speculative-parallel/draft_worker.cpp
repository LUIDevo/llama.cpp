#include "draft_worker.h"

#include "log.h"

#include <stdexcept>

// ----------------------------------------------------------------------------
// draft_batch_view
// ----------------------------------------------------------------------------

void draft_batch_view::clear() {
    tokens.clear();
    pos.clear();
}

void draft_batch_view::assign_from(const llama_batch & batch) {
    tokens.resize(batch.n_tokens);
    pos.resize(batch.n_tokens);

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        tokens[i] = batch.token[i];
        pos[i]    = batch.pos[i];
    }
}

void draft_batch_view::fill(llama_batch & dst, llama_seq_id seq_id) const {
    common_batch_clear(dst);

    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(dst, tokens[i], pos[i], { seq_id }, true);
    }
}

// ----------------------------------------------------------------------------
// draft_channel
// ----------------------------------------------------------------------------

void draft_channel::send_request(draft_request && req_new) {
    {
        std::lock_guard<std::mutex> lock(mtx);

        req     = std::move(req_new);
        has_req = true;
    }

    cv.notify_all();
}

bool draft_channel::wait_draft(llama_tokens & out) {
    std::unique_lock<std::mutex> lock(mtx);

    cv.wait(lock, [this] { return has_draft || quit; });

    if (quit) {
        return false;
    }

    out       = std::move(draft);
    has_draft = false;

    return true;
}

bool draft_channel::wait_request(draft_request & out) {
    std::unique_lock<std::mutex> lock(mtx);

    cv.wait(lock, [this] { return has_req || quit; });

    if (quit) {
        return false;
    }

    out     = std::move(req);
    has_req = false;

    return true;
}

void draft_channel::publish(llama_tokens && draft_new) {
    {
        std::lock_guard<std::mutex> lock(mtx);

        draft     = std::move(draft_new);
        has_draft = true;
    }

    cv.notify_all();
}

void draft_channel::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mtx);

        quit = true;
    }

    cv.notify_all();
}

// ----------------------------------------------------------------------------
// draft_setup
// ----------------------------------------------------------------------------

bool draft_setup(
        draft_worker        & w,
        const common_params & params_base,
        llama_context       * ctx_tgt,
        const llama_tokens  & inp) {
    // NONE stays in the list by default and is skipped when the impls are built
    // (speculative.cpp:2391), so only the real implementations are counted here
    size_t n_impl = 0;
    for (const auto type : params_base.speculative.types) {
        if (type == COMMON_SPECULATIVE_TYPE_NONE) {
            continue;
        }

        n_impl++;

        GGML_ASSERT(type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE &&
                    "parallel speculation requires the plain draft-model implementation");
    }

    GGML_ASSERT(n_impl == 1 && "parallel speculation requires exactly one draft-simple implementation");

    const auto & spec_draft = params_base.speculative.draft;

    common_params params_dft = params_base;

    params_dft.devices      = spec_draft.devices;
    params_dft.model        = spec_draft.mparams;
    params_dft.n_gpu_layers = spec_draft.n_gpu_layers;

    if (spec_draft.cpuparams.n_threads > 0) {
        params_dft.cpuparams.n_threads       = spec_draft.cpuparams.n_threads;
        params_dft.cpuparams_batch.n_threads = spec_draft.cpuparams.n_threads;
    }

    params_dft.tensor_buft_overrides = spec_draft.tensor_buft_overrides;

    auto mparams_dft = common_model_params_to_llama(params_dft);

    w.model.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
    if (w.model == nullptr) {
        LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
        return false;
    }

    auto cparams_dft = common_context_params_to_llama(params_dft);

    w.ctx.reset(llama_init_from_model(w.model.get(), cparams_dft));
    if (w.ctx == nullptr) {
        LOG_ERR("failed to create context for draft model, '%s'\n", params_dft.model.path.c_str());
        return false;
    }

    common_params_speculative sparams = params_base.speculative;

    sparams.draft.ctx_tgt = ctx_tgt;
    sparams.draft.ctx_dft = w.ctx.get();

    w.spec.reset(common_speculative_init(sparams, 1));
    if (w.spec == nullptr) {
        LOG_ERR("%s: failed to init the speculative context\n", __func__);
        return false;
    }

    w.batch = llama_batch_init(llama_n_batch(w.ctx.get()), 0, 1);

    w.prompt.assign(inp.begin(), inp.end() - 1);
    w.prompt.reserve(llama_n_ctx(w.ctx.get()));

    w.id_last = inp.back();
    w.n_past  = (int) w.prompt.size();

    if (llama_decode(w.ctx.get(), llama_batch_get_one(w.prompt.data(), (int32_t) w.prompt.size())) != 0) {
        LOG_ERR("failed to prefill the draft model prompt (%d tokens)\n", (int) w.prompt.size());
        return false;
    }

    common_speculative_begin(w.spec.get(), w.seq_id, w.prompt);

    w.timing.n_threads_dft = params_dft.cpuparams.n_threads;

    return true;
}

// ----------------------------------------------------------------------------
// draft_thread_main
// ----------------------------------------------------------------------------

void draft_thread_main(
        draft_worker        & w,
        draft_channel       & ch,
        const common_params & params_base,
        llama_context       * ctx_tgt,
        const llama_tokens  & inp,
        std::promise<bool>    ready) {
    bool ok = false;

    try {
        ok = draft_setup(w, params_base, ctx_tgt, inp);
    } catch (const std::exception & e) {
        LOG_ERR("draft worker setup failed: %s\n", e.what());
        ok = false;
    }

    ready.set_value(ok);

    if (!ok) {
        if (w.batch.token != nullptr) {
            llama_batch_free(w.batch);
            w.batch = {};
        }

        return;
    }

    auto do_draft = [&w]() {
        llama_tokens out;

        const int n_past_pre = w.n_past;

        common_speculative_get_draft_params(w.spec.get(), w.seq_id) = {
            /* .drafting = */ true,
            /* .n_max    = */ -1,
            /* .n_past   = */ w.n_past,
            /* .id_last  = */ w.id_last,
            /* .prompt   = */ &w.prompt,
            /* .result   = */ &out,
        };

        const int64_t t_start_us = ggml_time_us();

        common_speculative_draft(w.spec.get());

        w.timing.t_draft_us += ggml_time_us() - t_start_us;

        if (!out.empty()) {
            w.has_drafted = true;

            w.timing.n_draft_calls++;
            w.timing.n_draft_toks += out.size();
        }

        // draft() decodes id_last at n_past and its speculative tokens above it.
        // the verified batch replays exactly those positions, so the scratch has
        // to go or the replay collides with occupied cells
        llama_memory_seq_rm(llama_get_memory(w.ctx.get()), w.seq_id, n_past_pre, -1);

        return out;
    };

    ch.publish(do_draft());

    while (true) {
        draft_request req;

        const int64_t t_wait_start_us = ggml_time_us();

        const bool have_req = ch.wait_request(req);

        w.timing.t_wait_us += ggml_time_us() - t_wait_start_us;

        if (!have_req) {
            break;
        }

        if (!req.verified.empty()) {
            req.verified.fill(w.batch, w.seq_id);

            const int64_t t_start_us = ggml_time_us();

            common_speculative_process(w.spec.get(), w.batch);

            w.timing.t_process_us += ggml_time_us() - t_start_us;
        }

        if (w.has_drafted) {
            common_speculative_accept(w.spec.get(), w.seq_id, req.n_accepted);
        }

        w.prompt.insert(w.prompt.end(), req.accepted.begin(), req.accepted.end());

        w.id_last = req.id_last;
        w.n_past  = req.n_past;

        llama_memory_seq_rm(llama_get_memory(w.ctx.get()), w.seq_id, w.n_past, -1);

        ch.publish(do_draft());
    }

    if (w.batch.token != nullptr) {
        llama_batch_free(w.batch);
        w.batch = {};
    }
}
