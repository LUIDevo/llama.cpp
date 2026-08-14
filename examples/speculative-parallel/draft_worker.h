#pragma once

#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <vector>

// ============================================================================
// Threaded speculative decoding: draft worker contract
// ============================================================================
//
// Ownership rule (the reason this file exists):
//
//   llama_context is NOT thread safe. The draft thread exclusively owns
//   `ctx_dft` and `spec`. No pointer to either may be observed by main() after
//   setup returns. Everything main() needs from the draft model crosses this
//   channel as plain data.
//
// This also means the raw `llama_decode(ctx_dft, batch_tgt)` that the serial
// implementation performs during verification must move onto the worker; it
// appears here as draft_request::verified, replayed via
// common_speculative_process().
//
// v1 scope (per the TODO block at the top of speculative-parallel.cpp):
//
//   On rejection the worker discards its draft entirely and regenerates. There
//   is deliberately no checkpoint/rollback path and no partial draft reuse, so
//   common_prompt_checkpoint does not appear in this header. Partial reuse is
//   v2 work.
//
// Constraint (verified in common/speculative.cpp):
//
//   Only COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE can be decoupled onto a thread.
//   EAGLE3 / DFlash / MTP read target hidden states inside draft() itself
//   (speculative.cpp:600, 1052, 1402), so there is nothing to run ahead of.
//   draft_setup() asserts this.

// ----------------------------------------------------------------------------
// draft_batch_view
// ----------------------------------------------------------------------------
//
// A llama_batch holds raw pointers into buffers owned by main()'s `batch_tgt`,
// which is cleared and refilled every iteration. It therefore cannot be handed
// across the thread boundary as-is. This is a value copy of the contents.

struct draft_batch_view {
    std::vector<llama_token> tokens;
    std::vector<llama_pos>   pos;

    void clear();

    // copy batch.n_tokens entries out of `batch`
    void assign_from(const llama_batch & batch);

    // refill `dst` (a batch owned by the caller) from this view, all entries
    // targeting `seq_id` with logits enabled
    void fill(llama_batch & dst, llama_seq_id seq_id) const;

    bool empty() const { return tokens.empty(); }
};

// ----------------------------------------------------------------------------
// draft_request : main -> worker
// ----------------------------------------------------------------------------
//
// Sent after every verification round. Tells the worker what the target model
// actually committed, so the draft KV cache can be brought back into agreement
// before the next draft is generated.

struct draft_request {
    // exactly the batch the target model decoded this round, i.e.
    // [id_last, draft[0] .. draft[n-1]]. Replayed into ctx_dft so both caches
    // have seen identical tokens at identical positions.
    draft_batch_view verified;

    // how many of the worker's previously proposed draft tokens the target
    // accepted (ids.size() - 1 in the serial code)
    uint16_t n_accepted = 0;

    // tokens the target committed this round, to be appended to the worker's
    // local prompt mirror
    llama_tokens accepted;

    // post-commit cursor state, mirroring main()'s own
    llama_token id_last = 0;
    int         n_past  = 0;
};

// ----------------------------------------------------------------------------
// draft_timing
// ----------------------------------------------------------------------------
//
// Populated by the worker, read by main() only after join(). The point of the
// exercise is finding the bottleneck, so drafting and verification are timed
// separately and the thread split is recorded alongside them -- under OpenMP
// the two models' teams contend, and a timing number is meaningless without
// knowing how the cores were divided.

struct draft_timing {
    int64_t t_draft_us   = 0;  // time inside common_speculative_draft
    int64_t t_process_us = 0;  // time replaying the verified batch into ctx_dft
    int64_t t_wait_us    = 0;  // time blocked waiting on main -- if this
                               // dominates, the draft model is not the
                               // bottleneck and n_draft should grow

    int64_t n_draft_calls = 0;
    int64_t n_draft_toks  = 0;

    int n_threads_dft = 0;     // recorded so runs are comparable
};

// ----------------------------------------------------------------------------
// draft_channel
// ----------------------------------------------------------------------------
//
// One slot in each direction, guarded by a single mutex. Deliberately the
// simplest thing that works: v1 has no pipelining, so a deeper queue would
// only hide ordering mistakes.

class draft_channel {
public:
    // -- main side ----------------------------------------------------------

    // hand a request to the worker, overwriting any unconsumed one
    void send_request(draft_request && req);

    // block until the worker publishes a draft.
    // returns false if the worker exited (setup failure or shutdown).
    bool wait_draft(llama_tokens & out);

    // -- worker side --------------------------------------------------------

    // block until main sends a request.
    // returns false if shutdown was requested -- the worker must then exit.
    bool wait_request(draft_request & out);

    // publish a freshly generated draft to main
    void publish(llama_tokens && draft);

    // -- either side --------------------------------------------------------

    // wake both sides and cause every subsequent wait to return false
    void shutdown();

private:
    std::mutex              mtx;
    std::condition_variable cv;

    draft_request req;
    bool          has_req = false;

    llama_tokens draft;
    bool         has_draft = false;

    bool quit = false;
};

// ----------------------------------------------------------------------------
// draft_worker
// ----------------------------------------------------------------------------
//
// All fields are touched only by the draft thread once the thread is spawned.

struct draft_worker {
    llama_model_ptr        model;
    llama_context_ptr      ctx;
    common_speculative_ptr spec;

    // local mirror of the committed token history.
    //
    // This must NOT alias main()'s prompt_tgt. common_speculative_draft_params
    // holds `const llama_tokens * prompt` and the implementation dereferences
    // it across a full llama_decode; a concurrent push_back on main's vector
    // would reallocate the buffer mid-draft.
    llama_tokens prompt;

    llama_token  id_last = 0;
    int          n_past  = 0;
    llama_seq_id seq_id  = 0;

    // batch owned by the worker, used to replay draft_request::verified
    llama_batch batch = {};

    // common_speculative_accept() asserts on spec->impl_last[seq_id]
    // (speculative.cpp:2624-2626), which is assigned only by a draft() call
    // that produced a non-empty result (speculative.cpp:2580, 2590-2596).
    // Calling accept before that has happened aborts, so the loop must draft
    // first and this guard must be honoured.
    bool has_drafted = false;

    draft_timing timing;
};

// ----------------------------------------------------------------------------
// entry points
// ----------------------------------------------------------------------------

// Runs on the draft thread before the work loop. Loads the draft model, builds
// ctx_dft and the speculator, prefills the prompt, and establishes the same
// invariant main() holds: the KV cache contains `prompt`, and `id_last` is
// tokenized but not yet decoded.
//
// `ctx_tgt` is consumed here only, for common_speculative_init's vocab
// compatibility check (speculative.cpp:236), and must not be retained.
//
// Returns false on any failure; the caller reports it through the promise.
bool draft_setup(
        draft_worker        & w,
        const common_params & params_base,
        llama_context       * ctx_tgt,
        const llama_tokens  & inp);

// Thread entry point. Calls draft_setup, reports the result through `ready`,
// and on success runs the work loop until shutdown.
//
// Loop ordering is fixed by the accept-after-draft constraint above:
//
//     draft once  ->  publish  ->  wait request  ->  process + accept
//         ^                                              |
//         +----------------------------------------------+
//
// Exceptions must not escape: common_speculative_init throws on incompatible
// vocab (speculative.cpp:241) and an exception leaving a thread entry point
// calls std::terminate.
void draft_thread_main(
        draft_worker        & w,
        draft_channel       & ch,
        const common_params & params_base,
        llama_context       * ctx_tgt,
        const llama_tokens  & inp,
        std::promise<bool>    ready);
