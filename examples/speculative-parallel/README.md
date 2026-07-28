# llama.cpp/examples/speculative-parallel

Speculative decoding where the drafter and the target run concurrently instead of alternating serially.

Starting point is a copy of `speculative-simple`, kept alongside it so the serial version stays available
as an A/B timing reference.

```bash
./bin/llama-speculative-parallel \
    -m  ~/dev/arm-spec/models/qwen2.5-3b-instruct-q4_0.gguf \
    -md ~/dev/arm-spec/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --spec-type draft-simple --spec-draft-n-max 4 \
    -t 12 -td 4 --temp 0 -n 256 -c 4096 \
    -p "Write a detailed explanation of how a compiler lowers a for-loop to assembly."
```

`--spec-type draft-simple` is required — `-md` on its own does not enable a speculator.

`-t` sets the target threads, `-td` the drafter threads. To run the drafter on a separate device instead,
use `-devd` / `-ngld`.

```
  cmake --build build -j$(nproc)

  note: --spec-type draft-simple is REQUIRED. params.speculative.types defaults to
  { COMMON_SPECULATIVE_TYPE_NONE }, so passing -md alone leaves no speculator enabled,
  common_speculative_init() returns nullptr and we assert in get_draft_params().

  serial reference (examples/speculative-simple, identical logic for now):

    ./build/bin/llama-speculative-simple \
      -m ~/dev/arm-spec/models/qwen2.5-3b-instruct-q4_0.gguf \
      -md ~/dev/arm-spec/models/qwen2.5-0.5b-instruct-q4_0.gguf \
      --spec-type draft-simple --spec-draft-n-max 4 \
      -t 16 --temp 0 -n 256 -c 4096 \
      -p "Write a detailed explanation of how a compiler lowers a for-loop to assembly."

  this example, config A (split 16 cores: target 12, drafter 4):

    ./build/bin/llama-speculative-parallel \
      -m ~/dev/arm-spec/models/qwen2.5-3b-instruct-q4_0.gguf \
      -md ~/dev/arm-spec/models/qwen2.5-0.5b-instruct-q4_0.gguf \
      --spec-type draft-simple --spec-draft-n-max 4 \
      -t 12 -td 4 --temp 0 -n 256 -c 4096 \
      -p "Write a detailed explanation of how a compiler lowers a for-loop to assembly."
```
