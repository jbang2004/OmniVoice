# Online Batch Serving

OmniVoice includes a single-process HTTP server for resident GPU inference and
online micro-batching. It keeps one model instance loaded, accepts concurrent
HTTP requests, groups compatible requests, and executes batched
`model.generate(...)` calls.

Install serving dependencies:

```bash
pip install "omnivoice[serve]"
```

Start a production-oriented local server:

```bash
omnivoice-serve-online-batch \
  --model k2-fsa/OmniVoice \
  --host 0.0.0.0 \
  --port 8000 \
  --scheduler_profile balanced12 \
  --generation_mode optimized \
  --compile_llm true \
  --warmup_test_list warmup.jsonl \
  --warmup_fill_batch true
```

For exact comparisons with the original per-item path, use:

```bash
omnivoice-serve-online-batch \
  --model k2-fsa/OmniVoice \
  --scheduler_profile balanced12 \
  --generation_mode official_compatible \
  --compile_llm false
```

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/healthz` | GET | Health, warmup status, server limits, scheduler snapshot, and voice registry summary. |
| `/v1/server` | GET | Server runtime settings, scheduler snapshot, registered voices, and startup warmup state. |
| `/v1/scheduler` | GET | Scheduler-only snapshot for dashboards and benchmark tooling. |
| `/v1/scheduler/reset_metrics` | POST | Reset scheduler metrics. |
| `/v1/tts` | POST | Generate one WAV response. |
| `/v1/tts_batch` | POST | Submit many requests and receive base64 WAV payloads. |
| `/v1/voices` | POST | Register a reusable voice-clone prompt. |
| `/v1/voices` | GET | List the in-memory voice registry. |
| `/v1/voices/{voice_id}` | DELETE | Delete a registered voice prompt. |

`/v1/tts_batch` atomically enqueues the submitted list: if the scheduler queue
does not have enough capacity for every item, the endpoint returns `429` and no
item from that HTTP request is submitted for generation.

Request bodies are strict. Unknown JSON fields return `422` instead of being
ignored, so misspelled controls such as `duraton` fail fast instead of silently
dropping duration or speed settings.

Server-side request errors use structured `detail` payloads with `code` and
`message` fields. For example, text longer than `--max_request_text_chars`
returns `413` with `code: "text_too_long"`.

## Voice Registration

Register a reference voice once, then reuse its `voice_id` in later requests.
This avoids repeatedly tokenizing the same reference audio.

```bash
curl -s http://127.0.0.1:8000/v1/voices \
  -H 'content-type: application/json' \
  -d '{
    "voice_id": "speaker-a",
    "ref_audio": "/path/to/ref.wav",
    "ref_text": "Reference transcript."
  }'
```

Generate with the registered voice:

```bash
curl -s http://127.0.0.1:8000/v1/tts \
  -H 'content-type: application/json' \
  -o out.wav \
  -d '{
    "request_id": "demo-1",
    "text": "Hello from OmniVoice.",
    "language_id": "en",
    "voice_id": "speaker-a"
  }'
```

The voice registry is in-memory and LRU-bounded. Configure its capacity with
`--max_voice_prompts` (default: `256`). Registering more voices evicts the least
recently used entries. Use `GET /v1/voices` or `GET /v1/server` to inspect
`size`, `max_entries`, and `evictions`.

## Time-Slot Generation

`duration` sets the target audio-token count. Post-processing can still trim
silence, so use `enforce_output_duration=true` when the returned WAV must fit a
fixed time slot exactly.

```bash
curl -s http://127.0.0.1:8000/v1/tts \
  -H 'content-type: application/json' \
  -o exact.wav \
  -d '{
    "text": "This line should fit exactly into the requested slot.",
    "language_id": "en",
    "voice_id": "speaker-a",
    "duration": 3.5,
    "enforce_output_duration": true
  }'
```

You can enable strict output duration as the server default:

```bash
omnivoice-serve-online-batch \
  --model k2-fsa/OmniVoice \
  --generation_mode optimized \
  --enforce_output_duration true
```

Requests can also override the server default per item:

```bash
curl -s http://127.0.0.1:8000/v1/tts \
  -H 'content-type: application/json' \
  -o exact.wav \
  -d '{
    "text": "This request enforces its own output duration.",
    "duration": 3.5,
    "enforce_output_duration": true
  }'
```

The online scheduler applies strict duration after model generation per request,
so strict and non-strict requests can still share the same micro-batch.

## Recommended Inference Path

Use the optimized online batch path as the default resident-GPU runtime:

```bash
omnivoice-serve-online-batch \
  --model k2-fsa/OmniVoice \
  --scheduler_profile balanced12 \
  --generation_mode optimized \
  --num_step 32 \
  --compile_llm true \
  --enforce_output_duration true
```

Keep `num_step=32` for production-quality comparisons. Lower values are useful
for scheduler stress tests, but should not be used to judge final audio quality.
Use `official_compatible` only when you need an A/B comparison against the
original item-by-item path.

## Runtime Surface

The stable serving surface is intentionally small:

- `omnivoice.serving` exports the request-level online micro-batcher, HTTP app
  factory, serving profiles, and voice registry.
- `omnivoice-serve-online-batch` is the recommended resident-GPU server.
- `omnivoice-infer-online-batch` is the matching in-process benchmark path.

The step-level scheduler prototype lives in `omnivoice.experimental`. It exposes
the inner diffusion/unmasking loop one step at a time so active requests can be
repacked between steps. It is useful for continuous-batching research, but local
benchmarks showed it was slower than request-level online batching on the
context-outlier workload while keeping GPU utilization high. Keep it out of
production serving unless a workload-specific benchmark proves an improvement.

You can still run the experimental benchmark directly:

```bash
python -m omnivoice.experimental.infer_stepwise_online_batch --help
```

## Benchmarking

The serving tools are split by purpose:

- `omnivoice-recommend-serving-profile` inspects a representative JSONL and
  emits a measured profile plus the matching `omnivoice-serve-online-batch`
  command.
- `omnivoice-make-heterogeneous-test-list` creates short/long mixed JSONL
  traffic for scheduler experiments.
- `omnivoice-benchmark-scheduler-packing` simulates packing policies without
  loading the model or touching the GPU.
- `omnivoice-infer-online-batch`, `omnivoice-benchmark-http-batch`,
  `omnivoice-benchmark-http-sweep`, and `omnivoice-benchmark-server-sweep`
  measure real generation throughput.

For a first-pass recommendation:

```bash
omnivoice-recommend-serving-profile \
  --test_list validation.jsonl \
  --concurrency 12
```

To build a small context-outlier workload and compare packing policies before
running GPU benchmarks:

```bash
omnivoice-make-heterogeneous-test-list \
  --output results/heterogeneous.jsonl \
  --ref_audio ref.wav \
  --ref_text "Reference transcript."

omnivoice-benchmark-scheduler-packing \
  --test_list results/heterogeneous.jsonl \
  --policies target,target_context \
  --batch_size 12
```

Use the included server sweep helper to run repeatable load tests:

```bash
omnivoice-benchmark-server-sweep \
  --model k2-fsa/OmniVoice \
  --test_list test.jsonl \
  --res_dir results/server_sweep \
  --generation_mode optimized \
  --enforce_output_duration true \
  --client_concurrency_values 4,8,16
```

The benchmark writes per-profile summaries and uses `/v1/scheduler` plus
response headers to capture queue wait, batch size, token costs, and generation
profile data.

Benchmark summaries count a request as successful only when it explicitly
returns a valid, non-empty WAV. HTTP 200 responses with invalid audio, failed
requests, malformed metric headers, and per-request scheduler errors are kept in
`results` as failed rows and excluded from `num_successful`, `audio_s`,
`rtf_wall`, and request metric distributions. This keeps throughput and RTF
numbers tied to usable generated audio instead of transport success alone.

For a lighter in-process benchmark without HTTP, use
`omnivoice-infer-online-batch`. It exposes the same core scheduler controls as
`omnivoice-serve-online-batch`, including packing policy, model duration
estimation, split retry, adaptive memory batch caps, and control-queue
fairness. Keep those flags aligned when comparing CLI benchmark results with
HTTP serving results.

## Final Validation Checklist

Before treating a serving profile as ready, run the same JSONL through these
checks and keep the generated WAVs for listening:

```bash
# 1. Original-compatible baseline
omnivoice-infer-online-batch \
  --model k2-fsa/OmniVoice \
  --test_list validation.jsonl \
  --res_dir results/original_compatible \
  --generation_mode official_compatible \
  --num_step 32 \
  --concurrency 1 \
  --enforce_output_duration true

# 2. Optimized in-process scheduler path
omnivoice-infer-online-batch \
  --model k2-fsa/OmniVoice \
  --test_list validation.jsonl \
  --res_dir results/optimized_online \
  --generation_mode optimized \
  --num_step 32 \
  --concurrency 8 \
  --enforce_output_duration true

# 3. Resident HTTP server and load sweep
omnivoice-benchmark-server-sweep \
  --model k2-fsa/OmniVoice \
  --test_list validation.jsonl \
  --res_dir results/server_sweep \
  --generation_mode optimized \
  --num_step_values 32 \
  --client_concurrency_values 1,4,8,16 \
  --save_wavs true \
  --client_pre_register_voices true \
  --enforce_output_duration true
```

The validation JSONL should include at least: same cloned voice with the same
text at different `duration` values, different cloned voices, Chinese reference
voice cloning for non-Chinese target text, and several concurrent short/long
requests. Compare `num_successful == num_requests`, duration fit, transcription
consistency, and listening quality before changing scheduler limits again.
