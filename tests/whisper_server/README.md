# Whisper Server Test Suite

End-to-end checks for `build/examples/multimodal/whisper_server`.

Stdlib Python only — no `pytest`, no `requests`, no `ffmpeg`. It runs on a bare
Jetson with nothing installed. Each check starts its own server on a free port
and stops it afterwards, so runs never collide and a crashed check cannot leave
a process behind.

---

## Quick start

```bash
export TRT_PACKAGE_DIR=/path/to/tensorrt      # needed for LD_LIBRARY_PATH
cmake --build build --target whisper_server -j$(nproc)

./tests/whisper_server/run_tests.sh           # everything
```

```bash
./tests/whisper_server/run_tests.sh --list                    # what's available
./tests/whisper_server/run_tests.sh --only slot_isolation     # one check
./tests/whisper_server/run_tests.sh --only equivalence --only smoke
./tests/whisper_server/run_tests.sh --skip memory_leak        # skip the slow one
./tests/whisper_server/run_tests.sh --slots 1                 # force single slot
./tests/whisper_server/run_tests.sh --bench                   # throughput, not pass/fail
```

Exit code is 0 when everything passed, 1 on any failure, 2 on a setup problem
(missing binary or engine).

---

## The checks

| Name | What it proves | Time |
|---|---|---|
| `smoke` | `/health` and `/v1/models` answer with the expected shape | ~10 s |
| `equivalence` | Server transcript is **byte-identical** to the `whisper_runtime` CLI — the server changed no numerics | ~20 s |
| `slot_reuse` | 20 sequential requests on one slot give identical output, so the per-request cache reset holds | ~20 s |
| `slot_isolation` | 4 distinct clips × 15 rounds, 12 in flight: every response matches *its own* clip's baseline | ~60 s |
| `input_rejection` | 8 malformed/oversize/boundary inputs get the right status, and none occupies a slot | ~25 s |
| `backpressure` | Load past the queue is shed with 503 + `Retry-After`, never queued or hung | ~20 s |
| `language_task` | `language` and `task` actually change the forced prompt; bad values are refused | ~20 s |
| `memory_leak` | Resident memory **plateaus** rather than growing per request | ~3 min |
| `long_audio` | Audio past one 30 s window is windowed, seeked and joined; text grows with duration | ~40 s |
| `long_audio_modes` | Timestamp seeking retains repeated speech that overlap matching drops | ~60 s |
| `cli_regression` | `whisper_runtime` still works after changes to the shared runner classes | ~15 s |

`equivalence` and `cli_regression` skip automatically when `whisper_runtime`
has not been built.

---

## Configuration

Paths default to the layout in `COMMANDS.txt`, resolved relative to the repo
root. Override any of them:

| Variable | Default |
|---|---|
| `WHISPER_SERVER_BIN` | `build/examples/multimodal/whisper_server` |
| `WHISPER_RUNTIME_BIN` | `build/examples/multimodal/whisper_runtime` |
| `WHISPER_ENCODER_ENGINE` | `whisper_small_native_engine/audio/audio_encoder.engine` |
| `WHISPER_CROSS_KV_ENGINE` | `whisper_small_cross_cache_engine/cross_kv/cross_kv.engine` |
| `WHISPER_DECODER_ENGINE` | `whisper_small_cross_cache_onnx_fixed_cache/decoder/decoder.engine` |
| `WHISPER_TOKENIZER_DIR` | `whisper_onnx/decoder` |
| `WHISPER_TEST_AUDIO` | `whisper_runtime_16k.wav` |

Fixtures and server logs land in `build/whisper_server_tests/`. A failing check
prints the server log path; fixtures are cached and rebuilt with
`--rebuild-fixtures`.

---

## Files

| File | Role |
|---|---|
| `run_tests.sh` | Entry point. Sets `LD_LIBRARY_PATH` from `TRT_PACKAGE_DIR`, runs from the repo root. |
| `run_tests.py` | Argument parsing, path resolution, check selection, reporting. |
| `checks.py` | The checks. One function each, registered by name via `@check`. |
| `harness.py` | Server lifecycle, multipart HTTP client, `/proc` sampling, PASS/FAIL output. |
| `fixtures.py` | Audio fixture generation. Hand-rolled RIFF parsing — the reference clip is IEEE-float WAV, which Python's `wave` module cannot read. |
| `bench.py` | Throughput and latency benchmark. Reports numbers, never passes or fails. |

Files are named `check_*`/`run_*` rather than `test_*` so the repo's pytest
collection does not pick them up — this suite is deliberately independent of the
CI pytest tree, which needs `pytest`, remote-runner config and the full
`LLM_SDK_DIR` environment.

---

## Adding a check

```python
@check("my_check", "one line shown by --list")
def check_my_check(config, fixtures):
    with WhisperServer(config, slots=2, queue_depth=16) as server:
        response = transcribe(server.port, fixtures["clip_a"], language="en")

    ok = response.status == 200

    return CheckResult("my_check", ok, "summary line", ["detail", "lines"])
```

---

## Two traps worth knowing

**Check the status code before reading a body.** An early version of this suite
reported 22 false failures because it fired 40 concurrent requests at a queue
depth of 16 and then read `["text"]` out of the 503 error bodies. The server was
correct; the test was not. `Response.text` now raises on any non-200 rather than
letting that happen silently.

**The fixtures are deliberately repetitive.** They are built from ~10 s of
unique source speech, so a 90 s clip repeats it — which is precisely the input
that overlap matching cannot handle and timestamp seeking can. `long_audio_modes`
exists to pin that difference. If you switch the server to
`--long-audio-mode overlap`, expect `long_audio` to fail: that is the mode's
known limitation, documented in `SERVER_HOST.md` §17.5 and §18.

**Keep in-flight ≤ queue depth when measuring correctness.** Exceeding it is
*correct* shedding, and it will look like data loss. `backpressure` is the only
check that deliberately over-subscribes.

---

## Benchmarking

```bash
./tests/whisper_server/run_tests.sh --bench                          # slots 1 and 4
./tests/whisper_server/run_tests.sh --bench --bench-slots 1,2,4,8
./tests/whisper_server/run_tests.sh --bench --bench-count 100 --bench-reps 3
```

Jetson clocks are not pinned (`Performance.md` §2) and GPU stages vary by more
than 2× run to run, so every point is repeated and printed as a range. **Serial
rows (1 in flight) are too noisy to compare between slot counts** — one early
reading of that row suggested 4 slots were *slower* than 1, which repeating it
disproved. Compare the saturated rows.
