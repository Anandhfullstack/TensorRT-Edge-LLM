# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Correctness checks for the Whisper transcription server.

Each check takes (config, fixtures) and returns a CheckResult. Checks own their
server instance, so one can be run alone and so slot count and queue depth can
differ per check.

Load-generating checks keep in-flight <= queue depth on purpose: exceeding it is
*correct* shedding, and reading a 503 body as a transcript is what made the
first version of this suite report 22 false failures.
"""

import concurrent.futures
import subprocess
import time

from harness import CheckResult, WhisperServer, get, transcribe, transcribe_no_file

# Checks are registered here; run_tests.sh --list prints this table.
REGISTRY = {}


def check(name, description):

    def wrap(function):
        function.description = description
        REGISTRY[name] = function

        return function

    return wrap


def _parallel(jobs, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: job(), jobs))


# ----------------------------------------------------------------------------


@check("smoke", "/health and /v1/models answer with the expected shape")
def check_smoke(config, fixtures):
    with WhisperServer(config, slots=2) as server:
        health = server.health()
        models = get(server.port, "/v1/models")

        problems = []

        for field in ("status", "slots", "busy", "queue_depth"):
            if field not in health:
                problems.append(f"/health missing '{field}'")

        if health.get("slots") != 2:
            problems.append(f"/health slots={health.get('slots')}, expected 2")

        if models.status != 200 or "whisper" not in models.body:
            problems.append(f"/v1/models returned {models.status}: {models.body[:120]}")

        return CheckResult("smoke", not problems,
                           "endpoints healthy" if not problems else "endpoint problems", problems)


@check("equivalence", "server transcript is byte-identical to the whisper_runtime CLI")
def check_equivalence(config, fixtures):
    if not config.runtime_bin:
        return CheckResult("equivalence", True, "whisper_runtime binary not found", skipped=True)

    completed = subprocess.run([
        config.runtime_bin, config.encoder_engine, config.cross_kv_engine,
        config.decoder_engine, config.tokenizer_dir, config.reference_audio, "128"
    ],
                               capture_output=True,
                               text=True,
                               env=config.env,
                               cwd=config.repo_root,
                               timeout=600)

    lines = completed.stdout.splitlines()

    if " Transcription" not in lines:
        return CheckResult("equivalence", False, "CLI produced no transcript",
                           [completed.stdout[-400:]])

    # The banner rule sits between the heading and the text.
    cli_text = lines[lines.index(" Transcription") + 2]

    with WhisperServer(config, slots=1) as server:
        server_text = transcribe(server.port, config.reference_audio).text

    passed = cli_text == server_text

    return CheckResult(
        "equivalence", passed,
        "byte-identical" if passed else "server and CLI disagree",
        [f"CLI   : {cli_text!r}", f"server: {server_text!r}"] if not passed else
        [f"{len(server_text)} chars: {server_text[:72]!r}"])


@check("slot_reuse", "a slot returns identical output across sequential requests")
def check_slot_reuse(config, fixtures):
    rounds = 20

    with WhisperServer(config, slots=1) as server:
        texts = {transcribe(server.port, config.reference_audio).text for _ in range(rounds)}

    passed = len(texts) == 1

    return CheckResult(
        "slot_reuse", passed,
        f"{rounds} sequential requests -> {len(texts)} distinct result(s)",
        [] if passed else [f"{t!r}" for t in sorted(texts)])


@check("slot_isolation", "concurrent slots never return another request's transcript")
def check_slot_isolation(config, fixtures):
    slots = max(2, config.slots)
    queue_depth = 16
    in_flight = min(12, queue_depth)
    rounds = 15
    clips = ["clip_a", "clip_b", "clip_c", "clip_d"]

    with WhisperServer(config, slots=slots, queue_depth=queue_depth) as server:
        # Baselines one at a time, so each is that clip's uncontended answer.
        baseline = {name: transcribe(server.port, fixtures[name]).text for name in clips}

        if len(set(baseline.values())) != len(clips):
            return CheckResult("slot_isolation", False,
                               "fixture clips do not have distinct transcripts",
                               [f"{k}: {v!r}" for k, v in baseline.items()])

        jobs = [(lambda n=name: (n, transcribe(server.port, fixtures[n])))
                for _ in range(rounds) for name in clips]
        results = _parallel(jobs, in_flight)

    mismatches = []
    shed = 0

    for name, response in results:
        if response.status != 200:
            shed += 1
            continue

        if response.text != baseline[name]:
            mismatches.append(f"{name}: expected {baseline[name]!r}, got {response.text!r}")

    passed = not mismatches and shed == 0

    detail = mismatches[:5]

    if shed:
        detail.append(f"{shed} request(s) shed with 503 - lower in-flight or raise queue depth")

    return CheckResult(
        "slot_isolation", passed,
        f"{len(results) - shed}/{len(results)} correct across {slots} slots, "
        f"{in_flight} in flight", detail)


@check("input_rejection", "bad, empty and over-long uploads are refused with the right status")
def check_input_rejection(config, fixtures):
    with WhisperServer(config, slots=1) as server:
        cases = [
            ("missing 'file' field", 400, lambda: transcribe_no_file(server.port, model="x")),
            ("empty file", 400, lambda: transcribe(server.port, fixtures["empty"])),
            ("undecodable bytes", 400, lambda: transcribe(server.port, fixtures["garbage"])),
            # Since long-audio support, exceeding one window is transcribed
            # across several windows rather than refused.
            ("29 s audio (one window)", 200,
             lambda: transcribe(server.port, fixtures["audio_29s"])),
            ("35 s audio (two windows)", 200,
             lambda: transcribe(server.port, fixtures["audio_35s"])),
            # Past the shared loader's decode cap it cannot be buffered at all.
            ("601 s audio (past the decode cap)", 400,
             lambda: transcribe(server.port, fixtures["audio_601s"])),
            ("bad response_format", 400,
             lambda: transcribe(server.port, config.reference_audio, response_format="srt")),
            ("unknown language", 400,
             lambda: transcribe(server.port, config.reference_audio, language="zz")),
            ("unknown task", 400,
             lambda: transcribe(server.port, config.reference_audio, task="summarize")),
        ]

        detail = []
        failures = 0

        for label, expected, call in cases:
            actual = call().status

            if actual != expected:
                failures += 1
                detail.append(f"{label}: expected {expected}, got {actual}")
            else:
                detail.append(f"{label}: {actual}")

        busy = server.health()["busy"]

        if busy != 0:
            failures += 1
            detail.append(f"busy={busy} after rejections - a refused request held a slot")

    return CheckResult("input_rejection", failures == 0,
                       f"{len(cases) - failures}/{len(cases)} cases correct", detail)


@check("backpressure", "load beyond the queue is shed with 503 + Retry-After, not queued")
def check_backpressure(config, fixtures):
    slots = config.slots
    queue_depth = max(slots, 4)
    flood = queue_depth * 5

    with WhisperServer(config, slots=slots, queue_depth=queue_depth) as server:
        jobs = [(lambda: transcribe(server.port, config.reference_audio)) for _ in range(flood)]
        responses = _parallel(jobs, flood)

        codes = {}

        for response in responses:
            codes[response.status] = codes.get(response.status, 0) + 1

        shed = [r for r in responses if r.status == 503]
        missing_header = [r for r in shed if r.header("Retry-After") is None]

        recovered = server.health()["busy"]

    problems = []

    if not shed:
        problems.append(f"nothing shed at queue depth {queue_depth} under {flood} concurrent "
                        "- is the socket pool larger than the admission cap?")

    if missing_header:
        problems.append(f"{len(missing_header)} of {len(shed)} 503s lacked Retry-After")

    if set(codes) - {200, 503}:
        problems.append(f"unexpected status codes: {sorted(set(codes) - {200, 503})}")

    if recovered != 0:
        problems.append(f"busy={recovered} after the flood drained")

    return CheckResult("backpressure", not problems,
                       f"{flood} concurrent at depth {queue_depth} -> " +
                       ", ".join(f"{n}x {c}" for c, n in sorted(codes.items())), problems)


@check("language_task", "language and task change the forced prompt and are validated")
def check_language_task(config, fixtures):
    with WhisperServer(config, slots=1) as server:
        default = transcribe(server.port, config.reference_audio).text
        english = transcribe(server.port, config.reference_audio, language="en").text
        upper = transcribe(server.port, config.reference_audio, language="EN")
        german = transcribe(server.port, config.reference_audio, language="de").text
        french = transcribe(server.port, config.reference_audio, language="fr").text
        translated = transcribe(server.port,
                                config.reference_audio,
                                language="de",
                                task="translate").text

    problems = []

    if default != english:
        problems.append("language=en differs from the default prompt")

    if upper.status != 200:
        problems.append(f"language=EN should case-fold, got {upper.status}")

    if german == english:
        problems.append("language=de produced the English transcript - prompt not applied")

    if french == english or french == german:
        problems.append("language=fr did not produce a distinct transcript")

    # Whisper's translate task always targets English, so a German-tagged
    # translate should come back looking like the English transcription.
    if translated == german:
        problems.append("task=translate behaved like transcribe")

    return CheckResult("language_task", not problems,
                       "en / de / fr / translate all distinct; validation enforced", [
                           f"en       : {english[:60]!r}",
                           f"de       : {german[:60]!r}",
                           f"fr       : {french[:60]!r}",
                           f"translate: {translated[:60]!r}",
                       ] if not problems else problems)


@check("memory_leak", "resident memory plateaus instead of growing per request")
def check_memory_leak(config, fixtures):
    """Two consecutive windows, and only the second is asserted.

    A server's working set forms over its first few hundred requests -- glibc
    grows a malloc arena per socket thread, upload buffers and per-slot scratch
    reach full size. Measured here that is a one-time ~90 MiB, after which
    VmRSS oscillates within +/-1.5 MiB per 100 requests, in both directions.

    Growth in window 1 therefore proves nothing. A real leak is linear, so it
    would show up in window 2 just as strongly; a plateau would not. That is the
    property this check tests.
    """
    window = config.leak_requests
    in_flight = min(8, config.slots * 2)

    with WhisperServer(config, slots=config.slots, queue_depth=16) as server:
        def burst(count):
            return _parallel([(lambda: transcribe(server.port, fixtures["clip_a"]))
                              for _ in range(count)], in_flight)

        burst(8)

        start_rss = server.proc_status("VmRSS")
        responses = burst(window)
        mid_rss = server.proc_status("VmRSS")
        responses += burst(window)
        end_rss = server.proc_status("VmRSS")
        end_hwm = server.proc_status("VmHWM")

    warm_growth = mid_rss - start_rss
    steady_growth = end_rss - mid_rss

    # Window-2 jitter is ~1.5 MiB per 100 requests; a leak of even 100 kB per
    # request would be 20x this budget at the default window.
    budget = max(8192, mid_rss // 64)

    failed = [r for r in responses if r.status != 200]
    problems = []

    if failed:
        problems.append(f"{len(failed)} request(s) did not return 200")

    if steady_growth > budget:
        problems.append(f"VmRSS still growing {steady_growth} kB in window 2 "
                        f"(budget {budget} kB) - this looks like a real leak")

    return CheckResult(
        "memory_leak", not problems,
        f"2 x {window} requests: warm-up {warm_growth:+d} kB, then {steady_growth:+d} kB "
        f"(budget {budget} kB)", problems or [
            f"VmRSS {start_rss} -> {mid_rss} -> {end_rss} kB, VmHWM {end_hwm} kB",
            "window 2 flat => working-set plateau, not a per-request leak",
        ])


@check("long_audio", "audio past one 30 s window is windowed, seeked and joined")
def check_long_audio(config, fixtures):
    """Long clips are decoded window by window, the next window's start taken
    from the last timestamp the model emitted.

    Text length *is* asserted to grow with duration. That only became a valid
    assertion with timestamp seeking: the previous overlap-stitching mode could
    not tell repeated speech from a duplicated overlap, so it merged real text
    away and a 90 s loop produced less text than a 45 s one. See
    SERVER_HOST.md S18.
    """
    marker = "<|endoftext|>"

    with WhisperServer(config, slots=1, max_new_tokens=220) as server:
        transcribe(server.port, config.reference_audio)  # warm

        begin = time.perf_counter()
        short = transcribe(server.port, config.reference_audio)
        short_elapsed = time.perf_counter() - begin

        mid = transcribe(server.port, fixtures["speech_45s"])

        begin = time.perf_counter()
        long = transcribe(server.port, fixtures["speech_90s"])
        long_elapsed = time.perf_counter() - begin

    problems = []

    for label, response in (("9.94 s", short), ("45 s", mid), ("90 s", long)):
        if response.status != 200:
            problems.append(f"{label}: expected 200, got {response.status}")

    if problems:
        return CheckResult("long_audio", False, "a request failed", problems)

    short_text, mid_text, long_text = short.text, mid.text, long.text

    # The single-window path must stay exactly what it was; that is what
    # `equivalence` pins against the CLI.
    if marker not in short_text:
        problems.append("single-window output lost its end-of-text marker - "
                        "equivalence with the CLI is no longer guaranteed")

    if marker in mid_text or marker in long_text:
        problems.append("joined output contains an interior end-of-text marker")

    if len(mid_text) <= len(short_text):
        problems.append(f"45 s produced {len(mid_text)} chars, not more than one window "
                        f"({len(short_text)}) - windowing may not be running")

    # The regression that timestamp seeking exists to prevent.
    if len(long_text) <= len(mid_text):
        problems.append(f"90 s produced {len(long_text)} chars, not more than 45 s "
                        f"({len(mid_text)}) - speech is being merged away")

    # Elapsed work is content-independent proof that every window decoded.
    ratio = long_elapsed / short_elapsed if short_elapsed > 0 else 0.0

    if ratio < 2.0:
        problems.append(f"90 s took {long_elapsed:.2f} s against {short_elapsed:.2f} s for one "
                        f"window ({ratio:.1f}x) - later windows may not be decoded")

    return CheckResult(
        "long_audio", not problems,
        f"9.94 s -> {len(short_text)} chars, 45 s -> {len(mid_text)}, "
        f"90 s -> {len(long_text)} ({ratio:.1f}x work)", problems or
        [f"45 s: {mid_text[:88]}..."])


@check("long_audio_modes", "timestamp seeking keeps repeated speech that overlap matching drops")
def check_long_audio_modes(config, fixtures):
    """Runs the same repetitive clip through both long-audio strategies.

    The fixtures repeat ~10 s of source speech, which is the exact input
    overlap matching cannot handle: repeated text is indistinguishable from a
    duplicated overlap. Timestamp seeking takes its boundary from the model's
    predicted time instead, so the repetitions survive.
    """
    clip = fixtures["speech_90s"]

    with WhisperServer(config, slots=1, max_new_tokens=220) as server:
        timestamps = transcribe(server.port, clip)

    with WhisperServer(config, slots=1, max_new_tokens=220,
                       extra_args=["--long-audio-mode", "overlap"]) as server:
        overlap = transcribe(server.port, clip)

    problems = []

    for label, response in (("timestamps", timestamps), ("overlap", overlap)):
        if response.status != 200:
            problems.append(f"{label} mode: expected 200, got {response.status}")

    if problems:
        return CheckResult("long_audio_modes", False, "a request failed", problems)

    ts_len = len(timestamps.text)
    ov_len = len(overlap.text)

    if ts_len <= ov_len:
        problems.append(f"timestamp mode produced {ts_len} chars against overlap's {ov_len} - "
                        "timestamp seeking should retain strictly more on repetitive speech")

    return CheckResult("long_audio_modes", not problems,
                       f"90 s repetitive clip: timestamps {ts_len} chars, "
                       f"overlap {ov_len} chars ({ts_len / max(ov_len, 1):.1f}x)", problems)


@check("cli_regression", "whisper_runtime still runs and transcribes after server changes")
def check_cli_regression(config, fixtures):
    if not config.runtime_bin:
        return CheckResult("cli_regression", True, "whisper_runtime binary not found", skipped=True)

    completed = subprocess.run([
        config.runtime_bin, config.encoder_engine, config.cross_kv_engine,
        config.decoder_engine, config.tokenizer_dir, config.reference_audio, "128"
    ],
                               capture_output=True,
                               text=True,
                               env=config.env,
                               cwd=config.repo_root,
                               timeout=600)

    lines = completed.stdout.splitlines()
    ok = completed.returncode == 0 and " Transcription" in lines

    text = lines[lines.index(" Transcription") + 2] if " Transcription" in lines else ""

    return CheckResult("cli_regression", ok,
                       "CLI works" if ok else f"CLI exited {completed.returncode}",
                       [f"{text[:72]!r}"] if ok else [completed.stdout[-400:]])
