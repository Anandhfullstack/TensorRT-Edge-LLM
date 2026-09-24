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
Server lifecycle, HTTP client and reporting for the Whisper server test suite.

Stdlib only, so the suite runs on a bare device with no pip install.
"""

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


class Response:
    """One HTTP response. ``text`` is only meaningful once ``status`` is checked.

    The suite's own first load test misread 503 bodies as transcripts, so
    ``json_text`` refuses to parse anything that is not a 200.
    """

    def __init__(self, status, body, headers):
        self.status = status
        self.body = body
        self.headers = headers

    @property
    def text(self):
        """Transcript from a 200 JSON body. Raises on any other status."""
        if self.status != 200:
            raise ValueError(f"expected 200, got {self.status}: {self.body[:200]}")

        return json.loads(self.body)["text"]

    def header(self, name):
        return self.headers.get(name.lower())


def _multipart(fields, file_field=None, file_path=None):
    """Encode a multipart/form-data body. Returns (content_type, body bytes)."""
    boundary = "----WhisperTest" + uuid.uuid4().hex
    parts = []

    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode())

    if file_field is not None:
        with open(file_path, "rb") as handle:
            payload = handle.read()

        filename = os.path.basename(file_path)
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode())
        parts.append(payload)
        parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())

    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def transcribe(port, wav_path, timeout=120, **form):
    """POST /v1/audio/transcriptions. ``form`` carries language, task, etc."""
    content_type, body = _multipart(form, "file", wav_path)

    return _request(port, "POST", "/v1/audio/transcriptions", body, content_type, timeout)


def transcribe_no_file(port, timeout=30, **form):
    """POST with no ``file`` part, to exercise the missing-field path."""
    content_type, body = _multipart(form)

    return _request(port, "POST", "/v1/audio/transcriptions", body, content_type, timeout)


def get(port, path, timeout=30):
    return _request(port, "GET", path, None, None, timeout)


def _request(port, method, path, body, content_type, timeout):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    try:
        headers = {"Content-Type": content_type} if content_type else {}
        connection.request(method, path, body=body, headers=headers)
        raw = connection.getresponse()

        return Response(raw.status, raw.read().decode("utf-8", "replace"),
                        {k.lower(): v for k, v in raw.getheaders()})
    finally:
        connection.close()


# ----------------------------------------------------------------------------
# Server lifecycle
# ----------------------------------------------------------------------------


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))

        return probe.getsockname()[1]


class WhisperServer:
    """Starts whisper_server on a free port and stops it on exit.

    Startup is ~3-5 s: engine deserialization, one CUDA-graph capture per slot,
    and a warm-up inference per slot.
    """

    def __init__(self, config, slots=1, queue_depth=16, max_new_tokens=128, log_path=None,
                 extra_args=None):
        self.config = config
        self.slots = slots
        self.queue_depth = queue_depth
        self.max_new_tokens = max_new_tokens
        self.extra_args = list(extra_args or [])
        self.port = free_port()
        self.log_path = log_path or os.path.join(config.workdir, f"server_{self.port}.log")
        self.process = None

    def __enter__(self):
        command = [
            self.config.server_bin,
            self.config.encoder_engine,
            self.config.cross_kv_engine,
            self.config.decoder_engine,
            self.config.tokenizer_dir,
            "--port", str(self.port),
            "--slots", str(self.slots),
            "--queue-depth", str(self.queue_depth),
            "--max-new-tokens", str(self.max_new_tokens),
        ] + self.extra_args

        self.log_handle = open(self.log_path, "w")
        self.process = subprocess.Popen(command,
                                        stdout=self.log_handle,
                                        stderr=subprocess.STDOUT,
                                        env=self.config.env,
                                        cwd=self.config.repo_root)

        deadline = time.time() + 180

        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.process.returncode}; see {self.log_path}")

            try:
                if get(self.port, "/health", timeout=2).status == 200:
                    return self
            except OSError:
                time.sleep(0.5)

        raise RuntimeError(f"server did not become ready; see {self.log_path}")

    def __exit__(self, *_):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)

            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

        self.log_handle.close()

        return False

    def proc_status(self, field):
        """Read one /proc/<pid>/status field in kB (e.g. VmHWM, VmRSS)."""
        with open(f"/proc/{self.process.pid}/status") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])

        return None

    def health(self):
        return json.loads(get(self.port, "/health").body)

    def log_tail(self, lines=20):
        try:
            with open(self.log_path) as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError:
            return ""


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _colour(text, code):
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


class CheckResult:

    def __init__(self, name, passed, summary, detail=None, skipped=False):
        self.name = name
        self.passed = passed
        self.summary = summary
        self.detail = detail or []
        self.skipped = skipped


def print_header(title):
    print()
    print(_colour(f"── {title} " + "─" * max(0, 68 - len(title)), DIM))


def print_result(result):
    if result.skipped:
        tag = _colour("SKIP", YELLOW)
    elif result.passed:
        tag = _colour("PASS", GREEN)
    else:
        tag = _colour("FAIL", RED)

    print(f"  [{tag}] {result.name}: {result.summary}")

    for line in result.detail:
        print(f"         {line}")


def print_summary(results):
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = [r for r in results if not r.passed and not r.skipped]
    skipped = sum(1 for r in results if r.skipped)

    print()
    print("=" * 72)
    line = f"{passed} passed"

    if failed:
        line += f", {len(failed)} failed"

    if skipped:
        line += f", {skipped} skipped"

    print(_colour(line, RED if failed else GREEN))

    for result in failed:
        print(_colour(f"  FAILED: {result.name} — {result.summary}", RED))

    print("=" * 72)

    return 1 if failed else 0
