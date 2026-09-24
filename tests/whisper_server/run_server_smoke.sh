#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Build whisper_server, start it, send N transcription requests and report
# latency and RSS per request. Every setting can be overridden from the
# environment, e.g. `PORT=8001 NUM_REQUESTS=50 ./run_server_smoke.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-/usr}"
BUILD_DIR="${BUILD_DIR:-build}"
SKIP_BUILD="${SKIP_BUILD:-0}"

ENCODER_ENGINE="${ENCODER_ENGINE:-whisper_small_native_engine/audio/audio_encoder.engine}"
CROSS_KV_ENGINE="${CROSS_KV_ENGINE:-whisper_small_cross_cache_engine/cross_kv/cross_kv.engine}"
DECODER_ENGINE="${DECODER_ENGINE:-whisper_small_cross_cache_onnx_fixed_cache/decoder/decoder.engine}"
TOKENIZER_DIR="${TOKENIZER_DIR:-whisper_onnx/decoder}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SLOTS="${SLOTS:-2}"
AUDIO_FILE="${AUDIO_FILE:-whisper_runtime_16k.wav}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-120}"
LOG_FILE="${LOG_FILE:-${BUILD_DIR}/whisper_server_smoke.log}"

SERVER_BIN="${BUILD_DIR}/examples/multimodal/whisper_server"
URL="http://${HOST}:${PORT}"

export TRT_PACKAGE_DIR
export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib:/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

for path in "${ENCODER_ENGINE}" "${CROSS_KV_ENGINE}" "${DECODER_ENGINE}" "${TOKENIZER_DIR}" "${AUDIO_FILE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing: ${path}" >&2
        exit 1
    fi
done

if curl -s -o /dev/null "${URL}/health"; then
    echo "Something is already listening on ${URL}; stop it or set PORT." >&2
    exit 1
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo "==> Building whisper_server"
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        cmake -S . -B "${BUILD_DIR}" -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}"
    fi
    cmake --build "${BUILD_DIR}" --target whisper_server -j"$(nproc)"
fi

echo "==> Starting server on ${URL} (slots ${SLOTS}); log: ${LOG_FILE}"
"${SERVER_BIN}" \
    "${ENCODER_ENGINE}" "${CROSS_KV_ENGINE}" "${DECODER_ENGINE}" "${TOKENIZER_DIR}" \
    --host "${HOST}" --port "${PORT}" --slots "${SLOTS}" \
    >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}"
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

rss_kb() {
    awk '/VmRSS/ {print $2}' "/proc/${SERVER_PID}/status"
}

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -s -o /dev/null "${URL}/health"; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited during startup; last log lines:" >&2
        tail -n 30 "${LOG_FILE}" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Server not ready after ${STARTUP_TIMEOUT_S} s" >&2
        exit 1
    fi
    sleep 1
done

baseline_rss=$(rss_kb)
echo "==> Server ready (pid ${SERVER_PID}, RSS ${baseline_rss} kB)"
echo "==> Sending ${NUM_REQUESTS} requests with ${AUDIO_FILE}"
printf '%-5s %-6s %-10s %-12s %-10s %s\n' "req" "http" "time_s" "rss_kB" "delta_kB" "text"

failures=0
body_file="$(mktemp)"
trap 'rm -f "${body_file}"; cleanup' EXIT

for ((i = 1; i <= NUM_REQUESTS; ++i)); do
    read -r http_code time_s < <(curl -s -o "${body_file}" -w '%{http_code} %{time_total}\n' \
        -X POST "${URL}/v1/audio/transcriptions" -F "file=@${AUDIO_FILE}")

    rss=$(rss_kb)
    text=$(tr -d '\n' <"${body_file}" | cut -c1-60)
    printf '%-5s %-6s %-10s %-12s %-10s %s\n' "${i}" "${http_code}" "${time_s}" "${rss}" \
        "$((rss - baseline_rss))" "${text}"

    if [[ "${http_code}" != "200" ]]; then
        failures=$((failures + 1))
    fi
done

final_rss=$(rss_kb)
echo "==> Done: $((NUM_REQUESTS - failures))/${NUM_REQUESTS} succeeded, RSS grew $((final_rss - baseline_rss)) kB"

if ((failures > 0)); then
    exit 1
fi
