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
Throughput and latency benchmark. Reports numbers; it never passes or fails.

Jetson clocks are not pinned (see Performance.md S2), and GPU-dependent stages
vary by more than 2x run to run. Every point is therefore repeated and reported
as a range. Serial (one in flight) figures in particular are too noisy to
compare between server configurations -- use the saturated rows.
"""

import concurrent.futures
import statistics
import time

from harness import WhisperServer, transcribe


def _run(server, audio, count, in_flight):
    def one(_):
        start = time.perf_counter()
        response = transcribe(server.port, audio)

        return time.perf_counter() - start, response.status

    with concurrent.futures.ThreadPoolExecutor(max_workers=in_flight) as pool:
        list(pool.map(one, range(in_flight)))  # warm

        start = time.perf_counter()
        results = list(pool.map(one, range(count)))
        wall = time.perf_counter() - start

    latencies = sorted(latency for latency, status in results if status == 200)
    shed = sum(1 for _, status in results if status != 200)

    return {
        "req_per_s": len(latencies) / wall if wall else 0.0,
        "median": statistics.median(latencies) if latencies else 0.0,
        "p95": latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        "shed": shed,
    }


def run(config, fixtures, slots_list=(1, 4), in_flight_list=(1, 4, 8), count=40, reps=2):
    audio = config.reference_audio

    print()
    print(f"Benchmark: {count} requests per point, {reps} reps, audio={audio}")
    print("Clocks are unpinned; serial rows are noisy by design. See Performance.md S2.")
    print()
    print(f"{'slots':>6} {'in flight':>10} {'req/s (range)':>22} {'median s':>10} {'p95 s':>8} {'shed':>5}")
    print("-" * 68)

    for slots in slots_list:
        queue_depth = max(16, max(in_flight_list))

        with WhisperServer(config, slots=slots, queue_depth=queue_depth) as server:
            for in_flight in in_flight_list:
                points = [_run(server, audio, count, in_flight) for _ in range(reps)]

                rates = [p["req_per_s"] for p in points]
                rate_text = (f"{min(rates):.2f}" if reps == 1 else
                             f"{min(rates):.2f} - {max(rates):.2f}")

                best = max(points, key=lambda p: p["req_per_s"])

                print(f"{slots:>6} {in_flight:>10} {rate_text:>22} "
                      f"{best['median']:>10.3f} {best['p95']:>8.3f} "
                      f"{sum(p['shed'] for p in points):>5}")

    print()
    print("Peak throughput is the meaningful comparison between slot counts;")
    print("latency growth at fixed slots shows requests queueing behind the pool.")
