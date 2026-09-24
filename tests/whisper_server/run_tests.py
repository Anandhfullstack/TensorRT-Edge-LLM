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
Entry point for the Whisper server test suite. Prefer ./run_tests.sh, which
fills in the default engine paths and LD_LIBRARY_PATH.
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench  # noqa: E402
import checks  # noqa: E402
import fixtures as fixtures_module  # noqa: E402
from harness import CheckResult, print_header, print_result, print_summary  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DEFAULTS = {
    "server_bin": "build/examples/multimodal/whisper_server",
    "runtime_bin": "build/examples/multimodal/whisper_runtime",
    "encoder_engine": "whisper_small_native_engine/audio/audio_encoder.engine",
    "cross_kv_engine": "whisper_small_cross_cache_engine/cross_kv/cross_kv.engine",
    "decoder_engine": "whisper_small_cross_cache_onnx_fixed_cache/decoder/decoder.engine",
    "tokenizer_dir": "whisper_onnx/decoder",
    "reference_audio": "whisper_runtime_16k.wav",
}

ENV_OVERRIDES = {
    "server_bin": "WHISPER_SERVER_BIN",
    "runtime_bin": "WHISPER_RUNTIME_BIN",
    "encoder_engine": "WHISPER_ENCODER_ENGINE",
    "cross_kv_engine": "WHISPER_CROSS_KV_ENGINE",
    "decoder_engine": "WHISPER_DECODER_ENGINE",
    "tokenizer_dir": "WHISPER_TOKENIZER_DIR",
    "reference_audio": "WHISPER_TEST_AUDIO",
}


class Config:

    def __init__(self, args):
        self.repo_root = REPO_ROOT
        self.env = dict(os.environ)

        for key, default in DEFAULTS.items():
            value = os.environ.get(ENV_OVERRIDES[key], default)
            setattr(self, key, value if os.path.isabs(value) else os.path.join(REPO_ROOT, value))

        self.slots = args.slots
        self.leak_requests = args.leak_requests
        self.workdir = args.workdir or os.path.join(REPO_ROOT, "build", "whisper_server_tests")

        os.makedirs(self.workdir, exist_ok=True)

        # whisper_runtime is optional: the two checks that use it skip without it.
        if not os.path.exists(self.runtime_bin):
            self.runtime_bin = None

    def missing(self):
        required = [
            ("server binary", self.server_bin),
            ("encoder engine", self.encoder_engine),
            ("cross-KV engine", self.cross_kv_engine),
            ("decoder engine", self.decoder_engine),
            ("tokenizer dir", self.tokenizer_dir),
            ("reference audio", self.reference_audio),
        ]

        return [f"{label}: {path}" for label, path in required if not os.path.exists(path)]


def main():
    parser = argparse.ArgumentParser(
        description="Whisper transcription server test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Paths come from WHISPER_* environment variables; see README.md.")
    parser.add_argument("--list", action="store_true", help="list the available checks and exit")
    parser.add_argument("--only",
                        metavar="NAME",
                        action="append",
                        help="run only this check (repeatable)")
    parser.add_argument("--skip", metavar="NAME", action="append", help="skip this check")
    parser.add_argument("--slots",
                        type=int,
                        default=4,
                        help="slots for checks that exercise concurrency (default 4)")
    parser.add_argument("--leak-requests",
                        type=int,
                        default=200,
                        help="requests in the memory_leak check (default 200)")
    parser.add_argument("--bench", action="store_true", help="run the throughput benchmark instead")
    parser.add_argument("--bench-slots",
                        default="1,4",
                        help="comma-separated slot counts to benchmark (default 1,4)")
    parser.add_argument("--bench-count", type=int, default=40, help="requests per point")
    parser.add_argument("--bench-reps", type=int, default=2, help="repeats per point")
    parser.add_argument("--workdir", help="where to write fixtures and server logs")
    parser.add_argument("--rebuild-fixtures", action="store_true", help="regenerate audio fixtures")
    args = parser.parse_args()

    if args.list:
        print("Available checks:\n")

        for name, function in checks.REGISTRY.items():
            print(f"  {name:18} {function.description}")

        print("\nRun one with: ./run_tests.sh --only <name>")

        return 0

    config = Config(args)
    missing = config.missing()

    if missing:
        print("Cannot run - these paths do not exist:\n")

        for line in missing:
            print(f"  {line}")

        print("\nBuild with:  cmake --build build --target whisper_server -j$(nproc)")
        print("Override paths with the WHISPER_* environment variables (see README.md).")

        return 2

    print_header("Whisper server test suite")
    print(f"  server   : {os.path.relpath(config.server_bin, REPO_ROOT)}")
    print(f"  slots    : {config.slots}")
    print(f"  workdir  : {config.workdir}")

    fixture_paths = fixtures_module.build(os.path.join(config.workdir, "fixtures"),
                                          config.reference_audio,
                                          rebuild=args.rebuild_fixtures)

    if args.bench:
        bench.run(config,
                  fixture_paths,
                  slots_list=[int(s) for s in args.bench_slots.split(",")],
                  count=args.bench_count,
                  reps=args.bench_reps)

        return 0

    selected = list(checks.REGISTRY)

    if args.only:
        unknown = [name for name in args.only if name not in checks.REGISTRY]

        if unknown:
            print(f"\nUnknown check(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(checks.REGISTRY)}")

            return 2

        selected = [name for name in selected if name in args.only]

    if args.skip:
        selected = [name for name in selected if name not in args.skip]

    results = []

    for name in selected:
        print_header(name)

        try:
            result = checks.REGISTRY[name](config, fixture_paths)
        except Exception as error:  # noqa: BLE001 - a crashed check is a failed check
            result = CheckResult(name, False, f"raised {type(error).__name__}: {error}",
                                 traceback.format_exc().splitlines()[-6:])

        print_result(result)
        results.append(result)

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
