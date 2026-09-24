#!/usr/bin/env bash
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
#
# Whisper transcription server test suite.
#
#   ./run_tests.sh                      # every check
#   ./run_tests.sh --list               # what is available
#   ./run_tests.sh --only slot_isolation
#   ./run_tests.sh --bench              # throughput instead of pass/fail
#
# Engine paths default to the layout in COMMANDS.txt and can be overridden with
# the WHISPER_* variables documented in README.md.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The server links against TensorRT; without this it fails to start and every
# check reports a startup timeout instead of the real cause.
if [[ -n "${TRT_PACKAGE_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

cd "${REPO_ROOT}" || exit 1

exec python3 "${SCRIPT_DIR}/run_tests.py" "$@"
