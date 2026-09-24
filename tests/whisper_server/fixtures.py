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
Audio fixtures for the Whisper server test suite.

Pure stdlib: the reference clip is an IEEE-float WAV that Python's ``wave``
module cannot read, and depending on ffmpeg would make the suite unrunnable on a
bare device. RIFF parsing and PCM16 writing are hand-rolled here instead.

Fixtures are cached -- rebuilt only when missing.
"""

import math
import os
import struct

SAMPLE_RATE = 16000

# Four segments of the reference clip. Different offsets produce genuinely
# different transcripts, so a cross-slot leak shows up as the wrong text rather
# than as a coincidental match.
CLIP_SEGMENTS = {
    "clip_a": (0.0, 9.9),
    "clip_b": (2.0, 6.0),
    "clip_c": (4.0, 5.0),
    "clip_d": (6.0, 3.5),
}


def read_wav(path):
    """Decode a RIFF/WAVE file to (mono float samples, sample rate).

    Handles format tag 1 (integer PCM, 16/32-bit) and 3 (IEEE float 32-bit),
    which covers both the reference clip and everything this module writes.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    fmt = None
    raw = None
    pos = 12

    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + chunk_size]

        if chunk_id == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            raw = body

        # RIFF chunks are word-aligned.
        pos += 8 + chunk_size + (chunk_size & 1)

    if fmt is None or raw is None:
        raise ValueError(f"{path}: missing fmt or data chunk")

    tag, channels, rate, bits = fmt

    if tag == 3 and bits == 32:
        interleaved = struct.unpack(f"<{len(raw) // 4}f", raw[:len(raw) // 4 * 4])
    elif tag == 1 and bits == 16:
        ints = struct.unpack(f"<{len(raw) // 2}h", raw[:len(raw) // 2 * 2])
        interleaved = [value / 32768.0 for value in ints]
    elif tag == 1 and bits == 32:
        ints = struct.unpack(f"<{len(raw) // 4}i", raw[:len(raw) // 4 * 4])
        interleaved = [value / 2147483648.0 for value in ints]
    else:
        raise ValueError(f"{path}: unsupported format tag={tag} bits={bits}")

    if channels > 1:
        mono = [
            sum(interleaved[i:i + channels]) / channels
            for i in range(0, len(interleaved) - channels + 1, channels)
        ]
    else:
        mono = list(interleaved)

    return mono, rate


def write_wav_pcm16(path, samples, rate=SAMPLE_RATE):
    """Write mono float samples as a 16-bit PCM WAV."""
    clipped = bytearray()

    for value in samples:
        scaled = int(max(-1.0, min(1.0, value)) * 32767.0)
        clipped += struct.pack("<h", scaled)

    byte_rate = rate * 2
    header = (b"RIFF" + struct.pack("<I", 36 + len(clipped)) + b"WAVE" + b"fmt " +
              struct.pack("<IHHIIHH", 16, 1, 1, rate, byte_rate, 2, 16) + b"data" +
              struct.pack("<I", len(clipped)))

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(bytes(clipped))


def write_silence_pcm16(path, seconds, rate=SAMPLE_RATE):
    """Write `seconds` of digital silence without materialising the samples.

    The over-the-cap fixture is ~19 MB; packing 9.6M zeros one at a time takes
    seconds, and every byte of it is zero anyway.
    """
    payload = bytes(2 * int(seconds * rate))
    byte_rate = rate * 2
    header = (b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE" + b"fmt " +
              struct.pack("<IHHIIHH", 16, 1, 1, rate, byte_rate, 2, 16) + b"data" +
              struct.pack("<I", len(payload)))

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(payload)


def _tone(seconds, rate=SAMPLE_RATE, frequency=220.0):
    return [0.25 * math.sin(2 * math.pi * frequency * t / rate) for t in range(int(rate * seconds))]


def build(outdir, source_wav, rebuild=False):
    """Create every fixture the suite needs. Returns a name -> path mapping."""
    os.makedirs(outdir, exist_ok=True)

    paths = {name: os.path.join(outdir, name + ".wav") for name in CLIP_SEGMENTS}
    paths["audio_29s"] = os.path.join(outdir, "audio_29s.wav")
    paths["audio_35s"] = os.path.join(outdir, "audio_35s.wav")
    paths["speech_45s"] = os.path.join(outdir, "speech_45s.wav")
    paths["speech_90s"] = os.path.join(outdir, "speech_90s.wav")
    paths["audio_601s"] = os.path.join(outdir, "audio_601s.wav")
    paths["empty"] = os.path.join(outdir, "empty.wav")
    paths["garbage"] = os.path.join(outdir, "garbage.bin")

    if not rebuild and all(os.path.exists(p) for p in paths.values()):
        return paths

    samples, rate = read_wav(source_wav)

    if rate != SAMPLE_RATE:
        raise ValueError(f"{source_wav}: expected {SAMPLE_RATE} Hz, got {rate}")

    for name, (start, length) in CLIP_SEGMENTS.items():
        begin = int(start * rate)
        end = min(len(samples), begin + int(length * rate))
        write_wav_pcm16(paths[name], samples[begin:end], rate)

    # 29 s must be accepted and 35 s rejected, so the duration gate is shown to
    # be a boundary rather than a blanket refusal. Tone, not speech: only the
    # length matters.
    write_wav_pcm16(paths["audio_29s"], _tone(29), rate)
    write_wav_pcm16(paths["audio_35s"], _tone(35), rate)

    # Long-audio fixtures. Built by rotating through the four segments rather
    # than repeating one sentence: identical repeated speech is indistinguishable
    # from a window overlap, so the stitcher legitimately merges it away and the
    # fixture would measure that artefact instead of the windowing.
    segments = [
        samples[int(start * rate):int((start + length) * rate)]
        for start, length in CLIP_SEGMENTS.values()
    ]
    gap = [0.0] * int(0.4 * rate)

    for name, seconds in (("speech_45s", 45), ("speech_90s", 90)):
        needed = int(seconds * rate)
        track = []
        index = 0

        while len(track) < needed:
            track.extend(segments[index % len(segments)])
            track.extend(gap)
            index += 1

        write_wav_pcm16(paths[name], track[:needed], rate)

    # One second past the shared audio loader's kMaxDecodedSeconds, which
    # refuses the decode outright.
    write_silence_pcm16(paths["audio_601s"], 601, rate)

    open(paths["empty"], "wb").close()

    with open(paths["garbage"], "wb") as handle:
        handle.write(os.urandom(4096))

    return paths
