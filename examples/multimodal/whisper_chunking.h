/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace examples
{
namespace whisper
{

//! Default overlap between consecutive windows. Long enough to contain a word
//! straddling a boundary plus enough context either side for the two windows to
//! agree on it, short enough that the redundant decode cost stays near 17%.
constexpr double kDefaultOverlapSeconds = 5.0;

//! Fast speech, in words per second. Used to size the overlap search from the
//! overlap duration: a run longer than the overlap could possibly contain is
//! not the overlap, it is genuinely repeated speech, and merging it would
//! delete real text.
constexpr double kMaxWordsPerSecond = 4.0;

//! Words of overlap the stitcher will look for, for a given overlap duration.
//! Floors at 8 so a very short overlap can still match something.
inline std::size_t maxOverlapWordsFor(
    double overlapSeconds)
{
    auto const scaled = static_cast<std::size_t>(overlapSeconds * kMaxWordsPerSecond);

    return scaled < 8U ? 8U : scaled;
}

//! How far into the next window the shared run may begin, to absorb a window
//! pair disagreeing on the first word or two of the region they share.
constexpr std::size_t kDefaultMaxOffsetWords = 8;

//! Shortest run accepted when it starts at a non-zero offset. One or two words
//! in common are not evidence of overlap, and acting on them would delete real
//! text from the next window.
constexpr std::size_t kMinOffsetMatchWords = 3;

//! One analysis window over a recording longer than a single Whisper chunk.
//!
//! ``sampleCount`` may be shorter than the window for the final window; the
//! audio processor zero-pads it to the full chunk, exactly as it does for a
//! short standalone clip.
struct AudioWindow
{
    std::size_t startSample;
    std::size_t sampleCount;
};

//! Split ``totalSamples`` into windows of ``windowSamples`` advancing by
//! ``windowSamples - overlapSamples``.
//!
//! Audio that already fits yields exactly one window covering all of it, so the
//! single-window path stays bit-for-bit what it was before chunking existed.
//! Windowing stops as soon as a window's full extent reaches the end, so no
//! window is emitted that an earlier one already covers.
//!
//! \param overlapSamples Clamped to half ``windowSamples``: a larger overlap
//!        decodes every sample three or more times, and one approaching the
//!        window size would emit a window per sample.
std::vector<AudioWindow> planWindows(
    std::size_t totalSamples,
    std::size_t windowSamples,
    std::size_t overlapSamples);

//! How much of a window's output is final, and where the next window starts.
struct WindowSeek
{
    //! Tokens from the start of the window whose text is settled. Anything
    //! after this belongs to a segment the window cut short; the next window
    //! re-decodes that audio, so emitting it here would duplicate it.
    std::size_t keepTokens;

    //! Seconds to advance the read position. Always > 0 so the seek cannot
    //! stall, and never more than one window so no audio is skipped unheard.
    double advanceSeconds;
};

//! Decide what to keep and where to seek from one window's tokens.
//!
//! This is Whisper's own long-form strategy, mirroring ``transcribe.py``. The
//! model brackets each segment with timestamp tokens, so a *pair* of adjacent
//! timestamps marks one segment ending and the next beginning.
//!
//!   - Ending in ``text <ts>`` means the last segment closed cleanly: everything
//!     is final and the next window starts a full window later.
//!   - Ending otherwise, with at least one adjacent pair, means the last
//!     segment was cut off by the window edge. Text up to the final pair is
//!     kept and the seek returns to where that segment began.
//!   - With no adjacent pair at all there is no segment structure to trust, so
//!     the whole window is kept and the seek falls back to its last timestamp.
//!
//! Because the boundary comes from the model's predicted *time*, no comparison
//! of window texts is needed -- which is what lets genuinely repeated speech
//! survive, where overlap matching cannot tell it from a duplicated overlap.
//!
//! \param tokens Token ids from one window, timestamps included.
//! \param timestampBegin Id of ``<|0.00|>``.
//! \param precisionSeconds Seconds per timestamp id (0.02 for Whisper).
//! \param windowSeconds Full window length.
WindowSeek planWindowSeek(
    std::vector<int64_t> const& tokens,
    int64_t timestampBegin,
    double precisionSeconds,
    double windowSeconds);

//! Drop a trailing ``<|endoftext|>`` marker, if present.
//!
//! The tokenizer emits this despite ``skipSpecialTokens``; harmless at the end
//! of a single transcript, but stitching windows would bury one mid-sentence.
std::string stripEndOfTextMarker(
    std::string text);

//! Leading words of \p right that restate the tail of \p left.
//!
//! Comparison ignores ASCII case and punctuation: the two windows decode the
//! shared audio independently and routinely punctuate and capitalise it
//! differently.
//!
//! They also sometimes disagree on the first word or two of the shared region
//! ("Sea waves" against "The waves" on real output), which defeats a strict
//! prefix match. The run is therefore allowed to begin up to
//! \p maxOffsetWords into \p right, and everything up to and including it is
//! reported as duplicated. A run found at a non-zero offset must be at least
//! ``kMinOffsetMatchWords`` long, so a single common word cannot cause real
//! text to be dropped.
//!
//! \return Words of \p right to skip, or 0 when the windows share no boundary
//!         text.
std::size_t duplicatedPrefixWords(
    std::string const& left,
    std::string const& right,
    std::size_t maxWords,
    std::size_t maxOffsetWords);

//! Append one window's text to \p transcript, dropping the words the overlap
//! duplicated. Falls back to plain concatenation when no overlap is detected.
void appendWindowText(
    std::string& transcript,
    std::string const& windowText,
    std::size_t maxOverlapWords);

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm
