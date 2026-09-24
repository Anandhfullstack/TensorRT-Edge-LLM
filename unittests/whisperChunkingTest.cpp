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

#include "whisper_chunking.h"

#include <gtest/gtest.h>

using namespace trt_edgellm::examples::whisper;

namespace
{

constexpr std::size_t kRate = 16000;
constexpr std::size_t kWindow = 30 * kRate;
constexpr std::size_t kOverlap = 5 * kRate;

std::size_t seconds(double value)
{
    return static_cast<std::size_t>(value * kRate);
}

} // namespace

// ============================================================
// planWindows
// ============================================================

TEST(WhisperChunkingTest, EmptyAudioYieldsNoWindows)
{
    EXPECT_TRUE(planWindows(0, kWindow, kOverlap).empty());
}

TEST(WhisperChunkingTest, ShortAudioYieldsOneWindowCoveringEverything)
{
    auto const windows = planWindows(seconds(9.94), kWindow, kOverlap);

    ASSERT_EQ(windows.size(), 1U);
    EXPECT_EQ(windows[0].startSample, 0U);
    EXPECT_EQ(windows[0].sampleCount, seconds(9.94));
}

//! The single-window path is what keeps short-clip output bit-for-bit identical
//! to the pre-chunking server, so an exactly-full window must not split.
TEST(WhisperChunkingTest, ExactlyOneWindowDoesNotSplit)
{
    auto const windows = planWindows(kWindow, kWindow, kOverlap);

    ASSERT_EQ(windows.size(), 1U);
    EXPECT_EQ(windows[0].sampleCount, kWindow);
}

TEST(WhisperChunkingTest, LongAudioAdvancesByStride)
{
    auto const windows = planWindows(seconds(60), kWindow, kOverlap);

    ASSERT_EQ(windows.size(), 3U);
    EXPECT_EQ(windows[0].startSample, 0U);
    EXPECT_EQ(windows[1].startSample, seconds(25));
    EXPECT_EQ(windows[2].startSample, seconds(50));
    EXPECT_EQ(windows[2].sampleCount, seconds(10));
}

TEST(WhisperChunkingTest, WindowsCoverEverySample)
{
    std::size_t const total = seconds(137);
    auto const windows = planWindows(total, kWindow, kOverlap);

    ASSERT_FALSE(windows.empty());
    EXPECT_EQ(windows.front().startSample, 0U);

    for (std::size_t i = 1; i < windows.size(); ++i)
    {
        // No gap: each window starts no later than the previous window's end.
        EXPECT_LE(windows[i].startSample,
            windows[i - 1].startSample + windows[i - 1].sampleCount);
    }

    auto const& last = windows.back();
    EXPECT_EQ(last.startSample + last.sampleCount, total);
}

TEST(WhisperChunkingTest, NoWindowIsFullyCoveredByItsPredecessor)
{
    for (double length : {30.5, 31.0, 45.0, 55.0, 60.0, 75.0})
    {
        auto const windows = planWindows(seconds(length), kWindow, kOverlap);

        for (std::size_t i = 1; i < windows.size(); ++i)
        {
            EXPECT_GT(windows[i].startSample + windows[i].sampleCount,
                windows[i - 1].startSample + windows[i - 1].sampleCount)
                << "length=" << length << " window=" << i;
        }
    }
}

TEST(WhisperChunkingTest, ZeroOverlapStillTiles)
{
    auto const windows = planWindows(seconds(90), kWindow, 0);

    ASSERT_EQ(windows.size(), 3U);
    EXPECT_EQ(windows[1].startSample, kWindow);
    EXPECT_EQ(windows[2].startSample, 2 * kWindow);
}

//! An overlap at or above the window size must not collapse the stride. Before
//! the clamp was bounded to half a window this produced 960 001 windows for 90
//! seconds of audio.
TEST(WhisperChunkingTest, OverlapAtOrAboveWindowIsClamped)
{
    auto const windows = planWindows(seconds(90), kWindow, kWindow);

    // Stride floors at half a window: 0, 15, 30, 45, 60 s.
    ASSERT_EQ(windows.size(), 5U);
    EXPECT_EQ(windows[1].startSample, kWindow / 2);
}

TEST(WhisperChunkingTest, DefaultOverlapIsBelowTheClamp)
{
    // The clamp must not perturb the configured default.
    auto const clamped = planWindows(seconds(60), kWindow, kOverlap);
    auto const explicitStride = planWindows(seconds(60), kWindow, 5 * kRate);

    ASSERT_EQ(clamped.size(), explicitStride.size());
    EXPECT_EQ(clamped[1].startSample, explicitStride[1].startSample);
}

// ============================================================
// stripEndOfTextMarker
// ============================================================

TEST(WhisperChunkingTest, StripsTrailingEndOfTextMarker)
{
    EXPECT_EQ(stripEndOfTextMarker(" hello there<|endoftext|>"), " hello there");
    EXPECT_EQ(stripEndOfTextMarker(" hello there"), " hello there");
    EXPECT_EQ(stripEndOfTextMarker("<|endoftext|>"), "");
}

TEST(WhisperChunkingTest, LeavesInteriorMarkerAlone)
{
    EXPECT_EQ(stripEndOfTextMarker("a<|endoftext|>b"), "a<|endoftext|>b");
}

// ============================================================
// duplicatedPrefixWords
// ============================================================

TEST(WhisperChunkingTest, FindsSharedBoundaryRun)
{
    EXPECT_EQ(duplicatedPrefixWords("the cat sat on the mat", "on the mat is flat", 32, 0), 3U);
}

TEST(WhisperChunkingTest, ReturnsZeroWhenNothingShared)
{
    EXPECT_EQ(duplicatedPrefixWords("alpha beta gamma", "delta epsilon", 32, 0), 0U);
}

//! Whisper punctuates and capitalises the shared region differently between
//! windows, which is exactly why matching is normalized.
TEST(WhisperChunkingTest, IgnoresCaseAndPunctuation)
{
    EXPECT_EQ(duplicatedPrefixWords("passing through the water.", "Through the water, most are", 32, 0), 3U);
}

//! maxWords bounds how far back the search looks; it does not truncate a
//! match. A suffix/prefix match cannot be partial, so a true overlap longer
//! than the bound is simply not found.
TEST(WhisperChunkingTest, MaxWordsBoundsTheSearch)
{
    EXPECT_EQ(duplicatedPrefixWords("x y a b", "a b z w", 4, 0), 2U);
    EXPECT_EQ(duplicatedPrefixWords("x y a b", "a b z w", 1, 0), 0U);
}

TEST(WhisperChunkingTest, PunctuationOnlyRunIsNotAMatch)
{
    // Both normalize to empty; treating that as an overlap would delete text.
    EXPECT_EQ(duplicatedPrefixWords("hello ---", "--- world", 32, 0), 0U);
}

TEST(WhisperChunkingTest, HandlesEmptyInputs)
{
    EXPECT_EQ(duplicatedPrefixWords("", "anything", 32, 0), 0U);
    EXPECT_EQ(duplicatedPrefixWords("anything", "", 32, 0), 0U);
}

// ============================================================
// appendWindowText
// ============================================================

TEST(WhisperChunkingTest, FirstWindowIsTakenVerbatim)
{
    std::string transcript;
    appendWindowText(transcript, " Sea waves are powerful<|endoftext|>", 32);

    EXPECT_EQ(transcript, " Sea waves are powerful");
}

TEST(WhisperChunkingTest, MergesOverlapBetweenWindows)
{
    std::string transcript;
    appendWindowText(transcript, " caused by energy passing through the water", 32);
    appendWindowText(transcript, " through the water most are generated by wind", 32);

    EXPECT_EQ(transcript, " caused by energy passing through the water most are generated by wind");
}

TEST(WhisperChunkingTest, ConcatenatesWhenNoOverlapFound)
{
    std::string transcript;
    appendWindowText(transcript, " first part", 32);
    appendWindowText(transcript, " unrelated second part", 32);

    EXPECT_EQ(transcript, " first part unrelated second part");
}

TEST(WhisperChunkingTest, SkipsEmptyWindow)
{
    std::string transcript;
    appendWindowText(transcript, " only text", 32);
    appendWindowText(transcript, "<|endoftext|>", 32);

    EXPECT_EQ(transcript, " only text");
}

TEST(WhisperChunkingTest, FullyDuplicateWindowAddsNothing)
{
    std::string transcript;
    appendWindowText(transcript, " the same words", 32);
    appendWindowText(transcript, " the same words", 32);

    EXPECT_EQ(transcript, " the same words");
}

TEST(WhisperChunkingTest, PreservesNonAsciiText)
{
    std::string transcript;
    appendWindowText(transcript, " Die Wälder sind", 32);
    appendWindowText(transcript, " Wälder sind stärker", 32);

    EXPECT_EQ(transcript, " Die Wälder sind stärker");
}

//! Real failure from a 60 s clip: the two windows disagreed on the first word
//! of the region they share ("Sea waves" / "The waves"), so a strict prefix
//! match found nothing and the sentence was emitted twice.
TEST(WhisperChunkingTest, MatchesWhenWindowsDisagreeOnTheFirstWord)
{
    std::string const left = "energy passing through the water. Sea waves are "
                             "powerful disturbances on the ocean surface.";
    std::string const right = "The waves are powerful disturbances on the ocean "
                              "surface caused by energy.";

    // Skips "The waves are powerful disturbances on the ocean surface" = 9 words.
    EXPECT_EQ(duplicatedPrefixWords(left, right, 32, 8), 9U);
    EXPECT_EQ(duplicatedPrefixWords(left, right, 32, 0), 0U);
}

TEST(WhisperChunkingTest, ShortRunAtAnOffsetIsRejected)
{
    // "the" alone is not evidence of overlap; acting on it would delete text.
    EXPECT_EQ(duplicatedPrefixWords("ends with the", "brand new words the end", 32, 8), 0U);
}

TEST(WhisperChunkingTest, PrefersTheLongestRun)
{
    // A 1-word match sits at offset 0 and a 3-word match at offset 1; the
    // longer run must win even though it starts later.
    EXPECT_EQ(duplicatedPrefixWords("x alpha beta gamma", "gamma alpha beta gamma delta", 32, 8),
        4U);
}

TEST(WhisperChunkingTest, StitchesWindowsThatDisagreeOnTheBoundaryWord)
{
    std::string transcript;
    appendWindowText(transcript, " Sea waves are powerful disturbances on the ocean surface.", 32);
    appendWindowText(transcript, " The waves are powerful disturbances on the ocean surface "
                                 "caused by energy.", 32);

    EXPECT_EQ(transcript,
        " Sea waves are powerful disturbances on the ocean surface. caused by energy.");
}

//! The search bound must track the overlap: a run longer than the overlap can
//! hold is repeated speech, not the shared region, and merging it deletes text.
TEST(WhisperChunkingTest, OverlapWordBoundTracksOverlapDuration)
{
    EXPECT_EQ(maxOverlapWordsFor(5.0), 20U);
    EXPECT_EQ(maxOverlapWordsFor(10.0), 40U);
    EXPECT_EQ(maxOverlapWordsFor(0.0), 8U);
    EXPECT_EQ(maxOverlapWordsFor(1.0), 8U);
}

// ============================================================
// planWindowSeek
// ============================================================

namespace
{

constexpr int64_t kTsBegin = 50364;
constexpr double kTsPrecision = 0.02;
constexpr double kWindowSeconds = 30.0;

//! Token id for a timestamp at `seconds`.
int64_t ts(double seconds)
{
    return kTsBegin + static_cast<int64_t>(seconds / kTsPrecision);
}

WindowSeek seekOf(std::vector<int64_t> const& tokens)
{
    return planWindowSeek(tokens, kTsBegin, kTsPrecision, kWindowSeconds);
}

} // namespace

//! Ending in "text <ts>" means the final segment closed cleanly, so the whole
//! window is final and the next one starts a full window later. Matches
//! openai/whisper transcribe.py's `single_timestamp_ending` branch.
TEST(WhisperChunkingTest, ClosedEndingKeepsEverythingAndAdvancesAFullWindow)
{
    std::vector<int64_t> const tokens
        = {ts(0.0), 100, 200, ts(24.82), ts(24.82), 300, ts(28.64)};

    auto const seek = seekOf(tokens);

    EXPECT_EQ(seek.keepTokens, tokens.size());
    EXPECT_DOUBLE_EQ(seek.advanceSeconds, kWindowSeconds);
}

//! The case that matters: the window edge cut a segment off mid-speech. The
//! incomplete tail must be dropped, not emitted, because the next window
//! re-decodes that audio -- emitting both is how text gets duplicated.
TEST(WhisperChunkingTest, CutOffTailIsDroppedAndSeekRewindsToItsStart)
{
    std::vector<int64_t> const tokens
        = {ts(0.0), 100, 200, ts(24.82), ts(24.82), 300, 400};

    auto const seek = seekOf(tokens);

    EXPECT_EQ(seek.keepTokens, 4U);
    EXPECT_NEAR(seek.advanceSeconds, 24.82, 1e-6);
}

TEST(WhisperChunkingTest, UsesTheLastPairNotAnEarlierOne)
{
    std::vector<int64_t> const tokens
        = {ts(0.0), 100, ts(10.0), ts(10.0), 200, ts(22.0), ts(22.0), 300, 400};

    auto const seek = seekOf(tokens);

    EXPECT_EQ(seek.keepTokens, 6U);
    EXPECT_NEAR(seek.advanceSeconds, 22.0, 1e-6);
}

//! No pair to anchor on: keep the window and fall back to its last timestamp.
TEST(WhisperChunkingTest, FallsBackToTheLastTimestampWithoutAPair)
{
    std::vector<int64_t> const tokens = {ts(0.0), 100, 200, ts(17.5), 300, 400};

    auto const seek = seekOf(tokens);

    EXPECT_EQ(seek.keepTokens, tokens.size());
    EXPECT_NEAR(seek.advanceSeconds, 17.5, 1e-6);
}

TEST(WhisperChunkingTest, AdvancesAFullWindowWithoutTimestamps)
{
    EXPECT_DOUBLE_EQ(seekOf({100, 200, 300}).advanceSeconds, kWindowSeconds);
    EXPECT_DOUBLE_EQ(seekOf({}).advanceSeconds, kWindowSeconds);
    EXPECT_EQ(seekOf({}).keepTokens, 0U);
}

//! A window whose only timestamp is <|0.00|> would otherwise advance by zero
//! and re-decode the same audio forever.
TEST(WhisperChunkingTest, NeverAdvancesByZero)
{
    EXPECT_GT(seekOf({ts(0.0)}).advanceSeconds, 0.0);
    EXPECT_GT(seekOf({ts(0.0), 100, 200}).advanceSeconds, 0.0);
    EXPECT_GT(seekOf({ts(0.0), ts(0.0), 100}).advanceSeconds, 0.0);
}

TEST(WhisperChunkingTest, ClampsAdvanceToTheWindow)
{
    // A malformed timestamp past the window must not skip unheard audio.
    EXPECT_DOUBLE_EQ(seekOf({ts(0.0), 100, kTsBegin + 5000}).advanceSeconds, kWindowSeconds);
}

TEST(WhisperChunkingTest, KeptPrefixNeverExceedsTheTokenCount)
{
    std::vector<std::vector<int64_t>> const cases = {
        {ts(0.0)},
        {ts(0.0), ts(0.0)},
        {100},
        {ts(0.0), 100, ts(5.0), ts(5.0)},
        {ts(0.0), 100, ts(5.0), ts(5.0), 200},
    };

    for (auto const& tokens : cases)
    {
        auto const seek = seekOf(tokens);

        EXPECT_LE(seek.keepTokens, tokens.size());
        EXPECT_GT(seek.advanceSeconds, 0.0);
        EXPECT_LE(seek.advanceSeconds, kWindowSeconds);
    }
}
