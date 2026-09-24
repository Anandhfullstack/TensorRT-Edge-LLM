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

#include <algorithm>
#include <cctype>

namespace trt_edgellm
{
namespace examples
{
namespace whisper
{

namespace
{

constexpr char const* kEndOfTextMarker = "<|endoftext|>";

std::vector<std::string> splitWords(
    std::string const& text)
{
    std::vector<std::string> words;

    std::size_t index = 0;

    while (index < text.size())
    {
        while (index < text.size()
            && static_cast<unsigned char>(text[index]) <= ' ')
        {
            ++index;
        }

        std::size_t const begin = index;

        while (index < text.size()
            && static_cast<unsigned char>(text[index]) > ' ')
        {
            ++index;
        }

        if (index > begin)
        {
            words.push_back(text.substr(begin, index - begin));
        }
    }

    return words;
}

//! Case- and punctuation-insensitive form used only for matching. Bytes with
//! the high bit set are passed through untouched, so UTF-8 sequences in
//! non-English output survive intact.
std::string normalizeWord(
    std::string const& word)
{
    std::string result;

    result.reserve(word.size());

    for (char const character : word)
    {
        auto const byte = static_cast<unsigned char>(character);

        if (byte >= 0x80)
        {
            result.push_back(character);
        }
        else if (std::isalnum(byte) != 0)
        {
            result.push_back(static_cast<char>(std::tolower(byte)));
        }
    }

    return result;
}

std::vector<std::string> normalizeWords(
    std::vector<std::string> const& words)
{
    std::vector<std::string> result;

    result.reserve(words.size());

    for (auto const& word : words)
    {
        result.push_back(normalizeWord(word));
    }

    return result;
}

} // namespace

std::vector<AudioWindow> planWindows(
    std::size_t const totalSamples,
    std::size_t const windowSamples,
    std::size_t overlapSamples)
{
    if (totalSamples == 0 || windowSamples == 0)
    {
        return {};
    }

    if (totalSamples <= windowSamples)
    {
        return {AudioWindow{0, totalSamples}};
    }

    // Cap the overlap at half the window. Beyond that every sample would be
    // decoded three or more times, and an overlap approaching the window size
    // collapses the stride towards one sample and emits a window per sample.
    overlapSamples = std::min(overlapSamples, windowSamples / 2);

    std::size_t const stride = windowSamples - overlapSamples;

    std::vector<AudioWindow> windows;

    for (std::size_t start = 0; start < totalSamples; start += stride)
    {
        std::size_t const count
            = std::min(windowSamples, totalSamples - start);

        windows.push_back(AudioWindow{start, count});

        // This window's full extent already reaches the end of the audio, so
        // any further window would cover only ground it has taken.
        if (start + windowSamples >= totalSamples)
        {
            break;
        }
    }

    return windows;
}

WindowSeek planWindowSeek(
    std::vector<int64_t> const& tokens,
    int64_t const timestampBegin,
    double const precisionSeconds,
    double const windowSeconds)
{
    auto const count = tokens.size();

    auto isTimestamp = [&](std::size_t index)
    { return tokens[index] >= timestampBegin; };

    auto toSeconds = [&](int64_t token)
    { return static_cast<double>(token - timestampBegin) * precisionSeconds; };

    auto clamped = [&](double seconds)
    {
        return (seconds > 0.0 && seconds <= windowSeconds) ? seconds : windowSeconds;
    };

    if (count == 0)
    {
        return WindowSeek{0, windowSeconds};
    }

    // Index of the second timestamp in each adjacent pair.
    std::size_t lastPairEnd = 0;
    bool hasPair = false;

    for (std::size_t index = 1; index < count; ++index)
    {
        if (isTimestamp(index - 1) && isTimestamp(index))
        {
            lastPairEnd = index;
            hasPair = true;
        }
    }

    // "text <ts>" at the tail: the final segment closed, so the window is fully
    // transcribed and the next one starts a whole window later.
    bool const closedEnding
        = count >= 2 && !isTimestamp(count - 2) && isTimestamp(count - 1);

    if (hasPair && !closedEnding)
    {
        // The tail segment ran into the window edge. Keep everything before the
        // pair that opened it and rewind to that point.
        return WindowSeek{lastPairEnd, clamped(toSeconds(tokens[lastPairEnd - 1]))};
    }

    if (closedEnding)
    {
        return WindowSeek{count, windowSeconds};
    }

    // No pair to anchor on: keep the window whole and use its last timestamp,
    // which is as far as the model claimed to have got.
    for (std::size_t index = count; index > 0; --index)
    {
        if (isTimestamp(index - 1))
        {
            return WindowSeek{count, clamped(toSeconds(tokens[index - 1]))};
        }
    }

    return WindowSeek{count, windowSeconds};
}

std::string stripEndOfTextMarker(
    std::string text)
{
    std::string const marker{kEndOfTextMarker};

    while (text.size() >= marker.size()
        && text.compare(text.size() - marker.size(), marker.size(), marker) == 0)
    {
        text.erase(text.size() - marker.size());
    }

    return text;
}

std::size_t duplicatedPrefixWords(
    std::string const& left,
    std::string const& right,
    std::size_t const maxWords,
    std::size_t const maxOffsetWords)
{
    std::vector<std::string> const leftWords = normalizeWords(splitWords(left));
    std::vector<std::string> const rightWords = normalizeWords(splitWords(right));

    if (leftWords.empty() || rightWords.empty())
    {
        return 0;
    }

    std::size_t const longest
        = std::min({maxWords, leftWords.size(), rightWords.size()});

    // Longest run wins; among equal lengths the one nearest the start of
    // `right` wins, since that drops the least text.
    for (std::size_t length = longest; length > 0; --length)
    {
        std::size_t const maxOffset
            = std::min(maxOffsetWords, rightWords.size() - length);

        for (std::size_t offset = 0; offset <= maxOffset; ++offset)
        {
            if (offset > 0 && length < kMinOffsetMatchWords)
            {
                break;
            }

            bool matched = true;
            bool allEmpty = true;

            for (std::size_t index = 0; index < length; ++index)
            {
                std::string const& leftWord
                    = leftWords[leftWords.size() - length + index];
                std::string const& rightWord = rightWords[offset + index];

                if (leftWord != rightWord)
                {
                    matched = false;
                    break;
                }

                if (!leftWord.empty())
                {
                    allEmpty = false;
                }
            }

            // A run of pure punctuation normalizes to empty strings and would
            // match anything; it is not evidence that the windows share text.
            if (matched && !allEmpty)
            {
                return offset + length;
            }
        }
    }

    return 0;
}

void appendWindowText(
    std::string& transcript,
    std::string const& windowText,
    std::size_t const maxOverlapWords)
{
    std::string const cleaned = stripEndOfTextMarker(windowText);

    std::vector<std::string> const words = splitWords(cleaned);

    if (words.empty())
    {
        return;
    }

    if (transcript.empty())
    {
        transcript = cleaned;

        return;
    }

    std::size_t const shared = duplicatedPrefixWords(
        transcript, cleaned, maxOverlapWords, kDefaultMaxOffsetWords);

    for (std::size_t index = shared; index < words.size(); ++index)
    {
        transcript.push_back(' ');
        transcript.append(words[index]);
    }
}

} // namespace whisper
} // namespace examples
} // namespace trt_edgellm
