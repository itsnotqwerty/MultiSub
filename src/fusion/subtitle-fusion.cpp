#include "subtitle-fusion.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string_view>

namespace {

std::string normalize_dialogue_text(std::string text)
{
	auto is_trim_char = [](unsigned char ch) {
		return std::isspace(ch) != 0 || ch == ',' || ch == '.' || ch == '!' || ch == '?' || ch == '"' || ch == '\'' || ch == '`';
	};

	while (!text.empty() && is_trim_char(static_cast<unsigned char>(text.front()))) {
		text.erase(text.begin());
	}

	while (!text.empty() && is_trim_char(static_cast<unsigned char>(text.back()))) {
		text.pop_back();
	}

	std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
	return text;
}

bool looks_like_spurious_dialogue(const std::string &text)
{
	const std::string normalized = normalize_dialogue_text(text);
	if (normalized.empty()) {
		return true;
	}

	static const std::string_view kFillerWords[] = {
		"yeah",
		"okay",
		"ok",
		"yep",
		"sure",
		"right",
		"uh",
		"um",
		"hmm",
		"mhm",
		"mm-hmm",
		"uh-huh",
		"mm hmm",
		"mmhmm",
	};

	for (std::string_view filler : kFillerWords) {
		if (normalized == filler) {
			return true;
		}
	}

	size_t word_count = 0;
	bool in_word = false;
	for (char ch : normalized) {
		const bool is_word_char = std::isalnum(static_cast<unsigned char>(ch)) != 0;
		if (is_word_char && !in_word) {
			++word_count;
			in_word = true;
		} else if (!is_word_char) {
			in_word = false;
		}
	}

	if (word_count <= 1 && normalized.size() < 12) {
		return true;
	}

	return false;
}

size_t count_words(const std::string &text)
{
	size_t word_count = 0;
	bool in_word = false;
	for (char ch : text) {
		const bool is_word_char = std::isalnum(static_cast<unsigned char>(ch)) != 0;
		if (is_word_char && !in_word) {
			++word_count;
			in_word = true;
		} else if (!is_word_char) {
			in_word = false;
		}
	}
	return word_count;
}

bool has_acknowledgement_word(const std::string &text)
{
	static const std::string_view kAckWords[] = {
		"yeah",
		"okay",
		"ok",
		"yep",
		"sure",
		"right",
		"uh",
		"um",
		"hmm",
		"mhm",
		"mm-hmm",
		"uh-huh",
		"mm",
	};

	std::istringstream stream(text);
	std::string token;
	while (stream >> token) {
		const std::string normalized = normalize_dialogue_text(token);
		for (std::string_view word : kAckWords) {
			if (normalized == word) {
				return true;
			}
		}
	}

	return false;
}

bool should_suppress_low_confidence_dialogue(const multisub::AudioResult &result)
{
	const std::string normalized = normalize_dialogue_text(result.transcript);
	if (normalized.empty()) {
		return true;
	}

	if (result.transcript_confidence >= 0.35f) {
		return false;
	}

	if (has_acknowledgement_word(normalized) && count_words(normalized) <= 3) {
		return true;
	}

	return false;
}

} // namespace

namespace multisub {

std::vector<SubtitleEvent> SubtitleFusion::fuse_audio(const std::vector<AudioResult> &results)
{
	std::vector<SubtitleEvent> events;
	events.reserve(results.size() * 2);
	constexpr uint64_t kEnvironmentalCooldownNs = 2200000000ULL;

	for (const AudioResult &result : results) {
		const bool has_dialogue = !result.transcript.empty();
		if (!result.transcript.empty() && !looks_like_spurious_dialogue(result.transcript) && !should_suppress_low_confidence_dialogue(result)) {
			events.push_back(SubtitleEvent{
				.channel = SubtitleChannel::Dialogue,
				.start_ns = result.timestamp_ns,
				.end_ns = result.timestamp_ns + 1500000000ULL,
				.confidence = result.transcript_confidence,
				.text = result.transcript,
			});
		}

		for (const std::string &event : result.environmental_events) {
			if (event.empty()) {
				continue;
			}

			if (event == "Ambient room tone") {
				continue;
			}

			if (result.environmental_confidence < 0.55f) {
				continue;
			}

			if (has_dialogue && event == "Audio activity") {
				continue;
			}

			if (event == last_environmental_text_ &&
			    result.timestamp_ns < last_environmental_ts_ns_ + kEnvironmentalCooldownNs) {
				continue;
			}

			events.push_back(SubtitleEvent{
				.channel = SubtitleChannel::Environmental,
				.start_ns = result.timestamp_ns,
				.end_ns = result.timestamp_ns + 1200000000ULL,
				.confidence = result.environmental_confidence,
				.text = event,
			});

			last_environmental_text_ = event;
			last_environmental_ts_ns_ = result.timestamp_ns;
			break;
		}
	}

	return events;
}

} // namespace multisub
