#include "subtitle-fusion.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
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

bool should_suppress_low_confidence_dialogue(std::string_view transcript, float confidence)
{
	const std::string normalized = normalize_dialogue_text(std::string{transcript});
	if (normalized.empty()) {
		return true;
	}

	const size_t word_count = count_words(normalized);
	if (word_count <= 3) {
		if (confidence < 0.70f) {
			return true;
		}
	} else if (confidence < 0.50f) {
		return true;
	}

	if (has_acknowledgement_word(normalized) && word_count <= 3 && confidence < 0.80f) {
		return true;
	}

	return false;
}

bool overlaps_in_time(const multisub::TimelineEvent &lhs, const multisub::TimelineEvent &rhs, uint64_t slack_ns)
{
	if (lhs.end_ns + slack_ns < rhs.start_ns) {
		return false;
	}
	if (rhs.end_ns + slack_ns < lhs.start_ns) {
		return false;
	}
	return true;
}

bool text_is_compatible(std::string_view lhs, std::string_view rhs)
{
	const std::string left = normalize_dialogue_text(std::string{lhs});
	const std::string right = normalize_dialogue_text(std::string{rhs});
	if (left.empty() || right.empty()) {
		return false;
	}
	if (left == right) {
		return true;
	}

	if (left.size() >= 4 && right.find(left) != std::string::npos) {
		return true;
	}

	if (right.size() >= 4 && left.find(right) != std::string::npos) {
		return true;
	}

	return false;
}

float clamp_confidence(float value)
{
	return std::clamp(value, 0.0f, 0.99f);
}

} // namespace

namespace multisub {

std::vector<SubtitleEvent> SubtitleFusion::fuse_audio(const std::vector<AudioResult> &results)
{
	ingest_timeline(convert_audio_results(results));
	return choreograph_pending();
}

std::vector<SubtitleEvent> SubtitleFusion::fuse_timeline(const std::vector<TimelineEvent> &events)
{
	ingest_timeline(events);
	return choreograph_pending();
}

std::vector<TimelineEvent> SubtitleFusion::convert_audio_results(const std::vector<AudioResult> &results)
{
	std::vector<TimelineEvent> timeline_events;
	timeline_events.reserve(results.size() * 2);

	for (const AudioResult &result : results) {
		const bool has_dialogue = !result.transcript.empty();
		if (!result.transcript.empty()) {
			timeline_events.push_back(TimelineEvent{
				.source = TimelineSource::AudioAsr,
				.channel = SubtitleChannel::Dialogue,
				.observed_ns = result.timestamp_ns,
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

			timeline_events.push_back(TimelineEvent{
				.source = TimelineSource::AudioEnvironmental,
				.channel = SubtitleChannel::Environmental,
				.observed_ns = result.timestamp_ns,
				.start_ns = result.timestamp_ns,
				.end_ns = result.timestamp_ns + 1200000000ULL,
				.confidence = result.environmental_confidence,
				.text = event,
			});

			break;
		}
	}

	return timeline_events;
}

void SubtitleFusion::ingest_timeline(const std::vector<TimelineEvent> &events)
{
	if (events.empty()) {
		return;
	}

	timeline_.insert(timeline_.end(), events.begin(), events.end());
	pending_timeline_.insert(pending_timeline_.end(), events.begin(), events.end());

	constexpr size_t kTimelineRetentionLimit = 2048;
	if (timeline_.size() > kTimelineRetentionLimit) {
		const size_t trim_count = timeline_.size() - kTimelineRetentionLimit;
		timeline_.erase(timeline_.begin(), timeline_.begin() + static_cast<std::vector<TimelineEvent>::difference_type>(trim_count));
	}
}

std::vector<SubtitleEvent> SubtitleFusion::choreograph_pending()
{
	if (pending_timeline_.empty()) {
		return {};
	}

	std::vector<TimelineEvent> window = std::move(pending_timeline_);
	pending_timeline_.clear();

	std::stable_sort(window.begin(), window.end(), [](const TimelineEvent &lhs, const TimelineEvent &rhs) {
		if (lhs.start_ns != rhs.start_ns) {
			return lhs.start_ns < rhs.start_ns;
		}
		return lhs.observed_ns < rhs.observed_ns;
	});

	std::vector<SubtitleEvent> events;
	events.reserve(window.size());
	std::vector<bool> consumed(window.size(), false);
	constexpr uint64_t kDialogueAlignmentSlackNs = 400000000ULL;
	constexpr uint64_t kDialogueRepeatCooldownNs = 2800000000ULL;
	constexpr uint64_t kEnvironmentalCooldownNs = 2200000000ULL;

	for (size_t i = 0; i < window.size(); ++i) {
		if (consumed[i]) {
			continue;
		}

		const TimelineEvent &base = window[i];
		if (base.channel == SubtitleChannel::Dialogue) {
			uint64_t fused_start = base.start_ns;
			uint64_t fused_end = base.end_ns;
			float fused_confidence = base.confidence;
			std::string fused_text = base.text;
			std::set<TimelineSource> corroborated_sources;
			corroborated_sources.insert(base.source);
			consumed[i] = true;

			for (size_t j = i + 1; j < window.size(); ++j) {
				if (consumed[j]) {
					continue;
				}

				const TimelineEvent &candidate = window[j];
				if (candidate.channel != SubtitleChannel::Dialogue) {
					continue;
				}

				if (!overlaps_in_time(base, candidate, kDialogueAlignmentSlackNs)) {
					continue;
				}

				if (!text_is_compatible(base.text, candidate.text)) {
					continue;
				}

				corroborated_sources.insert(candidate.source);
				fused_start = std::min(fused_start, candidate.start_ns);
				fused_end = std::max(fused_end, candidate.end_ns);
				if (candidate.confidence > fused_confidence || fused_text.empty()) {
					fused_confidence = candidate.confidence;
					fused_text = candidate.text;
				}
				consumed[j] = true;
			}

			if (fused_text.empty()) {
				continue;
			}

			if (corroborated_sources.size() == 1U && looks_like_spurious_dialogue(fused_text)) {
				continue;
			}

			if (base.source == TimelineSource::AudioAsr && corroborated_sources.size() == 1U &&
			    should_suppress_low_confidence_dialogue(fused_text, fused_confidence)) {
				continue;
			}

			const std::string normalized_fused_text = normalize_dialogue_text(fused_text);
			const bool repeated_short_phrase =
				!normalized_fused_text.empty() &&
				normalized_fused_text == last_dialogue_text_ &&
				fused_start < last_dialogue_ts_ns_ + kDialogueRepeatCooldownNs &&
				count_words(normalized_fused_text) <= 3 &&
				fused_confidence < 0.82f &&
				corroborated_sources.size() == 1U;
			if (repeated_short_phrase) {
				continue;
			}

			const float reinforcement = static_cast<float>(corroborated_sources.size() - 1U) * 0.08f;
			events.push_back(SubtitleEvent{
				.channel = SubtitleChannel::Dialogue,
				.start_ns = fused_start,
				.end_ns = std::max(fused_end, fused_start + static_cast<uint64_t>(400000000ULL)),
				.confidence = clamp_confidence(fused_confidence + reinforcement),
				.text = fused_text,
			});

			last_dialogue_text_ = normalized_fused_text;
			last_dialogue_ts_ns_ = fused_start;
			continue;
		}

		if (base.channel == SubtitleChannel::Environmental) {
			consumed[i] = true;
			if (base.text.empty()) {
				continue;
			}
			if (base.text == "Ambient room tone") {
				continue;
			}
			if (base.confidence < 0.55f) {
				continue;
			}
			if (base.text == last_environmental_text_ && base.start_ns < last_environmental_ts_ns_ + kEnvironmentalCooldownNs) {
				continue;
			}

			events.push_back(SubtitleEvent{
				.channel = SubtitleChannel::Environmental,
				.start_ns = base.start_ns,
				.end_ns = std::max(base.end_ns, base.start_ns + static_cast<uint64_t>(300000000ULL)),
				.confidence = clamp_confidence(base.confidence),
				.text = base.text,
			});

			last_environmental_text_ = base.text;
			last_environmental_ts_ns_ = base.start_ns;
		}
	}

	return events;
}

} // namespace multisub
