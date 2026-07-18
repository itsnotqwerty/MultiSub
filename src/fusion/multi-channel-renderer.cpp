#include "multi-channel-renderer.h"

#include <algorithm>
#include <sstream>

namespace multisub {

std::string MultiChannelRenderer::format_for_log(const std::vector<SubtitleEvent> &events) const
{
	std::ostringstream out;

	for (const SubtitleEvent &event : events) {
		const char *channel = event.channel == SubtitleChannel::Dialogue ? "Dialogue" : "Environmental";
		out << '[' << channel << "] " << event.text << " (" << event.confidence << ")\n";
	}

	return out.str();
}

std::string MultiChannelRenderer::format_for_overlay(const std::vector<SubtitleEvent> &events, size_t max_lines) const
{
	return format_for_overlay(events, true, true, true, max_lines);
}

std::string MultiChannelRenderer::format_for_overlay(const std::vector<SubtitleEvent> &events,
	bool show_environmental,
	bool show_dialogue,
	bool show_channel_labels,
	size_t max_lines) const
{
	if (events.empty() || max_lines == 0) {
		return {};
	}
	if (!show_environmental && !show_dialogue) {
		return {};
	}

	std::string latest_environmental;
	std::string latest_dialogue;

	for (auto it = events.rbegin(); it != events.rend(); ++it) {
		if (show_environmental && latest_environmental.empty() && it->channel == SubtitleChannel::Environmental && !it->text.empty()) {
			latest_environmental = it->text;
		}
		if (show_dialogue && latest_dialogue.empty() && it->channel == SubtitleChannel::Dialogue && !it->text.empty()) {
			latest_dialogue = it->text;
		}
		if ((!show_environmental || !latest_environmental.empty()) && (!show_dialogue || !latest_dialogue.empty())) {
			break;
		}
	}

	std::ostringstream out;
	if (show_channel_labels) {
		out << "[Environmental] ";
	}
	out << (show_environmental ? latest_environmental : "") << '\n';
	if (show_channel_labels) {
		out << "[Dialogue] ";
	}
	out << (show_dialogue ? latest_dialogue : "");
	return out.str();
}

} // namespace multisub
