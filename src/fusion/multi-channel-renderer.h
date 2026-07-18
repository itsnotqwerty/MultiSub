#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "multisub/fusion-engine.h"

namespace multisub {

class MultiChannelRenderer {
public:
	std::string format_for_log(const std::vector<SubtitleEvent> &events) const;
	std::string format_for_overlay(const std::vector<SubtitleEvent> &events, size_t max_lines = 3) const;
	std::string format_for_overlay(const std::vector<SubtitleEvent> &events,
		bool show_environmental,
		bool show_dialogue,
		bool show_channel_labels,
		size_t max_lines = 3) const;
};

} // namespace multisub
