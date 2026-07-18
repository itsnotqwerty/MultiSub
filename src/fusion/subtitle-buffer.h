#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

#include "multisub/fusion-engine.h"

namespace multisub {

class SubtitleBuffer {
public:
	explicit SubtitleBuffer(size_t max_events);

	void push(const std::vector<SubtitleEvent> &events);
	std::vector<SubtitleEvent> snapshot() const;

private:
	size_t max_events_;
	mutable std::mutex mutex_;
	std::deque<SubtitleEvent> events_;
};

} // namespace multisub
