#pragma once

#include <cstddef>
#include <cstdint>
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
	std::vector<SubtitleEvent> snapshot_recent(uint64_t max_age_ns) const;

private:
	struct TimedSubtitleEvent {
		SubtitleEvent event;
		uint64_t inserted_ns = 0;
	};

	size_t max_events_;
	mutable std::mutex mutex_;
	std::deque<TimedSubtitleEvent> events_;
};

} // namespace multisub
