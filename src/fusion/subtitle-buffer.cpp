#include "subtitle-buffer.h"

namespace multisub {

SubtitleBuffer::SubtitleBuffer(size_t max_events) : max_events_(max_events) {}

void SubtitleBuffer::push(const std::vector<SubtitleEvent> &events)
{
	std::lock_guard<std::mutex> lock(mutex_);
	for (const SubtitleEvent &event : events) {
		events_.push_back(event);
	}

	while (events_.size() > max_events_) {
		events_.pop_front();
	}
}

std::vector<SubtitleEvent> SubtitleBuffer::snapshot() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return std::vector<SubtitleEvent>(events_.begin(), events_.end());
}

} // namespace multisub
