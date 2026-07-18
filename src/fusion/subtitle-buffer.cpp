#include "subtitle-buffer.h"

#include <chrono>

namespace multisub {

namespace {

uint64_t now_steady_ns()
{
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch())
		.count());
}

} // namespace

SubtitleBuffer::SubtitleBuffer(size_t max_events) : max_events_(max_events) {}

void SubtitleBuffer::push(const std::vector<SubtitleEvent> &events)
{
	std::lock_guard<std::mutex> lock(mutex_);
	const uint64_t inserted_ns = now_steady_ns();
	for (const SubtitleEvent &event : events) {
		events_.push_back(TimedSubtitleEvent{
			.event = event,
			.inserted_ns = inserted_ns,
		});
	}

	while (events_.size() > max_events_) {
		events_.pop_front();
	}
}

std::vector<SubtitleEvent> SubtitleBuffer::snapshot() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	std::vector<SubtitleEvent> out;
	out.reserve(events_.size());
	for (const TimedSubtitleEvent &entry : events_) {
		out.push_back(entry.event);
	}
	return out;
}

std::vector<SubtitleEvent> SubtitleBuffer::snapshot_recent(uint64_t max_age_ns) const
{
	std::lock_guard<std::mutex> lock(mutex_);
	const uint64_t now_ns = now_steady_ns();

	std::vector<SubtitleEvent> out;
	out.reserve(events_.size());
	for (const TimedSubtitleEvent &entry : events_) {
		if (entry.inserted_ns > now_ns || now_ns - entry.inserted_ns <= max_age_ns) {
			out.push_back(entry.event);
		}
	}
	return out;
}

} // namespace multisub
