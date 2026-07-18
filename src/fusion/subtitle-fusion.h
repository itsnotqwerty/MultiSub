#pragma once

#include "multisub/fusion-engine.h"

namespace multisub {

class SubtitleFusion {
public:
	std::vector<SubtitleEvent> fuse_audio(const std::vector<AudioResult> &results);
	std::vector<SubtitleEvent> fuse_timeline(const std::vector<TimelineEvent> &events);

private:
	std::vector<TimelineEvent> convert_audio_results(const std::vector<AudioResult> &results);
	void ingest_timeline(const std::vector<TimelineEvent> &events);
	std::vector<SubtitleEvent> choreograph_pending();

	std::vector<TimelineEvent> timeline_;
	std::vector<TimelineEvent> pending_timeline_;
	std::string last_environmental_text_;
	uint64_t last_environmental_ts_ns_ = 0;
	std::string last_dialogue_text_;
	uint64_t last_dialogue_ts_ns_ = 0;
};

} // namespace multisub
