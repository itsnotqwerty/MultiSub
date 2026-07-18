#pragma once

#include "multisub/fusion-engine.h"

namespace multisub {

class SubtitleFusion {
public:
	std::vector<SubtitleEvent> fuse_audio(const std::vector<AudioResult> &results);

private:
	std::string last_environmental_text_;
	uint64_t last_environmental_ts_ns_ = 0;
};

} // namespace multisub
