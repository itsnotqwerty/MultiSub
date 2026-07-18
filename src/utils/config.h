#pragma once

#include <string>

namespace multisub {

struct RuntimeConfig {
	bool dialogue_enabled = true;
	bool environmental_enabled = true;
	bool show_channel_labels = true;
	bool debug_asr_output = false;
	int max_latency_ms = 300;
	std::string asr_model_path;
	std::string noise_model_path;
};

} // namespace multisub
