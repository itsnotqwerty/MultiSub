#pragma once

#include <string>

namespace multisub {

struct RuntimeConfig {
	bool dialogue_enabled = true;
	bool environmental_enabled = true;
	bool vision_enabled = false;
	bool show_channel_labels = true;
	bool debug_asr_output = false;
	int max_latency_ms = 300;
	float noise_gate_db = -40.0f;
	std::string asr_model_path;
	std::string noise_model_path;
	std::string lipread_model_path;
	std::string lipread_runner_python = "python3";
	float lipread_min_confidence = 0.45f;
	int lipread_min_frames = 12;
};

} // namespace multisub
