#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "multisub/fusion-engine.h"
#include "multisub/vision-engine.h"

namespace multisub {

class LipReadEngine {
public:
	struct RuntimeConfig {
		std::string model_path;
		std::string runner_python;
		float min_confidence = 0.45f;
		size_t min_inference_frames = 12;
		bool enabled = true;
	};

	struct InferenceResult {
		std::string text;
		float confidence = 0.0f;
		uint64_t start_ns = 0;
		uint64_t end_ns = 0;
	};

	bool load_model(const std::string &model_path);
	void set_runtime_config(RuntimeConfig config);
	std::vector<TimelineEvent> infer(const std::vector<VideoFrame> &frames);
	bool is_loaded() const;
	bool uses_native_avhubert() const;

private:
	std::vector<InferenceResult> run_native_bridge(const std::vector<VideoFrame> &frames) const;
	std::string default_runner_script_path() const;
	std::string resolve_runner_script_path() const;

	enum class Backend {
		None,
		Onnx,
		AvHubertNative,
	};

	Backend backend_ = Backend::None;
	RuntimeConfig config_;
	bool loaded_ = false;
	std::string model_path_;
};

} // namespace multisub
