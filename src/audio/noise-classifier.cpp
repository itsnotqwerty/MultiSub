#include "noise-classifier.h"

#include <algorithm>
#include <array>
#include <vector>

#include "audio-utils.h"
#include "utils/model-loader.h"

namespace {

constexpr std::array<const char *, 4> kEventLabels = {
	"Ambient room tone",
	"Keyboard activity",
	"Music",
	"Applause",
};

} // namespace

namespace multisub {

struct NoiseClassifier::SessionState {
	std::string model_path;

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
	std::unique_ptr<Ort::Session> session;
	std::string input_name;
	std::string output_name;
#endif
};

NoiseClassifier::NoiseClassifier() : state_(std::make_unique<SessionState>()) {}

NoiseClassifier::~NoiseClassifier() = default;

bool NoiseClassifier::load_model(const std::string &model_path)
{
	state_->model_path = normalize_model_path(model_path);

	if (state_->model_path.empty() || !model_path_exists(state_->model_path)) {
#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
		state_->session.reset();
		state_->input_name.clear();
		state_->output_name.clear();
#endif
		return false;
	}

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
	try {
		Ort::AllocatorWithDefaultOptions allocator;
		const std::basic_string<ORTCHAR_T> native_path = to_ort_path(state_->model_path);
		state_->session = std::make_unique<Ort::Session>(
			get_onnx_env(), native_path.c_str(), create_default_session_options());

		auto input_name = state_->session->GetInputNameAllocated(0, allocator);
		auto output_name = state_->session->GetOutputNameAllocated(0, allocator);
		state_->input_name = input_name.get();
		state_->output_name = output_name.get();
		return true;
	} catch (...) {
		state_->session.reset();
		state_->input_name.clear();
		state_->output_name.clear();
		return false;
	}
#else
	return false;
#endif
}

bool NoiseClassifier::is_loaded() const
{
#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
	return state_->session != nullptr;
#else
	return false;
#endif
}

std::vector<std::string> NoiseClassifier::classify(const AudioChunk &chunk, float &confidence)
{
	const float energy = estimate_chunk_energy(chunk);

	if (chunk.sample_count == 0) {
		confidence = 0.0f;
		return {};
	}

	if (is_loaded()) {
#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
		try {
			const std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(chunk.mono_samples.size())};
			std::vector<float> input(chunk.mono_samples.begin(), chunk.mono_samples.end());
			Ort::MemoryInfo memory_info =
				Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
			Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
				memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());

			const char *input_names[] = {state_->input_name.c_str()};
			const char *output_names[] = {state_->output_name.c_str()};
			std::vector<Ort::Value> outputs = state_->session->Run(
				Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

			if (!outputs.empty() && outputs[0].IsTensor()) {
				const Ort::TensorTypeAndShapeInfo shape_info = outputs[0].GetTensorTypeAndShapeInfo();
				const size_t value_count = shape_info.GetElementCount();
				if (value_count > 0) {
					const float *scores = outputs[0].GetTensorData<float>();
					size_t top_index = 0;
					float top_value = scores[0];
					for (size_t i = 1; i < value_count; ++i) {
						if (scores[i] > top_value) {
							top_value = scores[i];
							top_index = i;
						}
					}
					confidence = std::clamp(top_value, 0.0f, 1.0f);
					if (confidence >= 0.5f) {
						return {kEventLabels[top_index % kEventLabels.size()]};
					}
				}
			}
		} catch (...) {
			confidence = 0.0f;
		}
#endif
	}

	if (energy < 0.02f) {
		confidence = 0.65f;
		return {"Ambient room tone"};
	}

	if (energy > 0.012f) {
		confidence = 0.62f;
		return {"Audio activity"};
	}

	confidence = 0.28f;
	return {};
}

} // namespace multisub
