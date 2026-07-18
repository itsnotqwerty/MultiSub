#include "model-loader.h"

#include <filesystem>

namespace multisub {

bool onnx_backend_available()
{
#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
	return true;
#else
	return false;
#endif
}

std::string normalize_model_path(const std::string &path)
{
	if (path.empty()) {
		return {};
	}

	std::filesystem::path native(path);
	return std::filesystem::absolute(native).string();
}

bool model_path_exists(const std::string &path)
{
	if (path.empty()) {
		return false;
	}
	return std::filesystem::exists(std::filesystem::path(path));
}

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME

Ort::Env &get_onnx_env()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "multisub-onnx");
	return env;
}

Ort::SessionOptions create_default_session_options()
{
	Ort::SessionOptions options;
	options.SetIntraOpNumThreads(1);
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	return options;
}

std::basic_string<ORTCHAR_T> to_ort_path(const std::string &path)
{
#if defined(_WIN32)
	return std::basic_string<ORTCHAR_T>(path.begin(), path.end());
#else
	return path;
#endif
}

#endif

} // namespace multisub
