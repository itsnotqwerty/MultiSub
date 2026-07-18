#pragma once

#include <string>

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace multisub {

bool onnx_backend_available();
std::string normalize_model_path(const std::string &path);
bool model_path_exists(const std::string &path);

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME

Ort::Env &get_onnx_env();
Ort::SessionOptions create_default_session_options();
std::basic_string<ORTCHAR_T> to_ort_path(const std::string &path);

#endif

} // namespace multisub
