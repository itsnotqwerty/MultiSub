#include "lipread-engine.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "utils/logger.h"
#include "utils/model-loader.h"

namespace multisub {

namespace {

bool looks_like_avhubert_repo(const std::filesystem::path &path)
{
	return std::filesystem::exists(path / "avhubert" / "infer_s2s.py");
}

bool looks_like_avhubert_checkpoint(const std::filesystem::path &path)
{
	return path.extension() == ".pt" && std::filesystem::exists(path);
}

std::string escape_json(const std::string &value)
{
	std::string escaped;
	escaped.reserve(value.size());
	for (char ch : value) {
		switch (ch) {
		case '\\':
			escaped += "\\\\";
			break;
		case '"':
			escaped += "\\\"";
			break;
		case '\n':
			escaped += "\\n";
			break;
		case '\r':
			escaped += "\\r";
			break;
		case '\t':
			escaped += "\\t";
			break;
		default:
			escaped += ch;
			break;
		}
	}
	return escaped;
}

float clamp_confidence(float value)
{
	return std::clamp(value, 0.0f, 1.0f);
}

uint64_t parse_u64_with_default(const std::string &input, uint64_t fallback)
{
	try {
		return static_cast<uint64_t>(std::stoull(input));
	} catch (...) {
		return fallback;
	}
}

float parse_float_with_default(const std::string &input, float fallback)
{
	try {
		return std::stof(input);
	} catch (...) {
		return fallback;
	}
}

std::string extract_json_string(const std::string &json, const std::string &key)
{
	const std::string token = "\"" + key + "\"";
	const size_t token_pos = json.find(token);
	if (token_pos == std::string::npos) {
		return {};
	}

	const size_t colon = json.find(':', token_pos + token.size());
	if (colon == std::string::npos) {
		return {};
	}

	const size_t first_quote = json.find('"', colon + 1);
	if (first_quote == std::string::npos) {
		return {};
	}

	const size_t second_quote = json.find('"', first_quote + 1);
	if (second_quote == std::string::npos || second_quote <= first_quote + 1) {
		return {};
	}

	return json.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::string extract_json_number(const std::string &json, const std::string &key)
{
	const std::string token = "\"" + key + "\"";
	const size_t token_pos = json.find(token);
	if (token_pos == std::string::npos) {
		return {};
	}

	const size_t colon = json.find(':', token_pos + token.size());
	if (colon == std::string::npos) {
		return {};
	}

	const size_t number_start = json.find_first_of("-0123456789.", colon + 1);
	if (number_start == std::string::npos) {
		return {};
	}

	const size_t number_end = json.find_first_not_of("-0123456789.eE+", number_start);
	if (number_end == std::string::npos) {
		return json.substr(number_start);
	}

	return json.substr(number_start, number_end - number_start);
}

} // namespace

void LipReadEngine::set_runtime_config(RuntimeConfig config)
{
	config_ = std::move(config);
}

bool LipReadEngine::load_model(const std::string &model_path)
{
	backend_ = Backend::None;
	loaded_ = false;
	model_path_.clear();

	if (model_path.empty()) {
		log_warn("Lip-read model path is empty; vision backend disabled");
		return false;
	}

	const std::filesystem::path candidate_path = std::filesystem::path(normalize_model_path(model_path));
	if (looks_like_avhubert_repo(candidate_path) || looks_like_avhubert_checkpoint(candidate_path)) {
		backend_ = Backend::AvHubertNative;
		loaded_ = true;
		model_path_ = candidate_path.string();
		log_info("Lip-read backend configured for native AV-HuBERT path: " + model_path_);
		return true;
	}

	if (model_path_exists(model_path) && candidate_path.extension() == ".onnx") {
		backend_ = Backend::Onnx;
		loaded_ = true;
		model_path_ = candidate_path.string();
		log_info("Lip-read backend configured for ONNX model: " + model_path_);
		return true;
	}

	log_warn("Lip-read model path is not a supported AV-HuBERT repo/checkpoint or ONNX file: " + candidate_path.string());

	return loaded_;
}

std::vector<TimelineEvent> LipReadEngine::infer(const std::vector<VideoFrame> &frames)
{
	if (!config_.enabled || !loaded_ || frames.empty()) {
		return {};
	}

	if (frames.size() < config_.min_inference_frames) {
		return {};
	}

	std::vector<InferenceResult> hypotheses;
	if (backend_ == Backend::AvHubertNative) {
		hypotheses = run_native_bridge(frames);
	} else {
		const VideoFrame &first = frames.front();
		const VideoFrame &last = frames.back();
		hypotheses = {
			InferenceResult{
				.text = "",
				.confidence = 0.0f,
				.start_ns = first.timestamp_ns,
				.end_ns = last.timestamp_ns,
			}
		};
	}

	std::vector<TimelineEvent> events;
	events.reserve(hypotheses.size());
	for (const InferenceResult &hypothesis : hypotheses) {
		if (hypothesis.text.empty()) {
			continue;
		}

		if (hypothesis.confidence < config_.min_confidence) {
			continue;
		}

		events.push_back(TimelineEvent{
			.source = TimelineSource::VisualLipRead,
			.channel = SubtitleChannel::Dialogue,
			.observed_ns = hypothesis.end_ns,
			.start_ns = hypothesis.start_ns,
			.end_ns = hypothesis.end_ns,
			.confidence = clamp_confidence(hypothesis.confidence),
			.text = hypothesis.text,
		});
	}

	return events;
}

bool LipReadEngine::is_loaded() const
{
	return loaded_;
}

bool LipReadEngine::uses_native_avhubert() const
{
	return backend_ == Backend::AvHubertNative;
}

std::vector<LipReadEngine::InferenceResult> LipReadEngine::run_native_bridge(const std::vector<VideoFrame> &frames) const
{
	if (frames.empty()) {
		return {};
	}

	const VideoFrame &first = frames.front();
	const VideoFrame &last = frames.back();

	const char *mock_text = std::getenv("MULTISUB_LIPREAD_MOCK_RESPONSE_TEXT");
	if (mock_text != nullptr && mock_text[0] != '\0') {
		const char *mock_conf = std::getenv("MULTISUB_LIPREAD_MOCK_RESPONSE_CONFIDENCE");
		const float confidence = parse_float_with_default(mock_conf != nullptr ? mock_conf : "0.8", 0.8f);
		return {
			InferenceResult{
				.text = mock_text,
				.confidence = clamp_confidence(confidence),
				.start_ns = first.timestamp_ns,
				.end_ns = last.timestamp_ns,
			}
		};
	}

	const std::filesystem::path temp_root = std::filesystem::temp_directory_path();
	const auto unique_id = std::to_string(last.timestamp_ns) + "-" + std::to_string(frames.size());
	const std::filesystem::path request_path = temp_root / ("multisub-lipread-request-" + unique_id + ".json");
	const std::filesystem::path response_path = temp_root / ("multisub-lipread-response-" + unique_id + ".json");

	std::ostringstream request;
	request << "{\n";
	request << "  \"version\": \"1\",\n";
	request << "  \"backend\": \"av_hubert_native\",\n";
	request << "  \"model\": {\"path\": \"" << escape_json(model_path_) << "\"},\n";
	request << "  \"window\": {\n";
	request << "    \"start_ns\": " << first.timestamp_ns << ",\n";
	request << "    \"end_ns\": " << last.timestamp_ns << ",\n";
	request << "    \"frame_count\": " << frames.size() << ",\n";
	request << "    \"frame_width\": " << first.width << ",\n";
	request << "    \"frame_height\": " << first.height << "\n";
	request << "  },\n";
	request << "  \"hints\": {\"mode\": \"video\", \"max_hypotheses\": 3}\n";
	request << "}\n";

	{
		std::ofstream out(request_path);
		if (!out) {
			log_error("Failed to write lip-read IPC request file: " + request_path.string());
			return {};
		}
		out << request.str();
	}

	const std::string runner_python = config_.runner_python.empty() ? "python3" : config_.runner_python;
	const std::string runner_script = resolve_runner_script_path();
	if (runner_script.empty()) {
		log_warn("Lip-read runner script is unavailable; no visual hypotheses emitted");
		std::error_code ec;
		std::filesystem::remove(request_path, ec);
		return {};
	}

	std::ostringstream command;
	command << '"' << runner_python << '"'
		<< " \"" << runner_script << "\""
		<< " --request \"" << request_path.string() << "\""
		<< " --response \"" << response_path.string() << "\"";

	const int exit_code = std::system(command.str().c_str());
	if (exit_code != 0) {
		log_warn("Lip-read runner exited with non-zero status: " + std::to_string(exit_code));
		std::error_code ec;
		std::filesystem::remove(request_path, ec);
		std::filesystem::remove(response_path, ec);
		return {};
	}

	std::ifstream response_stream(response_path);
	if (!response_stream) {
		log_warn("Lip-read runner did not produce a response file");
		std::error_code ec;
		std::filesystem::remove(request_path, ec);
		return {};
	}

	std::string response_json((std::istreambuf_iterator<char>(response_stream)), std::istreambuf_iterator<char>());
	std::vector<InferenceResult> outputs;
	const std::string status = extract_json_string(response_json, "status");
	const std::string runner_error = extract_json_string(response_json, "error");
	if (!status.empty() && status != "ok") {
		log_warn("Lip-read runner returned non-ok status: " + status + (runner_error.empty() ? std::string{} : (" (" + runner_error + ")")));
	}

	const std::string hypothesis_array_key = "\"hypotheses\"";
	const size_t array_pos = response_json.find(hypothesis_array_key);
	if (array_pos != std::string::npos) {
		const size_t open_bracket = response_json.find('[', array_pos);
		const size_t close_bracket = response_json.find(']', open_bracket == std::string::npos ? array_pos : open_bracket);
		if (open_bracket != std::string::npos && close_bracket != std::string::npos && close_bracket > open_bracket) {
			const std::string first_hypothesis = response_json.substr(open_bracket + 1, close_bracket - open_bracket - 1);
			const std::string text = extract_json_string(first_hypothesis, "text");
			if (!text.empty()) {
				InferenceResult parsed;
				parsed.text = text;
				parsed.confidence = clamp_confidence(parse_float_with_default(extract_json_number(first_hypothesis, "confidence"), 0.0f));
				parsed.start_ns = parse_u64_with_default(extract_json_number(first_hypothesis, "start_ns"), first.timestamp_ns);
				parsed.end_ns = parse_u64_with_default(extract_json_number(first_hypothesis, "end_ns"), last.timestamp_ns);
				outputs.push_back(std::move(parsed));
			}
		}
	}

	std::error_code ec;
	std::filesystem::remove(request_path, ec);
	std::filesystem::remove(response_path, ec);

	if (outputs.empty()) {
		if (!runner_error.empty() && runner_error != "decode_completed_no_inline_hypotheses") {
			log_warn("Lip-read runner produced no hypotheses: " + runner_error);
		} else {
			static size_t no_hypothesis_log_counter = 0;
			++no_hypothesis_log_counter;
			if (no_hypothesis_log_counter >= 30) {
				log_info("Lip-read runner produced no hypotheses for recent frame windows");
				no_hypothesis_log_counter = 0;
			}
		}
	}

	return outputs;
}

std::string LipReadEngine::default_runner_script_path() const
{
#ifdef MULTISUB_SOURCE_DIR
	return std::string(MULTISUB_SOURCE_DIR) + "/src/vision/avhubert_native_runner.py";
#else
	return "src/vision/avhubert_native_runner.py";
#endif
}

std::string LipReadEngine::resolve_runner_script_path() const
{
	const char *env_runner = std::getenv("MULTISUB_AVHUBERT_RUNNER_PATH");
	if (env_runner != nullptr && env_runner[0] != '\0' && std::filesystem::exists(env_runner)) {
		return env_runner;
	}

	const std::string script_path = default_runner_script_path();
	if (std::filesystem::exists(script_path)) {
		return script_path;
	}

	return {};
}

} // namespace multisub
