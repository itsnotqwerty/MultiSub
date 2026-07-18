#include "asr-engine.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "audio-utils.h"
#include "utils/logger.h"
#include "utils/model-loader.h"

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace multisub {

namespace {

constexpr uint32_t kTargetSampleRateHz = 16000;
constexpr size_t kCanaryFeatureBins = 128;
constexpr size_t kCanaryFeatureSampleRateHop = 160;
constexpr size_t kCanaryFeatureWindow = 400;
constexpr size_t kCanaryFeatureFft = 512;
constexpr size_t kCanaryMaxSeconds = 20;
constexpr size_t kCanaryMaxSamples = kTargetSampleRateHz * kCanaryMaxSeconds;
constexpr size_t kCanaryMinSamples = kTargetSampleRateHz / 2;
constexpr size_t kCanaryMinSpeechSamples = kTargetSampleRateHz / 2;
constexpr size_t kCanaryPreferredUtteranceSamples = kTargetSampleRateHz * 2;
constexpr size_t kCanaryForceFinalizeSamples = kTargetSampleRateHz * 8;
constexpr size_t kCanaryEndSilenceSamples = static_cast<size_t>(kTargetSampleRateHz * 0.4f);
constexpr size_t kCanaryMaxDecoderSteps = 256;
constexpr float kSilenceEnergyThreshold = 0.008f;
constexpr float kSpeechEnergyThreshold = 0.004f;

struct BufferedUtterance {
	std::vector<float> samples;
	uint64_t start_timestamp_ns = 0;
	uint64_t end_timestamp_ns = 0;
	size_t speech_samples = 0;
	size_t trailing_silence_samples = 0;
	bool active = false;
};

struct BundleInspection {
	bool has_config = false;
	bool has_encoder = false;
	bool has_decoder = false;
	bool has_vocab = false;
	std::string model_type = "unknown";
	std::string config_path;
	std::string encoder_path;
	std::string decoder_path;
	std::string vocab_path;
};

struct CanaryPromptIds {
	int64_t space = -1;
	int64_t start_context = -1;
	int64_t start_transcript = -1;
	int64_t emo_undefined = -1;
	int64_t language = -1;
	int64_t target_language = -1;
	int64_t pnc = -1;
	int64_t itn = -1;
	int64_t timestamp = -1;
	int64_t diarize = -1;
	int64_t endoftext = -1;
};

std::string trim_copy(std::string value)
{
	auto is_trim_char = [](unsigned char ch) {
		return std::isspace(ch) != 0 || ch == '[' || ch == ']' || ch == '{' || ch == '}' || ch == '"' || ch == '\'';
	};

	while (!value.empty() && is_trim_char(static_cast<unsigned char>(value.front()))) {
		value.erase(value.begin());
	}

	while (!value.empty() && is_trim_char(static_cast<unsigned char>(value.back()))) {
		value.pop_back();
	}

	return value;
}

std::string trim_whitespace_copy(std::string value)
{
	auto is_space = [](unsigned char ch) {
		return std::isspace(ch) != 0;
	};

	while (!value.empty() && is_space(static_cast<unsigned char>(value.front()))) {
		value.erase(value.begin());
	}

	while (!value.empty() && is_space(static_cast<unsigned char>(value.back()))) {
		value.pop_back();
	}

	return value;
}

std::vector<std::string> split_label_line(const std::string &line)
{
	std::vector<std::string> labels;
	std::string token;
	for (char ch : line) {
		if (std::isspace(static_cast<unsigned char>(ch)) != 0 || ch == ',' || ch == ';') {
			if (!token.empty()) {
				labels.push_back(trim_copy(std::move(token)));
				token.clear();
			}
			continue;
		}
		token.push_back(ch);
	}
	if (!token.empty()) {
		labels.push_back(trim_copy(std::move(token)));
	}
	labels.erase(std::remove_if(labels.begin(), labels.end(), [](const std::string &token) {
		return token.empty() || token == "<blank>" || token == "<pad>";
	}), labels.end());
	return labels;
}

int64_t parse_int64(const std::string &value, int64_t fallback)
{
	const std::string trimmed = trim_copy(value);
	if (trimmed.empty()) {
		return fallback;
	}

	int64_t parsed = fallback;
	const char *begin = trimmed.c_str();
	const char *end = begin + trimmed.size();
	const auto [ptr, ec] = std::from_chars(begin, end, parsed);
	if (ec != std::errc{} || ptr != end) {
		return fallback;
	}

	return parsed;
}

float clamp_audio_sample(float value)
{
	if (!std::isfinite(value)) {
		return 0.0f;
	}
	return std::clamp(value, -1.0f, 1.0f);
}

float decode_audio_sample(const uint8_t *plane, uint32_t index)
{
	if (plane == nullptr) {
		return 0.0f;
	}

	const float *as_float = reinterpret_cast<const float *>(plane);
	const float float_value = as_float[index];
	if (std::isfinite(float_value) && std::fabs(float_value) <= 8.0f) {
		return clamp_audio_sample(float_value);
	}

	const int16_t *as_s16 = reinterpret_cast<const int16_t *>(plane);
	return static_cast<float>(as_s16[index]) / 32768.0f;
}

float estimate_energy(const std::vector<float> &samples)
{
	if (samples.empty()) {
		return 0.0f;
	}

	float sum = 0.0f;
	for (float sample : samples) {
		sum += sample * sample;
	}
	return std::sqrt(sum / static_cast<float>(samples.size()));
}

float top_token_probability_from_logits(const float *logits, size_t count, size_t top_index)
{
	if (logits == nullptr || count == 0 || top_index >= count) {
		return 0.0f;
	}

	float max_logit = logits[0];
	for (size_t i = 1; i < count; ++i) {
		if (logits[i] > max_logit) {
			max_logit = logits[i];
		}
	}

	double denom = 0.0;
	for (size_t i = 0; i < count; ++i) {
		denom += std::exp(static_cast<double>(logits[i] - max_logit));
	}

	if (!(denom > 0.0) || !std::isfinite(denom)) {
		return 0.0f;
	}

	const double numer = std::exp(static_cast<double>(logits[top_index] - max_logit));
	const double probability = numer / denom;
	if (!std::isfinite(probability)) {
		return 0.0f;
	}

	return std::clamp(static_cast<float>(probability), 0.0f, 1.0f);
}

float hz_to_mel(float hz)
{
	return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel)
{
	return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> build_mel_filter_bank(size_t fft_size, size_t mel_bins, uint32_t sample_rate_hz)
{
	const size_t spectrum_bins = fft_size / 2 + 1;
	std::vector<float> bank(mel_bins * spectrum_bins, 0.0f);

	const float mel_min = hz_to_mel(0.0f);
	const float mel_max = hz_to_mel(static_cast<float>(sample_rate_hz) / 2.0f);

	std::vector<float> mel_points(mel_bins + 2, 0.0f);
	for (size_t i = 0; i < mel_points.size(); ++i) {
		mel_points[i] = mel_min + (mel_max - mel_min) * static_cast<float>(i) / static_cast<float>(mel_bins + 1);
	}

	std::vector<size_t> fft_points(mel_bins + 2, 0);
	for (size_t i = 0; i < mel_points.size(); ++i) {
		const float hz = mel_to_hz(mel_points[i]);
		const float bin = (static_cast<float>(fft_size) + 1.0f) * hz / static_cast<float>(sample_rate_hz);
		fft_points[i] = static_cast<size_t>(std::clamp(bin, 0.0f, static_cast<float>(spectrum_bins - 1)));
	}

	for (size_t mel = 0; mel < mel_bins; ++mel) {
		const size_t left = fft_points[mel];
		const size_t center = std::max(left + 1, fft_points[mel + 1]);
		const size_t right = std::max(center + 1, fft_points[mel + 2]);

		for (size_t bin = left; bin < center && bin < spectrum_bins; ++bin) {
			const float denom = static_cast<float>(std::max<size_t>(1, center - left));
			bank[mel * spectrum_bins + bin] = static_cast<float>(bin - left) / denom;
		}

		for (size_t bin = center; bin < right && bin < spectrum_bins; ++bin) {
			const float denom = static_cast<float>(std::max<size_t>(1, right - center));
			bank[mel * spectrum_bins + bin] = static_cast<float>(right - bin) / denom;
		}
	}

	return bank;
}

std::vector<float> fft_power_spectrum(const std::vector<float> &frame, size_t fft_size)
{
	std::vector<float> spectrum(fft_size / 2 + 1, 0.0f);
	constexpr float kPi = 3.14159265358979323846f;
	for (size_t k = 0; k < spectrum.size(); ++k) {
		float real = 0.0f;
		float imag = 0.0f;
		for (size_t n = 0; n < frame.size(); ++n) {
			const float angle = 2.0f * kPi * static_cast<float>(k * n) / static_cast<float>(fft_size);
			real += frame[n] * std::cos(angle);
			imag -= frame[n] * std::sin(angle);
		}
		spectrum[k] = real * real + imag * imag;
	}
	return spectrum;
}

std::vector<float> resample_linear(const std::vector<float> &input, uint32_t input_rate_hz, uint32_t output_rate_hz)
{
	if (input.empty() || input_rate_hz == 0 || output_rate_hz == 0) {
		return {};
	}

	if (input_rate_hz == output_rate_hz) {
		return input;
	}

	const double ratio = static_cast<double>(input_rate_hz) / static_cast<double>(output_rate_hz);
	const size_t output_size = static_cast<size_t>(static_cast<double>(input.size()) / ratio);
	std::vector<float> output(output_size);

	for (size_t i = 0; i < output_size; ++i) {
		const double src_index = static_cast<double>(i) * ratio;
		const size_t left = static_cast<size_t>(src_index);
		const size_t right = std::min(left + 1, input.size() - 1);
		const double mix = src_index - static_cast<double>(left);
		output[i] = static_cast<float>(input[left] * (1.0 - mix) + input[right] * mix);
	}

	return output;
}

std::vector<float> build_nemo_features(const std::vector<float> &samples, uint32_t sample_rate_hz, uint32_t &feature_frames)
{
	feature_frames = 0;
	if (samples.empty()) {
		return {};
	}

	const std::vector<float> resampled = resample_linear(samples, sample_rate_hz, kTargetSampleRateHz);
	if (resampled.size() < kCanaryFeatureWindow) {
		return {};
	}

	const size_t frame_count = 1 + (resampled.size() - kCanaryFeatureWindow) / kCanaryFeatureSampleRateHop;
	feature_frames = static_cast<uint32_t>(frame_count);

	std::vector<float> features(kCanaryFeatureBins * frame_count, 0.0f);
	std::vector<float> window(kCanaryFeatureWindow, 0.0f);
	constexpr float kPi = 3.14159265358979323846f;
	for (size_t i = 0; i < window.size(); ++i) {
		window[i] = 0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(window.size() - 1));
	}

	const std::vector<float> mel_bank = build_mel_filter_bank(kCanaryFeatureFft, kCanaryFeatureBins, kTargetSampleRateHz);
	const size_t spectrum_bins = kCanaryFeatureFft / 2 + 1;

	for (size_t frame = 0; frame < frame_count; ++frame) {
		std::vector<float> frame_samples(kCanaryFeatureWindow, 0.0f);
		const size_t offset = frame * kCanaryFeatureSampleRateHop;
		for (size_t i = 0; i < kCanaryFeatureWindow; ++i) {
			frame_samples[i] = resampled[offset + i] * window[i];
		}

		const std::vector<float> spectrum = fft_power_spectrum(frame_samples, kCanaryFeatureFft);
		for (size_t mel = 0; mel < kCanaryFeatureBins; ++mel) {
			float energy = 0.0f;
			for (size_t bin = 0; bin < spectrum_bins; ++bin) {
				energy += spectrum[bin] * mel_bank[mel * spectrum_bins + bin];
			}
			features[mel * frame_count + frame] = std::log(std::max(energy, 1e-10f));
		}
	}

	std::vector<float> mean(kCanaryFeatureBins, 0.0f);
	std::vector<float> variance(kCanaryFeatureBins, 0.0f);
	for (size_t mel = 0; mel < kCanaryFeatureBins; ++mel) {
		for (size_t frame = 0; frame < frame_count; ++frame) {
			mean[mel] += features[mel * frame_count + frame];
		}
		mean[mel] /= static_cast<float>(frame_count);
		for (size_t frame = 0; frame < frame_count; ++frame) {
			const float centered = features[mel * frame_count + frame] - mean[mel];
			variance[mel] += centered * centered;
		}
		variance[mel] = std::sqrt(variance[mel] / std::max<size_t>(1, frame_count - 1));
		for (size_t frame = 0; frame < frame_count; ++frame) {
			const size_t index = mel * frame_count + frame;
			features[index] = (features[index] - mean[mel]) / (variance[mel] + 1e-5f);
		}
	}

	return features;
}

BundleInspection inspect_asr_bundle(const std::string &model_path)
{
	BundleInspection bundle;
	const std::filesystem::path model_file(model_path);
	const std::filesystem::path model_dir = std::filesystem::is_directory(model_file)
		? model_file
		: (model_file.has_parent_path() ? model_file.parent_path() : std::filesystem::path{});

	const std::filesystem::path config_path = model_dir / "config.json";
	const std::filesystem::path encoder_path = model_dir / "encoder-model.onnx";
	const std::filesystem::path decoder_path = model_dir / "decoder-model.onnx";
	const std::filesystem::path vocab_path = model_dir / "vocab.txt";

	bundle.has_config = std::filesystem::exists(config_path);
	bundle.has_encoder = std::filesystem::exists(encoder_path);
	bundle.has_decoder = std::filesystem::exists(decoder_path);
	bundle.has_vocab = std::filesystem::exists(vocab_path);
	bundle.config_path = config_path.string();
	bundle.encoder_path = encoder_path.string();
	bundle.decoder_path = decoder_path.string();
	bundle.vocab_path = vocab_path.string();
	if (bundle.has_config && bundle.has_encoder && bundle.has_decoder && bundle.has_vocab) {
		bundle.model_type = "nemo-conformer-aed";
	}

	return bundle;
}

std::unordered_map<std::string, int64_t> load_vocab_map(const std::string &vocab_path, std::vector<std::string> &ordered_tokens)
{
	std::unordered_map<std::string, int64_t> token_to_id;
	std::ifstream input(vocab_path);
	std::string line;
	int64_t line_index = 0;
	while (std::getline(input, line)) {
		line = trim_whitespace_copy(std::move(line));
		if (line.empty()) {
			++line_index;
			continue;
		}

		std::string token = line;
		int64_t id = line_index;
		const size_t last_space = line.find_last_of(' ');
		if (last_space != std::string::npos) {
			const std::string suffix = trim_whitespace_copy(line.substr(last_space + 1));
			const int64_t parsed = parse_int64(suffix, line_index);
			if (parsed >= 0) {
				id = parsed;
				token = trim_whitespace_copy(line.substr(0, last_space));
			}
		}

		if (token.empty()) {
			++line_index;
			continue;
		}

		if (static_cast<size_t>(id) >= ordered_tokens.size()) {
			ordered_tokens.resize(static_cast<size_t>(id) + 1);
		}
		ordered_tokens[static_cast<size_t>(id)] = token;
		token_to_id[token] = id;
		++line_index;
	}

	return token_to_id;
}

std::string decode_tokens_to_text(const std::vector<int64_t> &tokens, const std::vector<std::string> &vocab)
{
	std::string out;
	for (int64_t id : tokens) {
		if (id < 0 || static_cast<size_t>(id) >= vocab.size()) {
			continue;
		}
		const std::string &token = vocab[static_cast<size_t>(id)];
		if (token.empty()) {
			continue;
		}
		if (token == "<|endoftext|>" || token == "<|startoftext|>" || token == "<|startoftranscript|>" ||
		    token == "<|startofcontext|>" || token == "<|timestamp|>" || token == "<|notimestamp|>" ||
		    token == "<|diarize|>" || token == "<|nodiarize|>" || token == "<|pnc|>" || token == "<|nopnc|>" ||
		    token == "<|itn|>" || token == "<|noitn|>" || token == "<|emo:undefined|>" || token == "<|spkchange|>" ||
		    token == "<|audioseparator|>" || token == "<|predict_lang|>" || token == "<|nopredict_lang|>" ||
		    token == "<|unklang|>" || token == "<|nospeech|>") {
			continue;
		}
		if (token == "<blk>" || token == "<blank>" || token == "<pad>") {
			continue;
		}
		if (token == "|" || token == "▁" || token == "<space>") {
			if (!out.empty() && out.back() != ' ') {
				out.push_back(' ');
			}
			continue;
		}
		if (token.rfind("##", 0) == 0) {
			out.append(token.substr(2));
			continue;
		}

		std::string normalized_token = token;
		for (size_t marker = normalized_token.find("▁"); marker != std::string::npos; marker = normalized_token.find("▁", marker)) {
			normalized_token.replace(marker, std::string("▁").size(), " ");
		}
		for (size_t marker = normalized_token.find("Ġ"); marker != std::string::npos; marker = normalized_token.find("Ġ", marker)) {
			normalized_token.replace(marker, std::string("Ġ").size(), " ");
		}

		out.append(normalized_token);
	}

	std::string collapsed;
	collapsed.reserve(out.size());
	bool in_space = false;
	for (char ch : out) {
		if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
			if (!in_space) {
				collapsed.push_back(' ');
				in_space = true;
			}
			continue;
		}
		collapsed.push_back(ch);
		in_space = false;
	}

	return trim_copy(std::move(collapsed));
}

CanaryPromptIds build_prompt_ids(const std::unordered_map<std::string, int64_t> &tokens)
{
	CanaryPromptIds ids;
	auto lookup = [&tokens](std::string_view token) -> int64_t {
		const auto it = tokens.find(std::string(token));
		return it == tokens.end() ? -1 : it->second;
	};

	ids.space = lookup("▁");
	if (ids.space < 0) {
		ids.space = lookup(" ");
	}
	ids.start_context = lookup("<|startofcontext|>");
	ids.start_transcript = lookup("<|startoftranscript|>");
	ids.emo_undefined = lookup("<|emo:undefined|>");
	ids.language = lookup("<|en|>");
	ids.target_language = lookup("<|en|>");
	ids.pnc = lookup("<|pnc|>");
	ids.itn = lookup("<|noitn|>");
	ids.timestamp = lookup("<|notimestamp|>");
	ids.diarize = lookup("<|nodiarize|>");
	ids.endoftext = lookup("<|endoftext|>");
	return ids;
}

std::vector<int64_t> build_prompt_sequence(const CanaryPromptIds &prompt)
{
	std::vector<int64_t> prompt_tokens = {
		prompt.space,
		prompt.start_context,
		prompt.start_transcript,
		prompt.emo_undefined,
		prompt.language,
		prompt.target_language,
		prompt.pnc,
		prompt.itn,
		prompt.timestamp,
		prompt.diarize,
	};
	prompt_tokens.erase(std::remove_if(prompt_tokens.begin(), prompt_tokens.end(), [](int64_t id) { return id < 0; }), prompt_tokens.end());
	return prompt_tokens;
}

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME

std::vector<int64_t> empty_decoder_mems(const Ort::Session &decoder_session, size_t batch_size)
{
	Ort::AllocatorWithDefaultOptions allocator;
	const Ort::TypeInfo type_info = decoder_session.GetInputTypeInfo(3);
	const auto shape_info = type_info.GetTensorTypeAndShapeInfo();
	const std::vector<int64_t> shape = shape_info.GetShape();
	std::vector<int64_t> empty_shape = shape;
	for (int64_t &dim : empty_shape) {
		if (dim < 0) {
			dim = 0;
		}
	}
	if (empty_shape.size() < 4) {
		empty_shape = {1, static_cast<int64_t>(batch_size), 0, 1};
	} else {
		empty_shape[1] = static_cast<int64_t>(batch_size);
		empty_shape[2] = 0;
	}
	return empty_shape;
}

#endif

} // namespace

struct AsrEngine::SessionState {
	std::string model_path;
	BundleInspection bundle;
	std::unordered_map<std::string, int64_t> token_to_id;
	std::vector<std::string> vocab_tokens;
	CanaryPromptIds prompt_ids;
	BufferedUtterance buffered_utterance;
	bool loaded = false;

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
	std::unique_ptr<Ort::Session> encoder_session;
	std::unique_ptr<Ort::Session> decoder_session;
#endif
};

AsrEngine::AsrEngine() : state_(std::make_unique<SessionState>()) {}

AsrEngine::~AsrEngine() = default;

bool AsrEngine::load_model(const std::string &model_path)
{
	state_->model_path = normalize_model_path(model_path);
	state_->buffered_utterance = {};
	state_->token_to_id.clear();
	state_->vocab_tokens.clear();
	state_->prompt_ids = {};
	state_->loaded = false;
	state_->bundle = inspect_asr_bundle(state_->model_path);

	if (state_->model_path.empty() || !model_path_exists(state_->model_path)) {
		return false;
	}

	if (state_->bundle.model_type != "nemo-conformer-aed") {
		log_warn("ASR model bundle was detected but is not Canary AED; subtitles will stay unavailable for this model");
		return false;
	}

	state_->token_to_id = load_vocab_map(state_->bundle.vocab_path, state_->vocab_tokens);
	state_->prompt_ids = build_prompt_ids(state_->token_to_id);
	if (state_->vocab_tokens.empty()) {
		log_error("ASR model vocab.txt could not be loaded or was empty");
		return false;
	}

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
	try {
		const std::basic_string<ORTCHAR_T> encoder_path = to_ort_path(state_->bundle.encoder_path);
		const std::basic_string<ORTCHAR_T> decoder_path = to_ort_path(state_->bundle.decoder_path);
		state_->encoder_session = std::make_unique<Ort::Session>(get_onnx_env(), encoder_path.c_str(), create_default_session_options());
		state_->decoder_session = std::make_unique<Ort::Session>(get_onnx_env(), decoder_path.c_str(), create_default_session_options());
		state_->loaded = true;
		log_info("Loaded Canary AED ASR bundle with encoder-model.onnx, decoder-model.onnx, config.json, and vocab.txt");
		return true;
	} catch (const std::exception &error) {
		state_->encoder_session.reset();
		state_->decoder_session.reset();
		state_->loaded = false;
		log_error(std::string("Failed to initialize Canary ASR encoder/decoder sessions: ") + error.what());
		return false;
	} catch (...) {
		state_->encoder_session.reset();
		state_->decoder_session.reset();
		state_->loaded = false;
		log_error("Failed to initialize Canary ASR encoder/decoder sessions");
		return false;
	}
#else
	return false;
#endif
}

bool AsrEngine::is_loaded() const
{
	return state_ != nullptr && state_->loaded;
}

std::string AsrEngine::transcribe(const AudioChunk &chunk, float &confidence)
{
	if (chunk.sample_count == 0) {
		confidence = 0.0f;
		return {};
	}

	std::vector<float> chunk_samples = chunk.mono_samples;
	if (chunk.sample_rate_hz != 0 && chunk.sample_rate_hz != kTargetSampleRateHz) {
		chunk_samples = resample_linear(chunk.mono_samples, chunk.sample_rate_hz, kTargetSampleRateHz);
	}

	if (chunk_samples.empty()) {
		confidence = 0.0f;
		return {};
	}

	state_->buffered_utterance.samples.insert(state_->buffered_utterance.samples.end(), chunk_samples.begin(), chunk_samples.end());
	if (state_->buffered_utterance.samples.size() > kCanaryMaxSamples) {
		state_->buffered_utterance.samples.erase(
			state_->buffered_utterance.samples.begin(),
			state_->buffered_utterance.samples.begin() + static_cast<std::vector<float>::difference_type>(state_->buffered_utterance.samples.size() - kCanaryMaxSamples));
	}
	if (!state_->buffered_utterance.active) {
		state_->buffered_utterance.start_timestamp_ns = chunk.timestamp_ns;
		state_->buffered_utterance.active = true;
	}
	state_->buffered_utterance.end_timestamp_ns = chunk.timestamp_ns;

	const float chunk_energy = estimate_energy(chunk_samples);
	if (chunk_energy >= kSpeechEnergyThreshold) {
		state_->buffered_utterance.speech_samples += chunk_samples.size();
		state_->buffered_utterance.trailing_silence_samples = 0;
	} else {
		state_->buffered_utterance.trailing_silence_samples += chunk_samples.size();
	}

	if (chunk_energy < kSilenceEnergyThreshold && state_->buffered_utterance.samples.size() < kCanaryMinSamples) {
		confidence = 0.0f;
		return {};
	}

	const size_t buffered_samples = state_->buffered_utterance.samples.size();
	const float buffered_energy = estimate_energy(state_->buffered_utterance.samples);
	const bool has_enough_speech = state_->buffered_utterance.speech_samples >= kCanaryMinSpeechSamples ||
		(buffered_samples >= kCanaryPreferredUtteranceSamples && buffered_energy >= kSilenceEnergyThreshold * 1.1f);
	const bool silence_endpoint = has_enough_speech && state_->buffered_utterance.trailing_silence_samples >= kCanaryEndSilenceSamples;
	const bool preferred_window_reached = has_enough_speech && buffered_samples >= kCanaryPreferredUtteranceSamples;
	const bool force_window_reached = buffered_samples >= kCanaryForceFinalizeSamples;

	if (!silence_endpoint && !preferred_window_reached && !force_window_reached) {
		confidence = estimate_energy(state_->buffered_utterance.samples);
		return {};
	}

	if (!has_enough_speech) {
		state_->buffered_utterance = {};
		confidence = 0.0f;
		return {};
	}

	const std::vector<float> samples = std::move(state_->buffered_utterance.samples);
	state_->buffered_utterance = {};

#if defined(MULTISUB_ENABLE_ONNX_RUNTIME) && MULTISUB_ENABLE_ONNX_RUNTIME
	if (!state_->loaded || state_->encoder_session == nullptr || state_->decoder_session == nullptr) {
		confidence = 0.0f;
		return {};
	}

	try {
		uint32_t feature_frames = 0;
		std::vector<float> features = build_nemo_features(samples, kTargetSampleRateHz, feature_frames);
		if (features.empty() || feature_frames == 0) {
			confidence = 0.0f;
			return {};
		}

		const std::array<int64_t, 3> feature_shape = {1, static_cast<int64_t>(kCanaryFeatureBins), static_cast<int64_t>(feature_frames)};
		const std::array<int64_t, 1> length_shape = {1};
		const int64_t feature_length = static_cast<int64_t>(feature_frames);

		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value feature_tensor = Ort::Value::CreateTensor<float>(memory_info, features.data(), features.size(), feature_shape.data(), feature_shape.size());
		Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t *>(&feature_length), 1, length_shape.data(), length_shape.size());

		const char *encoder_input_names[] = {"audio_signal", "length"};
		const char *encoder_output_names[] = {"encoder_embeddings", "encoder_mask"};
		std::array<Ort::Value, 2> encoder_inputs = {
			std::move(feature_tensor),
			std::move(length_tensor),
		};
		std::vector<Ort::Value> encoder_outputs = state_->encoder_session->Run(
			Ort::RunOptions{nullptr}, encoder_input_names, encoder_inputs.data(), encoder_inputs.size(), encoder_output_names, 2);

		if (encoder_outputs.size() != 2 || !encoder_outputs[0].IsTensor() || !encoder_outputs[1].IsTensor()) {
			log_warn("Canary encoder did not return the expected encoder_embeddings and encoder_mask tensors");
			confidence = 0.0f;
			return {};
		}

		const Ort::TensorTypeAndShapeInfo encoder_embeddings_info = encoder_outputs[0].GetTensorTypeAndShapeInfo();
		const Ort::TensorTypeAndShapeInfo encoder_mask_info = encoder_outputs[1].GetTensorTypeAndShapeInfo();
		const std::vector<int64_t> encoder_embeddings_shape = encoder_embeddings_info.GetShape();
		const std::vector<int64_t> encoder_mask_shape = encoder_mask_info.GetShape();
		float *encoder_embeddings_data = encoder_outputs[0].GetTensorMutableData<float>();
		int64_t *encoder_mask_data = encoder_outputs[1].GetTensorMutableData<int64_t>();

		if (encoder_embeddings_data == nullptr || encoder_mask_data == nullptr || encoder_embeddings_shape.size() != 3 || encoder_mask_shape.size() != 2) {
			log_warn("Canary encoder outputs had unexpected shapes or null data");
			confidence = 0.0f;
			return {};
		}

		const std::vector<int64_t> prompt_tokens = build_prompt_sequence(state_->prompt_ids);
		if (prompt_tokens.empty()) {
			log_error("Canary prompt token set is incomplete; no transcript can be produced");
			confidence = 0.0f;
			return {};
		}

		std::vector<int64_t> current_tokens = prompt_tokens;
		std::vector<int64_t> generated_tokens;
		float cumulative_confidence = 0.0f;
		size_t confidence_steps = 0;
		std::vector<float> decoder_mems_data;
		std::vector<int64_t> decoder_mems_shape = empty_decoder_mems(*state_->decoder_session, 1);

		const char *decoder_input_names[] = {"input_ids", "encoder_embeddings", "encoder_mask", "decoder_mems"};
		const char *decoder_output_names[] = {"logits", "decoder_hidden_states"};

		for (size_t step = 0; step < kCanaryMaxDecoderSteps; ++step) {
				std::vector<int64_t> input_ids_tokens = decoder_mems_data.empty() ? current_tokens : std::vector<int64_t>{current_tokens.back()};
			Ort::Value decoder_mems_tensor = decoder_mems_data.empty()
				? Ort::Value::CreateTensor<float>(memory_info, nullptr, 0, decoder_mems_shape.data(), decoder_mems_shape.size())
				: Ort::Value::CreateTensor<float>(memory_info, decoder_mems_data.data(), decoder_mems_data.size(), decoder_mems_shape.data(), decoder_mems_shape.size());

				const std::array<int64_t, 2> input_ids_shape = {1, static_cast<int64_t>(input_ids_tokens.size())};
				Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_ids_tokens.data(), input_ids_tokens.size(), input_ids_shape.data(), input_ids_shape.size());
			Ort::Value embeddings_tensor = Ort::Value::CreateTensor<float>(memory_info, encoder_embeddings_data, encoder_embeddings_info.GetElementCount(), encoder_embeddings_shape.data(), encoder_embeddings_shape.size());
			Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, encoder_mask_data, encoder_mask_info.GetElementCount(), encoder_mask_shape.data(), encoder_mask_shape.size());

			std::array<Ort::Value, 4> decoder_inputs = {
				std::move(input_ids_tensor),
				std::move(embeddings_tensor),
				std::move(mask_tensor),
				std::move(decoder_mems_tensor),
			};

			std::vector<Ort::Value> decoder_outputs = state_->decoder_session->Run(
				Ort::RunOptions{nullptr}, decoder_input_names, decoder_inputs.data(), decoder_inputs.size(), decoder_output_names, 2);

			if (decoder_outputs.size() != 2 || !decoder_outputs[0].IsTensor() || !decoder_outputs[1].IsTensor()) {
				log_warn("Canary decoder did not return logits and decoder_hidden_states tensors");
				break;
			}

			const Ort::TensorTypeAndShapeInfo logits_info = decoder_outputs[0].GetTensorTypeAndShapeInfo();
			const std::vector<int64_t> logits_shape = logits_info.GetShape();
			const Ort::TensorTypeAndShapeInfo decoder_hidden_info = decoder_outputs[1].GetTensorTypeAndShapeInfo();
			decoder_mems_shape = decoder_hidden_info.GetShape();
			decoder_mems_data.assign(
				decoder_outputs[1].GetTensorData<float>(),
				decoder_outputs[1].GetTensorData<float>() + decoder_hidden_info.GetElementCount());
			if (logits_shape.size() != 3) {
				log_warn("Canary decoder logits tensor did not have rank 3");
				break;
			}

			const float *logits = decoder_outputs[0].GetTensorData<float>();
			const size_t vocab_size = static_cast<size_t>(logits_shape[2]);
			const size_t last_index = static_cast<size_t>(logits_shape[1] - 1);
			const float *step_logits = logits + (last_index * vocab_size);

			size_t best_index = 0;
			float best_value = step_logits[0];
			for (size_t i = 1; i < vocab_size; ++i) {
				if (step_logits[i] > best_value) {
					best_value = step_logits[i];
					best_index = i;
				}
			}

			cumulative_confidence += top_token_probability_from_logits(step_logits, vocab_size, best_index);
			++confidence_steps;

			if (static_cast<int64_t>(best_index) == state_->prompt_ids.endoftext) {
				break;
			}

			current_tokens.push_back(static_cast<int64_t>(best_index));
			generated_tokens.push_back(static_cast<int64_t>(best_index));
		}

		confidence = confidence_steps > 0 ? cumulative_confidence / static_cast<float>(confidence_steps) : 0.0f;
		std::string text = decode_tokens_to_text(generated_tokens, state_->vocab_tokens);
		if (text.empty()) {
			log_warn("Canary decode completed without text output for the current utterance");
		}
		return text;
	} catch (...) {
		confidence = 0.0f;
		return {};
	}
#else
	confidence = estimate_energy(samples);
	return {};
#endif
}

} // namespace multisub
