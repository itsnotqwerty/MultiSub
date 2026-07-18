#include "multisub-filter.h"

#include <obs-module.h>

#include <chrono>
#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

#include "audio/audio-capture.h"
#include "audio/audio-utils.h"
#include "fusion/multi-channel-renderer.h"
#include "fusion/subtitle-buffer.h"
#include "fusion/subtitle-fusion.h"
#include "multisub/audio-engine.h"
#include "utils/config.h"
#include "utils/logger.h"
#include "utils/model-loader.h"

namespace multisub {

struct MultiSubFilter {
	obs_source_t *source = nullptr;
	obs_source_t *capture_source = nullptr;
	obs_source_t *overlay_source = nullptr;
	RuntimeConfig config;
	AudioEngine audio_engine;
	SubtitleFusion fusion;
	SubtitleBuffer subtitle_buffer{256};
	MultiChannelRenderer renderer;
	std::string last_overlay_text;
	std::string capture_source_name;
	std::chrono::steady_clock::time_point last_capture_source_audio_at{};
	uint32_t overlay_layout_width = 0;
	uint32_t overlay_layout_height = 0;
	uint64_t captured_chunks = 0;
	float last_capture_energy = 0.0f;
};

static bool is_recent_capture_source_audio(const MultiSubFilter *filter)
{
	if (filter == nullptr || filter->capture_source == nullptr) {
		return false;
	}

	if (filter->last_capture_source_audio_at == std::chrono::steady_clock::time_point{}) {
		return false;
	}

	const auto elapsed = std::chrono::steady_clock::now() - filter->last_capture_source_audio_at;
	return elapsed < std::chrono::seconds(2);
}

static void log_audio_results(const std::vector<AudioResult> &results)
{
	for (const AudioResult &result : results) {
		if (!result.transcript.empty()) {
			log_info("Live ASR transcript: '" + result.transcript + "' (confidence=" + std::to_string(result.transcript_confidence) + ")");
		} else {
			log_info("Live ASR transcript: <empty> (confidence=" + std::to_string(result.transcript_confidence) + ")");
		}

		if (!result.environmental_events.empty()) {
			log_info("Live environmental labels: '" + result.environmental_events.front() + "' (confidence=" + std::to_string(result.environmental_confidence) + ")");
		}
	}
}

static void multisub_filter_audio_capture(void *param, obs_source_t *, const audio_data *audio_data, bool muted)
{
	auto *filter = static_cast<MultiSubFilter *>(param);
	UNUSED_PARAMETER(muted);
	if (filter == nullptr || audio_data == nullptr) {
		return;
	}

	AudioChunk chunk = capture_audio_chunk(audio_data);
	filter->last_capture_energy = estimate_chunk_energy(chunk);
	filter->captured_chunks += 1;
	filter->last_capture_source_audio_at = std::chrono::steady_clock::now();
	filter->audio_engine.submit_chunk(std::move(chunk));
}

static void clear_audio_capture_source(MultiSubFilter *filter)
{
	if (filter == nullptr || filter->capture_source == nullptr) {
		return;
	}

	obs_source_remove_audio_capture_callback(filter->capture_source, multisub_filter_audio_capture, filter);
	obs_source_release(filter->capture_source);
	filter->capture_source = nullptr;
	filter->capture_source_name.clear();
}

static void set_audio_capture_source(MultiSubFilter *filter, const char *source_name)
{
	if (filter == nullptr) {
		return;
	}

	const std::string desired_name = source_name != nullptr ? source_name : "";
	if (desired_name == filter->capture_source_name) {
		return;
	}

	clear_audio_capture_source(filter);

	if (desired_name.empty()) {
		return;
	}

	obs_source_t *selected_source = obs_get_source_by_name(desired_name.c_str());
	if (selected_source == nullptr) {
		log_info("Configured audio source not found");
		return;
	}

	obs_source_add_audio_capture_callback(selected_source, multisub_filter_audio_capture, filter);
	filter->capture_source = selected_source;
	filter->capture_source_name = desired_name;
	filter->last_capture_source_audio_at = std::chrono::steady_clock::time_point{};
	log_info("Attached audio capture callback to source: " + filter->capture_source_name);
}

static bool enum_audio_source_list(void *param, obs_source_t *source)
{
	auto *list_prop = static_cast<obs_property_t *>(param);
	if (list_prop == nullptr || source == nullptr) {
		return true;
	}

	if ((obs_source_get_output_flags(source) & OBS_SOURCE_AUDIO) == 0) {
		return true;
	}

	const char *source_name = obs_source_get_name(source);
	if (source_name == nullptr || source_name[0] == '\0') {
		return true;
	}

	obs_property_list_add_string(list_prop, source_name, source_name);
	return true;
}

struct AudioSourcePickState {
	std::string preferred_name;
	std::string fallback_name;
};

static std::string to_lower_copy(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

static bool enum_find_default_audio_source(void *param, obs_source_t *source)
{
	auto *state = static_cast<AudioSourcePickState *>(param);
	if (state == nullptr || source == nullptr) {
		return true;
	}

	if ((obs_source_get_output_flags(source) & OBS_SOURCE_AUDIO) == 0) {
		return true;
	}

	const char *name_cstr = obs_source_get_name(source);
	if (name_cstr == nullptr || name_cstr[0] == '\0') {
		return true;
	}

	const std::string name = name_cstr;
	if (state->fallback_name.empty()) {
		state->fallback_name = name;
	}

	const std::string lower_name = to_lower_copy(name);
	if (lower_name.find("mic") != std::string::npos || lower_name.find("microphone") != std::string::npos ||
	    lower_name.find("input capture") != std::string::npos || lower_name.find("audio input") != std::string::npos) {
		state->preferred_name = name;
		return false;
	}

	return true;
}

static std::string get_default_audio_source_name()
{
	AudioSourcePickState state;
	obs_enum_sources(enum_find_default_audio_source, &state);
	if (!state.preferred_name.empty()) {
		return state.preferred_name;
	}
	return state.fallback_name;
}

static struct obs_audio_data *multisub_filter_filter_audio(void *data, struct obs_audio_data *audio)
{
	auto *filter = static_cast<MultiSubFilter *>(data);
	if (filter == nullptr || audio == nullptr) {
		return audio;
	}

	if (is_recent_capture_source_audio(filter)) {
		return audio;
	}

	AudioChunk chunk = capture_audio_chunk(audio);
	filter->last_capture_energy = estimate_chunk_energy(chunk);
	filter->captured_chunks += 1;
	filter->audio_engine.submit_chunk(std::move(chunk));

	return audio;
}

static obs_source_t *create_overlay_source()
{
	obs_data_t *settings = obs_get_source_defaults("text_ft2_source");
	if (settings == nullptr) {
		return nullptr;
	}

	obs_data_t *font = obs_data_create();
	obs_data_set_string(font, "face", "Sans Serif");
	obs_data_set_string(font, "style", "");
	obs_data_set_int(font, "size", 28);
	obs_data_set_int(font, "flags", 0);

	obs_data_set_string(settings, "text", "");
	obs_data_set_obj(settings, "font", font);
	obs_data_set_int(settings, "color1", 0xFFFFFFFF);
	obs_data_set_int(settings, "color2", 0xFFFFFFFF);
	obs_data_set_bool(settings, "outline", true);
	obs_data_set_bool(settings, "drop_shadow", true);
	obs_data_set_bool(settings, "word_wrap", true);

	obs_data_release(font);

	obs_source_t *overlay_source = obs_source_create_private("text_ft2_source", "MultiSub Overlay", settings);
	obs_data_release(settings);
	return overlay_source;
}

static void update_overlay_source_text(MultiSubFilter *filter, const std::string &text)
{
	if (filter == nullptr || filter->overlay_source == nullptr || text == filter->last_overlay_text) {
		return;
	}

	obs_data_t *settings = obs_source_get_settings(filter->overlay_source);
	if (settings == nullptr) {
		return;
	}

	obs_data_set_string(settings, "text", text.c_str());
	obs_source_update(filter->overlay_source, settings);
	obs_data_release(settings);
	filter->last_overlay_text = text;
}

static void update_overlay_source_layout(MultiSubFilter *filter, uint32_t width, uint32_t height)
{
	if (filter == nullptr || filter->overlay_source == nullptr || width == 0 || height == 0) {
		return;
	}

	if (filter->overlay_layout_width == width && filter->overlay_layout_height == height) {
		return;
	}

	obs_data_t *settings = obs_source_get_settings(filter->overlay_source);
	if (settings == nullptr) {
		return;
	}

	obs_data_set_bool(settings, "extents", true);
	obs_data_set_int(settings, "extents_cx", static_cast<long long>(width));
	obs_data_set_int(settings, "extents_cy", static_cast<long long>(height));
	obs_data_set_bool(settings, "word_wrap", true);
	obs_source_update(filter->overlay_source, settings);
	obs_data_release(settings);

	filter->overlay_layout_width = width;
	filter->overlay_layout_height = height;
}

static void multisub_filter_video_tick(void *data, float)
{
	auto *filter = static_cast<MultiSubFilter *>(data);
	if (filter == nullptr) {
		return;
	}

	std::vector<AudioResult> results = filter->audio_engine.consume_results();
	if (results.empty()) {
		if (filter->captured_chunks == 0) {
			update_overlay_source_text(filter, "[MultiSub] No audio captured - choose Audio Source in filter settings");
		} else if (!filter->subtitle_buffer.snapshot().empty()) {
			update_overlay_source_text(filter,
				filter->renderer.format_for_overlay(
					filter->subtitle_buffer.snapshot(),
					filter->config.environmental_enabled,
					filter->config.dialogue_enabled,
					filter->config.show_channel_labels));
		} else {
			update_overlay_source_text(filter, "");
		}
		return;
	}

	if (filter->config.debug_asr_output) {
		log_audio_results(results);
	}

	std::vector<SubtitleEvent> events = filter->fusion.fuse_audio(results);
	if (events.empty()) {
		if (!filter->subtitle_buffer.snapshot().empty()) {
			update_overlay_source_text(filter,
				filter->renderer.format_for_overlay(
					filter->subtitle_buffer.snapshot(),
					filter->config.environmental_enabled,
					filter->config.dialogue_enabled,
					filter->config.show_channel_labels));
			return;
		}

		if (!onnx_backend_available() && !filter->config.asr_model_path.empty()) {
			update_overlay_source_text(filter, "[MultiSub] ONNX Runtime backend unavailable in this build");
			return;
		}

		const bool has_asr_model = !filter->config.asr_model_path.empty() && model_path_exists(filter->config.asr_model_path);
		if (filter->last_capture_energy > 0.0008f && !has_asr_model) {
			update_overlay_source_text(filter, "[MultiSub] Speech detected - set ASR model path for transcription");
		} else {
			update_overlay_source_text(filter, "");
		}
		return;
	}

	filter->subtitle_buffer.push(events);
	const std::string overlay_text = filter->renderer.format_for_overlay(
		filter->subtitle_buffer.snapshot(),
		filter->config.environmental_enabled,
		filter->config.dialogue_enabled,
		filter->config.show_channel_labels);
	update_overlay_source_text(filter, overlay_text);
}

static const char *multisub_filter_get_name(void *)
{
	return obs_module_text("MultiSub.Filter");
}

static void *multisub_filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *filter = new MultiSubFilter;
	filter->source = source;

	if (settings != nullptr) {
		filter->config.dialogue_enabled = obs_data_get_bool(settings, "dialogue_enabled");
		filter->config.environmental_enabled = obs_data_get_bool(settings, "environmental_enabled");
		filter->config.show_channel_labels = obs_data_get_bool(settings, "show_channel_labels");
		filter->config.debug_asr_output = obs_data_get_bool(settings, "debug_asr_output");
		filter->config.max_latency_ms = static_cast<int>(obs_data_get_int(settings, "max_latency_ms"));
		filter->config.asr_model_path = obs_data_get_string(settings, "asr_model_path");
		filter->config.noise_model_path = obs_data_get_string(settings, "noise_model_path");
		const char *configured_audio_source = obs_data_get_string(settings, "audio_source_name");
		if (configured_audio_source == nullptr || configured_audio_source[0] == '\0') {
			const std::string default_audio_source = get_default_audio_source_name();
			if (!default_audio_source.empty()) {
				obs_data_set_string(settings, "audio_source_name", default_audio_source.c_str());
				configured_audio_source = obs_data_get_string(settings, "audio_source_name");
			}
		}
		set_audio_capture_source(filter, configured_audio_source);
	}

	filter->overlay_source = create_overlay_source();
	if (filter->overlay_source == nullptr) {
		log_info("Subtitle overlay source unavailable; subtitles will only be logged");
	} else {
		update_overlay_source_text(filter, "[MultiSub] Listening for audio...");
	}

	filter->audio_engine.configure_models(filter->config.asr_model_path, filter->config.noise_model_path);
	filter->audio_engine.start();

	log_info("Created MultiSub filter instance");
	return filter;
}

static void multisub_filter_destroy(void *data)
{
	auto *filter = static_cast<MultiSubFilter *>(data);
	if (filter == nullptr) {
		return;
	}

	filter->audio_engine.stop();
	clear_audio_capture_source(filter);
	if (filter->overlay_source != nullptr) {
		obs_source_release(filter->overlay_source);
		filter->overlay_source = nullptr;
	}
	delete filter;
	log_info("Destroyed MultiSub filter instance");
}

static void multisub_filter_update(void *data, obs_data_t *settings)
{
	auto *filter = static_cast<MultiSubFilter *>(data);
	if (filter == nullptr || settings == nullptr) {
		return;
	}

	filter->config.dialogue_enabled = obs_data_get_bool(settings, "dialogue_enabled");
	filter->config.environmental_enabled = obs_data_get_bool(settings, "environmental_enabled");
	filter->config.show_channel_labels = obs_data_get_bool(settings, "show_channel_labels");
	filter->config.debug_asr_output = obs_data_get_bool(settings, "debug_asr_output");
	filter->config.max_latency_ms = static_cast<int>(obs_data_get_int(settings, "max_latency_ms"));
	filter->config.asr_model_path = obs_data_get_string(settings, "asr_model_path");
	filter->config.noise_model_path = obs_data_get_string(settings, "noise_model_path");
	const char *configured_audio_source = obs_data_get_string(settings, "audio_source_name");
	if (configured_audio_source == nullptr || configured_audio_source[0] == '\0') {
		const std::string default_audio_source = get_default_audio_source_name();
		if (!default_audio_source.empty()) {
			obs_data_set_string(settings, "audio_source_name", default_audio_source.c_str());
			configured_audio_source = obs_data_get_string(settings, "audio_source_name");
		}
	}
	set_audio_capture_source(filter, configured_audio_source);
	filter->audio_engine.configure_models(filter->config.asr_model_path, filter->config.noise_model_path);
}

static obs_properties_t *multisub_filter_properties(void *)
{
	obs_properties_t *props = obs_properties_create();
	obs_properties_add_bool(props, "dialogue_enabled", obs_module_text("MultiSub.DialogueEnabled"));
	obs_properties_add_bool(props, "environmental_enabled", obs_module_text("MultiSub.EnvironmentalEnabled"));
	obs_properties_add_bool(props, "show_channel_labels", obs_module_text("MultiSub.ShowChannelLabels"));
	obs_properties_add_bool(props, "debug_asr_output", obs_module_text("MultiSub.DebugAsrOutput"));
	obs_properties_add_int(props, "max_latency_ms", obs_module_text("MultiSub.MaxLatencyMs"), 50, 2000, 10);
	obs_property_t *audio_source_prop = obs_properties_add_list(props, "audio_source_name",
		obs_module_text("MultiSub.AudioSource"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(audio_source_prop, "Auto (filtered source)", "");
	obs_enum_sources(enum_audio_source_list, audio_source_prop);
	obs_properties_add_path(props, "asr_model_path", obs_module_text("MultiSub.AsrModelPath"), OBS_PATH_DIRECTORY,
		"", nullptr);
	obs_properties_add_path(props, "noise_model_path", obs_module_text("MultiSub.NoiseModelPath"), OBS_PATH_FILE,
		"ONNX model (*.onnx);;All files (*.*)", nullptr);
	return props;
}

static void multisub_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "dialogue_enabled", true);
	obs_data_set_default_bool(settings, "environmental_enabled", true);
	obs_data_set_default_bool(settings, "show_channel_labels", true);
	obs_data_set_default_bool(settings, "debug_asr_output", false);
	obs_data_set_default_int(settings, "max_latency_ms", 300);
	obs_data_set_default_string(settings, "audio_source_name", "");
	obs_data_set_default_string(settings, "asr_model_path", "");
	obs_data_set_default_string(settings, "noise_model_path", "");
}

static void multisub_filter_video_render(void *data, gs_effect_t *effect)
{
	auto *filter = static_cast<MultiSubFilter *>(data);
	UNUSED_PARAMETER(effect);
	if (filter == nullptr) {
		return;
	}

	if (!obs_source_process_filter_begin(filter->source, GS_RGBA, OBS_NO_DIRECT_RENDERING)) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	gs_effect_t *base_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	if (base_effect == nullptr) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	obs_source_process_filter_end(filter->source, base_effect, 0, 0);

	const uint32_t source_width = obs_source_get_base_width(filter->source);
	const uint32_t source_height = obs_source_get_base_height(filter->source);
	update_overlay_source_layout(filter, source_width, source_height);

	if (filter->overlay_source != nullptr && !filter->last_overlay_text.empty()) {
		const uint32_t overlay_width = obs_source_get_width(filter->overlay_source);
		const uint32_t overlay_height = obs_source_get_height(filter->overlay_source);
		const float x = source_width > overlay_width ? static_cast<float>(source_width - overlay_width) * 0.5f : 0.0f;
		const float y = source_height > overlay_height + 36 ? static_cast<float>(source_height - overlay_height - 36) : 0.0f;

		gs_matrix_push();
		gs_matrix_translate3f(x, y, 0.0f);
		obs_source_video_render(filter->overlay_source);
		gs_matrix_pop();
	}
}

static struct obs_source_info multisub_filter_info = {};

void register_multisub_filter()
{
	multisub_filter_info.id = "multisub_filter";
	multisub_filter_info.type = OBS_SOURCE_TYPE_FILTER;
	multisub_filter_info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
	multisub_filter_info.get_name = multisub_filter_get_name;
	multisub_filter_info.create = multisub_filter_create;
	multisub_filter_info.destroy = multisub_filter_destroy;
	multisub_filter_info.update = multisub_filter_update;
	multisub_filter_info.get_defaults = multisub_filter_defaults;
	multisub_filter_info.get_properties = multisub_filter_properties;
	multisub_filter_info.filter_audio = multisub_filter_filter_audio;
	multisub_filter_info.video_tick = multisub_filter_video_tick;
	multisub_filter_info.video_render = multisub_filter_video_render;

	obs_register_source(&multisub_filter_info);
}

} // namespace multisub
