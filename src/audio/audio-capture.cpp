#include "audio-capture.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

float decode_audio_sample(const uint8_t *plane, uint32_t index)
{
	if (plane == nullptr) {
		return 0.0f;
	}

	const float *as_float = reinterpret_cast<const float *>(plane);
	const float float_value = as_float[index];
	if (std::isfinite(float_value) && std::fabs(float_value) <= 8.0f) {
		return std::clamp(float_value, -1.0f, 1.0f);
	}

	const int16_t *as_s16 = reinterpret_cast<const int16_t *>(plane);
	return static_cast<float>(as_s16[index]) / 32768.0f;
}

} // namespace

namespace multisub {

AudioChunk capture_audio_chunk(const obs_audio_data *audio)
{
	AudioChunk chunk;
	chunk.timestamp_ns = static_cast<uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch())
			.count());
	chunk.sample_rate_hz = 48000;

	if (audio == nullptr) {
		return chunk;
	}

	chunk.sample_count = std::min<uint32_t>(audio->frames, 1600);
	chunk.mono_samples.resize(chunk.sample_count, 0.0f);

	const uint8_t *plane = audio->data[0];
	for (uint32_t i = 0; i < chunk.sample_count; ++i) {
		chunk.mono_samples[i] = decode_audio_sample(plane, i);
	}

	return chunk;
}

AudioChunk capture_audio_chunk(const audio_data *audio)
{
	AudioChunk chunk;
	chunk.timestamp_ns = audio != nullptr ? audio->timestamp : 0;
	chunk.sample_rate_hz = 48000;

	if (audio == nullptr) {
		return chunk;
	}

	chunk.sample_count = std::min<uint32_t>(audio->frames, 1600);
	chunk.mono_samples.resize(chunk.sample_count, 0.0f);

	const uint8_t *plane = audio->data[0];
	if (plane == nullptr) {
		return chunk;
	}

	for (uint32_t i = 0; i < chunk.sample_count; ++i) {
		chunk.mono_samples[i] = decode_audio_sample(plane, i);
	}

	return chunk;
}

} // namespace multisub
