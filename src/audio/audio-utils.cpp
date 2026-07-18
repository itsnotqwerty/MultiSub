#include "audio-utils.h"

#include <cmath>

namespace multisub {

float estimate_chunk_energy(const AudioChunk &chunk)
{
	if (chunk.mono_samples.empty()) {
		return 0.0f;
	}

	float sum = 0.0f;
	for (float sample : chunk.mono_samples) {
		sum += sample * sample;
	}

	return std::sqrt(sum / static_cast<float>(chunk.mono_samples.size()));
}

} // namespace multisub
