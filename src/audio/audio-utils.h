#pragma once

#include "multisub/audio-engine.h"

namespace multisub {

float estimate_chunk_energy(const AudioChunk &chunk);

} // namespace multisub
