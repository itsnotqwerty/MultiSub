#pragma once

#include <obs-module.h>

#include "multisub/audio-engine.h"

struct audio_data;

namespace multisub {

AudioChunk capture_audio_chunk(const obs_audio_data *audio);
AudioChunk capture_audio_chunk(const audio_data *audio);

} // namespace multisub
