#pragma once

#include <memory>
#include <string>
#include <vector>

#include "multisub/audio-engine.h"

namespace multisub {

class NoiseClassifier {
public:
	NoiseClassifier();
	~NoiseClassifier();

	bool load_model(const std::string &model_path);
	bool is_loaded() const;
	std::vector<std::string> classify(const AudioChunk &chunk, float &confidence);

private:
	struct SessionState;
	std::unique_ptr<SessionState> state_;
};

} // namespace multisub
