#pragma once

#include <memory>
#include <string>
#include <vector>

#include "multisub/audio-engine.h"

namespace multisub {

class AsrEngine {
public:
	AsrEngine();
	~AsrEngine();

	bool load_model(const std::string &model_path);
	bool is_loaded() const;
	void set_noise_gate_db(float noise_gate_db);
	std::string transcribe(const AudioChunk &chunk, float &confidence);

private:
	struct AsrBundleInfo {
		bool supported = false;
		bool has_encoder = false;
		bool has_decoder = false;
		bool has_vocab = false;
		std::string model_type;
		std::string vocab_path;
		std::vector<std::string> vocab_tokens;
	};

	struct SessionState;
	std::unique_ptr<SessionState> state_;
};

} // namespace multisub
