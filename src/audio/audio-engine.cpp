#include "multisub/audio-engine.h"

#include <condition_variable>
#include <thread>
#include <utility>

#include "asr-engine.h"
#include "noise-classifier.h"

namespace multisub {

AudioEngine::AudioEngine() = default;

AudioEngine::AudioEngine(AsrTranscriber asr_transcriber, NoiseClassifierFn noise_classifier)
	: asr_transcriber_(std::move(asr_transcriber)), noise_classifier_(std::move(noise_classifier))
{
}

AudioEngine::~AudioEngine()
{
	stop();
}

void AudioEngine::start()
{
	if (running_) {
		return;
	}

	running_ = true;
	worker_ = new std::thread(&AudioEngine::process_loop, this);
}

void AudioEngine::stop()
{
	{
		std::lock_guard<std::mutex> lock(queue_mutex_);
		running_ = false;
	}

	queue_cv_.notify_all();

	if (worker_ != nullptr) {
		if (worker_->joinable()) {
			worker_->join();
		}
		delete worker_;
		worker_ = nullptr;
	}
}

void AudioEngine::configure_models(std::string asr_model_path, std::string noise_model_path)
{
	{
		std::lock_guard<std::mutex> lock(config_mutex_);
		asr_model_path_ = std::move(asr_model_path);
		noise_model_path_ = std::move(noise_model_path);
		reload_models_ = true;
	}
	queue_cv_.notify_one();
}

void AudioEngine::submit_chunk(AudioChunk chunk)
{
	{
		std::lock_guard<std::mutex> lock(queue_mutex_);
		queue_.push_back(std::move(chunk));
	}
	queue_cv_.notify_one();
}

std::vector<AudioResult> AudioEngine::consume_results()
{
	std::lock_guard<std::mutex> lock(output_mutex_);
	std::vector<AudioResult> out(output_.begin(), output_.end());
	output_.clear();
	return out;
}

void AudioEngine::process_loop()
{
	AsrEngine asr;
	NoiseClassifier classifier;
	const bool use_custom_asr = static_cast<bool>(asr_transcriber_);
	const bool use_custom_noise_classifier = static_cast<bool>(noise_classifier_);
	std::string active_asr_path;
	std::string active_noise_path;

	for (;;) {
		std::string pending_asr_path;
		std::string pending_noise_path;
		bool should_reload = false;
		{
			std::lock_guard<std::mutex> lock(config_mutex_);
			if (reload_models_) {
				pending_asr_path = asr_model_path_;
				pending_noise_path = noise_model_path_;
				reload_models_ = false;
				should_reload = true;
			}
		}

		if (should_reload) {
			if (!use_custom_asr && pending_asr_path != active_asr_path) {
				asr.load_model(pending_asr_path);
				active_asr_path = pending_asr_path;
			}

			if (!use_custom_noise_classifier && pending_noise_path != active_noise_path) {
				classifier.load_model(pending_noise_path);
				active_noise_path = pending_noise_path;
			}
		}

		AudioChunk chunk;
		{
			std::unique_lock<std::mutex> lock(queue_mutex_);
			queue_cv_.wait(lock, [this] { return !running_ || !queue_.empty(); });
			if (!running_ && queue_.empty()) {
				return;
			}

			chunk = std::move(queue_.front());
			queue_.pop_front();
		}

		AudioResult result;
		result.timestamp_ns = chunk.timestamp_ns;
		if (use_custom_asr) {
			result.transcript = asr_transcriber_(chunk, result.transcript_confidence);
		} else {
			result.transcript = asr.transcribe(chunk, result.transcript_confidence);
		}

		if (use_custom_noise_classifier) {
			result.environmental_events = noise_classifier_(chunk, result.environmental_confidence);
		} else {
			result.environmental_events = classifier.classify(chunk, result.environmental_confidence);
		}

		std::lock_guard<std::mutex> output_lock(output_mutex_);
		output_.push_back(std::move(result));
		if (output_.size() > 128) {
			output_.pop_front();
		}
	}
}

} // namespace multisub
