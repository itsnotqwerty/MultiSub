#include "multisub/vision-engine.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

#include "lipread-engine.h"
#include "utils/logger.h"

namespace multisub {

namespace {

VideoFrame extract_mouth_roi(const VideoFrame &input)
{
	if (input.width == 0 || input.height == 0 || input.rgba.empty()) {
		return {};
	}

	const uint32_t roi_w = std::max<uint32_t>(1, input.width / 3);
	const uint32_t roi_h = std::max<uint32_t>(1, input.height / 4);
	const uint32_t roi_x = (input.width > roi_w) ? (input.width - roi_w) / 2 : 0;
	const uint32_t roi_y = (input.height > roi_h) ? ((input.height * 11) / 20) : 0;

	VideoFrame roi;
	roi.timestamp_ns = input.timestamp_ns;
	roi.width = 96;
	roi.height = 96;
	roi.rgba.resize(static_cast<size_t>(roi.width) * static_cast<size_t>(roi.height) * 4U, 0);

	for (uint32_t y = 0; y < roi.height; ++y) {
		const uint32_t src_y = roi_y + ((y * roi_h) / roi.height);
		for (uint32_t x = 0; x < roi.width; ++x) {
			const uint32_t src_x = roi_x + ((x * roi_w) / roi.width);
			const size_t src_idx = (static_cast<size_t>(src_y) * static_cast<size_t>(input.width) + static_cast<size_t>(src_x)) * 4U;
			const size_t dst_idx = (static_cast<size_t>(y) * static_cast<size_t>(roi.width) + static_cast<size_t>(x)) * 4U;
			if (src_idx + 3U >= input.rgba.size() || dst_idx + 3U >= roi.rgba.size()) {
				continue;
			}
			roi.rgba[dst_idx + 0U] = input.rgba[src_idx + 0U];
			roi.rgba[dst_idx + 1U] = input.rgba[src_idx + 1U];
			roi.rgba[dst_idx + 2U] = input.rgba[src_idx + 2U];
			roi.rgba[dst_idx + 3U] = 255U;
		}
	}

	return roi;
}

} // namespace

class VisionEngine::Impl {
public:
	void start()
	{
		running_ = true;
	}

	void stop()
	{
		running_ = false;
		std::lock_guard<std::mutex> lock(mutex_);
		frames_.clear();
	}

	void configure(VisionRuntimeConfig config)
	{
		config_ = std::move(config);
		LipReadEngine::RuntimeConfig lipread_config;
		lipread_config.model_path = config_.lipread_model_path;
		lipread_config.runner_python = config_.lipread_runner_python;
		lipread_config.min_confidence = config_.lipread_min_confidence;
		lipread_config.min_inference_frames = config_.lipread_min_inference_frames;
		lipread_config.enabled = config_.enabled;
		lipread_.set_runtime_config(std::move(lipread_config));

		if (!config_.enabled) {
			log_info("Vision engine disabled via runtime settings");
			return;
		}

		if (!lipread_.load_model(config_.lipread_model_path)) {
			log_warn("Vision engine could not load lip-read model path; continuing without visual hypotheses");
		}
	}

	void submit_frame(VideoFrame frame)
	{
		if (!running_ || !config_.enabled) {
			return;
		}

		VideoFrame preprocessed = extract_mouth_roi(frame);
		if (preprocessed.width == 0 || preprocessed.height == 0 || preprocessed.rgba.empty()) {
			return;
		}

		std::lock_guard<std::mutex> lock(mutex_);
		frames_.push_back(std::move(preprocessed));
		while (frames_.size() > kMaxWindowFrames) {
			frames_.pop_front();
		}
	}

	std::vector<TimelineEvent> consume_timeline_events()
	{
		if (!running_ || !config_.enabled || !lipread_.is_loaded()) {
			return {};
		}

		std::vector<VideoFrame> window;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (frames_.size() < config_.lipread_min_inference_frames) {
				return {};
			}
			window.assign(frames_.begin(), frames_.end());
			frames_.clear();
		}

		std::vector<TimelineEvent> events = lipread_.infer(window);
		events.erase(
			std::remove_if(events.begin(), events.end(), [](const TimelineEvent &event) {
				return event.text.empty();
			}),
			events.end());
		return events;
	}

private:
	static constexpr size_t kMaxWindowFrames = 48;

	bool running_ = false;
	VisionRuntimeConfig config_;
	std::mutex mutex_;
	std::deque<VideoFrame> frames_;
	LipReadEngine lipread_;
};

VisionEngine::VisionEngine() : impl_(new Impl) {}

VisionEngine::~VisionEngine()
{
	delete impl_;
	impl_ = nullptr;
}

void VisionEngine::start()
{
	impl_->start();
}

void VisionEngine::stop()
{
	impl_->stop();
}

void VisionEngine::configure(VisionRuntimeConfig config)
{
	impl_->configure(std::move(config));
}

void VisionEngine::submit_frame(VideoFrame frame)
{
	impl_->submit_frame(std::move(frame));
}

std::vector<TimelineEvent> VisionEngine::consume_timeline_events()
{
	return impl_->consume_timeline_events();
}

} // namespace multisub
