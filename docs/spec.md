# MultiSub Technical Specification

## 1. Product Goal

MultiSub provides real-time subtitle generation in OBS Studio by combining multiple input modalities. The output distinguishes semantic channels such as spoken dialogue and environmental events.

## 2. Plugin Form Factor

- Primary mode: OBS filter attached to a video source.
- Optional mode: source plugin for standalone subtitle composition.
- Audio input: associated source audio and optional dedicated mic routing.

## 3. Functional Requirements

### 3.1 Input Processing

- Capture audio buffers from OBS callbacks.
- Run preprocessing (VAD/noise reduction where available).
- Produce dialogue transcript candidates.
- Produce environmental event candidates.
- In later phases, ingest frame data for lip and sign processing.
- Phase 2 scaffold: preprocess video frame windows with center-lower lip ROI extraction and normalization before lip-read inference.

### 3.2 Fusion

- Merge candidates by timestamp window.
- Score by confidence and freshness.
- Emit subtitle records tagged by channel:
	- Dialogue
	- Environmental

### 3.3 Output

- Internal buffer for immediate rendering/consumption.
- Export-ready subtitle event format (SRT/WebVTT planned).
- OBS text rendering integration (planned).

### 3.4 Settings

- Toggle channels.
- Configure max pipeline latency.
- Configure model backend paths (planned).
- Configure visual speech backend paths and thresholds:
	- AV-HuBERT model path (third_party repo/checkpoint or ONNX)
	- runner Python command
	- minimum lip-read confidence
	- minimum frame window size

### 3.5 Vision Runner IPC (Phase 2 Scaffold)

Native AV-HuBERT execution uses a JSON request/response bridge via `src/vision/avhubert_native_runner.py`.

Request schema:

```json
{
	"version": "1",
	"backend": "av_hubert_native",
	"model": {"path": "..."},
	"window": {
		"start_ns": 0,
		"end_ns": 0,
		"frame_count": 12,
		"frame_width": 96,
		"frame_height": 96
	},
	"hints": {"mode": "video", "max_hypotheses": 3}
}
```

Response schema:

```json
{
	"version": "1",
	"status": "ok",
	"backend": "av_hubert_native",
	"hypotheses": [
		{
			"text": "decoded text",
			"confidence": 0.82,
			"start_ns": 0,
			"end_ns": 0
		}
	],
	"error": ""
}
```

If `status` is `error` or `hypotheses` is empty, the visual stream degrades gracefully and fusion continues using available modalities.

Native decode activation:

- Set `MULTISUB_AVHUBERT_ENABLE_DECODE=1` to force third_party decode execution.
- Provide AV-HuBERT checkpoint and dataset paths through request `model.checkpoint_path` / `decode.{data_dir,label_dir}` or environment variables:
	- `MULTISUB_AVHUBERT_REPO`
	- `MULTISUB_AVHUBERT_CHECKPOINT`

Without these decode inputs, the runner returns a safe empty hypothesis set.

## 4. Non-Functional Requirements

- Real-time safe: OBS callbacks remain lightweight.
- Threaded inference to avoid render/audio callback stalls.
- Portable C++20 implementation.
- Pluggable ML backends through narrow interfaces.

## 5. Initial Data Model

- `AudioChunk`: PCM slice metadata and optional mono samples.
- `AudioResult`: transcript text/confidence and event labels.
- `SubtitleEvent`: channel, text, confidence, start/end timestamps.

## 6. Error Handling

- Log backend/model failures via OBS logger.
- Degrade gracefully to available channels.
- Continue operation when one model path is unavailable.

## 7. Phase 1 Scope (Implemented Scaffold)

- OBS filter registration and property skeleton.
- Audio callback ingestion into worker queue.
- Placeholder ASR and noise classifier implementations.
- Fusion to subtitle events and bounded buffer.

## 8. Out of Scope for Phase 1

- Real model inference runtime integration.
- Vision processing.
- Persistent subtitle export files.
- Full UI/Qt settings panel.
