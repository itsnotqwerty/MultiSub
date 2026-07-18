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
