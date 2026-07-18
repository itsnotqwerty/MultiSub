# MultiSub

MultiSub is an OBS Studio plugin for real-time multi-modal subtitle generation.
It combines:

- microphone speech-to-text (ASR)
- environmental sound event classification
- lip reading (planned)
- sign language recognition (planned)

to produce richer subtitles with channel-aware output (Dialogue and Environmental).

## Status

The repository started from the official OBS plugin template.
Phase 1 audio MVP scaffold is implemented, with on-screen subtitle rendering still in progress:

- OBS filter registration and lifecycle
- threaded audio processing pipeline (placeholder inference)
- subtitle fusion and in-memory subtitle buffer
- architecture/spec/roadmap docs

## High-Level Architecture

- Audio Pipeline: mic audio -> preprocessing -> ASR + sound event classifier
- Vision Pipeline: frames -> lip reading + sign language recognition (future phases)
- Fusion Engine: confidence weighted merge into channel-tagged subtitle events
- Renderer: on-screen overlay in Phase 1, export pipeline in future phases
- OBS Integration: filter properties and runtime controls

Detailed design: [docs/architecture.md](docs/architecture.md)

## Build

This project uses the OBS plugin template CMake infrastructure.

Typical local flow:

1. Configure with your platform preset.
2. Build target using CMake.
3. Install/package using template helpers.

Example:

```bash
cmake --preset linux-x86_64
cmake --build --preset linux-x86_64
```

## Phase Plan

- Phase 1: audio-only ASR + environmental classification plus on-screen subtitle rendering
- Phase 2: lip reading fusion
- Phase 3: sign language recognition
- Phase 4: multi-channel renderer + UI polish

Roadmap details: [docs/roadmap.md](docs/roadmap.md)

## Dependencies (Planned)

- Core: libobs, CMake 3.28+
- ML runtime: ONNX Runtime (planned)
- Audio preprocessing: RNNoise/FFmpeg options (planned)
- Vision: OpenCV and/or MediaPipe (planned)

## Model Distribution

Models are not bundled in source by default. See [docs/models.md](docs/models.md) for licensing and download policy.

## Recommended ASR Decoder

For best subtitle quality, use the Canary ONNX bundle/decoder:

- Recommended model: canary-1b-v2-onnx
- Hugging Face repo: [istupakov/canary-1b-v2-onnx](https://huggingface.co/istupakov/canary-1b-v2-onnx)

When configuring the filter in OBS, set the ASR model path to the folder containing the ONNX bundle files (for example: encoder-model.onnx, decoder-model.onnx, config.json, and vocab.txt).
