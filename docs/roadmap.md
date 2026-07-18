# MultiSub Roadmap

## Phase 0: Foundation (current)

- [x] Start from OBS plugin template baseline.
- [x] Define architecture, spec, and implementation roadmap.
- [x] Introduce C++ module layout for audio/fusion scaffolding.

## Phase 1: Audio MVP

Objective: produce usable channel-tagged subtitles from audio only.

- [x] Register MultiSub as OBS filter.
- [x] Add threaded audio ingestion and processing queue.
- [x] Implement placeholder ASR and noise event classifier wrappers.
- [x] Add fusion and subtitle buffer components.
- [x] Add on-screen subtitle rendering path in filter render callback.
- [x] Add runtime settings for latency and channel toggles.

Phase 1 does not include file export; that is deferred to Phase 4.

Exit criteria:

- Plugin runs stably as filter.
- Dialogue/environment records are produced continuously from audio callback data.
- No callback-time blocking from inference work.

## Phase 2: Visual Speech

Objective: improve dialogue confidence with lip-reading signals.

- [ ] Frame processor and face/lip ROI extraction.
- [ ] Lip-reading model wrapper.
- [ ] Timestamp alignment and confidence fusion with ASR.

## Phase 3: Sign Language

Objective: add text channel from sign-language gestures.

- [ ] Hand/pose tracking pipeline.
- [ ] Sign recognition model wrapper.
- [ ] Fusion policy updates for sign + speech conflict handling.

## Phase 4: UX and Distribution

- [ ] SRT/WebVTT exporter.
- [ ] Multi-channel styled renderer.
- [ ] Advanced properties panel.
- [ ] Model download/validation flow.
- [ ] Packaging and release automation hardening.
