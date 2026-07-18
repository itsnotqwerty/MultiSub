# Model Requirements and Licensing

## Policy

Model binaries are not committed by default. MultiSub should load models from user-configured paths or downloaded model cache directories.

## Planned Models

- ASR model (ONNX)
- Environmental sound classifier (ONNX)
- Lip-reading model (ONNX, phase 2)
- Sign-language model (ONNX, phase 3)

## Licensing Constraints

Before enabling any default model download:

- Verify redistribution rights.
- Include upstream license notices.
- Preserve model card metadata where required.

## Runtime Compatibility

- Preferred backend: ONNX Runtime.
- Quantized variants (INT8/FP16) should be selectable for latency control.
- Fallback behavior must be explicit in logs when model loading fails.

## Local Setup

1. Install ONNX Runtime development files for your platform.
2. Configure CMake with `ENABLE_ONNX_RUNTIME=ON`.
3. Build the plugin and verify CMake reports ONNX Runtime linkage.
4. In OBS filter properties, set:
	- ASR ONNX Model Path
	- Noise Classifier ONNX Model Path

If ONNX Runtime is not found at configure time, MultiSub builds with placeholder inference and logs a warning.

## Runtime Loading Behavior

- Model paths are read from OBS filter settings.
- Audio worker reloads model sessions when paths change.
- Invalid paths disable only the affected model while keeping the filter active.
- Inference errors are handled per-chunk and do not stop processing.
