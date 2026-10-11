# Contributing

## Development Workflow

1. Build with the template CMake presets.
2. Run formatter helpers from `build-aux/`.
3. Keep module boundaries intact (`audio`, `vision`, `fusion`, `utils`).

## Arch Linux Build and Install

Use the Arch preset to install into the system OBS plugin directories:

```sh
cmake --preset arch-x86_64
cmake --build --preset arch-x86_64 --clean-first
ctest --test-dir build_arch_x86_64 -R '^multisub-unit$' --output-on-failure
sudo cmake --install build_arch_x86_64
```

Restart OBS after installation. Rebuild and reinstall after ONNX Runtime upgrades:
an installed plugin linked against an older `VERS_1.x` symbol version may fail to
load even when the source still compiles against the current OBS release.

## Code Guidelines

- Keep OBS callbacks lightweight and non-blocking.
- Run inference on worker threads.
- Use interfaces to isolate backend-specific dependencies.

## Pull Request Expectations

- Describe which phase milestone is advanced.
- Include performance notes if callback or queue behavior changes.
- Add/update tests for deterministic logic (fusion, buffering, parser behavior).
