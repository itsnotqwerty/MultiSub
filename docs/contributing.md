# Contributing

## Development Workflow

1. Build with the template CMake presets.
2. Run formatter helpers from `build-aux/`.
3. Keep module boundaries intact (`audio`, `vision`, `fusion`, `utils`).

## Code Guidelines

- Keep OBS callbacks lightweight and non-blocking.
- Run inference on worker threads.
- Use interfaces to isolate backend-specific dependencies.

## Pull Request Expectations

- Describe which phase milestone is advanced.
- Include performance notes if callback or queue behavior changes.
- Add/update tests for deterministic logic (fusion, buffering, parser behavior).
