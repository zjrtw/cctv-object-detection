# Repository Guidelines

## Project Structure & Module Organization

This repository is currently documentation-first. `docs/PRD.md` defines the product scope and target architecture, while `docs/TASKS.md` tracks implementation work. The planned Python application should keep `main.py` as a thin orchestration layer and separate responsibilities into `camera/`, `detector/`, `tracker/`, `recognition/`, `matcher/`, `notification/`, and `utils/`. Store runtime configuration in `config/`, reference face images in `faces/`, generated detection captures in `captures/`, and automated tests in `tests/`.

## Build, Test, and Development Commands

No executable source or dependency manifest has been committed yet. Once the Python scaffold exists, use the documented workflow:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python -m pytest
```

The first two commands create and activate an isolated environment. Install pinned dependencies before running the pipeline. Use `pytest` for the complete test suite. Update this section when project-specific scripts or required CLI arguments are introduced.

## Coding Style & Naming Conventions

Use Python 3.13 and four-space indentation. Follow PEP 8: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Keep model loading, frame processing, matching, persistence, and notification logic in their corresponding modules. Prefer type hints and small functions with explicit inputs and return values. Do not embed thresholds, model paths, camera indexes, or RTSP URLs in source code; place them in configuration.

## Testing Guidelines

Use `pytest`, with files named `tests/test_<module>.py` and tests named `test_<behavior>()`. Cover each module independently before adding pipeline tests. Include cases for missing frames, empty detections, absent faces, failed embedding extraction, threshold boundaries, duplicate-alert suppression, and webcam versus MP4 input. Keep test media small and exclude generated captures and large model weights from version control.

## Task Completion Tracking

After completing each step in `docs/TASKS.md`, immediately mark that step as completed by changing its checkbox from `[ ]` to `[x]`.

## Commit & Pull Request Guidelines

Git history is not available in the current directory, so no repository-specific commit convention can be inferred. Use short, imperative subjects such as `Add ByteTrack adapter` and keep each commit focused. Pull requests should describe the change, list verification commands, link the relevant task or issue, and note configuration or model requirements. Include screenshots or sample output when detection overlays or capture behavior changes.

## Security & Configuration

Never commit RTSP credentials, personal images, generated captures, or model secrets. Provide sanitized example configuration and keep local values in ignored files. Treat face embeddings and CCTV footage as sensitive data.
