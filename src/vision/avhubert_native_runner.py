#!/usr/bin/env python3
"""Native AV-HuBERT runner bridge for MultiSub.

This script consumes a JSON request file and writes a JSON response file.
It currently supports a deterministic mock path (for tests and local bring-up)
and validates third_party AV-HuBERT paths before optional native execution.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("request root must be an object")
    return data


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)


def run_mock_response(request: dict[str, Any]) -> dict[str, Any]:
    text = os.environ.get("MULTISUB_LIPREAD_MOCK_RESPONSE_TEXT", "")
    confidence_raw = os.environ.get("MULTISUB_LIPREAD_MOCK_RESPONSE_CONFIDENCE", "0.0")
    try:
        confidence = float(confidence_raw)
    except ValueError:
        confidence = 0.0

    window = request.get("window") if isinstance(request.get("window"), dict) else {}
    start_ns = int(window.get("start_ns", 0))
    end_ns = int(window.get("end_ns", start_ns))

    hypotheses: list[dict[str, Any]] = []
    if text:
        hypotheses.append(
            {
                "text": text,
                "confidence": confidence,
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
        )

    return {
        "version": "1",
        "status": "ok",
        "backend": "mock",
        "hypotheses": hypotheses,
        "error": "",
    }


def validate_avhubert_layout(model_path: pathlib.Path) -> bool:
    if model_path.is_file() and model_path.suffix == ".pt":
        return True
    if model_path.is_dir() and (model_path / "avhubert" / "infer_s2s.py").exists():
        return True
    return False


def normalize_repo_and_checkpoint(
    model_path: pathlib.Path, request: dict[str, Any]
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    model = request.get("model") if isinstance(request.get("model"), dict) else {}
    checkpoint_override = model.get("checkpoint_path")
    repo_override = model.get("repo_path")

    repo_path: pathlib.Path | None = None
    checkpoint_path: pathlib.Path | None = None

    if model_path.is_file() and model_path.suffix == ".pt":
        checkpoint_path = model_path
    elif model_path.is_dir() and (model_path / "avhubert" / "infer_s2s.py").exists():
        repo_path = model_path

    if isinstance(repo_override, str) and repo_override.strip():
        repo_path = pathlib.Path(repo_override).expanduser()
    if isinstance(checkpoint_override, str) and checkpoint_override.strip():
        checkpoint_path = pathlib.Path(checkpoint_override).expanduser()

    if repo_path is None:
        env_repo = os.environ.get("MULTISUB_AVHUBERT_REPO")
        if env_repo:
            repo_path = pathlib.Path(env_repo).expanduser()

    if checkpoint_path is None:
        env_ckpt = os.environ.get("MULTISUB_AVHUBERT_CHECKPOINT")
        if env_ckpt:
            checkpoint_path = pathlib.Path(env_ckpt).expanduser()

    return repo_path, checkpoint_path


def run_third_party_decode(
    request: dict[str, Any], repo_path: pathlib.Path, checkpoint_path: pathlib.Path
) -> dict[str, Any]:
    decode = request.get("decode") if isinstance(request.get("decode"), dict) else {}
    data_dir = decode.get("data_dir")
    label_dir = decode.get("label_dir")
    results_dir = decode.get("results_dir")
    gen_subset = str(decode.get("gen_subset", "test"))
    config_name = str(decode.get("config_name", "conf-name"))

    if not data_dir or not label_dir:
        return {
            "version": "1",
            "status": "error",
            "backend": "av_hubert_native",
            "hypotheses": [],
            "error": "missing_decode_data_or_label_dir",
        }

    if not checkpoint_path.exists() or checkpoint_path.suffix != ".pt":
        return {
            "version": "1",
            "status": "error",
            "backend": "av_hubert_native",
            "hypotheses": [],
            "error": "missing_or_invalid_checkpoint_pt",
        }

    infer_script = repo_path / "avhubert" / "infer_s2s.py"
    if not infer_script.exists():
        return {
            "version": "1",
            "status": "error",
            "backend": "av_hubert_native",
            "hypotheses": [],
            "error": "infer_s2s_not_found",
        }

    results_path = (
        pathlib.Path(results_dir).expanduser()
        if results_dir
        else (pathlib.Path(data_dir).expanduser() / "decode-out")
    )
    results_path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(infer_script),
        "--config-dir",
        str(repo_path / "avhubert" / "conf"),
        "--config-name",
        config_name,
        f"dataset.gen_subset={gen_subset}",
        f"common_eval.path={checkpoint_path}",
        f"common_eval.results_path={results_path}",
        "override.modalities=['video']",
        f"override.data={pathlib.Path(data_dir).expanduser()}",
        f"override.label_dir={pathlib.Path(label_dir).expanduser()}",
        f"common.user_dir={repo_path / 'avhubert'}",
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout or "decode_failed").strip()
        return {
            "version": "1",
            "status": "error",
            "backend": "av_hubert_native",
            "hypotheses": [],
            "error": error_text[:500],
        }

    return {
        "version": "1",
        "status": "ok",
        "backend": "av_hubert_native",
        "hypotheses": [],
        "error": "decode_completed_no_inline_hypotheses",
    }


def run_native(request: dict[str, Any]) -> dict[str, Any]:
    model = request.get("model") if isinstance(request.get("model"), dict) else {}
    model_path = pathlib.Path(str(model.get("path", ""))).expanduser()

    if not validate_avhubert_layout(model_path):
        return {
            "version": "1",
            "status": "error",
            "backend": "av_hubert_native",
            "hypotheses": [],
            "error": "invalid_avhubert_model_path",
        }

    force_decode = (
        request.get("force_decode") is True
        or os.environ.get("MULTISUB_AVHUBERT_ENABLE_DECODE") == "1"
    )
    if force_decode:
        repo_path, checkpoint_path = normalize_repo_and_checkpoint(model_path, request)
        if repo_path is None or not repo_path.exists():
            return {
                "version": "1",
                "status": "error",
                "backend": "av_hubert_native",
                "hypotheses": [],
                "error": "missing_avhubert_repo_path",
            }
        if checkpoint_path is None:
            return {
                "version": "1",
                "status": "error",
                "backend": "av_hubert_native",
                "hypotheses": [],
                "error": "missing_checkpoint_path",
            }
        return run_third_party_decode(request, repo_path, checkpoint_path)

    # Future step: call AV-HuBERT decode using third_party implementation.
    # For now return a valid empty hypothesis set to keep the real-time path safe.
    return {
        "version": "1",
        "status": "ok",
        "backend": "av_hubert_native",
        "hypotheses": [],
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AV-HuBERT native inference bridge"
    )
    parser.add_argument("--request", required=True, help="Path to request JSON")
    parser.add_argument("--response", required=True, help="Path to response JSON")
    args = parser.parse_args()

    request_path = pathlib.Path(args.request)
    response_path = pathlib.Path(args.response)

    try:
        request = load_json(request_path)
        force_mock = request.get("force_mock") is True or bool(
            os.environ.get("MULTISUB_LIPREAD_MOCK_RESPONSE_TEXT")
        )
        payload = run_mock_response(request) if force_mock else run_native(request)
    except Exception as exc:  # pylint: disable=broad-except
        payload = {
            "version": "1",
            "status": "error",
            "backend": "runner",
            "hypotheses": [],
            "error": str(exc),
        }

    write_json(response_path, payload)
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
