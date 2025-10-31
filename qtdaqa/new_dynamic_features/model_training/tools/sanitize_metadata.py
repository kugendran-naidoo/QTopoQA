#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

SENSITIVE_PATTERNS = [
    re.compile(r"openai", re.IGNORECASE),
    re.compile(r"anthropic", re.IGNORECASE),
    re.compile(r"huggingface", re.IGNORECASE),
    re.compile(r"azure", re.IGNORECASE),
    re.compile(r"aws", re.IGNORECASE),
    re.compile(r"gcp|google", re.IGNORECASE),
    re.compile(r"cohere", re.IGNORECASE),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"access[_-]?key", re.IGNORECASE),
    re.compile(r"client[_-]?secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"bearer", re.IGNORECASE),
]

RUN_ROOT = Path(__file__).resolve().parents[1] / "training_runs"
TARGET_KEYS = {
    "OPENAI_API_KEY",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "ANTHROPIC_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "COHERE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_API_KEY",
    "GCP_API_KEY",
    "HF_HOME",
    "HF_TOKEN",
    "HF_API_TOKEN",
    "HF_ENDPOINT",
    "HF_INFERENCE_API_KEY",
}  # explicit blacklist; extend as needed


def is_sensitive(key: str, value: str | None) -> bool:
    if key in TARGET_KEYS:
        return True
    text = f"{key}={value}".lower() if value else key.lower()
    return any(pat.search(text) for pat in SENSITIVE_PATTERNS)


def scrub_run_metadata(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    env = data.get("environment")
    if not isinstance(env, dict):
        return False
    sanitized = {k: v for k, v in env.items() if not is_sensitive(k, v if isinstance(v, str) else None)}
    removed = set(env) - set(sanitized)
    if removed:
        data["environment"] = sanitized
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return bool(removed)


def main() -> None:
    scrubbed = 0
    for json_path in RUN_ROOT.rglob("run_metadata.json"):
        if scrub_run_metadata(json_path):
            print(f"Scrubbed: {json_path}")
            scrubbed += 1
    print(f"Done. Files scrubbed: {scrubbed}")


if __name__ == "__main__":
    main()
