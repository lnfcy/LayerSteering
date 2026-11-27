#!/usr/bin/env python3
"""Restore final d-Tuned Lens weights from their original binary fragments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
MODELS = ("llada-inst", "llada-1.5")
PART_COUNT = 11
ORIGINAL_BYTES = 1_074_010_376
MAX_PART_BYTES = 100 * 1024 * 1024
CHUNK = 4 * 1024 * 1024


class DirectDownload(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


DOWNLOAD = build_opener(DirectDownload())


def mirror_base(repo_url):
    match = re.fullmatch(r"https://anonymous\.4open\.science/r/([A-Za-z0-9_-]+)/?", repo_url)
    if not match:
        raise ValueError("Use the anonymous repository's root /r/ URL.")
    repo_id = match.group(1)
    return f"https://anonymous.4open.science/api/repo/{repo_id}/file/checkpoints/"


def artifact(relative_path, mirror):
    local = ROOT / relative_path
    if local.is_file():
        return local.open("rb")
    if mirror is None:
        raise FileNotFoundError(f"Missing {local}. Supply --repo-url to download it.")
    url = mirror + relative_path
    return DOWNLOAD.open(Request(url, headers={"Accept-Encoding": "identity"}), timeout=60)


def manifest_models(mirror):
    with artifact("manifest.json", mirror) as stream:
        data = stream.read(64 * 1024 + 1)
    if len(data) > 64 * 1024:
        raise ValueError("Invalid checkpoint manifest size.")
    manifest = json.loads(data)
    if manifest["version"] != 1 or set(manifest["models"]) != set(MODELS):
        raise ValueError("Unexpected checkpoint manifest format.")
    for name, model in manifest["models"].items():
        parts = model["parts"]
        if len(parts) != PART_COUNT or model["bytes"] != ORIGINAL_BYTES:
            raise ValueError(f"{name}: unexpected model or part count.")
        for entry in [model, *parts]:
            if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise ValueError(f"{name}: invalid SHA-256 in manifest.")
        if any(type(part["bytes"]) is not int or not 0 < part["bytes"] <= MAX_PART_BYTES
               for part in parts):
            raise ValueError(f"{name}: invalid part size.")
        if sum(part["bytes"] for part in parts) != ORIGINAL_BYTES:
            raise ValueError(f"{name}: part sizes do not sum to the original size.")
    return manifest["models"]


def restore(model, record, mirror, output_root):
    target = output_root / model / "tuned_lens" / "checkpoint-final" / "model.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = hashlib.sha256()
        if target.stat().st_size == record["bytes"]:
            with target.open("rb") as original:
                for block in iter(lambda: original.read(CHUNK), b""):
                    digest.update(block)
            if digest.hexdigest() == record["sha256"]:
                print(f"Verified existing {target}", flush=True)
                return
        raise FileExistsError(f"{target} already exists with different contents.")

    whole_hash = hashlib.sha256()
    output = tempfile.NamedTemporaryFile(dir=target.parent, prefix=".lens-", delete=False)
    pending = Path(output.name)
    try:
        with output:
            for index, part in enumerate(record["parts"]):
                relative = f"shards/{model}/part-{index:02d}.bin"
                size = 0
                part_hash = hashlib.sha256()
                with artifact(relative, mirror) as stream:
                    for block in iter(lambda: stream.read(CHUNK), b""):
                        size += len(block)
                        if size > part["bytes"]:
                            raise ValueError(f"{relative}: larger than expected.")
                        output.write(block)
                        part_hash.update(block)
                        whole_hash.update(block)
                if size != part["bytes"] or part_hash.hexdigest() != part["sha256"]:
                    raise ValueError(f"{relative}: byte count or SHA-256 mismatch.")
                print(f"{model}: verified part {index + 1}/{PART_COUNT}", flush=True)

            if whole_hash.hexdigest() != record["sha256"]:
                raise ValueError(f"{model}: restored SHA-256 mismatch.")
        os.link(pending, target)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Restored {target} (SHA-256 {record['sha256']})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", help="Anonymous review repository URL, to fetch missing files.")
    parser.add_argument("--model", choices=MODELS, help="Restore one model. The default restores both.")
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Checkpoint destination directory.")
    args = parser.parse_args()
    mirror = mirror_base(args.repo_url) if args.repo_url else None
    models = manifest_models(mirror)
    for model in (args.model,) if args.model else MODELS:
        restore(model, models[model], mirror, args.output_dir)


if __name__ == "__main__":
    main()
