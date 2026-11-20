"""Mirror the release payload to Hugging Face and ModelScope.

``tools/make_release_assets.py`` writes ``dist/release/``; this script pushes the
same files (byte identical, so one ``SHA256SUMS.txt`` covers all mirrors) to the
two model hubs and drops a model card next to them.

    python tools/upload_mirrors.py --hf-token <token> --modelscope-token <token>
    python tools/upload_mirrors.py --only hf

Tokens can also come from ``HF_TOKEN`` / ``MODELSCOPE_TOKEN``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "dist" / "release"
MODEL_CARD = ROOT / "docs" / "MODEL_CARD.md"

HF_REPO = "Mark42IRP/is3D-BlenderScript"
MODELSCOPE_REPO = "Mark42IRPC/is3D-BlenderScript"
HF_ENDPOINT = "https://hf-mirror.com"

PAYLOAD = [
    "SHA256SUMS.txt",
    "manifest.json",
    "is3d-addon-0.4.0.zip",
    "idol_engine.dll",
    "idol_weights.bin.part00",
    "idol_weights.bin.part01",
]


def payload(version):
    files = [name for name in PAYLOAD if (RELEASE / name).is_file()]
    if not files:
        raise SystemExit(f"nothing to upload, run tools/make_release_assets.py first ({RELEASE})")
    missing = [name for name in PAYLOAD if name.startswith("is3d-addon-") and version not in name]
    if missing and not any(name.startswith("is3d-addon-") for name in files):
        raise SystemExit("the add-on zip is missing from dist/release")
    return files


def upload_hf(token, files, card):
    from huggingface_hub import HfApi

    api = HfApi(endpoint=HF_ENDPOINT, token=token)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    print(f"hf: {HF_REPO}")
    steps = [(RELEASE / name, name) for name in files]
    if card:
        steps.append((card, "README.md"))
    for path, name in steps:
        started = time.time()
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=HF_REPO, repo_type="model")
        print(f"  {name}  {path.stat().st_size / 2**20:.1f} MiB  {time.time() - started:.0f}s", flush=True)


def upload_modelscope(token, files, card):
    from modelscope_hub.api import HubApi

    api = HubApi(token=token)
    api.create_repo(MODELSCOPE_REPO, repo_type="model", private=False)
    print(f"modelscope: {MODELSCOPE_REPO}")
    steps = [(RELEASE / name, name) for name in files]
    if card:
        steps.append((card, "README.md"))
    for path, name in steps:
        started = time.time()
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=MODELSCOPE_REPO, repo_type="model")
        print(f"  {name}  {path.stat().st_size / 2**20:.1f} MiB  {time.time() - started:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="0.4.0")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--modelscope-token", default=os.environ.get("MODELSCOPE_TOKEN"))
    parser.add_argument("--only", choices=["hf", "modelscope"], help="upload to one hub only")
    parser.add_argument("--skip-card", action="store_true")
    args = parser.parse_args()

    files = payload(args.version)
    card = None if args.skip_card or not MODEL_CARD.is_file() else MODEL_CARD
    errors = 0

    if args.only in (None, "hf"):
        if not args.hf_token:
            print("hf: skipped, no token", file=sys.stderr)
            errors += 1
        else:
            try:
                upload_hf(args.hf_token, files, card)
            except Exception as exc:  # noqa: BLE001 - maintainer script, report and continue
                print(f"hf: failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                errors += 1

    if args.only in (None, "modelscope"):
        if not args.modelscope_token:
            print("modelscope: skipped, no token", file=sys.stderr)
            errors += 1
        else:
            try:
                upload_modelscope(args.modelscope_token, files, card)
            except Exception as exc:  # noqa: BLE001
                print(f"modelscope: failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                errors += 1

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
