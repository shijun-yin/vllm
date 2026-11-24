import argparse
import base64
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: expected SHA256 {expected}, got {actual}")
    print(f"OK {path} {actual}", flush=True)


def download_github(entry, path):
    url = (
        f"https://api.github.com/repos/{entry['repo']}/contents/{entry['file']}"
        f"?ref={entry['revision']}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "suffix-splice-h20-validation"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = base64.b64decode(json.load(response)["content"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("SPLICE_DATA_ROOT", "/root/qwen35-9b-deploy/data")),
    )
    parser.add_argument(
        "--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(__file__).with_name("datasets.json")
    manifest = json.loads(manifest_path.read_text())

    for name, entry in manifest.items():
        path = args.root / entry["local"]
        if args.check_only:
            verify(path, entry["sha256"])
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and sha256(path) == entry["sha256"]:
            verify(path, entry["sha256"])
            continue
        print(f"Downloading {name} at {entry['revision']}", flush=True)
        if entry["type"] == "huggingface":
            downloaded = hf_hub_download(
                entry["repo"],
                entry["file"],
                repo_type="dataset",
                revision=entry["revision"],
                endpoint=args.endpoint,
                local_dir=args.root / Path(entry["local"]).parts[0],
            )
            if Path(downloaded) != path:
                raise ValueError(f"Unexpected download path: {downloaded}")
        elif entry["type"] == "github_contents":
            download_github(entry, path)
        else:
            raise ValueError(f"Unknown source type: {entry['type']}")
        verify(path, entry["sha256"])


if __name__ == "__main__":
    main()
