"""Download parquet data files from Cloudflare R2 to the local repo.

Mirrors R2 object keys back to local paths. Keys are version-rooted (no
top-level package prefix), so an object at:
    v1/data/base_xg/train/even_strength.parquet
is written to:
    <repo_root>/chickenstats_xg/v1/data/base_xg/train/even_strength.parquet

Deduplication: downloads are skipped when the local file's MD5 matches the
remote ETag. This relies on objects having been uploaded with multipart disabled
(as upload_data.py does), so ETag == MD5(file) always holds.

Reliability: adaptive retries (up to 5 attempts with exponential backoff) and
a 5-minute read timeout handle R2 connection resets and slow transfers.

Required env vars (add to .env):
    R2_ENDPOINT_URL       https://<account_id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID      R2 API token access key
    R2_SECRET_ACCESS_KEY  R2 API token secret
    R2_BUCKET_NAME        source bucket name

Usage:
    uv run download-data --all
    uv run download-data --raw-pbp
    uv run download-data --v1 base_xg context_xg rapm
    uv run download-data --all --dry-run
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from chickenstats.utilities import ChickenProgress
from dotenv import load_dotenv

# Match upload_data.py: single-part only so ETag == MD5(file) for dedup.
_TRANSFER_CONFIG = TransferConfig(multipart_threshold=5 * 1024**3)

# Adaptive retries + 5-min read timeout for R2 reliability.
_CLIENT_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=300,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_V1_DATA = _REPO_ROOT / "chickenstats_xg" / "v1" / "data"
_RAW_PBP = _REPO_ROOT / "raw_data" / "pbp"

_R2_PREFIXES: dict[str, str] = {
    "raw_pbp": "raw_data/pbp/",
    "base_xg": "v1/data/base_xg/",
    "context_xg": "v1/data/context_xg/",
    "pred_goal": "v1/data/pred_goal/",
    "rapm": "v1/data/rapm/",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_objects(s3, bucket: str, prefix: str) -> list[dict]:
    """Return all objects under prefix (handles pagination)."""
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects


def _download_prefix(
    s3,
    bucket: str,
    prefix: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Download all .parquet objects under prefix. Returns (downloaded, skipped)."""
    objects = [o for o in _list_objects(s3, bucket, prefix) if o["Key"].endswith(".parquet")]
    if not objects:
        print(f"  (no parquet objects found under {prefix!r})")
        return 0, 0

    downloaded = skipped = 0
    with ChickenProgress() as progress:
        task = progress.add_task(f"Downloading {prefix}...", total=len(objects))
        for obj in objects:
            key = obj["Key"]
            remote_etag = obj["ETag"].strip('"')
            size_mb = obj["Size"] / 1_048_576

            # v1/... keys live under chickenstats_xg/ locally; raw_data/... keys are repo-root relative.
            if key.startswith("v1/"):
                local_path = _REPO_ROOT / "chickenstats_xg" / key
            else:
                local_path = _REPO_ROOT / key

            if local_path.exists() and _md5(local_path) == remote_etag:
                skipped += 1
                progress.update(task, advance=1)
                continue

            progress.update(task, description=f"{'[dry] ' if dry_run else ''}↓ {Path(key).name} ({size_mb:.1f} MB)")
            if not dry_run:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local_path), Config=_TRANSFER_CONFIG)
            downloaded += 1
            progress.update(task, advance=1)

    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Download parquet data files from Cloudflare R2.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download everything: raw PBP + all v1 data tiers.",
    )
    group.add_argument(
        "--raw-pbp",
        action="store_true",
        help="Download raw_data/pbp/ only.",
    )
    group.add_argument(
        "--v1",
        nargs="+",
        choices=["base_xg", "context_xg", "pred_goal", "rapm"],
        metavar="TIER",
        help="Download specific v1 data tier(s): base_xg context_xg pred_goal rapm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading.",
    )
    args = parser.parse_args()

    load_dotenv()

    endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("R2_BUCKET_NAME", "").strip()

    missing = [
        k
        for k, v in [
            ("R2_ENDPOINT_URL", endpoint),
            ("R2_ACCESS_KEY_ID", access_key),
            ("R2_SECRET_ACCESS_KEY", secret_key),
            ("R2_BUCKET_NAME", bucket),
        ]
        if not v
    ]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=_CLIENT_CONFIG,
    )

    if args.all:
        prefixes = list(_R2_PREFIXES.items())
    elif args.raw_pbp:
        prefixes = [("raw_pbp", _R2_PREFIXES["raw_pbp"])]
    else:
        prefixes = [(t, _R2_PREFIXES[t]) for t in args.v1]

    total_downloaded = total_skipped = 0
    for name, prefix in prefixes:
        print(f"\n[{name}] {prefix}")
        dn, sk = _download_prefix(s3, bucket, prefix, dry_run=args.dry_run)
        total_downloaded += dn
        total_skipped += sk

    action = "would download" if args.dry_run else "downloaded"
    print(f"\nDone — {action} {total_downloaded} file(s), skipped {total_skipped} unchanged.")


if __name__ == "__main__":
    main()
