"""Upload local parquet data files to Cloudflare R2.

Mirrors the local repo path structure as R2 object keys, stripping the
top-level package directory so keys are version-rooted, e.g.:
    <repo_root>/chickenstats_xg/v1/data/base_xg/train/even_strength.parquet
is uploaded as:
    v1/data/base_xg/train/even_strength.parquet

Deduplication: uploads are skipped when the remote object's ETag matches the
local MD5. To keep ETag == MD5(file) always true, multipart upload is disabled
(threshold set to 5 GB — larger than any parquet this repo will produce).
Multipart ETags are composite hashes and would break this check.

Reliability: adaptive retries (up to 5 attempts with exponential backoff) and
a 5-minute read timeout handle R2 connection resets and slow transfers.

Required env vars (add to .env):
    R2_ENDPOINT_URL       https://<account_id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID      R2 API token access key
    R2_SECRET_ACCESS_KEY  R2 API token secret
    R2_BUCKET_NAME        target bucket name

Usage:
    uv run upload-data --all
    uv run upload-data --raw-pbp
    uv run upload-data --v1 base_xg context_xg rapm
    uv run upload-data --all --dry-run
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

# Force single-part uploads so ETag == MD5(file) always.
# Multipart ETags are composite hashes (md5_part1+...+"-N") and would break dedup.
_TRANSFER_CONFIG = TransferConfig(multipart_threshold=5 * 1024 ** 3)

# Adaptive retries handle R2 connection resets; 5-min read timeout covers large files.
_CLIENT_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=300,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_V1_DATA   = _REPO_ROOT / "chickenstats_xg" / "v1" / "data"
_RAW_PBP   = _REPO_ROOT / "raw_data" / "pbp"

# Data directories keyed by logical name
_DATA_DIRS: dict[str, Path] = {
    "raw_pbp":   _RAW_PBP,
    "base_xg":   _V1_DATA / "base_xg",
    "context_xg": _V1_DATA / "context_xg",
    "pred_goal": _V1_DATA / "pred_goal",
    "rapm":      _V1_DATA / "rapm",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _r2_key(local_path: Path) -> str:
    rel = local_path.relative_to(_REPO_ROOT)
    # Strip the top-level package dir (chickenstats_xg/) so keys are version-rooted.
    if rel.parts[0] == "chickenstats_xg":
        rel = Path(*rel.parts[1:])
    return rel.as_posix()


def _upload_dir(
    s3,
    bucket: str,
    directory: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """Upload all .parquet files under directory. Returns (uploaded, skipped)."""
    uploaded = skipped = 0
    parquets = sorted(directory.rglob("*.parquet"))
    if not parquets:
        print(f"  (no parquet files found in {directory.relative_to(_REPO_ROOT)})")
        return 0, 0

    label = directory.relative_to(_REPO_ROOT).as_posix()
    with ChickenProgress() as progress:
        task = progress.add_task(f"Uploading {label}...", total=len(parquets))
        for path in parquets:
            key = _r2_key(path)
            local_md5 = _md5(path)
            size_mb = path.stat().st_size / 1_048_576

            try:
                head = s3.head_object(Bucket=bucket, Key=key)
                remote_etag = head["ETag"].strip('"')
                if remote_etag == local_md5:
                    skipped += 1
                    progress.update(task, advance=1)
                    continue
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                    raise

            progress.update(task, description=f"{'[dry] ' if dry_run else ''}↑ {path.name} ({size_mb:.1f} MB)")
            if not dry_run:
                s3.upload_file(str(path), bucket, key, Config=_TRANSFER_CONFIG)
            uploaded += 1
            progress.update(task, advance=1)

    return uploaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload parquet data files to Cloudflare R2."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", "-a", action="store_true",
        help="Upload everything: raw PBP + all v1 data tiers.",
    )
    group.add_argument(
        "--raw-pbp", action="store_true",
        help="Upload raw_data/pbp/ only.",
    )
    group.add_argument(
        "--v1", nargs="+",
        choices=["base_xg", "context_xg", "pred_goal", "rapm"],
        metavar="TIER",
        help="Upload specific v1 data tier(s): base_xg context_xg pred_goal rapm",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading.",
    )
    parser.add_argument(
    )
    args = parser.parse_args()

    load_dotenv()

    endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.environ.get("R2_BUCKET_NAME", "").strip()

    missing = [k for k, v in [
        ("R2_ENDPOINT_URL", endpoint),
        ("R2_ACCESS_KEY_ID", access_key),
        ("R2_SECRET_ACCESS_KEY", secret_key),
        ("R2_BUCKET_NAME", bucket),
    ] if not v]
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
        dirs_to_upload = list(_DATA_DIRS.items())
    elif args.raw_pbp:
        dirs_to_upload = [("raw_pbp", _DATA_DIRS["raw_pbp"])]
    else:
        dirs_to_upload = [(t, _DATA_DIRS[t]) for t in args.v1]

    total_uploaded = total_skipped = 0
    for name, directory in dirs_to_upload:
        if not directory.exists():
            print(f"  [{name}] directory not found: {directory} — skipping")
            continue
        print(f"\n[{name}] {directory.relative_to(_REPO_ROOT)}")
        up, sk = _upload_dir(s3, bucket, directory, dry_run=args.dry_run)
        total_uploaded += up
        total_skipped += sk

    action = "would upload" if args.dry_run else "uploaded"
    print(f"\nDone — {action} {total_uploaded} file(s), skipped {total_skipped} unchanged.")


if __name__ == "__main__":
    main()