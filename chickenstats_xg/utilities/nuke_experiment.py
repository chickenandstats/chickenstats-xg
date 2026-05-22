"""Delete an MLflow experiment and its Optuna study completely.

Deletion order:
  1. S3 artifacts — all objects under each run's artifact prefix (skips gracefully
     if already deleted or if the prefix never existed)
  2. MLflow soft-delete — runs and experiment marked DELETED via tracking API
  3. MLflow hard delete — two modes (controlled by --use-gc flag):
       Direct SQL (default): SQLAlchemy DELETE statements against the tracking
         PostgreSQL. Fast, no hang risk, but requires MLFLOW_BACKEND_STORE_URI
         and won't auto-adapt if MLflow adds new tables in future versions.
       mlflow gc (--use-gc): MLflow's official hard-delete CLI. Handles all
         current and future schema tables (model registry, traces, etc.)
         automatically, but re-attempts S3 artifact deletion which can hang
         if the endpoint is slow. Safe to use if S3 is responsive.
  4. Optuna study — hard-deleted from Optuna's PostgreSQL via optuna.delete_study()

Study name convention (matches experiments.py):
    {strength}-{version}-{base|informed}

Examples:
    # Always dry-run first
    python nuke_experiment.py --study even_strength-v1-base --dry-run

    # Nuke using direct SQL (default, no hang risk)
    python nuke_experiment.py --study even_strength-v1-base --confirm

    # Nuke using mlflow gc (official path, handles model registry / traces)
    python nuke_experiment.py --study even_strength-v1-base --confirm --use-gc

    # Build the study name from component parts
    python nuke_experiment.py --strength even_strength --version v1 --model base_xg --confirm

Environment variables (from .env):
    MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
        — remote MLflow tracking server
    MLFLOW_BACKEND_STORE_URI
        — direct PostgreSQL URI for the MLflow server's backend DB (required for hard delete)
        — format: postgresql+psycopg2://user:password@host:port/dbname
    MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
        — S3-compatible artifact store credentials
    DB_* — Optuna PostgreSQL connection
"""

import argparse
import os
import subprocess
import sys
import urllib.parse

import boto3
import mlflow
import mlflow.entities
import optuna
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text


# ── S3 helpers ────────────────────────────────────────────────────────────────


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """s3://bucket/key/path  →  (bucket, key/path)"""
    parsed = urllib.parse.urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _delete_s3_prefix(s3, bucket: str, prefix: str) -> int:
    """Delete all objects under prefix. Returns count deleted. Safe if prefix is empty."""
    paginator = s3.get_paginator("list_objects_v2")
    total = 0
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if not objects:
                continue
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
            total += len(objects)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        print(f"  [s3]     skipped (bucket error: {code})")
    return total


# ── MLflow hard-delete helpers ───────────────────────────────────────────────


def _mlflow_hard_delete_gc(backend_uri: str, experiment_id: str, dry_run: bool) -> None:
    """Hard-delete via `mlflow gc` (official path).

    Pro: handles model registry, traces, and any future schema additions automatically.
    Con: re-attempts S3 artifact deletion, which can hang on slow endpoints.
    """
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "gc",
        "--backend-store-uri",
        backend_uri,
        "--experiment-ids",
        experiment_id,
    ]
    if dry_run:
        print(f"  [mlflow] [dry] would run: {' '.join(cmd)}")
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [mlflow] WARNING: mlflow gc exited {result.returncode}")
        if result.stderr:
            print(f"           stderr: {result.stderr.strip()}")
    else:
        print("  [mlflow] Hard delete (gc) complete.")
        if result.stdout.strip():
            print(f"           {result.stdout.strip()}")


# ── MLflow hard-delete via direct SQL ────────────────────────────────────────


def _mlflow_hard_delete_sql(
    backend_uri: str,
    experiment_id: str,
    run_ids: list[str],
    dry_run: bool,
) -> None:
    """Permanently remove experiment + run records from MLflow's PostgreSQL.

    Avoids `mlflow gc` which re-attempts S3 artifact deletion and can hang.
    Handles missing tables gracefully (schema varies across MLflow versions).
    """
    if dry_run:
        print(
            f"  [mlflow] [dry] would hard delete {len(run_ids)} run records "
            f"and experiment id={experiment_id} from PostgreSQL."
        )
        return

    engine = create_engine(backend_uri)
    existing = set(inspect(engine).get_table_names())

    def _del(conn, table: str, where: str, params: dict) -> int:
        if table not in existing:
            return 0
        return conn.execute(text(f"DELETE FROM {table} WHERE {where}"), params).rowcount

    with engine.begin() as conn:
        counts: dict[str, int] = {}

        if run_ids:
            # Resolve logged_model IDs before deleting the models themselves
            if "logged_models" in existing:
                rows = conn.execute(
                    text("SELECT model_id FROM logged_models WHERE run_id = ANY(:ids)"),
                    {"ids": run_ids},
                ).fetchall()
                model_ids = [r[0] for r in rows]
                if model_ids:
                    for t in ("logged_model_params", "logged_model_tags", "logged_model_metrics"):
                        n = _del(conn, t, "model_id = ANY(:ids)", {"ids": model_ids})
                        if n:
                            counts[t] = n
                n = _del(conn, "logged_models", "run_id = ANY(:ids)", {"ids": run_ids})
                if n:
                    counts["logged_models"] = n

            # Resolve input UUIDs before deleting inputs
            if "inputs" in existing:
                rows = conn.execute(
                    text("SELECT input_uuid FROM inputs WHERE destination_id = ANY(:ids) AND destination_type = 'RUN'"),
                    {"ids": run_ids},
                ).fetchall()
                input_uuids = [r[0] for r in rows]
                if input_uuids:
                    _del(conn, "input_tags", "input_uuid = ANY(:ids)", {"ids": input_uuids})
                _del(conn, "inputs", "destination_id = ANY(:ids) AND destination_type = 'RUN'", {"ids": run_ids})

            for t in ("params", "metrics", "latest_metrics", "tags"):
                n = _del(conn, t, "run_uuid = ANY(:ids)", {"ids": run_ids})
                if n:
                    counts[t] = n

        n = _del(conn, "runs", "experiment_id = :eid", {"eid": int(experiment_id)})
        if n:
            counts["runs"] = n

        _del(conn, "experiment_tags", "experiment_id = :eid", {"eid": int(experiment_id)})
        _del(conn, "datasets", "experiment_id = :eid", {"eid": int(experiment_id)})
        n = _del(conn, "experiments", "experiment_id = :eid", {"eid": int(experiment_id)})
        if n:
            counts["experiments"] = n

    summary = ", ".join(f"{t}:{n}" for t, n in counts.items())
    print(f"  [mlflow] Hard deleted from DB — {summary or 'nothing (already clean)'}")


# ── Optuna helpers ────────────────────────────────────────────────────────────


def _optuna_storage() -> optuna.storages.RDBStorage:
    db_host = os.environ.get("DB_HOST")
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_PASSWORD"]
    db_name = os.environ["DB_NAME"]
    db_port = os.environ["DB_PORT"]
    return optuna.storages.RDBStorage(
        url=f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        skip_compatibility_check=True,
    )


# ── Main nuke logic ───────────────────────────────────────────────────────────


def nuke(study_name: str, *, dry_run: bool, use_gc: bool = False) -> None:
    client = mlflow.tracking.MlflowClient()
    backend_uri = os.environ.get("MLFLOW_BACKEND_STORE_URI", "").strip()

    # ── 1. S3 artifacts ───────────────────────────────────────────────────────
    experiment = client.get_experiment_by_name(study_name)

    if experiment is None:
        print(f"[mlflow]  No experiment named '{study_name}' found — skipping MLflow steps.")
        run_ids: list[str] = []
        exp_id = None
    else:
        exp_id = experiment.experiment_id
        runs = client.search_runs(
            experiment_ids=[exp_id],
            run_view_type=mlflow.entities.ViewType.ALL,
            max_results=50_000,
        )
        run_ids = [r.info.run_id for r in runs]
        print(f"[mlflow]  Experiment '{study_name}' (id={exp_id}) — {len(runs)} runs.")

        s3 = _s3_client()
        total_objects = 0

        for run in runs:
            artifact_uri = run.info.artifact_uri
            if not artifact_uri.startswith("s3://"):
                print(f"  [s3]     skipping run {run.info.run_id} — non-S3 URI: {artifact_uri}")
                continue
            bucket, key = _parse_s3_uri(artifact_uri)
            # artifact_uri ends in …/artifacts; step up one level to get the full run dir
            run_prefix = "/".join(key.rstrip("/").split("/")[:-1]) + "/"
            if dry_run:
                print(f"  [s3]     [dry] would delete s3://{bucket}/{run_prefix}")
            else:
                n = _delete_s3_prefix(s3, bucket, run_prefix)
                total_objects += n
                status = f"{n} objects deleted" if n else "already empty"
                print(f"  [s3]     run {run.info.run_id[:8]}… — {status}")

        if not dry_run:
            print(f"[mlflow]  S3 total: {total_objects} objects deleted.")

        # ── 2. Soft-delete via tracking API (marks rows DELETED; needed for gc compat) ──
        if dry_run:
            active = sum(1 for r in runs if r.info.lifecycle_stage == "active")
            print(f"[mlflow]  [dry] would soft-delete {active} active runs and experiment '{study_name}'.")
        else:
            deleted_runs = 0
            for run in runs:
                if run.info.lifecycle_stage == "active":
                    client.delete_run(run.info.run_id)
                    deleted_runs += 1
            if experiment.lifecycle_stage == "active":
                client.delete_experiment(exp_id)
                print(f"[mlflow]  Soft-deleted {deleted_runs} runs and experiment '{study_name}'.")
            else:
                print(
                    f"[mlflow]  Soft-deleted {deleted_runs} runs. "
                    f"Experiment already in '{experiment.lifecycle_stage}' state — skipping delete_experiment."
                )

        # ── 3. Hard-delete rows from MLflow's PostgreSQL ─────────────────────
        if backend_uri:
            if use_gc:
                print("[mlflow]  Hard deleting via mlflow gc (official path)...")
                _mlflow_hard_delete_gc(backend_uri, exp_id, dry_run)
            else:
                print("[mlflow]  Hard deleting via direct SQL (use --use-gc for official path)...")
                _mlflow_hard_delete_sql(backend_uri, exp_id, run_ids, dry_run)
        else:
            print(
                "[mlflow]  WARNING: MLFLOW_BACKEND_STORE_URI not set — rows remain in MLflow's\n"
                "          tracking DB (soft-deleted only). Add it to .env and re-run --confirm."
            )

    # ── 4. Optuna (always a hard delete) ─────────────────────────────────────
    storage = _optuna_storage()
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        n_trials = len(study.trials)
        if dry_run:
            print(f"[optuna]  [dry] would delete study '{study_name}' ({n_trials} trials).")
        else:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"[optuna]  Study '{study_name}' deleted ({n_trials} trials).")
    except KeyError:
        print(f"[optuna]  No study named '{study_name}' found — skipping.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Nuke an MLflow experiment and Optuna study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    name_group = parser.add_argument_group("study name (pick one form)")
    mutex = name_group.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--study", type=str, help="Full study name, e.g. even_strength-v1-base")
    mutex.add_argument("--strength", type=str, help="Strength name (requires --version and --model)")
    parser.add_argument("--version", "-v", type=str)
    parser.add_argument("--model", "-m", type=str, default="base_xg", choices=["base_xg", "informed_xg"])

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--confirm", action="store_true", help="Actually delete everything.")
    action.add_argument("--dry-run", action="store_true", help="Print what would be deleted; touch nothing.")

    parser.add_argument(
        "--use-gc",
        action="store_true",
        default=False,
        help=(
            "Use `mlflow gc` for hard delete (official path; handles model registry and traces). "
            "Default is direct SQL, which is faster and won't re-attempt S3 deletion."
        ),
    )

    args = parser.parse_args()

    if args.strength and not args.version:
        parser.error("--strength requires --version")

    study_name = args.study if args.study else f"{args.strength}-{args.version}-{args.model.replace('_xg', '')}"

    print(f"\nTarget: '{study_name}'")
    backend_uri = os.environ.get("MLFLOW_BACKEND_STORE_URI", "").strip()
    if backend_uri:
        method = "mlflow gc" if args.use_gc else "direct SQL"
        print(f"MLflow hard delete: yes ({method})")
    else:
        print("MLflow hard delete: no (MLFLOW_BACKEND_STORE_URI not set — soft delete only)")
    print(f"Mode: {'DRY RUN — nothing will be deleted' if args.dry_run else 'LIVE — irreversible'}\n")

    nuke(study_name, dry_run=args.dry_run, use_gc=args.use_gc)
    print("\nDone.")


if __name__ == "__main__":
    main()
