"""submit_job.py — Submit a Cloud Run Job execution for one video file."""
import argparse
import hashlib
import subprocess
import sys
import os
from datetime import datetime


def pick_region(job_id: str, regions: list[str]) -> str:
    """Deterministically pick a region from the list using job_id hash."""
    idx = int(hashlib.md5(job_id.encode()).hexdigest(), 16) % len(regions)
    return regions[idx]


def main():
    parser = argparse.ArgumentParser(description="Submit a dubbing pipeline Cloud Run Job")
    parser.add_argument("--file", required=True, help="gs:// URI of source video")
    parser.add_argument("--source-lang", default="ko")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--job-name", default="dubbing-pipeline-job")
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID", ""))
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET_NAME", ""))
    parser.add_argument("--max-iterations", default="3")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--batch-regions",
        default="us-central1",
        help="Comma-separated list of regions for quota rotation",
    )
    args = parser.parse_args()

    if not args.project:
        print("ERROR: --project or PROJECT_ID env var required")
        sys.exit(1)
    if not args.bucket:
        print("ERROR: --bucket or BUCKET_NAME env var required")
        sys.exit(1)

    # Build a unique job ID
    stem = args.file.split("/")[-1].replace(".mp4", "")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{stem}-{args.source_lang}-{args.target_lang}-{timestamp}"

    # Pick batch region via deterministic hash
    batch_regions = [r.strip() for r in args.batch_regions.split(",")]
    batch_region = pick_region(job_id, batch_regions)

    env_vars = (
        f"^;^"
        f"INPUT_FILE={args.file};"
        f"SOURCE_LANG={args.source_lang};"
        f"TARGET_LANG={args.target_lang};"
        f"JOB_ID={job_id};"
        f"PROJECT_ID={args.project};"
        f"GOOGLE_CLOUD_PROJECT={args.project};"
        f"GCP_REGION={batch_region};"
        f"BUCKET_NAME={args.bucket};"
        f"GEMINI_MODEL={args.model};"
        f"MAX_ITERATIONS={args.max_iterations};"
        f"NUM_QA_AGENTS=3;"
        f"QA_TEMPERATURES=0.1,0.5,0.9;"
        f"QA_THRESHOLD=0.80;"
        f"BURN_SUBTITLES=true;"
        f"SOFT_SUBTITLES=true;"
        f"RESUME_FROM_CHECKPOINTS={'true' if args.resume else 'false'}"
    )

    cmd = [
        "gcloud", "run", "jobs", "execute", args.job_name,
        f"--region={args.region}",
        f"--project={args.project}",
        f"--update-env-vars={env_vars}",
        "--async",
    ]

    print(f"\nSubmitting job: {job_id}")
    print(f"  Input:   {args.file}")
    print(f"  Langs:   {args.source_lang} → {args.target_lang}")
    print(f"  Region:  {batch_region}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: Job submission failed")
        sys.exit(1)

    print(f"\nJob submitted. Monitor with:")
    print(
        f"  gcloud run jobs executions list "
        f"--job={args.job_name} --region={args.region} --project={args.project}"
    )
    print(f"\nStream logs:")
    print(
        f'  gcloud logging read "resource.type=cloud_run_job '
        f'AND resource.labels.job_name={args.job_name}" '
        f"--project={args.project} --limit=100 --format='value(textPayload)'"
    )
    print(f"\nCheck outputs at:")
    print(
        f"  gcloud storage ls gs://{args.bucket}/output/{args.target_lang}/"
    )


if __name__ == "__main__":
    main()
