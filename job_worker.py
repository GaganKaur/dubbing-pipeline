"""job_worker.py — Cloud Run Job entry point. Reads config from env vars."""
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("job_worker")


def main():
    # Validate required env vars before importing anything heavy
    required = ["PROJECT_ID", "BUCKET_NAME", "INPUT_FILE", "SOURCE_LANG", "TARGET_LANG"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        logger.error(f"Missing required env vars: {missing}")
        sys.exit(1)

    from main import run_pipeline

    summary = run_pipeline(
        input_gcs_uri=os.environ["INPUT_FILE"],
        source_lang=os.environ.get("SOURCE_LANG", "ko"),
        target_lang=os.environ.get("TARGET_LANG", "en"),
        job_id=os.environ.get("JOB_ID", "cloud-run-job"),
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "3")),
        resume=os.environ.get("RESUME_FROM_CHECKPOINTS", "false").lower() == "true",
    )

    logger.info("Job complete. Outputs:")
    for key, val in summary.get("outputs", {}).items():
        logger.info(f"  {key}: {val}")


if __name__ == "__main__":
    main()
