"""api/main.py — HTTP trigger for the dubbing pipeline Cloud Run Job."""
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import run_v2
import os

app = FastAPI(title="Dubbing Pipeline API")

PROJECT_ID = os.environ["PROJECT_ID"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
REGION = os.environ.get("GCP_REGION", "us-central1")
JOB_NAME = os.environ.get("JOB_NAME", "dubbing-pipeline-job")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class RunRequest(BaseModel):
    input_file: str
    source_lang: str = "ko"
    target_lang: str = "en"
    resume: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_pipeline(req: RunRequest):
    stem = req.input_file.split("/")[-1].rsplit(".", 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{stem}-{req.source_lang}-{req.target_lang}-{timestamp}"

    env = {
        "INPUT_FILE": req.input_file,
        "SOURCE_LANG": req.source_lang,
        "TARGET_LANG": req.target_lang,
        "JOB_ID": job_id,
        "PROJECT_ID": PROJECT_ID,
        "GOOGLE_CLOUD_PROJECT": PROJECT_ID,
        "GCP_REGION": REGION,
        "BUCKET_NAME": BUCKET_NAME,
        "GEMINI_MODEL": GEMINI_MODEL,
        "MAX_ITERATIONS": "3",
        "NUM_QA_AGENTS": "3",
        "QA_TEMPERATURES": "0.1,0.5,0.9",
        "QA_THRESHOLD": "0.80",
        "BURN_SUBTITLES": "true",
        "SOFT_SUBTITLES": "true",
        "RESUME_FROM_CHECKPOINTS": "true" if req.resume else "false",
    }

    client = run_v2.JobsClient()
    job_resource = f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{JOB_NAME}"

    try:
        operation = client.run_job(
            request=run_v2.RunJobRequest(
                name=job_resource,
                overrides=run_v2.RunJobRequest.Overrides(
                    container_overrides=[
                        run_v2.RunJobRequest.Overrides.ContainerOverride(
                            env=[run_v2.EnvVar(name=k, value=v) for k, v in env.items()]
                        )
                    ]
                ),
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    execution = operation.metadata.name.split("/")[-1]

    return {
        "status": "submitted",
        "job_id": job_id,
        "execution": execution,
        "langs": f"{req.source_lang} → {req.target_lang}",
        "input": req.input_file,
        "outputs": f"gs://{BUCKET_NAME}/output/{req.target_lang}/",
        "logs": (
            f"https://console.cloud.google.com/run/jobs/executions/details"
            f"/{REGION}/{execution}/logs?project={PROJECT_ID}"
        ),
    }
