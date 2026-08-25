# Korean → English Dubbing & Localization Pipeline
# Phase 1: Transcription + Cultural Intelligence + Translation + Subtitles

## Architecture
![Architecture diagram: Cloud Storage feeds a Transcription Agent (Chirp + Gemini enrichment), then a Cultural Intelligence Agent (RAG-grounded nuance detection), then a Translation + QA Loop (3 iterations, multi-agent consensus), then Subtitle Generation (SRT, VTT, burned-in and soft subs)](docs/architecture.jpg)

## Stack
- Google ADK (agent framework)
- Gemini 2.0 Flash (transcription, translation, QA)
- Vertex AI (model hosting)
- Cloud Run Jobs (pipeline execution)
- GCS (input/output storage)
- FFmpeg (subtitle burning)

## Project Structure
```
dubbing-pipeline/
├── job_worker.py              # Cloud Run Job entry point
├── main.py                    # ADK pipeline orchestrator (local dev)
├── submit_job.py              # Submit a Cloud Run Job execution
├── setup_apis.sh              # Enable required GCP APIs
├── Dockerfile
├── cloudbuild.yaml
├── requirements.txt
│
├── agents/
│   ├── transcription_agent.py       # Gemini audio → timed segments
│   ├── cultural_intel_agent.py      # RAG + Gemini → cultural flags
│   ├── translation_agent.py         # Culturally-aware KO→EN translation
│   ├── qa_agent.py                  # Multi-temperature consensus QC
│   └── rag_builder_agent.py         # One-time: seeds cultural knowledge base
│
├── tools/
│   ├── gcs_tool.py                  # GCS upload/download helpers
│   ├── subtitle_tool.py             # Segments → SRT → FFmpeg burn
│   └── vector_search_tool.py        # Vertex AI Vector Search read/write
│
├── utils/
│   ├── segment_models.py            # Pydantic models for segments, flags, QC
│   ├── consensus.py                 # Multi-agent QC merge algorithm
│   └── srt_writer.py                # {start_ms, end_ms, text} → SRT
│
├── prompts/
│   ├── style_guides/
│   │   └── korean_english.txt       # KO→EN translation style guide
│   └── qc_rubrics/
│       └── subtitle_qc_rubric.txt   # QC scoring criteria
│
└── config/
    └── settings.py                  # All env vars in one place
```

## Quick Start (Local)
```bash
pip install -r requirements.txt
cp .env.example .env          # fill in your values
python main.py \
  --input gs://your-bucket/input/trailer.mp4 \
  --source-lang ko \
  --target-lang en \
  --job-id trailer-ko-en-001
```

## Cloud Run Deployment
```bash
bash setup_apis.sh -p YOUR_PROJECT_ID
gcloud builds submit --config cloudbuild.yaml
python submit_job.py \
  --file gs://your-bucket/input/trailer.mp4 \
  --source-lang ko \
  --target-lang en
```
