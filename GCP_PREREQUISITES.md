# GCP Prerequisites — Dubbing & Localization Pipeline

This document lists everything that needs to exist in your GCP project before
the pipeline can be deployed. Most of this is created automatically by a
single setup script we provide (`setup_apis.sh`) — this list is here so your
cloud/security team can review and pre-approve it before we run anything.

---

## 1. Project & billing

- A GCP project (existing or new), with billing enabled.
- Whoever runs the setup script needs **Owner**, or this specific combination
  of roles (since the script enables APIs, creates a service account, and
  grants IAM bindings):
  - `roles/serviceusage.serviceUsageAdmin` (enable APIs)
  - `roles/iam.serviceAccountAdmin` (create service accounts)
  - `roles/resourcemanager.projectIamAdmin` (grant IAM roles)
  - `roles/artifactregistry.admin` (create the image repo)
  - `roles/storage.admin` (create the GCS bucket)
- If your org has a policy restricting who can grant IAM roles
  (`constraints/iam.allowedPolicyMemberDomains` or similar), the service
  accounts below need to be allow-listed.

## 2. APIs to enable

| API | Purpose |
|---|---|
| `run.googleapis.com` | Runs the pipeline as a Cloud Run Job |
| `cloudbuild.googleapis.com` | Builds the container image from source |
| `artifactregistry.googleapis.com` | Stores the built container image |
| `aiplatform.googleapis.com` | Vertex AI — Gemini models for translation, cultural analysis, QA, and embeddings |
| `storage.googleapis.com` | Cloud Storage — input video, output subtitles/dubbed video |
| `speech.googleapis.com` | Cloud Speech-to-Text v2 (Chirp models) — word-level timestamp alignment |
| `texttospeech.googleapis.com` | Reserved for a future phase (synthesized voice dubbing); not used by Phase 1 |

All seven are enabled by a single command in `setup_apis.sh`.

## 3. Service accounts & IAM

**Runtime service account** — `dubbing-pipeline-sa@<PROJECT_ID>.iam.gserviceaccount.com`
Created by the setup script. Needs:

- `roles/aiplatform.user` — call Gemini models
- `roles/storage.objectAdmin` — read input video, write outputs
- `roles/run.developer` — the pipeline runs as a Cloud Run Job
- `roles/logging.logWriter` — write execution logs
- `roles/speech.client` — call Speech-to-Text

**Cloud Build's service account** (`<PROJECT_NUMBER>-compute@developer.gserviceaccount.com`,
already exists in every project) needs, for the build/deploy step only:

- `roles/storage.admin`
- `roles/artifactregistry.writer`
- `roles/logging.logWriter`
- `roles/run.admin`
- `roles/iam.serviceAccountUser` on `dubbing-pipeline-sa` (so Cloud Build is
  allowed to launch the job *as* that service account)

**If you also want the HTTP trigger API** (`api/main.py` — lets you kick off
a run with a REST call instead of the CLI): deploy it as its own Cloud Run
*service*. Its runtime identity needs `roles/run.developer` plus
`roles/iam.serviceAccountUser` on `dubbing-pipeline-sa`. This piece is
optional and not yet wired into the automated setup script — flag it if you
want it, and we'll account for it in the IAM grant.

No service account keys are created or downloaded anywhere in this
pipeline — everything runs on Google-managed identities (Application
Default Credentials), which is the more secure pattern most security teams
prefer.

## 4. Storage

- One GCS bucket, e.g. `dubbing-pipeline-<PROJECT_ID>`, regional, with
  **uniform bucket-level access** enabled (created by the setup script).
- Layout the pipeline uses inside that bucket:
  - `input/` — source video files you upload
  - `output/` — final subtitled/dubbed video and `.srt` files
  - `output/artifacts/` — intermediate transcription/translation/QA data

## 5. Artifact Registry

- One Docker repository, `cloud-run-source-deploy`, in your chosen region —
  holds the built container image (created by the setup script).

## 6. Region considerations

- Default region: `us-central1`. Can be any region where Vertex AI Gemini
  and Cloud Run Jobs are available — tell us if you have a data-residency
  requirement and we'll confirm model availability there.
- One nuance: the pipeline runs a second, independent speech-recognition
  pass (Chirp 3) as a quality/recall improvement, and that model is only GA
  in the `us` multi-region today (not `us-central1` specifically). This
  means Speech-to-Text calls will touch both your chosen region and the
  `us` multi-region. If your data-residency policy is strict about staying
  in a single region, tell us and we'll disable that fallback pass — it's
  an enhancement, not a requirement.
- Cloud Run Job resource footprint to plan quota for: 8 GiB memory, 4 vCPU,
  up to 6-hour task timeout per run.

## 7. Models used (no separate enablement beyond Vertex AI API above)

- `gemini-2.5-flash` — transcription enrichment, translation, QA scoring
- `gemini-embedding-001` — only used if cultural-knowledge-base search
  (below) is turned on; Google's current-generation multilingual
  embedding model (its predecessor, `text-embedding-004`, is already
  deprecated, so we didn't default to it)
- `chirp_2` / `chirp_3` — Cloud Speech-to-Text forced-alignment models

If your org restricts which Model Garden / Vertex models can be called
(some orgs gate this via an allowlist policy), Gemini 2.5 Flash and
gemini-embedding-001 need to be on that allowlist.

## 8. Optional — not required to launch Phase 1

- **Vertex AI Vector Search** (index + endpoint): powers the
  cultural-nuance knowledge base. If left unconfigured, the pipeline
  automatically falls back to a simple in-memory keyword match, so
  **nothing further is needed to run the pipeline end-to-end**. We'd only
  provision a real Vector Search index/endpoint if you want
  production-grade cultural RAG, which is a later conversation.
- **Text-to-Speech API**: already enabled by the setup script for a future
  phase (actual synthesized voice dubbing). Phase 1 only produces
  translated subtitles and burned-in captions — no synthesized audio yet.

## 9. Software dependencies (bundled — nothing for you to install)

Everything below ships inside the container image we build and deploy via
Cloud Build. Nothing needs to be pre-installed on your GCP project or any
of your machines — this section is here purely for your security/dependency
review, since some orgs scan container images for open-source components.

**System packages** (installed in the `Dockerfile`, on a `python:3.11-slim`
/ Debian base):
- `ffmpeg` — burns translated subtitles into the video and muxes soft-subtitle
  tracks
- `libsndfile1` — audio file reading, used by the transcription/audio-handling
  code

**Python packages** (pinned in `requirements.txt`):
| Package | Purpose |
|---|---|
| `google-cloud-aiplatform` | Vertex AI SDK — calls Gemini models |
| `vertexai` | Vertex AI generative-model and embedding client |
| `google-cloud-storage` | Reads/writes the GCS bucket |
| `google-cloud-speech` | Cloud Speech-to-Text v2 (Chirp) client |
| `google-generativeai` | Gemini client library |
| `pydantic` | Data validation for pipeline data structures |
| `python-dotenv` | Loads local `.env` config (dev convenience only, unused in Cloud Run) |

If you also deploy the optional HTTP trigger API (section 3), its own
container additionally includes `fastapi`, `uvicorn`, and `google-cloud-run`
(to submit Cloud Run Job executions via API instead of the CLI).

No system packages or Python libraries need to exist anywhere in your
project ahead of time — they arrive with the container each time we deploy.

## 10. What you provide us

- The video file(s) to localize, uploaded to `gs://<bucket>/input/`
  (any format ffmpeg reads — MP4/MOV are the common cases — with an audio
  track in the source language).
- Confirmation of source and target language(s).

---

### Summary for your infra team

Run one script (`setup_apis.sh -p YOUR_PROJECT_ID`) that enables 7 APIs,
creates one service account with 5 project-level roles, grants 5 roles to
the existing Cloud Build service account, creates one Artifact Registry
repo, and creates one GCS bucket. Nothing else touches your project. No
service account keys, no VPC changes, no org-policy changes required unless
your org already restricts IAM grants or Model Garden access (see sections
1 and 7).
