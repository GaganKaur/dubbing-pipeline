#!/bin/bash
# setup_apis.sh — Enable all required GCP APIs and create service account
set -e

usage() {
  echo "Usage: $0 -p PROJECT_ID [-r REGION] [-b BUCKET_NAME]"
  exit 1
}

PROJECT_ID=""
REGION="us-central1"
BUCKET_NAME=""

while getopts "p:r:b:" opt; do
  case $opt in
    p) PROJECT_ID="$OPTARG" ;;
    r) REGION="$OPTARG" ;;
    b) BUCKET_NAME="$OPTARG" ;;
    *) usage ;;
  esac
done

[ -z "$PROJECT_ID" ] && usage
[ -z "$BUCKET_NAME" ] && BUCKET_NAME="dubbing-pipeline-${PROJECT_ID}"

echo "==> Configuring project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "==> Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  texttospeech.googleapis.com \
  speech.googleapis.com \
  --project="$PROJECT_ID"

echo "==> Creating service account..."
SA_NAME="dubbing-pipeline-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Dubbing Pipeline Service Account" \
  --project="$PROJECT_ID" 2>/dev/null || echo "Service account already exists"

for ROLE in \
  roles/aiplatform.user \
  roles/storage.objectAdmin \
  roles/run.developer \
  roles/logging.logWriter \
  roles/speech.client; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" --quiet
done

echo "==> Granting Cloud Build service account permissions..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
CLOUDBUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for ROLE in roles/storage.admin roles/artifactregistry.writer roles/logging.logWriter roles/run.admin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="$ROLE" --quiet
done

# Allow Cloud Build to assign the pipeline SA to Cloud Run Jobs
gcloud iam service-accounts add-iam-policy-binding \
  "${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/iam.serviceAccountUser" --quiet

echo "==> Creating Artifact Registry repository..."
gcloud artifacts repositories create cloud-run-source-deploy \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --description="Dubbing pipeline container images" 2>/dev/null || echo "Artifact Registry repo already exists"

echo "==> Creating GCS bucket: gs://$BUCKET_NAME"
gcloud storage buckets create "gs://${BUCKET_NAME}" \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --uniform-bucket-level-access 2>/dev/null || echo "Bucket already exists"

echo ""
echo "==> Done. Add these to your .env:"
echo "PROJECT_ID=$PROJECT_ID"
echo "GCP_REGION=$REGION"
echo "BUCKET_NAME=$BUCKET_NAME"
echo "SA_EMAIL=$SA_EMAIL"
