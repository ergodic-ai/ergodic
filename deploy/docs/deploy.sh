#!/usr/bin/env bash
# Build the docs and deploy them to Cloud Run as a static nginx site.
#
# Route: local docker build, push to Artifact Registry, deploy by image.
# Do not switch to `gcloud run deploy --source`: on this project Cloud Build's
# default service account (PROJECT_NUMBER-compute@) cannot read the source
# staging bucket even with roles/cloudbuild.builds.builder granted, so source
# deploys fail. The local build sidesteps Cloud Build entirely.
#
# One-time setup already done on project `ergodic` (2026-06-10): run,
# cloudbuild, and artifactregistry APIs enabled; Artifact Registry docker repo
# `web` created in us-central1; `gcloud auth configure-docker
# us-central1-docker.pkg.dev` run once on this machine.
#
# Cloud Run wants amd64 images and dev boxes here are arm (OrbStack), so the
# build pins --platform linux/amd64.
#
# Usage: deploy/docs/deploy.sh [account] [project] [region]
set -euo pipefail
ACCOUNT="${1:-andre@ergodic.ai}"
PROJECT="${2:-ergodic}"
REGION="${3:-us-central1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/web/ergodic-docs:latest"

cd "$(dirname "$0")/../.."
uv run --group docs mkdocs build --strict

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp deploy/docs/Dockerfile deploy/docs/nginx.conf "$STAGE/"
cp -r site "$STAGE/site"

docker build --platform linux/amd64 -t "$IMAGE" "$STAGE"
docker push "$IMAGE"
gcloud run deploy ergodic-docs \
    --image "$IMAGE" \
    --account "$ACCOUNT" \
    --project "$PROJECT" \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 256Mi \
    --cpu 1 \
    --max-instances 2 \
    --quiet
