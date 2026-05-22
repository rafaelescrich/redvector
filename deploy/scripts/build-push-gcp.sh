#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_REGION:?Set GCP_REGION}"
: "${IMAGE_TAG:=$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse --short HEAD 2>/dev/null || echo latest)}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/redvector/redvector:${IMAGE_TAG}"

echo "Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

PLATFORMS="${PLATFORMS:-linux/amd64}"
echo "Building + pushing ${IMAGE} (platforms: ${PLATFORMS})"

# Use buildx so we can target the GKE node architecture.
# Common failure without this on Apple Silicon: "no match for platform in manifest".
docker buildx build \
  --platform "${PLATFORMS}" \
  -t "${IMAGE}" \
  -f "${ROOT}/Dockerfile" \
  --push \
  "${ROOT}"

echo "IMAGE=${IMAGE}" > "${ROOT}/deploy/.last-image.env"
echo "Pushed. Wrote ${ROOT}/deploy/.last-image.env"
