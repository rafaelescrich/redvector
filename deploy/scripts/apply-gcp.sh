#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GKE_CLUSTER:?Set GKE_CLUSTER}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY="${ROOT}/deploy/kubernetes/overlays/gcp"

if [[ -z "${IMAGE:-}" && -f "${ROOT}/deploy/.last-image.env" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/deploy/.last-image.env"
fi
: "${IMAGE:?Export IMAGE=... or run deploy/scripts/build-push-gcp.sh first (writes deploy/.last-image.env)}"

# Zonal cluster: set GCP_ZONE (e.g. us-central1-a). Regional: set GCP_REGION only.
if [[ -n "${GCP_ZONE:-}" ]]; then
  echo "Fetching GKE credentials: ${GKE_CLUSTER} (zone ${GCP_ZONE})"
  gcloud container clusters get-credentials "${GKE_CLUSTER}" \
    --zone "${GCP_ZONE}" \
    --project "${GCP_PROJECT}"
else
  : "${GCP_REGION:?Set GCP_REGION for regional clusters, or GCP_ZONE for zonal (e.g. us-central1-a)}"
  echo "Fetching GKE credentials: ${GKE_CLUSTER} (region ${GCP_REGION})"
  gcloud container clusters get-credentials "${GKE_CLUSTER}" \
    --region "${GCP_REGION}" \
    --project "${GCP_PROJECT}"
fi

echo "Applying overlay: gcp"
kubectl apply -k "${OVERLAY}"

echo "Setting StatefulSet image to ${IMAGE}"
kubectl -n redvector set image statefulset/redvector redvector="${IMAGE}"

kubectl -n redvector rollout status statefulset/redvector --timeout=600s
kubectl -n redvector get pods,pvc -o wide
