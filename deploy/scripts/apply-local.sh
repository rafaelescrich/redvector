#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY="${ROOT}/deploy/kubernetes/overlays/local"

echo "Applying Kustomize overlay: local (${OVERLAY})"
kubectl apply -k "${OVERLAY}"

echo "Waiting for StatefulSet rollout..."
kubectl -n redvector rollout status statefulset/redvector --timeout=300s

echo "Done. Pods:"
kubectl -n redvector get pods -o wide
