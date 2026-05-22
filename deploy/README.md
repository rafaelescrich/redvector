# RedVector Kubernetes deploy (GKE / local)

Durable data lives on a **StatefulSet volume** mounted at `/data/redvector` (RDB + AOF). Manifests live under `deploy/kubernetes/`.

## Layout

| Path | Purpose |
|------|---------|
| `kubernetes/base/` | Namespace, ConfigMap (`redvector.conf`), StatefulSet + PVC template, headless Service |
| `kubernetes/overlays/local/` | `imagePullPolicy: Never` for minikube / Docker Desktop |
| `kubernetes/overlays/gcp/` | `storageClassName: premium-rwo` for GKE SSD PD |

## Quick start (local cluster)

Requires Docker and a Kubernetes context (Docker Desktop Kubernetes, minikube, kind).

```bash
cd agentic-trading/graph-agentic-rag/redvector

# Minikube: build into minikube’s Docker daemon
# eval $(minikube docker-env)

# Docker Desktop K8s: plain docker build is enough
docker build -t redvector:latest -f Dockerfile .

./deploy/scripts/apply-local.sh
```

Verify:

```bash
kubectl -n redvector get pods,pvc
kubectl -n redvector port-forward pod/redvector-0 6379:6379 8888:8888
# redis-cli -p 6379 ping
curl -sf http://127.0.0.1:8888/health
```

In-cluster Redis URL: `redis://redvector-0.redvector.redvector.svc.cluster.local:6379` (headless Service + StatefulSet pod DNS).

## GCP (GKE)

1. Create Artifact Registry repository `redvector` (once per project):

```bash
gcloud artifacts repositories create redvector \
  --repository-format=docker \
  --location="${GCP_REGION}" \
  --project="${GCP_PROJECT}"
```
2. Create a zonal or regional GKE cluster with Workload Identity if you use other GCP APIs.
3. Run:

```bash
export GCP_PROJECT=your-project
export GCP_REGION=us-central1
export GKE_CLUSTER=your-cluster
# Zonal GKE only (omit for regional clusters):
export GCP_ZONE=us-central1-a
export IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || echo latest)"

./deploy/scripts/build-push-gcp.sh   # builds locally and pushes to Artifact Registry
./deploy/scripts/apply-gcp.sh      # applies overlay and sets the pushed image on the StatefulSet
```

`apply-gcp.sh` uses **`gcloud … get-credentials --zone`** when `GCP_ZONE` is set, and **`--region`** when it is not.

### Apple Silicon note (M1/M2/M3)

If you build on Apple Silicon and your GKE nodes are `amd64` (typical), you must push an `amd64` image. `build-push-gcp.sh` uses Docker buildx and defaults to:

- `PLATFORMS=linux/amd64`

If you need multi-arch, set:

- `PLATFORMS=linux/amd64,linux/arm64`

Or apply manually:

```bash
# Zonal:
gcloud container clusters get-credentials "$GKE_CLUSTER" --zone "$GCP_ZONE" --project "$GCP_PROJECT"
# Regional:
gcloud container clusters get-credentials "$GKE_CLUSTER" --region "$GCP_REGION" --project "$GCP_PROJECT"
kubectl apply -k deploy/kubernetes/overlays/gcp
kubectl -n redvector set image statefulset/redvector \
  redvector="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/redvector/redvector:${IMAGE_TAG}"
kubectl -n redvector rollout status statefulset/redvector
```

### Autopilot / storage class

If `premium-rwo` is not available, edit `overlays/gcp/storage-class.yaml` to your default (`standard-rwo`, `hyperdisk-balanced`, etc.) or remove the patch and rely on the cluster default.

## Optional: `requirepass`

1. Edit `base/configmap.yaml` and add a line `requirepass YOUR_STRONG_SECRET`, or merge a Kustomize patch from a local untracked file.
2. Prefer **Secret Manager + CSI driver** or **External Secrets** for production instead of committing secrets.

## Troubleshooting

### Docker build: `edition2024` / `getrandom` / “newer version of Cargo”

The builder image must be **Rust ≥ 1.85** (Rust 2024 edition). If you pinned an older `FROM rust:…` in a fork, bump it to match the repo `Dockerfile` (currently `rust:1.85-slim`).

## Tear down

```bash
kubectl delete namespace redvector
```

PVCs are deleted with the StatefulSet’s PVCs when using `persistentVolumeClaimRetentionPolicy` is not set—default is to retain or delete depending on policy; on `kubectl delete namespace`, PVCs in that namespace are removed.
