# Deploying To Google Cloud Run

This project already ships with a working `Dockerfile`. The steps below show how to build the image in Google Cloud, store uploads/results in Cloud Storage, and run the service on Cloud Run so you no longer need to host it locally.

## 1. Prerequisites

1. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and authenticate: `gcloud auth login`.
2. Select your project: `gcloud config set project <PROJECT_ID>`.
3. Enable required APIs:
   ```bash
   gcloud services enable \
     artifactregistry.googleapis.com \
     run.googleapis.com \
     cloudbuild.googleapis.com \
     secretmanager.googleapis.com
   ```

## 2. Prepare Cloud Storage buckets

Cloud Run instances have only ephemeral disk, so uploads and inference artifacts must be persisted in Cloud Storage.

```bash
gsutil mb -l us-central1 gs://<UPLOAD_BUCKET>
gsutil mb -l us-central1 gs://<RESULTS_BUCKET>
gsutil uniformbucketlevelaccess set on gs://<UPLOAD_BUCKET>
gsutil uniformbucketlevelaccess set on gs://<RESULTS_BUCKET>
```

The app defaults to generating publicly accessible URLs (set `GCS_MAKE_PUBLIC=false` if you prefer signed URLs). For public buckets:

```bash
gsutil iam ch allUsers:objectViewer gs://<UPLOAD_BUCKET>
gsutil iam ch allUsers:objectViewer gs://<RESULTS_BUCKET>
```

## 3. Build and push the container image

Create an Artifact Registry repo (one-time):

```bash
gcloud artifacts repositories create wheat-detect \
  --repository-format=docker \
  --location=us-central1 \
  --description="Wheat detection images"
```

Build and push using Cloud Build:

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/<PROJECT_ID>/wheat-detect/web:latest
```

This uses the existing `Dockerfile`, bundling `model.pth` and `best.pt` inside the image.

## 4. Deploy to Cloud Run

Run the container with 2 vCPUs / 4 GiB RAM (tune as needed), pointing the app at the new GCS buckets:

```bash
gcloud run deploy wheat-web \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/wheat-detect/web:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --set-env-vars STORAGE_BACKEND=gcs \
  --set-env-vars MODEL_PATH=/app/model.pth,YOLO_MODEL_PATH=/app/best.pt \
  --set-env-vars IMG_SIZE=1024,DEVICE=cpu,YOLO_DISEASE_CLASS=2,YOLO_WHEAT_CLASSES=1,2 \
  --set-env-vars GCS_UPLOAD_BUCKET=<UPLOAD_BUCKET>,GCS_RESULTS_BUCKET=<RESULTS_BUCKET>,GCS_MAKE_PUBLIC=true \
  --set-env-vars FLASK_SECRET=<generate_a_secret>
```

Key environment variables:

| Variable | Purpose |
| --- | --- |
| `STORAGE_BACKEND` | `gcs` enables Cloud Storage persistence (default `local`). |
| `GCS_UPLOAD_BUCKET` / `GCS_RESULTS_BUCKET` | Buckets that hold originals and inference artifacts. You can reuse one bucket if desired. |
| `GCS_UPLOAD_PREFIX` / `GCS_RESULTS_PREFIX` | Optional folder prefixes (default `uploads/` and `results/`). |
| `GCS_MAKE_PUBLIC` | `true` uploads objects with `blob.make_public()`. Set to `false` to use signed URLs instead. |
| `GCS_SIGNED_URL_TTL` | Lifetime (seconds) for signed URLs when `GCS_MAKE_PUBLIC=false`. |
| `WORK_ROOT` | Local scratch directory (defaults to `/tmp/wheat_work`). Cloud Run already exposes `/tmp` for writes—keep it under 512 MB per request. |

Cloud Run automatically provisions HTTPS. Visit the URL printed by the deploy command to test the app.

## 5. Optional improvements

1. **CI/CD** – Add a Cloud Build trigger so pushes to `main` rebuild and redeploy automatically.
2. **Secrets** – Move `FLASK_SECRET` and any future credentials into Secret Manager, then reference them with `--set-secrets`.
3. **Custom domain** – Map your own domain via `gcloud run domain-mappings create`.
4. **GPU workloads** – If you later need CUDA, migrate the same container to GKE or a Compute Engine VM with GPUs; Cloud Run (managed) is CPU-only.

## 6. Local sanity check

Before deploying, run the container locally (still uses the new storage abstraction):

```bash
docker build -t wheat-web .
docker run --rm -p 8080:5000 \
  -e STORAGE_BACKEND=local \
  -e LOCAL_UPLOAD_DIR=/data/uploads \
  -e LOCAL_RESULTS_DIR=/data/static/results \
  -v "$PWD/uploads":/data/uploads \
  -v "$PWD/static/results":/data/static/results \
  wheat-web
```

When tested locally with `STORAGE_BACKEND=local`, the Flask routes keep working exactly like the original docker-compose flow.
