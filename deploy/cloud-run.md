# Cloud Run deployment

The service is intentionally a single container so the exact same inference package used for assessment CSV generation serves the demo UI and REST API.

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/spotter/rate-intelligence:latest

gcloud run deploy spotter-rate-intelligence \
  --image REGION-docker.pkg.dev/PROJECT/spotter/rate-intelligence:latest \
  --region REGION \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 16 \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated
```

For a real company deployment, keep the service private behind IAM or an authenticated application layer. Large CSV jobs should move to a Cloud Run Job + Cloud Storage workflow instead of holding an HTTP request open.
