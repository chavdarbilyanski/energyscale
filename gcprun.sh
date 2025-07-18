gcloud run deploy webbidder \
    --image europe-central2-docker.pkg.dev/secret-node-420018/dockerrepochavdar/webapp:latest \
    --region europe-central2 \
    --platform managed \
    --allow-unauthenticated \
    --port 8000