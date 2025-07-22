#!/bin/bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination file:./mlruns/artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts