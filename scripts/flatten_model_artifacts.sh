#!/bin/bash
set -e

MODEL_ROOT="./models"
REQUIRED_FILES=("autoencoder.keras" "scaler.joblib" "autoencoder_config.json" "threshold.json")

# Find the directory containing all required files
FOUND_DIR=""
for d in "$MODEL_ROOT" "$MODEL_ROOT"/outputs "$MODEL_ROOT"/model_dir; do
  if [ -d "$d" ]; then
    missing=0
    for f in "${REQUIRED_FILES[@]}"; do
      if [ ! -f "$d/$f" ]; then
        missing=1
        break
      fi
    done
    if [ $missing -eq 0 ]; then
      FOUND_DIR="$d"
      break
    fi
  fi
done

if [ -z "$FOUND_DIR" ]; then
  echo "❌ Model artifacts not found in expected locations."
  exit 1
fi

echo "✅ Found model artifacts in: $FOUND_DIR"
# Copy to ./models/ root for Docker build, skip if source and destination are the same
for f in "${REQUIRED_FILES[@]}"; do
  if [ "$FOUND_DIR/$f" != "$MODEL_ROOT/$f" ]; then
    cp "$FOUND_DIR/$f" "$MODEL_ROOT/"
  fi
done

echo "✅ Model artifacts copied to ./models"