#!/bin/bash
set -e

# Apply all YAML manifests in the deployment/k8s and frontend directories
find deployment/k8s -name '*.yaml' -exec kubectl apply -f {} \;

if [ -d frontend ]; then
  find frontend -name '*.yaml' -exec kubectl apply -f {} \;
fi

echo "✅ All Kubernetes manifests applied."