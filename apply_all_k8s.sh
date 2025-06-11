#!/bin/bash
set -e

kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/ingress.yaml

# If frontend manifests exists apply them:
if [ -f frontend/frontend-nginx-deployment.yaml ]; then
  kubectl apply -f frontend/frontend-nginx-deployment.yaml
fi

echo "✅ All Kubernetes manifests applied."