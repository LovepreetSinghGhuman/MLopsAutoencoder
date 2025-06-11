#!/bin/bash
set -e

# Apply backend manifests
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/ingress.yaml

# Apply frontend manifests if they exist
if [ -f frontend/frontend-nginx-deployment.yaml ]; then
  kubectl apply -f frontend/frontend-nginx-deployment.yaml
fi

if [ -f frontend/frontend-nginx-service.yaml ]; then
  kubectl apply -f frontend/frontend-nginx-service.yaml
fi

echo "✅ All Kubernetes manifests applied."