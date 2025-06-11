#!/bin/bash
set -e

kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/ingress.yaml

# If you have frontend manifests, apply them too:
if [ -f frontend/frontend-nginx-deployment.yaml ]; then
  kubectl apply -f frontend/frontend-nginx-deployment.yaml
fi

echo "✅ All Kubernetes manifests applied."