# IEEE-CIS Fraud Detection with Autoencoders

## Project Overview

This repository demonstrates a **complete MLOps workflow** for fraud detection using an autoencoder neural network, deployed on Azure and Kubernetes, with CI/CD automation via GitHub Actions. The pipeline covers data preprocessing, model training, cloud retraining, containerized deployment, and automated updates.

---

## Quick Links

- **GitHub Repository:** https://github.com/LovepreetSinghGhuman/MLopsAutoencoder.git
- **Demo Video:** [YOUR_DEMO_VIDEO_LINK_HERE]

---

## Brief Explanation of the Project

- **Data:** IEEE-CIS fraud detection dataset (preprocessed as shown in [notebook_original/IEEE_CIS_AutoencoderV4.ipynb](notebook_original/IEEE_CIS_AutoencoderV4.ipynb)).
- **Model:** Autoencoder neural network for anomaly detection.
- **Preprocessing:** Data cleaning, feature engineering, and scaling are performed in the notebook and [src/train.py](src/train.py).
- **Pipeline:** End-to-end automation from data to deployment, using Azure ML for training and AKS for serving.

---

## Assignment Task Mapping

### Task 1: Cloud Training (Azure ML)

- **Cloud Service:** Azure ML is used for model training and retraining.
- **Automation:** Training is triggered by GitHub Actions ([.github/workflows/ci-cd.yaml](.github/workflows/ci-cd.yaml)), using [deployment/train-job.yaml](deployment/train-job.yaml).
- **Artifacts:** Model files (Keras model, scaler, config, threshold) are saved to [models/](models/) by the workflow.
- **Scripts:** Training logic is in [src/train.py](src/train.py). Data cleaning and feature engineering are in [notebook_original/IEEE_CIS_AutoencoderV4.ipynb](notebook_original/IEEE_CIS_AutoencoderV4.ipynb).
- **Reproducibility:** All preprocessing and feature engineering steps are consistent between notebook, training script, and API.
- **Screenshots to include in report:**  
  - Azure ML job submission and completion (from portal or CLI).
  - Pipeline run in GitHub Actions.

---

### Task 2: Kubernetes Deployment

- **Backend:** FastAPI app ([src/score.py](src/score.py)) containerized with [deployment/Dockerfile](deployment/Dockerfile).
- **Frontend:** Simple HTML ([frontend/index.html](frontend/index.html)) served by NGINX ([frontend/Dockerfile](frontend/Dockerfile)).
- **Kubernetes Manifests:**  
  - Backend: [deployment/k8s/deployment.yaml](deployment/k8s/deployment.yaml), [deployment/k8s/service.yaml](deployment/k8s/service.yaml)
  - Frontend: [frontend/frontend-nginx-deployment.yaml](frontend/frontend-nginx-deployment.yaml)
  - Ingress: [deployment/k8s/ingress.yaml](deployment/k8s/ingress.yaml) routes `/predict` to FastAPI and all other paths to the static frontend.
- **Reverse Proxy:** Ingress controller is used for routing and reverse proxy.
- **User Interaction:** Frontend allows file upload and displays fraud predictions.
- **Microservice Communication:**  
  - Ingress routes `/predict` to the backend API, all other paths to the frontend.
  - [Include a diagram in your report showing this flow.]
- **Screenshots to include in report:**  
  - Frontend UI (file upload and result).
  - FastAPI `/docs` page.
  - Output of `kubectl get pods,svc,ingress` to show running services.

---

### Task 3: CI/CD Automation (GitHub Actions)

- **3.1 Automatically retrain the model:**  
  - On every push to `main`, the workflow triggers a new Azure ML training job and waits for completion.
- **3.2 Automatically redeploy the model:**  
  - After training, the workflow builds and pushes a new backend Docker image to Azure Container Registry (ACR) and updates the AKS deployment.
- **3.3 Automatically redeploy the frontend/api:**  
  - When frontend code changes, a new Docker image is built and pushed, and the frontend deployment in AKS is updated.
- **Minimal Manual Steps:**  
  - After merging code to `main`, the pipeline retrains, builds, and redeploys both backend and frontend with no manual intervention.
- **Automation Details:**  
  - See [ci-cd.yaml](.github/workflows/ci-cd.yaml) for workflow logic.
  - Model versioning is handled via Azure ML's model registry.
  - All secrets are managed via GitHub Actions secrets.
- **Screenshots to include in report:**  
  - GitHub Actions workflow runs (showing retrain, build, deploy steps).
  - Any manual steps (if any) should be explained.

---

## Repository Structure

- **src/**: Training, inference, and pipeline scripts.
- **models/**: Model artifacts (populated by the workflow, not tracked in git).
- **deployment/**: Docker, conda, and Kubernetes manifests.
- **frontend/**: Static HTML frontend and NGINX configs.
- **notebook_original/**: Data exploration, cleaning, and pipeline prototyping.

---

## How to Run

### Cloud Workflow (Recommended)
1. **Push to main branch**  
   - Triggers the CI/CD pipeline automatically.
   - Azure ML retrains the model, downloads artifacts, builds and pushes Docker images, and updates the Kubernetes deployment.

### Manual (for local testing)
1. **Training:**  
   ```
   python src/train.py --train-data data/processed/cleaned_train.csv --output-dir models/
   ```
2. **Build & Deploy:**  
   ```
   docker build -f deployment/Dockerfile -t fraud-detector:latest .
   ```
   - Push to ACR and deploy to AKS using provided manifests.
3. **Frontend:**  
   - Access via the service/ingress endpoint.
   - Upload a CSV/Excel file and view predictions.

---

## Architecture Diagram

```
[User] --> [Frontend (NGINX)] --> [K8s Ingress] --> [FastAPI (Autoencoder)] --> [Model Artifacts]
```
- All services are containerized and orchestrated by Kubernetes.
- Model retraining and redeployment are fully automated via CI/CD.

---

## Reporting & Demo Guidance

- **Include links** to this repository and your demo video in your report.
- **Screenshots:**  
  - Azure ML job runs, GitHub Actions pipeline, FastAPI docs, frontend UI, and Kubernetes resources.
- **Explain:**  
  - How microservices communicate (see diagram above).
  - Any special Kubernetes setup or configuration.
  - Your CI/CD update strategy and why you chose it.
  - How you keep track of model versions (Azure ML registry).
- **Reflection:**  
  - Briefly describe what was easy, what was difficult, and how you solved any significant problems.
- **Demo Video:**  
  - Show the frontend, submit a file, get a result, and show your Kubernetes cluster is running (e.g., with `kubectl` commands).

---

## Additional Notes

- All files are UTF-8 encoded (see [src/convert_to_utf8.py](src/convert_to_utf8.py)).
- Data files are excluded from version control via [.gitignore](.gitignore) and managed with Git LFS ([.gitattributes](.gitattributes)).
- For any issues, see logs in GitHub Actions or Kubernetes pods.
- **Model artifacts in `models/` are managed by the pipeline and should not be tracked in git.**

---

**Author:**  
Lovepreet Singh  
MLOps and AI design patterns
