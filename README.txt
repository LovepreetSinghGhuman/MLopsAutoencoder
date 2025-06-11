# IEEE-CIS Fraud Detection with Autoencoders

## Project Overview

This repository demonstrates a **full MLOps workflow** for fraud detection using an autoencoder neural network, deployed on Azure and Kubernetes, with CI/CD automation via GitHub Actions. The pipeline covers data preprocessing, model training, cloud retraining, containerized deployment, and automated updates.

---

## 1. Project Explanation

- **Data:** IEEE-CIS Fraud Detection dataset (preprocessed and cleaned).
- **Model:** Deep autoencoder (TensorFlow/Keras) trained to reconstruct normal transactions; high reconstruction error signals potential fraud.
- **Preprocessing:** 
  - Drops IDs and target columns.
  - Feature engineering (e.g., time features, log-transform amounts, frequency encoding).
  - Handles missing values, encodes categoricals, winsorizes numerics, and scales features.
  - Ensures train/test feature alignment.
- **Pipeline:** 
  - Data cleaning and alignment in [notebook_original/IEEE_CIS_AutoencoderV4.ipynb](notebook_original/IEEE_CIS_AutoencoderV4.ipynb).
  - Model training in [src/train.py](src/train.py).
  - Inference API in [src/score.py](src/score.py).
  - Frontend for file upload in [frontend/index.html](frontend/index.html).

---

## 2. Cloud Training & Retraining

- **Cloud Service:** Azure ML.
- **Training Job:** Defined in [deployment/train-job.yaml](deployment/train-job.yaml).
- **Automation:** 
  - Training is triggered via GitHub Actions ([.github/workflows/ci-cd.yaml](.github/workflows/ci-cd.yaml)).
  - Model artifacts (Keras model, scaler, config, threshold) are saved to [models/](models/) by the workflow.
  - **No local model files are needed**; all artifacts are managed by the pipeline.
- **Reproducibility:** All steps (preprocessing, feature engineering, scaling) are consistent between notebook, training script, and API.

---

## 3. Kubernetes Deployment

- **Containerization:** Dockerfile in [deployment/Dockerfile](deployment/Dockerfile) builds a FastAPI app for inference.
- **Kubernetes Manifests:** 
  - Backend: [deployment/k8s/deployment.yaml](deployment/k8s/deployment.yaml), [deployment/k8s/service.yaml](deployment/k8s/service.yaml)
  - Frontend: [frontend/frontend-nginx-deployment.yaml](frontend/frontend-nginx-deployment.yaml)
  - Ingress: [deployment/k8s/ingress.yaml](deployment/k8s/ingress.yaml) routes `/predict` to FastAPI and all other paths to the static frontend.
- **Secrets:** ACR credentials managed via Kubernetes secret.

---

## 4. CI/CD Automation

- **GitHub Actions:** 
  - Checks UTF-8 encoding for all relevant files ([src/convert_to_utf8.py](src/convert_to_utf8.py)).
  - Submits Azure ML training jobs and waits for completion.
  - Downloads model artifacts from Azure ML and registers the new model version.
  - Builds and pushes Docker images to Azure Container Registry (ACR).
  - Updates Kubernetes deployment with the new image automatically after merging to `main`.
- **Minimal Manual Steps:** After merging code, the pipeline retrains, builds, and redeploys the model with no manual intervention.

---

## 5. Frontend & User Interaction

- **Frontend:** Simple HTML page ([frontend/index.html](frontend/index.html)) for uploading CSV/Excel files.
- **API:** `/predict` endpoint in FastAPI ([src/score.py](src/score.py)) returns fraud predictions for uploaded files.
- **Usage:** User uploads a file, receives a JSON with `TransactionID` and `isFraud` predictions.

---

## 6. Results

- **Model Performance:** 
  - ROC-AUC and F1 scores are logged during training and validation.
  - Hyperparameter tuning (Keras Tuner) further improves performance.
- **Kaggle Scores:** 
  - Public and private leaderboard scores are reported in the notebook.

---

## 7. Architecture Diagram

```
[User] --> [Frontend (NGINX)] --> [K8s Ingress] --> [FastAPI (Autoencoder)] --> [Model Artifacts]
```
- All services are containerized and orchestrated by Kubernetes.
- Model retraining and redeployment are fully automated via CI/CD.

---

## 8. Repository Structure

- **src/**: Training, inference, and main pipeline scripts.
- **models/**: Model artifacts (populated by the workflow, not tracked in git).
- **deployment/**: Docker, conda, and Kubernetes manifests.
- **frontend/**: Static HTML frontend and NGINX configs.
- **notebook_original/**: Data exploration, cleaning, and pipeline prototyping.

---

## 9. How to Run

### Cloud Workflow (Recommended)
1. **Push to main branch**  
   - Triggers the CI/CD pipeline automatically.
   - Azure ML retrains the model, downloads artifacts, builds and pushes Docker image, and updates the Kubernetes deployment.

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

## 10. References

- [notebook_original/IEEE_CIS_AutoencoderV4.ipynb](notebook_original/IEEE_CIS_AutoencoderV4.ipynb): Full pipeline, experiments, and results.
- [src/train.py](src/train.py): Training script.
- [src/score.py](src/score.py): FastAPI inference service.
- [deployment/k8s/](deployment/k8s/): Kubernetes manifests.
- [frontend/index.html](frontend/index.html): User interface.

---

## 11. Video Demonstration

- See attached video(s) for a walkthrough of the pipeline, cloud training, deployment, and frontend prediction.

---

## 12. Additional Notes

- All files are UTF-8 encoded (see [src/convert_to_utf8.py](src/convert_to_utf8.py)).
- Data files are excluded from version control via [.gitignore](.gitignore) and managed with Git LFS ([.gitattributes](.gitattributes)).
- For any issues, see logs in GitHub Actions or Kubernetes pods.
- **Model artifacts in `models/` are managed by the pipeline and should not be tracked in git.**

---

**Authors:**  
Lovepreet Singh  
MLOps and AI design patterns
