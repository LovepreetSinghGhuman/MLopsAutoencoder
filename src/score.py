# coding: utf-8
# ü
import os
import json
import joblib
import uvicorn
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import glob

# --- Preprocessing function (must match training pipeline exactly) ---
def preprocess_data(
    df,
    scaler=None,
    is_train=True,
    scaler_path="models/scaler.joblib",
    clip_values=True,
    feature_columns=None
):
    df = df.copy()

    # 1. Drop ID and target columns if present
    drop_cols = ['TransactionID', 'isFraud']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # 2. Feature Engineering: TransactionDT
    if 'TransactionDT' in df.columns:
        START_DATE = pd.Timestamp('2017-11-30')
        df['TransactionDT_datetime'] = START_DATE + pd.to_timedelta(df['TransactionDT'], unit='s')
        df['hour'] = df['TransactionDT_datetime'].dt.hour
        df['dayofweek'] = df['TransactionDT_datetime'].dt.dayofweek
        df['dayofmonth'] = df['TransactionDT_datetime'].dt.day
        df.drop(columns=['TransactionDT', 'TransactionDT_datetime'], inplace=True)

    # 3. Feature Engineering: TransactionAmt (log transform)
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt_Log'] = np.log1p(np.clip(df['TransactionAmt'], a_min=0, a_max=None))
        if 'card1' in df.columns:
            card1_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_card1_mean'] = df['TransactionAmt'] / (card1_mean + 1e-3)
        df.drop(columns=['TransactionAmt'], inplace=True)

    # 4. DeviceInfo grouping
    if 'DeviceInfo' in df.columns:
        df['DeviceInfo'] = df['DeviceInfo'].fillna('missing').str.lower()
        df['DeviceInfo_grouped'] = df['DeviceInfo'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else 'missing')

    # 5. DeviceType (fill missing, encode)
    if 'DeviceType' in df.columns:
        df['DeviceType'] = df['DeviceType'].fillna('missing').str.lower()

    # 6. Frequency encoding for high-cardinality features
    for col in ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']:
        if col in df.columns:
            freq = df[col].value_counts(dropna=False)
            df[col + '_freq'] = df[col].map(freq)
            df[col] = df[col].fillna(-999)

    # 7. Fill missing values for categorical/object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('missing')

    # 8. Fill missing values for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # 9. Encode categorical columns (robustly)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    # 10. Winsorize numeric columns (clip to 1st/99th percentile for robustness)
    for col in df.select_dtypes(include=[np.number]).columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    # 11. Select numeric columns only
    df = df.select_dtypes(include=[np.number]).astype(np.float32)

    # 12. Replace non-finite values with 0
    df[~np.isfinite(df)] = 0.0

    # 13. Optionally clip extreme values
    if clip_values:
        df = df.clip(lower=-1e6, upper=1e6)

    # 14. Ensure consistent columns between train and test
    if feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)
    elif is_train:
        feature_columns = df.columns.tolist()

    # 15. Scaling
    if scaler is None:
        if is_train:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df)
            joblib.dump(scaler, scaler_path)
            X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
        else:
            try:
                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(df)
                X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
            except FileNotFoundError:
                raise ValueError("Scaler required for inference. Provide a fitted scaler or ensure scaler.joblib exists.")
    else:
        X_scaled = scaler.transform(df)
        X_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

    return X_scaled, scaler, feature_columns

app = FastAPI(
    title="Fraud Detector Autoencoder",
    description="Upload an Excel file; returns fraud predictions.",
    version="1.0"
)

# Globals to hold loaded model components
model = None
scaler = None
feature_columns = None
threshold = None
config = None

# models directory in the container
MODELS_DIR = "models"

def find_model_output_dir():
    """
    Find the directory containing the model files.
    Checks for models/model_dir/, models/outputs/, then models/.
    """
    candidates = [
        os.path.join(MODELS_DIR, "model_dir"),
        os.path.join(MODELS_DIR, "outputs"),
        MODELS_DIR
    ]
    for d in candidates:
        if os.path.isdir(d) and any(os.path.isfile(os.path.join(d, f)) for f in ["autoencoder.keras", "scaler.joblib", "autoencoder_config.json", "threshold.json"]):
            return d
    # Fallback: just return models/
    return MODELS_DIR

@app.on_event("startup")
def load_model():
    global model, scaler, feature_columns, threshold, config

    model_dir = find_model_output_dir()
    print("Model directory used:", model_dir)
    print("Files in model_dir:", os.listdir(model_dir))

    # 1. Load Keras autoencoder
    model_path = os.path.join(model_dir, "autoencoder.keras")
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Make sure the model artifact is present in the Docker image or mounted volume."
        )
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # 2. Load scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise RuntimeError(
            f"Scaler file not found at '{scaler_path}'. "
            "Ensure the scaler artifact is present."
        )
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {str(e)}")

    # 3. Load config (feature_columns, version, etc.)
    config_path = os.path.join(model_dir, "autoencoder_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"Config file not found at '{config_path}'. "
            "Ensure the config artifact is present."
        )
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {str(e)}")

    feature_columns = config.get("feature_columns")
    if feature_columns is None:
        raise RuntimeError("feature_columns not found in config")

    # 4. Load threshold
    threshold_path = os.path.join(model_dir, "threshold.json")
    if not os.path.exists(threshold_path):
        raise RuntimeError(
            f"Threshold file not found at '{threshold_path}'. "
            "Ensure the threshold artifact is present."
        )
    try:
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
        threshold = threshold_data.get("threshold", None)
    except Exception as e:
        raise RuntimeError(f"Failed to load threshold: {str(e)}")
    if threshold is None:
        raise RuntimeError("Threshold not found in threshold.json")


class PredictionResult(BaseModel):
    TransactionID: List[str]
    isFraud: List[int]  # or float if you want probability

@app.post("/predict", response_model=PredictionResult)
def predict(file: UploadFile = File(...)):
    """
    Endpoint: Upload an Excel (.xlsx/.xls) or CSV (.csv) file
    with columns that match your training features (including TransactionID).
    Returns a JSON with TransactionID and predicted isFraud (0/1).
    """
    # 1. Ensure it’s an Excel or CSV file
    filename = file.filename.lower()
    if filename.endswith(".csv"):
        filetype = "csv"
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        filetype = "excel"
    else:
        raise HTTPException(status_code=400, detail="Please upload an Excel (.xlsx/.xls) or CSV (.csv) file.")

    # 2. Read into pandas
    try:
        if filetype == "csv":
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if "TransactionID" not in df.columns:
        raise HTTPException(status_code=400, detail="File must contain a 'TransactionID' column.")

    # 3. Preprocess (drop TransactionID internally, etc.)
    try:
        # Keep the IDs aside
        tx_ids = df["TransactionID"].astype(str).tolist()

        # Preprocess to numeric features: 
        X_input, _, _ = preprocess_data(
            df,
            scaler=scaler,
            is_train=False,
            feature_columns=feature_columns
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # 4. Compute reconstruction error (MSE)
    reconstructions = model.predict(X_input, batch_size=256)
    mse = np.mean(np.square(X_input.values - reconstructions), axis=1)

    # 5. Apply threshold → 0/1 classification
    is_fraud_preds = (mse > threshold).astype(int).tolist()

    return PredictionResult(TransactionID=tx_ids, isFraud=is_fraud_preds)

# Optional: a health check endpoint
@app.get("/health")
def health():
    return {"status": "Healthy"}


# Enable local debugging via `python score.py`
if __name__ == "__main__":
    uvicorn.run("score:app", host="0.0.0.0", port=8000, reload=True)