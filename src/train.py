# coding: utf-8
# Ã¼
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Use ONLY tf.keras imports below
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ----------------- Config -----------------
class Config:
    SEED = 42
    EPOCHS = 5  # Reduced from 50 to 5 for faster CI
    BATCH_SIZE = 256
    VAL_SIZE = 0.2
    LEARNING_RATE = 1e-4
    THRESHOLD_RANGE = (0.9, 0.99)
    NUM_THRESHOLDS = 100
    EARLY_STOPPING_PATIENCE = 3  # Also reduce patience (5) for faster stopping
    REDUCE_LR_PATIENCE = 2
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-6

# ----------------- Model -----------------
def build_autoencoder(input_dim: int) -> Model:
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(encoded)
    decoded = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(encoded)
    decoded = Dropout(0.3)(decoded)
    decoded = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(decoded)
    decoded = Dropout(0.3)(decoded)
    decoded = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(decoded)
    decoded = Dense(input_dim, activation="linear")(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

# ----------------- Preprocessing -----------------
def preprocess_data(df, scaler=None, is_train=True, feature_columns=None):
    df = df.copy()
    drop_cols = ['TransactionID', 'isFraud']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    df = df.select_dtypes(include=[np.number]).astype(np.float32)
    if feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)
    elif is_train:
        feature_columns = df.columns.tolist()
    if scaler is None and is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
    else:
        X_scaled = scaler.transform(df)
    return X_scaled, scaler, feature_columns

# ----------------- Threshold Selection -----------------
def get_reconstruction_errors(model, X):
    reconstructions = model.predict(X, batch_size=Config.BATCH_SIZE)
    return np.mean(np.power(X - reconstructions, 2), axis=1)

def select_best_threshold(model, X, y, threshold_range=(0.9, 0.99), optimize_by="roc_auc"):
    from sklearn.metrics import roc_auc_score, f1_score
    mse = get_reconstruction_errors(model, X)
    lower_q = np.quantile(mse, threshold_range[0])
    upper_q = np.quantile(mse, threshold_range[1])
    thresholds = np.linspace(lower_q, upper_q, Config.NUM_THRESHOLDS)
    best_score = -np.inf
    best_threshold = None
    for t in thresholds:
        y_pred = (mse > t).astype(int)
        if optimize_by == "roc_auc":
            score = roc_auc_score(y, mse)
        else:
            score = f1_score(y, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = t
    return float(best_threshold)

# ----------------- Main Training Logic -----------------
def main(args):
    np.random.seed(Config.SEED)
    tf.random.set_seed(Config.SEED)

    # Load data
    train_data = pd.read_csv(args.train_data)
    if args.val_data:
        val_data = pd.read_csv(args.val_data)
    else:
        train_data, val_data = train_test_split(
            train_data, test_size=Config.VAL_SIZE, stratify=train_data["isFraud"], random_state=Config.SEED
        )

    # Preprocess
    X_train = train_data[train_data["isFraud"] == 0]
    X_train_auto, scaler, feature_columns = preprocess_data(X_train, is_train=True)
    X_val_auto, _, _ = preprocess_data(val_data, scaler=scaler, is_train=False, feature_columns=feature_columns)
    y_val = val_data["isFraud"].values

    # Model
    input_dim = X_train_auto.shape[1]
    model = build_autoencoder(input_dim)
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss="mse")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=Config.REDUCE_LR_PATIENCE, factor=Config.REDUCE_LR_FACTOR, min_lr=Config.MIN_LR),
        ModelCheckpoint(filepath=os.path.join(args.output_dir, "autoencoder.keras"), monitor="val_loss", save_best_only=True)
    ]
    model.fit(
        X_train_auto, X_train_auto,
        validation_data=(X_val_auto, X_val_auto),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )

    # Threshold selection
    threshold = select_best_threshold(model, X_val_auto, y_val)

    # Save artifacts
    print("Output dir (as received):", args.output_dir)
    print("Output dir absolute path:", os.path.abspath(args.output_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Directory exists?", os.path.exists(args.output_dir))
    print("Directory contents before saving:", os.listdir(args.output_dir))
    model.save(os.path.join(args.output_dir, "autoencoder.keras"))
    with open(os.path.join(args.output_dir, "scaler.joblib"), "wb") as f:
        joblib.dump(scaler, f)
    with open(os.path.join(args.output_dir, "autoencoder_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_dim": input_dim,
            "feature_columns": feature_columns
        }, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, ensure_ascii=False)
    print(f"Training complete. Artifacts saved to {args.output_dir}")
    print("Files in output dir:", os.listdir(args.output_dir))
    print("Saving model to:", os.path.join(args.output_dir, "autoencoder.keras"))

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--val-data", type=str, default=None, help="Optional path to validation CSV file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save model artifacts")
    args = parser.parse_args()
    main(args)