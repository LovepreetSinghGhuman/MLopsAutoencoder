#!/usr/bin/env python
# coding: utf-8
# ü

# === Standard Library and Data Science Imports ===
import os
import logging
import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed

# === Sklearn Imports ===
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# === TensorFlow / Keras Imports ===
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# === Keras Tuner Imports ===
from kerastuner import HyperModel
from keras_tuner import HyperModel, HyperParameters, Hyperband, Objective, RandomSearch



# === Config Class === 
class Config:
    # General settings
    SEED = 42
    EPOCHS = 150
    BATCH_SIZE = 256
    VAL_SIZE = 0.2
    LEARNING_RATE = 1e-4
    OPTIMIZE_BY = "roc_auc"  # or "f1_score"
    THRESHOLD_RANGE = (0.9, 0.99)
    NUM_THRESHOLDS = 100

    # Early stopping and LR scheduling
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-6

    # Directory paths
    BASE_DIR = Path(".")
    DATA_DIR_PROCESS = BASE_DIR / "Data" / "processed"
    RESULTS_DIR = BASE_DIR / "Results"
    MODEL_DIR = BASE_DIR / "models"

    # Model and config file paths
    MODEL_PATH = MODEL_DIR / "best_autoencoder.keras"
    THRESHOLD_PATH = MODEL_DIR / "best_threshold.json"
    CONFIG_PATH = MODEL_DIR / "autoencoder_config.json"

    @staticmethod
    def prepare_dirs():
        for directory in [
            Config.DATA_DIR_PROCESS,
            Config.MODEL_DIR,
            Config.RESULTS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

# === Environment Setup ===
Config.prepare_dirs()
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)
warnings.filterwarnings('ignore')

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



try:
    train_data = pd.read_csv(Config.DATA_DIR_PROCESS /'merged_train.csv')
    test_data = pd.read_csv(Config.DATA_DIR_PROCESS/'merged_test.csv')
except FileNotFoundError as e:
    logging.error(f"Data files not found: {e}")
    raise



# Checking if there are any feature columns available in train but not in test
different_features = [features for features in train_data.columns if features not in test_data.columns]
different_features



# Define a function to clean the dataset
def clean_data(df):
    """
    Cleans the input DataFrame by:
    - Removing duplicate rows
    - Handling missing values (numeric: median, categorical: 'missing')
    - Removing constant columns
    - Removing columns with too many missing values (>90%)
    - Mapping email domains to groups using EMAIL_DOMAIN_MAP
    """
    df = df.copy()

    # Unify id feature names: id-XX -> id_XX
    df.columns = df.columns.str.replace(r'^id-', 'id_', regex=True)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove constant columns
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index
    df = df.drop(columns=constant_cols)

    # Remove columns with >90% missing values
    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.9].index
    df = df.drop(columns=high_missing_cols)

    # Map email domains to groups
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(EMAIL_DOMAIN_MAP).fillna('other')

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('missing')
        else:
            df[col] = df[col].fillna(df[col].median())

    return df

# Efficient email domain mapping using dictionary and pandas .map()
EMAIL_DOMAIN_MAP = {
    # Google
    'gmail.com': 'google', 'googlemail.com': 'google',
    # Yahoo
    'yahoo.com': 'yahoo', 'yahoo.com.mx': 'yahoo', 'yahoo.co.uk': 'yahoo', 'yahoo.co.jp': 'yahoo',
    'ymail.com': 'yahoo', 'rocketmail.com': 'yahoo',
    # Microsoft
    'hotmail.com': 'microsoft', 'outlook.com': 'microsoft', 'live.com': 'microsoft', 'msn.com': 'microsoft',
    # Apple
    'icloud.com': 'apple', 'me.com': 'apple', 'mac.com': 'apple',
    # AOL
    'aol.com': 'aol', 'aim.com': 'aol',
    # Protonmail
    'protonmail.com': 'protonmail',
    # Comcast
    'comcast.net': 'comcast',
    # Verizon
    'verizon.net': 'verizon',
    # Optonline
    'optonline.net': 'optonline',
    # Cox
    'cox.net': 'cox',
    # Charter
    'charter.net': 'charter',
    # AT&T
    'att.net': 'att', 'sbcglobal.net': 'att', 'bellsouth.net': 'att',
    # Earthlink
    'earthlink.net': 'earthlink',
    # Embarqmail
    'embarqmail.com': 'embarqmail',
    # Frontier
    'frontier.com': 'frontier', 'frontiernet.net': 'frontier',
    # Windstream
    'windstream.net': 'windstream',
    # Spectrum
    'twc.com': 'spectrum', 'roadrunner.com': 'spectrum',
    # Centurylink
    'centurylink.net': 'centurylink',
    # Suddenlink
    'suddenlink.net': 'suddenlink',
    # Netzero
    'netzero.net': 'netzero', 'netzero.com': 'netzero',
    # GMX
    'gmx.de': 'gmx', 'gmx.com': 'gmx',
    # Mail.ru
    'mail.ru': 'mailru',
    # Naver
    'naver.com': 'naver',
    # Yandex
    'yandex.ru': 'yandex', 'yandex.com': 'yandex',
    # Mail.com
    'mail.com': 'mail.com'
}


# Apply the cleaning function to both train_data and test_data
train_data = clean_data(train_data)
test_data = clean_data(test_data)


# Save the cleaned datasets
train_data.to_csv(Config.DATA_DIR_PROCESS / 'cleaned_train.csv', index=False)
test_data.to_csv(Config.DATA_DIR_PROCESS / 'cleaned_test.csv', index=False)

print("Cleaned train and test data saved successfully!")



train_data = pd.read_csv(Config.DATA_DIR_PROCESS / 'cleaned_train.csv')
test_data = pd.read_csv(Config.DATA_DIR_PROCESS / 'cleaned_test.csv')


# In[11]:


def preprocess_data(
    df,
    scaler=None,
    is_train=True,
    scaler_path=Config.MODEL_DIR/"scaler.joblib",
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

    # 3. Feature Engineering: TransactionAmt (log transform, remove outliers in train)
    if 'TransactionAmt' in df.columns:
        if is_train:
            df = df[df['TransactionAmt'] < 30000]
        df['TransactionAmt_Log'] = np.log1p(np.clip(df['TransactionAmt'], a_min=0, a_max=None))
        # TransactionAmt relative to card1 mean
        if 'card1' in df.columns:
            card1_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_card1_mean'] = df['TransactionAmt'] / (card1_mean + 1e-3)
        df.drop(columns=['TransactionAmt'], inplace=True)

    # 4. DeviceInfo grouping (prioritize device type)
    if 'DeviceInfo' in df.columns:
        df['DeviceInfo'] = df['DeviceInfo'].fillna('missing').str.lower()
        df['DeviceInfo_grouped'] = df['DeviceInfo'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else 'missing')

    # 5. DeviceType (fill missing, encode)
    if 'DeviceType' in df.columns:
        df['DeviceType'] = df['DeviceType'].fillna('missing').str.lower()

    # 6. Prioritize ProductCD, card1-card6, addr1/addr2, M1-M9, id_30, id_31, id_33, id_34, id_36, id_38
    # Frequency encoding for high-cardinality features
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

def stratified_split(data, val_ratio, seed):
    """
    Performs a stratified split on the dataset based on the 'isFraud' column.
    Stratified split means dividing your dataset into training and validation (or test) sets while preserving the proportion of each class label
    """
    return train_test_split(
        data,
        test_size=val_ratio,
        stratify=data["isFraud"],
        random_state=seed
    )

def split_normal_data(data, timestamp_col, val_ratio):
    """
    Splits normal (non-fraud) data into training and validation sets based on time.
    """
    normal_data = data[data["isFraud"] == 0].sort_values(by=timestamp_col)
    split_index = int(len(normal_data) * (1 - val_ratio))
    return normal_data.iloc[:split_index], normal_data.iloc[split_index:]


# === Preprocessing ===
# Step 1: Time-based split on normal (non-fraud) data
X_train, X_val = split_normal_data(train_data, timestamp_col="TransactionDT", val_ratio=Config.VAL_SIZE)

# Step 2: Preprocess training data (fit scaler here)
X_train_auto, scaler, feature_columns = preprocess_data(X_train, is_train=True)

# Step 3: Stratified split for validation (for final evaluation)
_, val_temp = stratified_split(train_data, val_ratio=Config.VAL_SIZE, seed=Config.SEED)

# Step 4: Preprocess validation data using saved scaler
X_val_auto, _, _ = preprocess_data(val_temp, scaler=scaler, is_train=False, feature_columns=feature_columns)
y_val_auto = val_temp["isFraud"].values


# --- Autoencoder Model Builder ---
def build_autoencoder(input_dim: int, version: str = "v1") -> Tuple[Model, Model]:
    """
    Architecture version to use. Options:
        - "v1": Basic encoder-decoder with dense layers and batch normalization.
        - "v2": Adds dropout and stronger L2 regularization for improved generalization.
        - "v3": Deeper model with more layers, dropout, and batch normalization.
        - "v4": Uses LeakyReLU activations, advanced initialization, and a deeper encoder/decoder.
        - "v5": Deepest model with an explicit bottleneck, heavy dropout, and batch normalization.
    """
    input_layer = Input(shape=(input_dim,), name="input_layer")
    if version == "v3":
        # Improved version of v2 with deeper layers, advanced activation, and better regularization
        encoded = Dense(512, activation="relu", kernel_regularizer=l2(1e-4), name="encoder_dense_1")(input_layer)
        encoded = BatchNormalization(name="encoder_bn_1")(encoded)
        encoded = Dense(256, activation="relu", kernel_regularizer=l2(1e-4), name="encoder_dense_2")(encoded)
        encoded = Dropout(0.3, name="encoder_dropout_1")(encoded)
        encoded = Dense(128, activation="relu", kernel_regularizer=l2(1e-4), name="encoder_dense_3")(encoded)
        encoded = BatchNormalization(name="encoder_bn_2")(encoded)
        encoded = Dense(64, activation="relu", kernel_regularizer=l2(1e-4), name="encoder_dense_4")(encoded)

        decoded = Dense(64, activation="relu", kernel_regularizer=l2(1e-4), name="decoder_dense_1")(encoded)
        decoded = Dropout(0.3, name="decoder_dropout_1")(decoded)
        decoded = Dense(128, activation="relu", kernel_regularizer=l2(1e-4), name="decoder_dense_2")(decoded)
        decoded = BatchNormalization(name="decoder_bn_1")(decoded)
        decoded = Dense(256, activation="relu", kernel_regularizer=l2(1e-4), name="decoder_dense_3")(decoded)
        decoded = Dropout(0.3, name="decoder_dropout_2")(decoded)
        decoded = Dense(512, activation="relu", kernel_regularizer=l2(1e-4), name="decoder_dense_4")(decoded)
        decoded = Dense(input_dim, activation="linear", name="output_layer")(decoded)
    else:
        raise ValueError(f"Unknown version: {version}")

    autoencoder = Model(inputs=input_layer, outputs=decoded, name=f"autoencoder_{version}")
    encoder = Model(inputs=input_layer, outputs=encoded, name=f"encoder_{version}")
    return autoencoder, encoder


# --- Training Function ---
def train_autoencoder(
    model: Model,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> tf.keras.callbacks.History:
    monitor_metric = "val_loss" if X_val is not None else "loss"
    checkpoint_path = save_path or Config.MODEL_DIR / "best_autoencoder.keras"

    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor=monitor_metric, patience=Config.REDUCE_LR_PATIENCE, factor=Config.REDUCE_LR_FACTOR, min_lr=Config.MIN_LR),
        ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_metric, save_best_only=True)
    ]

    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss="mse")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val) if X_val is not None else None,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    return history

# --- Helper Functions for Evaluation ---
def get_reconstruction_errors(model: Model, X: np.ndarray) -> np.ndarray:
    reconstructions = model.predict(X, batch_size=Config.BATCH_SIZE)
    return np.mean(np.power(X - reconstructions, 2), axis=1)

def evaluate_with_threshold(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    mse = get_reconstruction_errors(model, X)
    y_pred = (mse > threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y, mse),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, y_pred),
        "classification_report": classification_report(y, y_pred),
        "average_precision": average_precision_score(y, mse),
        "f1_score": f1_score(y, y_pred),
        "precision_score": precision_score(y, y_pred),
        "recall_score": recall_score(y, y_pred)
    }

def select_best_threshold(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    threshold_range: Tuple[float, float] = (0.9, 0.99),
    n_jobs: int = -1,
    optimize_by: str = "roc_auc"
) -> float:
    mse = get_reconstruction_errors(model, X)
    lower_q = np.quantile(mse, threshold_range[0])
    upper_q = np.quantile(mse, threshold_range[1])
    thresholds = np.linspace(lower_q, upper_q, Config.NUM_THRESHOLDS)

    def evaluate_threshold(t): 
        metrics = evaluate_with_threshold(model, X, y, t)
        return t, metrics.get(optimize_by, 0)

    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_threshold)(t) for t in thresholds)
    best_threshold, best_score = max(results, key=lambda x: x[1])
    logger.info(f"Best threshold selected by {optimize_by}: {best_threshold:.6f} (score: {best_score:.4f})")
    return best_threshold

def plot_loss(
    results: Dict[str, Any],
    save_path: Path = Config.RESULTS_DIR / "training_loss.png"
) -> None:
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for idx, (version, metrics) in enumerate(results.items()):
        plt.plot(
            metrics['history'].history['loss'],
            label=f'{version} Train',
            color=colors[idx],
            linestyle='-'
        )
        if 'val_loss' in metrics['history'].history:
            plt.plot(
                metrics['history'].history['val_loss'],
                label=f'{version} Val',
                color=colors[idx],
                linestyle='--'
            )

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss plot saved to {save_path}")


# --- Main Training & Selection Workflow ---
results = {}
best_model = None
best_f1 = 0.0
best_auc = 0.0
best_version = None

input_dim = X_train_auto.shape[1]

for version in ["v3"] :
    logger.info(f"Training autoencoder version {version}...")
    autoencoder, encoder = build_autoencoder(input_dim=input_dim, version=version)
    history = train_autoencoder(autoencoder, X_train_auto, X_val_auto)

    best_threshold = select_best_threshold(
        autoencoder, 
        X_val_auto, 
        y_val_auto,
        threshold_range=Config.THRESHOLD_RANGE,
        optimize_by=Config.OPTIMIZE_BY
    )

    val_metrics = evaluate_with_threshold(autoencoder, X_val_auto, y_val_auto, best_threshold)
    results[version] = {
        "threshold": best_threshold,
        "roc_auc": val_metrics["roc_auc"],
        "f1_score": val_metrics["f1_score"],
        "precision_score": val_metrics["precision_score"],
        "recall_score": val_metrics["recall_score"],
        "classification_report": val_metrics["classification_report"],
        "val_loss": min(history.history['val_loss']),
        "history": history
    }

    logger.info("="*50)
    logger.info(f"Version {version} Validation Metrics:\n{val_metrics['classification_report']}")
    logger.info(f"ROC-AUC: {val_metrics['roc_auc']:.4f}, F1: {val_metrics['f1_score']:.4f}")
    logger.info("="*50)

    # --- ROC-AUC based model selection ---
    if val_metrics["roc_auc"] > best_auc:
        best_auc = val_metrics["roc_auc"]
        best_f1 = val_metrics["f1_score"]
        best_model = autoencoder
        best_version = version
        logger.info(f"New best model: {version} (AUC: {best_auc:.4f}, F1: {best_f1:.4f})")
        best_model.save(Config.MODEL_PATH)
        with open(Config.THRESHOLD_PATH, "w") as f:
            json.dump({
                "threshold": best_threshold
            }, f)
        with open(Config.CONFIG_PATH, "w") as f:
            json.dump({
                "input_dim": input_dim,
                "version": version,
                "feature_columns": feature_columns
            }, f)

logger.info("\n=== Training Complete ===")
logger.info(f"Best model: {best_model.name if best_model else 'None'}")
logger.info(f"Validation ROC-AUC: {best_auc:.4f}, F1: {best_f1:.4f}")


# Plot loss curves for all models and returning the saved model and threshold.
plot_loss(results)
logger.info(f"Threshold saved: {Config.THRESHOLD_PATH}")
logger.info(f"Model saved: {Config.MODEL_PATH}")




# --- Efficient Hyperparameter Tuning with Keras Tuner on the Best Saved Model ---
# Load best model configuration (input_dim, version, l2_reg, dropout)
with open(Config.CONFIG_PATH, "r") as f:
    best_config = json.load(f)
input_dim = best_config["input_dim"]
best_version = best_config.get("version", "v3")
default_l2 = results.get(best_version, {}).get("l2_reg", 1e-4)
default_dropout = results.get(best_version, {}).get("dropout", 0.5)

class AutoencoderV3HyperModel(HyperModel):
    def __init__(self, input_dim, default_l2, default_dropout):
        self.input_dim = input_dim
        self.default_l2 = default_l2
        self.default_dropout = default_dropout

    def build(self, hp):
        l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log', default=self.default_l2)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.7, step=0.05, default=self.default_dropout)
        input_layer = Input(shape=(self.input_dim,), name="input_layer")
        # Encoder
        encoded = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(input_layer)
        encoded = LeakyReLU(alpha=0.1)(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(128, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(encoded)
        encoded = LeakyReLU(alpha=0.1)(encoded)
        encoded = Dropout(dropout_rate)(encoded)
        encoded = Dense(64, kernel_initializer="he_normal")(encoded)
        encoded = LeakyReLU(alpha=0.1)(encoded)
        # Bottleneck
        bottleneck = Dense(32, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(encoded)
        bottleneck = LeakyReLU(alpha=0.1)(bottleneck)
        # Decoder
        decoded = Dense(64, kernel_initializer="he_normal")(bottleneck)
        decoded = LeakyReLU(alpha=0.1)(decoded)
        decoded = Dropout(dropout_rate)(decoded)
        decoded = Dense(128, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(decoded)
        decoded = LeakyReLU(alpha=0.1)(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(l2_reg))(decoded)
        decoded = LeakyReLU(alpha=0.1)(decoded)
        decoded = Dense(self.input_dim, activation='linear', name="output_layer")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded, name="tuned_autoencoder_v3")
        autoencoder.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss='mse')
        return autoencoder

# Free up memory before tuning
gc.collect()
tf.keras.backend.clear_session()

# Use a smaller max_epochs and executions_per_trial for faster tuning
tuner = Hyperband(
    hypermodel=AutoencoderV3HyperModel(input_dim=input_dim, default_l2=default_l2, default_dropout=default_dropout),
    objective='val_loss',
    max_epochs=min(Config.EPOCHS, 50),  # Reduce epochs for tuning
    factor=3,
    executions_per_trial=1,  # Only 1 execution per trial for speed
    directory=str(Config.MODEL_DIR),
    project_name='autoencoder_tuning_v3',
    overwrite=True
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=max(3, Config.EARLY_STOPPING_PATIENCE // 2),  # Shorter patience for tuning
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.5,
    min_lr=Config.MIN_LR
)

logger.info("Efficient hyperparameter tuning on the best saved model...")
tuner.search(
    X_train_auto, X_train_auto,
    validation_data=(X_val_auto, X_val_auto),
    epochs=min(Config.EPOCHS, 50),
    batch_size=Config.BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Get and save the best tuned model
best_hp_model = tuner.get_best_models(num_models=1)[0]
tuned_model_path = Config.MODEL_DIR / 'tuned_autoencoder_v3.keras'
best_hp_model.save(tuned_model_path)
logger.info(f"Tuned model saved to {tuned_model_path}")

# Log best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
logger.info(f"Best hyperparameters (v3):\n- L2 regularization: {best_hps.get('l2_reg')}\n- Dropout rate: {best_hps.get('dropout_rate')}")

# Find and save the best threshold for the tuned model
best_threshold = select_best_threshold(
    best_hp_model,
    X_val_auto,
    y_val_auto,
    threshold_range=Config.THRESHOLD_RANGE,
    optimize_by=Config.OPTIMIZE_BY
)
with open(Config.MODEL_DIR / 'tuned_threshold_v3.json', 'w') as f:
    json.dump({'threshold': best_threshold}, f)

# Evaluate and log performance
val_metrics = evaluate_with_threshold(best_hp_model, X_val_auto, y_val_auto, best_threshold)
logger.info(f"Tuned Model (v3) Metrics:\n{val_metrics['classification_report']}")

# Clean up after tuning
gc.collect()
tf.keras.backend.clear_session()



def create_submission_tuned_best_model(test_df: pd.DataFrame) -> None:
    logger.info("Generating final submission...")

    try:
        # Load necessary components
        scaler = load_scaler(Config.MODEL_DIR / "scaler.joblib")
        model = load_model_safely(Config.MODEL_DIR / 'tuned_autoencoder.keras')
        version = "tuned_version"

        # Preprocess test data
        try:
            X_test = preprocess_and_validate(test_df, scaler)
        except ValueError as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return

        # Predict using reconstruction error
        mse = get_reconstruction_errors(model, X_test)
        logger.info(f"MSE stats - Min: {mse.min():.2f}, Max: {mse.max():.2f}, Mean: {mse.mean():.2f}")

         # --- Normalize MSE to [0, 1] for probability submission ---
        mse_norm = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)

        # Create submission DataFrame with normalized probabilities
        submission = pd.DataFrame({
            "TransactionID": test_df["TransactionID"].astype(str),
            "isFraud": mse  # Probability for ROC-AUC evaluation
        })

        # Save submission
        safe_version = version.replace(" ", "_").replace("/", "-")
        submission_path = Config.RESULTS_DIR / f"submission_{safe_version}.csv"
        try:
            submission.to_csv(submission_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save submission: {str(e)}")
            return

        logger.info(f"Submission created: {submission_path}")
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during submission generation: {str(e)}")


# === Helper Functions ===
def load_scaler(path: Path):
    if not path.exists():
        raise ValueError("Scaler file not found. Cannot proceed with submission.")
    return joblib.load(path)

def preprocess_and_validate(test_df: pd.DataFrame, scaler) -> np.ndarray:
    # Load feature columns from config
    with open(Config.CONFIG_PATH, "r") as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns")
    if feature_columns is None:
        raise ValueError("feature_columns not found in config. Cannot align test data.")
    X_test, _, _ = preprocess_data(test_df, is_train=False, scaler=scaler, feature_columns=feature_columns)
    if len(X_test) != len(test_df):
        raise ValueError(f"Data mismatch: {len(X_test)} vs {len(test_df)} rows")
    return X_test

def load_model_safely(path: Path):
    if not path.exists():
        raise ValueError(f"Model file {path} not found")
    try:
        model = tf.keras.models.load_model(path)
        logger.info(f"Loaded model from {path}")
        return model
    except Exception as e:
        raise ValueError(f"Model loading failed: {str(e)}")

def load_threshold(path: Path) -> float:
    if not path.exists():
        raise ValueError(f"Threshold file {path} missing")
    try:
        with open(path, "r") as f:
            return json.load(f)["threshold"]
    except Exception as e:
        raise ValueError(f"Threshold loading failed: {str(e)}")

# Executing the prediction on test data
transaction_ids = pd.read_csv(Config.DATA_DIR_PROCESS / 'merged_test.csv')[['TransactionID']]
test_data_with_id = test_data.copy()
test_data_with_id['TransactionID'] = transaction_ids
create_submission_tuned_best_model(test_data_with_id)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="Results")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # Override config with arguments
    Config.DATA_DIR_PROCESS = Path(args.data_dir)
    Config.MODEL_DIR = Path(args.model_dir)
    Config.RESULTS_DIR = Path(args.results_dir)
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.prepare_dirs()
