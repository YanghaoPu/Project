
# ======================== IMPORTS ======================== #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ======================== CONFIG ======================== #

DATA_FILE = "Data Morocco.xlsx"

TIME_COL = "timestamp"              # unified timestamp
CITY_COL = "city"                   # Laayoune, Boujdour, Foum eloued, Marrakech
ZONE_COL = "zone"                   # here we always use "zone1"
RAW_VALUE_COL = "value"             # original measurement (A or kW)
TARGET_COL = "load_kw"              # current load in kW (after conversion)
FUTURE_TARGET_COL = "load_kw_1h_ahead"  # 1-hour-ahead load in kW

HORIZON_STEPS = 6                   # 6 * 10min = 60 minutes = 1 hour
WINDOW_SIZE = 24                    # sequence length for deep models (4 hours)
RANDOM_STATE = 42                   # for reproducibility

# ======================== METRICS ======================== #

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (in %)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-6  # avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def evaluate_forecast(y_true, y_pred, label="model"):
    """Compute and print RMSE, MAE, MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    print(f"=== {label} ===")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape_val:.2f}%")
    print()
    return {"rmse": rmse, "mae": mae, "mape": mape_val}

# ======================== DATA LOADING ======================== #

def load_zone1_all_cities(path):
    """
    Load 'Data Morocco.xlsx', melt each sheet, and keep only zone1.

    Original per sheet:
        DateTime, zone1, zone2, ...

    After melting:
        timestamp, zone, value, city

    We then filter `zone == "zone1"` so our project focuses on
    ONE chosen zone per city.
    """
    xls = pd.ExcelFile(path)
    frames = []

    for sheet_name in xls.sheet_names:
        df_sheet = xls.parse(sheet_name)

        # Convert DateTime to pandas datetime
        df_sheet["DateTime"] = pd.to_datetime(df_sheet["DateTime"])

        # Melt zones to long format
        df_long = df_sheet.melt(
            id_vars=["DateTime"],
            var_name=ZONE_COL,
            value_name=RAW_VALUE_COL,
        )

        df_long[CITY_COL] = sheet_name
        frames.append(df_long)

    df_all = pd.concat(frames, ignore_index=True)

    # Rename DateTime to our unified TIME_COL
    df_all = df_all.rename(columns={"DateTime": TIME_COL})

    # Only keep zone1 (our chosen zone)
    df_all = df_all[df_all[ZONE_COL] == "zone1"].reset_index(drop=True)

    print(f"Loaded {len(df_all)} rows for zone1 across all cities.")
    return df_all

# ======================== CONVERSION A -> kW ======================== #

def convert_to_kw_zone1(df):
    """
    Convert non-Marrakech zone1 values from Amperes to kW.
    Assumptions (documented in the report):
    - Laayoune, Boujdour, Foum eloued zone1 are measured in Amperes.
    - Marrakech zone1 is already given in kW.
    - Use 220 V reference for non-Marrakech zone1.
    """
    df = df.copy()

    # Boolean for Marrakech rows
    is_marrakech = df[CITY_COL].str.lower() == "marrakech"

    # Start by copying the raw values
    df[TARGET_COL] = df[RAW_VALUE_COL].astype(float)

    # For non-Marrakech rows, treat 'value' as current (A) and convert:
    # P(kW) = I(A) * V(V) / 1000
    mask = ~is_marrakech
    df.loc[mask, TARGET_COL] = df.loc[mask, RAW_VALUE_COL].astype(float) * 220 / 1000.0

    print("Converted non-Marrakech zone1 currents from A to kW (220 V).")
    return df

# ======================== RESAMPLING & CLEANING ======================== #

def resample_to_10min(df):
    """
    Resample each city's zone1 series to a common 10-minute grid.

    - Some cities already have 10-minute data.
    - Marrakech has 30-minute data; resampling will interpolate in-between.
    - Duplicate timestamps (if any) are averaged before resampling.
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    resampled_frames = []

    for city, grp in df.groupby(CITY_COL):
        # Sort by time
        grp = grp.sort_values(TIME_COL)

        # If there are duplicate timestamps, average them
        grp = grp.groupby(TIME_COL, as_index=False)[TARGET_COL].mean()

        # Set timestamp as index for resampling
        grp = grp.set_index(TIME_COL)

        # Resample at 10-minute frequency and interpolate
        grp_resampled = grp.resample("10T").interpolate("time")

        # Add city name back
        grp_resampled[CITY_COL] = city

        resampled_frames.append(grp_resampled.reset_index())

    df_resampled = pd.concat(resampled_frames, ignore_index=True)
    print(f"Resampled to 10-minute grid; new length = {len(df_resampled)}")
    return df_resampled


def basic_cleaning(df):
    """Drop rows with missing load and sort chronologically."""
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df

# ======================== FEATURE ENGINEERING ======================== #

def add_time_features(df):
    """
    Add time-of-day and day-of-week features in cyclical form.
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df["hour"] = df[TIME_COL].dt.hour
    df["dayofweek"] = df[TIME_COL].dt.dayofweek

    # Hour of day encoded on a circle
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Day of week encoded on a circle
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    return df


def add_lag_and_rolling_features(df, lags=(1, 6, 24), rolling_window=24):
    """
    Add lagged load and rolling mean features for each city.

    Lags (10-min steps):
        lag1  =  10 minutes ago
        lag6  =  60 minutes ago (1 hour)
        lag24 = 240 minutes ago (4 hours)

    Rolling mean:
        24-step mean (4 hours) to capture local trend.
    """
    df = df.copy()
    df = df.sort_values([CITY_COL, TIME_COL])

    # Lag features
    for lag in lags:
        df[f"{TARGET_COL}_lag{lag}"] = df.groupby(CITY_COL)[TARGET_COL].shift(lag)

    # Rolling mean feature (4 hours window)
    df[f"{TARGET_COL}_rollmean{rolling_window}"] = (
        df.groupby(CITY_COL)[TARGET_COL]
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Drop rows where lag features are still NaN
    df = df.dropna().reset_index(drop=True)

    print("Added lag and rolling mean features.")
    return df


def add_future_target(df, horizon_steps=HORIZON_STEPS):
    """
    Add the 1-hour-ahead target.

    For each city:
        load_kw_1h_ahead(t) = load_kw(t + 6 steps)
    """
    df = df.copy()
    df = df.sort_values([CITY_COL, TIME_COL])

    df[FUTURE_TARGET_COL] = df.groupby(CITY_COL)[TARGET_COL].shift(-horizon_steps)

    # Drop rows at the *end* where the future value is not available
    df = df.dropna(subset=[FUTURE_TARGET_COL]).reset_index(drop=True)

    print(f"Added future target column ({horizon_steps} steps ahead = 1 hour).")
    return df

# ======================== FEATURES & SPLITS ======================== #

def prepare_feature_matrix(df):
    """
    Build the full feature matrix X and label vector y.

    Features:
      - load_kw_lag1, load_kw_lag6, load_kw_lag24
      - load_kw_rollmean24
      - hour_sin, hour_cos, dow_sin, dow_cos
      - one-hot encoding of city

    Target:
      - load_kw_1h_ahead
    """
    df = df.copy()

    # One-hot encode city
    city_dummies = pd.get_dummies(df[CITY_COL], prefix="city")

    # Concatenate features
    df_feat = pd.concat([df, city_dummies], axis=1)

    feature_cols = [
        f"{TARGET_COL}_lag1",
        f"{TARGET_COL}_lag6",
        f"{TARGET_COL}_lag24",
        f"{TARGET_COL}_rollmean24",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ] + list(city_dummies.columns)

    X = df_feat[feature_cols].values
    y = df_feat[FUTURE_TARGET_COL].values

    print(f"Feature matrix shape: {X.shape}")
    return X, y, feature_cols


def split_arrays_chronologically(X, y, train_ratio=0.7, val_ratio=0.1):
    """
    Chronological (no shuffle) split of feature arrays.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(
        f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(X_train, X_val, X_test):
    """
    Min–Max scaling of features for neural networks / linear models.
    """
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler

# ======================== SEQUENCE DATA (DEEP MODELS) ======================== #

def create_sequences_from_features(X, y, window_size):
    """
    Turn (N, F) feature matrix into sequences (N', window_size, F) for RNNs.

    For index i >= window_size-1:
        X_seq[k] = X[i-window_size+1 : i+1]
        y_seq[k] = y[i]

    So each sequence uses the previous `window_size` rows of features
    to predict the 1-hour-ahead target of the *last* row in that window.
    """
    X_seq, y_seq = [], []
    for i in range(window_size - 1, len(X)):
        X_seq.append(X[i - window_size + 1 : i + 1])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# ======================== BASELINE MODELS ======================== #

def run_baselines(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate Linear Regression and Random Forest baselines.
    """
    results = {}

    # ----- Linear Regression ----- #
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["LinearRegression"] = evaluate_forecast(y_test, y_pred_lr, "Linear Regression")

    # ----- Random Forest ----- #
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Use train + val for RF final training
    rf.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    y_pred_rf = rf.predict(X_test)
    results["RandomForest"] = evaluate_forecast(y_test, y_pred_rf, "Random Forest")

    return results

# ======================== DEEP LEARNING MODELS ======================== #

def build_lstm_model(input_shape, hidden_units=64, dropout_rate=0.2):
    """Simple LSTM regression model."""
    model = models.Sequential()
    model.add(
        layers.LSTM(
            hidden_units,
            return_sequences=False,
            input_shape=input_shape,
        )
    )
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_transformer_model(input_shape, d_model=64, num_heads=4, ff_dim=128, dropout_rate=0.1):
    """Simple Transformer encoder for time-series forecasting."""
    inputs = layers.Input(shape=input_shape)

    time_steps = input_shape[0]  # window size

    # Project inputs to d_model dimensions
    x = layers.Dense(d_model)(inputs)

    # Learnable positional embeddings (one embedding per time step)
    positions = tf.range(start=0, limit=time_steps, delta=1)
    pos_embed = layers.Embedding(input_dim=time_steps, output_dim=d_model)(positions)
    pos_embed = tf.expand_dims(pos_embed, axis=0)  # (1, time_steps, d_model)

    x = x + pos_embed

    # Self-attention block
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward block
    ffn = models.Sequential(
        [
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ]
    )
    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Global pooling + final dense
    x = layers.GlobalAveragePooling1D()(out2)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_keras_model(model, X_train, y_train, X_val, y_val, label="model"):
    """
    Train a Keras model with early stopping and plot training history.
    """
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[es],
        verbose=1,
    )

    # Plot training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"{label} – Training history")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

# ======================== MAIN PIPELINE ======================== #

def main():
    # ---------- 1. Load & pre-process data ---------- #
    df = load_zone1_all_cities(DATA_FILE)   # only zone1
    df = convert_to_kw_zone1(df)            # convert A -> kW for non-Marrakech
    df = resample_to_10min(df)              # unify to 10-minute grid
    df = basic_cleaning(df)                 # drop missing, sort

    # ---------- 2. Feature engineering ---------- #
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df, lags=(1, 6, 24), rolling_window=24)
    df = add_future_target(df, horizon_steps=HORIZON_STEPS)

    # ---------- 3. Build features & split ---------- #
    X_all, y_all, feature_names = prepare_feature_matrix(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_arrays_chronologically(
        X_all, y_all, train_ratio=0.7, val_ratio=0.1
    )

    # ---------- 4. Scale features ---------- #
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # ---------- 5. Baseline models ---------- #
    print("\n================ BASELINE MODELS ================\n")
    baseline_results = run_baselines(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )

    # ---------- 6. Deep models (LSTM + Transformer) ---------- #
    print("\n================ DEEP MODELS ================\n")

    # Create sequence datasets from scaled features
    X_seq_train, y_seq_train = create_sequences_from_features(
        X_train_s, y_train, WINDOW_SIZE
    )
    X_seq_val, y_seq_val = create_sequences_from_features(
        X_val_s, y_val, WINDOW_SIZE
    )
    X_seq_test, y_seq_test = create_sequences_from_features(
        X_test_s, y_test, WINDOW_SIZE
    )

    input_shape = X_seq_train.shape[1:]  # (window_size, num_features)

    # ----- LSTM ----- #
    print("\n----- Training LSTM model -----\n")
    lstm_model = build_lstm_model(input_shape, hidden_units=64)
    lstm_model = train_keras_model(
        lstm_model,
        X_seq_train,
        y_seq_train,
        X_seq_val,
        y_seq_val,
        label="LSTM",
    )
    y_pred_lstm = lstm_model.predict(X_seq_test).flatten()
    evaluate_forecast(y_seq_test, y_pred_lstm, "LSTM")

    # ----- Transformer ----- #
    print("\n----- Training Transformer model -----\n")
    transformer_model = build_transformer_model(input_shape)
    transformer_model = train_keras_model(
        transformer_model,
        X_seq_train,
        y_seq_train,
        X_seq_val,
        y_seq_val,
        label="Transformer",
    )
    y_pred_tr = transformer_model.predict(X_seq_test).flatten()
    evaluate_forecast(y_seq_test, y_pred_tr, "Transformer")

    # Optional: quick visual comparison for Transformer
    n_plot = min(200, len(y_seq_test))
    plt.figure(figsize=(8, 4))
    plt.plot(y_seq_test[:n_plot], label="True")
    plt.plot(y_pred_tr[:n_plot], label="Transformer prediction")
    plt.xlabel("Test sample index")
    plt.ylabel("Load (kW)")
    plt.title("Transformer – 1-hour-ahead forecast (zone1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Pipeline finished.")


if __name__ == "__main__":
    main()

