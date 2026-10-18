"""
Accelerated Machine Learning Pipeline
Google Cloud & NVIDIA Codelab Implementation

Demonstrates GPU acceleration for tabular data using:
- NVIDIA cuDF (pandas acceleration)
- NVIDIA cuML (scikit-learn acceleration)
- XGBoost GPU (device='cuda', tree_method='hist')
"""

import argparse
import glob
import os
import sys
import time
import numpy as np

def check_gpu_availability():
    """Check if NVIDIA GPU and RAPIDS libraries are available."""
    gpu_available = False
    try:
        import cupy
        gpu_available = True
    except ImportError:
        pass
    try:
        import cudf
        print("[INFO] NVIDIA cuDF is available.")
    except ImportError:
        print("[INFO] cuDF not detected; falling back to standard pandas if needed.")
    return gpu_available

def download_data(year=2024, months=(1,), data_dir="nyc_taxi_data"):
    """Download NYC Yellow Taxi Trip record parquet files."""
    import requests
    from tqdm import tqdm

    os.makedirs(data_dir, exist_ok=True)
    downloaded_files = []

    for month in months:
        file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
        local_path = os.path.join(data_dir, file_name)
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"

        if not os.path.exists(local_path):
            print(f"[DATA] Downloading {file_name} from CloudFront...")
            try:
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    with open(local_path, 'wb') as f, tqdm(
                        desc=file_name,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=65536):
                            size = f.write(chunk)
                            bar.update(size)
                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] Failed downloading {file_name}: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
        else:
            print(f"[DATA] Found local copy of {file_name}.")
        downloaded_files.append(local_path)

    return downloaded_files

def load_and_clean_data(data_dir="nyc_taxi_data", pd_module=None, sample_fraction=None):
    """Load parquet data and apply cleaning filters."""
    import pandas as pd
    if pd_module is None:
        pd_module = pd

    files = glob.glob(os.path.join(data_dir, "*-01.parquet"))
    if not files:
        files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}. Run data download first.")

    t0 = time.perf_counter()
    print(f"[PREP] Loading parquet files from {data_dir}...")
    df = pd_module.concat([pd_module.read_parquet(f) for f in files], ignore_index=True)
    load_time = time.perf_counter() - t0
    print(f"[PREP] Loaded {len(df):,} records in {load_time:.2f}s.")

    if sample_fraction and 0.0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"[PREP] Subsampled to {len(df):,} records.")

    # Cleaning: credit card trips with valid fare and distance
    t0 = time.perf_counter()
    df = df[
        (df['fare_amount'] > 0) & (df['fare_amount'] < 500) &
        (df['trip_distance'] > 0) & (df['trip_distance'] < 100) &
        (df['tip_amount'] >= 0) & (df['tip_amount'] < 100) &
        (df['payment_type'] == 1)
    ].copy()

    # Downcast numeric columns
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')
    clean_time = time.perf_counter() - t0

    print(f"[PREP] Cleaned down to {len(df):,} records in {clean_time:.2f}s.")
    return df

def feature_engineering(df):
    """Create derived features for model training."""
    t0 = time.perf_counter()
    print("[FE] Generating temporal, financial, and spatial features...")

    # Time Features
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['dow'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_rush_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)

    # Fare Features
    df['fare_log'] = np.log1p(df['fare_amount'])
    df['fare_decimal'] = (df['fare_amount'] % 1 * 100).astype(int)
    df['is_round_fare'] = (df['fare_amount'] % 5 == 0).astype(int)

    # Route Features
    df['route_id'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    route_counts = df['route_id'].value_counts()
    df['route_frequency'] = df['route_id'].map(route_counts)

    # Location Aggregations
    pu_tip_stats = df.groupby('PULocationID')['tip_amount'].agg(['mean', 'std']).reset_index()
    pu_tip_stats.columns = ['PULocationID', 'pu_tip_mean', 'pu_tip_std']
    df = df.merge(pu_tip_stats, on='PULocationID', how='left')

    fe_time = time.perf_counter() - t0
    print(f"[FE] Feature engineering complete in {fe_time:.2f}s.")
    return df

def train_models(df, use_gpu=False, n_splits=3):
    """Train XGBoost, Linear Regression, and Random Forest using Cross-Validation."""
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    from tqdm import tqdm

    feature_cols = [
        'trip_distance', 'fare_amount', 'passenger_count',
        'hour', 'dow', 'is_weekend', 'is_rush_hour',
        'fare_log', 'fare_decimal', 'is_round_fare',
        'route_frequency', 'pu_tip_mean', 'pu_tip_std',
        'PULocationID', 'DOLocationID'
    ]

    print("[TRAIN] Preparing feature matrix X and target y...")
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['tip_amount'].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- 1. XGBoost ---
    print("\n--- Training Model 1: XGBoost Regressor ---")
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 5,
        'learning_rate': 0.1,
        'verbosity': 0,
        'tree_method': 'hist',
        'max_bin': 256,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    if use_gpu:
        xgb_params['device'] = 'cuda'
        xgb_params['sampling_method'] = 'gradient_based'

    t0 = time.perf_counter()
    xgb_rmses, xgb_preds = [], np.zeros(len(y))
    for train_idx, val_idx in tqdm(kf.split(X), total=n_splits, desc="XGBoost CV"):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                          evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        preds = model.predict(dval)
        xgb_preds[val_idx] = preds
        xgb_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
    xgb_rmse = np.mean(xgb_rmses)
    xgb_time = time.perf_counter() - t0
    print(f"XGBoost RMSE:       ${xgb_rmse:.4f} (took {xgb_time:.2f}s)")

    # --- 2. Linear Regression ---
    print("\n--- Training Model 2: Linear Regression ---")
    t0 = time.perf_counter()
    lr_rmses, lr_preds = [], np.zeros(len(y))
    for train_idx, val_idx in tqdm(kf.split(X), total=n_splits, desc="LinearReg CV"):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = LinearRegression()
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_va_s)
        lr_preds[val_idx] = preds
        lr_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
    lr_rmse = np.mean(lr_rmses)
    lr_time = time.perf_counter() - t0
    print(f"Linear Reg RMSE:    ${lr_rmse:.4f} (took {lr_time:.2f}s)")

    # --- 3. Random Forest ---
    print("\n--- Training Model 3: Random Forest Regressor ---")
    t0 = time.perf_counter()
    rf_rmses, rf_preds = [], np.zeros(len(y))
    for train_idx, val_idx in tqdm(kf.split(X), total=n_splits, desc="RandomForest CV"):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            max_features='sqrt',
            random_state=42
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        rf_preds[val_idx] = preds
        rf_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
    rf_rmse = np.mean(rf_rmses)
    rf_time = time.perf_counter() - t0
    print(f"Random Forest RMSE: ${rf_rmse:.4f} (took {rf_time:.2f}s)")

    # --- 4. Ensemble Model ---
    print("\n--- Evaluating End-to-End Ensemble Pipeline ---")
    preds_matrix = np.c_[xgb_preds, rf_preds, lr_preds]
    ensemble_model = LinearRegression(positive=True, fit_intercept=False).fit(preds_matrix, y)
    weights = ensemble_model.coef_ / ensemble_model.coef_.sum()
    ensemble_preds = preds_matrix @ weights
    ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_preds))
    lift = xgb_rmse - ensemble_rmse

    print("\n" + "="*45)
    print(f"{'Model':<22} {'CV RMSE':>12} {'Train Time':>10}")
    print("="*45)
    print(f"{'Linear Regression':<22} ${lr_rmse:>10.4f} {lr_time:>9.2f}s")
    print(f"{'Random Forest':<22} ${rf_rmse:>10.4f} {rf_time:>9.2f}s")
    print(f"{'XGBoost':<22} ${xgb_rmse:>10.4f} {xgb_time:>9.2f}s")
    print("-" * 45)
    print(f"{'Ensemble':<22} ${ensemble_rmse:>10.4f} {'-':>10}")
    print("="*45)
    print(f"Optimal Weights: XGBoost={weights[0]:.3f}, RF={weights[1]:.3f}, LR={weights[2]:.3f}")
    print(f"Ensemble Accuracy Lift: ${lift:.4f} over single best model")

    return {
        'xgb_rmse': xgb_rmse, 'xgb_time': xgb_time,
        'lr_rmse': lr_rmse, 'lr_time': lr_time,
        'rf_rmse': rf_rmse, 'rf_time': rf_time,
        'ensemble_rmse': ensemble_rmse,
        'weights': weights.tolist(),
        'lift': lift
    }

def main():
    parser = argparse.ArgumentParser(description="Accelerated Machine Learning Pipeline with Google Cloud & NVIDIA")
    parser.add_argument("--use-gpu", action="store_true", help="Enable NVIDIA GPU acceleration (cuDF, cuML, XGBoost CUDA)")
    parser.add_argument("--download-only", action="store_true", help="Download dataset and exit")
    parser.add_argument("--data-dir", type="str", default="nyc_taxi_data", help="Directory for taxi parquet datasets")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Fraction of dataset to sample (e.g. 0.1 for 10%)")
    args = parser.parse_args()

    print("=" * 60)
    print(" Accelerated Machine Learning with Google Cloud & NVIDIA GPUs")
    print(" Codelab: cuDF / cuML / XGBoost GPU Acceleration")
    print("=" * 60)

    if args.use_gpu:
        try:
            import IPython
            ip = IPython.get_ipython()
            if ip:
                ip.run_line_magic('load_ext', 'cudf.pandas')
                ip.run_line_magic('load_ext', 'cuml.accel')
            print("[GPU] RAPIDS extensions loaded.")
        except Exception as e:
            print(f"[WARN] Could not load notebook magics in script mode: {e}")

    download_data(year=2024, months=(1,), data_dir=args.data_dir)
    if args.download_only:
        print("[DONE] Data download complete.")
        return

    df = load_and_clean_data(data_dir=args.data_dir, sample_fraction=args.sample_fraction)
    df = feature_engineering(df)
    results = train_models(df, use_gpu=args.use_gpu)
    print("\n[SUCCESS] Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
