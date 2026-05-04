"""
demand_model.py
---------------
Trains the LightGBM demand forecasting model, evaluates against a
naive baseline, trains quantile models for confidence intervals,
plots results, and saves all model artefacts.

Usage:
    python src/demand_model.py

    Or import individual functions:
        from src.demand_model import train_model, evaluate_model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engineering import build_features, FEATURE_COLS, FORECAST_TARGET


# ── Constants ───────────────────────────────────────────────────────────────
TRAIN_SPLIT   = 0.80   # 80% train, 20% test (chronological)
VAL_SPLIT     = 0.10   # 10% of train used for early stopping validation
MODELS_DIR    = 'models'
OUTPUT_DIR    = 'output'


# ── Model config ─────────────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators      = 3000,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 50,
    min_child_samples = 5,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    min_split_gain    = 0.01,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

QUANTILE_PARAMS = dict(
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 50,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_split_gain    = 0.01,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)


def compute_metrics(actual: np.ndarray,
                    predicted: np.ndarray,
                    label: str = '') -> dict:
    """
    Compute MAE, RMSE, MAPE for a set of predictions.

    Args:
        actual    : array of actual values
        predicted : array of predicted values
        label     : label for print output

    Returns:
        dict with keys: mae, rmse, mape
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = (np.abs(actual - predicted) / (actual + 1e-8)).mean() * 100

    if label:
        print(f"── {label} ──")
        print(f"  MAE  : {mae:.2f}")
        print(f"  RMSE : {rmse:.2f}")
        print(f"  MAPE : {mape:.1f}%")

    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def naive_baseline(df: pd.DataFrame) -> dict:
    """
    Evaluate naive persistence baseline: predict same hour yesterday.
    This is the benchmark all ML models must beat.

    Args:
        df: full model-ready DataFrame

    Returns:
        dict with baseline metrics and aligned arrays
    """
    mask    = df['demand_lag_24h'].notna()
    actual  = df.loc[mask, 'demand'].values
    pred    = df.loc[mask, 'demand_lag_24h'].values
    metrics = compute_metrics(actual, pred, 'Naive Baseline (lag_24h)')
    return {**metrics, 'actual': actual, 'pred': pred}


def split_data(df: pd.DataFrame) -> dict:
    """
    Chronological train/validation/test split.
    NEVER shuffle time series data — future leaks into past.

    Split:
        Train      : first 80% × 90% of rows
        Validation : first 80% × last 10% (for early stopping)
        Test       : last 20% (unseen, evaluated once)

    Args:
        df: model-ready DataFrame

    Returns:
        dict with X_tr, y_tr, X_val, y_val, X_test, y_test, split_idx
    """
    X = df[FEATURE_COLS]
    y = df[FORECAST_TARGET]

    split_idx = int(len(df) * TRAIN_SPLIT)
    val_size  = int(split_idx * VAL_SPLIT)

    X_train, X_test = X.iloc[:split_idx],  X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx],  y.iloc[split_idx:]
    X_tr,    X_val  = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr,    y_val  = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    print(f"Split date  : {df['timestamp'].iloc[split_idx].date()}")
    print(f"Train rows  : {len(X_tr):,}  |  Val: {len(X_val):,}  "
          f"|  Test: {len(X_test):,}")
    print(f"Train period: {df['timestamp'].iloc[0].date()} → "
          f"{df['timestamp'].iloc[split_idx - val_size - 1].date()}")
    print(f"Test period : {df['timestamp'].iloc[split_idx].date()} → "
          f"{df['timestamp'].iloc[-1].date()}")

    return dict(X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test, split_idx=split_idx)


def train_model(X_tr, y_tr, X_val, y_val) -> lgb.LGBMRegressor:
    """
    Train LightGBM point forecast model with early stopping.

    Early stopping monitors validation loss and halts training
    when it stops improving — prevents overfitting without manual tuning.

    Args:
        X_tr, y_tr : training features and target
        X_val, y_val: validation set for early stopping

    Returns:
        trained LGBMRegressor with best_iteration_ set
    """
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=100),
        ]
    )
    print(f"\nBest iteration : {model.best_iteration_} trees")
    return model


def train_quantile_models(X_tr, y_tr,
                           n_estimators: int) -> tuple:
    """
    Train lower (5th pct) and upper (95th pct) quantile models.

    Produces uncertainty-aware forecasts — the pricing engine uses
    the band width to discount the multiplier when uncertainty is high.

    Args:
        X_tr, y_tr   : training data
        n_estimators : use model.best_iteration_ from point forecast

    Returns:
        tuple: (model_low, model_high)
    """
    model_low = lgb.LGBMRegressor(
        objective='quantile', alpha=0.05,
        n_estimators=n_estimators, **QUANTILE_PARAMS
    )
    model_high = lgb.LGBMRegressor(
        objective='quantile', alpha=0.95,
        n_estimators=n_estimators, **QUANTILE_PARAMS
    )
    model_low.fit(X_tr, y_tr)
    model_high.fit(X_tr, y_tr)
    print("Quantile models trained (5th / 95th percentile)")
    return model_low, model_high


def evaluate_coverage(y_test, pred_low, pred_high) -> float:
    """
    Compute empirical coverage of the prediction interval.
    Target: ~80% of actuals should fall within the 5/95 band.

    Args:
        y_test    : actual test values
        pred_low  : lower bound predictions
        pred_high : upper bound predictions

    Returns:
        float: coverage percentage
    """
    coverage = ((y_test.values >= pred_low) &
                (y_test.values <= pred_high)).mean() * 100
    print(f"Prediction interval coverage : {coverage:.1f}%")
    return coverage


def plot_results(df, split_idx, y_test,
                 pred_mid, pred_low, pred_high,
                 model, output_dir: str = OUTPUT_DIR):
    """
    Generate 3 output charts:
        1. Actual vs predicted (first 2 weeks of test)
        2. 80% confidence band
        3. Feature importance (gain)

    Args:
        df         : full DataFrame
        split_idx  : index where test period starts
        y_test     : actual test values
        pred_mid   : point forecast
        pred_low   : lower bound forecast
        pred_high  : upper bound forecast
        model      : trained LGBMRegressor
        output_dir : folder to save PNGs
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamps = df['timestamp'].iloc[split_idx:].values
    n_show     = 24 * 14  # 2 weeks

    # ── Chart 1: actual vs predicted ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps[:n_show], y_test.values[:n_show],
            lw=1.2, color='steelblue', label='Actual demand', alpha=0.9)
    ax.plot(timestamps[:n_show], pred_mid[:n_show],
            lw=1.2, color='tomato', label='Predicted', alpha=0.9, ls='--')
    ax.set_title('Demand forecast — actual vs predicted (first 2 weeks of test)')
    ax.set_xlabel('Date'); ax.set_ylabel('Demand')
    ax.legend(); plt.tight_layout()
    plt.savefig(f'{output_dir}/demand_forecast_actual_vs_pred.png', dpi=150)
    plt.show()

    # ── Chart 2: confidence bands ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps[:n_show], y_test.values[:n_show],
            lw=1.2, color='steelblue', label='Actual', alpha=0.9)
    ax.plot(timestamps[:n_show], pred_mid[:n_show],
            lw=1.2, color='tomato', label='Predicted (median)', ls='--')
    ax.fill_between(timestamps[:n_show],
                    pred_low[:n_show], pred_high[:n_show],
                    alpha=0.15, color='tomato', label='80% prediction interval')
    ax.set_title('Demand forecast with 80% confidence band')
    ax.set_xlabel('Date'); ax.set_ylabel('Demand')
    ax.legend(); plt.tight_layout()
    plt.savefig(f'{output_dir}/demand_forecast_confidence.png', dpi=150)
    plt.show()

    # ── Chart 3: feature importance ──
    fig, ax = plt.subplots(figsize=(8, 6))
    lgb.plot_importance(model, ax=ax, importance_type='gain',
                        max_num_features=15,
                        title='Feature importance (gain)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150)
    plt.show()


def save_artefacts(model, model_low, model_high,
                   df, split_idx, y_test,
                   pred_mid, pred_low, pred_high,
                   models_dir: str = MODELS_DIR,
                   output_dir: str = OUTPUT_DIR):
    """
    Save trained models and predictions CSV for Phase 3.

    Applies quantile crossing fix: ensures pred_low ≤ pred_mid ≤ pred_high.

    Args:
        model, model_low, model_high : trained LightGBM models
        df         : full DataFrame
        split_idx  : test period start index
        y_test     : actual test values
        pred_*     : prediction arrays
        models_dir : folder for .pkl files
        output_dir : folder for predictions CSV
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model,      f'{models_dir}/lgbm_demand_forecast.pkl')
    joblib.dump(model_low,  f'{models_dir}/lgbm_demand_low.pkl')
    joblib.dump(model_high, f'{models_dir}/lgbm_demand_high.pkl')

    results = pd.DataFrame({
        'timestamp'    : df['timestamp'].iloc[split_idx:].values,
        'actual_demand': y_test.values,
        'pred_demand'  : pred_mid,
        'pred_low'     : np.minimum(pred_low,  pred_mid),  # crossing fix
        'pred_high'    : np.maximum(pred_high, pred_mid),  # crossing fix
        'supply'       : df['supply'].iloc[split_idx:].values,
        'DSR'          : df['DSR'].iloc[split_idx:].values,
        'is_event'     : df['is_event'].iloc[split_idx:].values,
    })
    results.to_csv(f'{output_dir}/phase2_predictions.csv', index=False)

    print(f"\nSaved models → {models_dir}/")
    print(f"Saved predictions → {output_dir}/phase2_predictions.csv "
          f"({len(results):,} rows)")


def run_pipeline(data_dir: str = 'data/comprehensive_data'):
    """
    Full demand forecasting pipeline — single entry point.

    Steps:
        1. Build features
        2. Naive baseline
        3. Train/test split
        4. Train LightGBM with early stopping
        5. Evaluate on test set
        6. Train quantile models
        7. Plot results
        8. Save artefacts

    Args:
        data_dir: path to raw CSV folder
    """
    # 1. Features
    df, _  = build_features(data_dir)
    splits = split_data(df)
    X_tr, y_tr     = splits['X_tr'],    splits['y_tr']
    X_val, y_val   = splits['X_val'],   splits['y_val']
    X_test, y_test = splits['X_test'],  splits['y_test']
    split_idx      = splits['split_idx']

    # 2. Baseline
    print("\n" + "=" * 40)
    base = naive_baseline(df)

    # 3. Train
    print("\n" + "=" * 40)
    model = train_model(X_tr, y_tr, X_val, y_val)

    # 4. Evaluate
    print("\n" + "=" * 40)
    pred_test  = model.predict(X_test)
    pred_train = model.predict(X_tr)
    tr_m  = compute_metrics(y_tr.values,    pred_train, 'LightGBM — Train')
    te_m  = compute_metrics(y_test.values,  pred_test,  'LightGBM — Test')
    print(f"\nMAPE improvement over baseline : "
          f"{base['mape'] - te_m['mape']:.1f} pp")
    print(f"Train vs test MAPE gap         : "
          f"{tr_m['mape'] - te_m['mape']:.1f} pp  (< 5pp = healthy)")

    # 5. Quantile models
    print("\n" + "=" * 40)
    model_low, model_high = train_quantile_models(
        X_tr, y_tr, model.best_iteration_)
    pred_low  = model_low.predict(X_test)
    pred_high = model_high.predict(X_test)
    coverage  = evaluate_coverage(y_test, pred_low, pred_high)

    # 6. Plots
    plot_results(df, split_idx, y_test,
                 pred_test, pred_low, pred_high, model)

    # 7. Save
    save_artefacts(model, model_low, model_high,
                   df, split_idx, y_test,
                   pred_test, pred_low, pred_high)

    # 8. Summary
    print(f"\n{'='*45}")
    print(f"  DEMAND MODEL COMPLETE")
    print(f"{'='*45}")
    print(f"  Baseline MAPE  : {base['mape']:.1f}%")
    print(f"  Model MAPE     : {te_m['mape']:.1f}%")
    print(f"  Improvement    : {base['mape'] - te_m['mape']:.1f} pp")
    print(f"  Coverage       : {coverage:.1f}%")
    print(f"  Best iteration : {model.best_iteration_} trees")
    print(f"  Features       : {len(FEATURE_COLS)}")
    print(f"{'='*45}")


if __name__ == '__main__':
    run_pipeline()