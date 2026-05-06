"""
feature_engineering.py
-----------------------
Loads raw CSVs, creates event type flags, fixes rolling std,
merges demand + supply, and returns a model-ready DataFrame.

Usage:
    from src.feature_engineering import build_features
    df, events = build_features(data_dir='data/comprehensive_data')
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Feature columns used by the model ──────────────────────────────────────
FEATURE_COLS = [
    # Time
    'hour', 'day_of_week', 'month', 'week', 'is_weekend',
    # Event flags
    'is_event', 'is_weather_event', 'is_concert_event', 'is_festival_event',
    # Demand lags
    'demand_lag_1h', 'demand_lag_3h', 'demand_lag_24h', 'demand_lag_7d',
    # Demand rolling
    'demand_rolling_3h', 'demand_rolling_6h',
    'demand_rolling_24h', 'demand_rolling_std_3h',
    # Supply & DSR
    'supply', 'supply_lag_1h', 'supply_rolling_3h',
    'DSR', 'DSR_lag_1h', 'DSR_rolling_3h',
]

FORECAST_TARGET   = 'demand'                # raw demand → forecasting model
ELASTICITY_TARGET = 'demand_price_adjusted' # price-adjusted → elasticity only


def load_raw_data(data_dir: str = 'data/comprehensive_data') -> dict:
    """
    Load all 4 raw CSVs from data_dir.

    Args:
        data_dir: path to folder containing demand.csv, supply.csv,
                  pricing.csv, events.csv

    Returns:
        dict with keys: demand, supply, pricing, events
    """
    base = Path(data_dir)
    data = {
        'demand' : pd.read_csv(base / 'demand.csv',
                               parse_dates=['timestamp']),
        'supply' : pd.read_csv(base / 'supply.csv',
                               parse_dates=['timestamp']),
        'pricing': pd.read_csv(base / 'pricing.csv',
                               parse_dates=['timestamp']),
        'events' : pd.read_csv(base / 'events.csv',
                               parse_dates=['date']),
    }
    print(f"Loaded: demand {data['demand'].shape}, "
          f"supply {data['supply'].shape}, "
          f"pricing {data['pricing'].shape}, "
          f"events {data['events'].shape}")
    return data


def add_event_type_flags(demand: pd.DataFrame,
                          events: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary event-type flags from the events calendar.

    Avoids using raw event_name / event_type strings as features —
    they are too sparse and high-cardinality for direct model use.
    One binary column per event category instead.

    Args:
        demand : demand DataFrame (must have 'timestamp' column)
        events : events calendar (must have 'event_type', 'date')

    Returns:
        demand DataFrame with 3 new binary columns added
    """
    demand = demand.copy()

    weather_dates  = set(
        events[events['event_type'] == 'weather']['date'].astype(str))
    concert_dates  = set(
        events[events['event_type'].isin(['concert', 'sports'])]['date'].astype(str))
    festival_dates = set(
        events[events['event_type'].isin(['festival', 'holiday'])]['date'].astype(str))

    demand['date_str'] = demand['timestamp'].dt.date.astype(str)
    demand['is_weather_event']  = demand['date_str'].isin(weather_dates).astype(int)
    demand['is_concert_event']  = demand['date_str'].isin(concert_dates).astype(int)
    demand['is_festival_event'] = demand['date_str'].isin(festival_dates).astype(int)
    demand.drop(columns=['date_str'], inplace=True)

    print(f"Event flags — weather: {demand['is_weather_event'].sum()}h, "
          f"concert: {demand['is_concert_event'].sum()}h, "
          f"festival: {demand['is_festival_event'].sum()}h")
    return demand


def fix_rolling_std(demand: pd.DataFrame) -> pd.DataFrame:
    """
    Fix demand_rolling_std_3h — fill NaNs with 0 and add +0.01 floor.

    Prevents zero-std values causing issues in log-scaled contexts
    and ensures the column is always usable as a model feature.

    Args:
        demand: demand DataFrame

    Returns:
        demand DataFrame with demand_rolling_std_3h cleaned
    """
    demand = demand.copy()
    demand['demand_rolling_std_3h'] = (
        demand['demand_rolling_std_3h'].fillna(0) + 0.01
    )
    return demand


def merge_and_build(demand: pd.DataFrame,
                    supply: pd.DataFrame) -> pd.DataFrame:
    """
    Merge demand and supply DataFrames on timestamp.
    Drop the 168-row lag warmup window (required for demand_lag_7d).

    Args:
        demand: demand DataFrame (with event flags already added)
        supply: supply DataFrame

    Returns:
        merged model-ready DataFrame
    """
    df = demand.merge(
        supply[['timestamp', 'supply', 'DSR', 'DSR_lag_1h',
                'DSR_rolling_3h', 'supply_lag_1h', 'supply_rolling_3h']],
        on='timestamp', how='left'
    )

    # Drop warmup: demand_lag_7d needs 168 rows (7d × 24h) to populate
    before = len(df)
    df = df.dropna(subset=['demand_lag_7d']).reset_index(drop=True)
    dropped = before - len(df)
    print(f"Warmup rows dropped: {dropped} ({dropped/before*100:.1f}%)")
    print(f"Final shape : {df.shape}")
    print(f"Date range  : {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")
    return df


def build_features(data_dir: str = 'data/comprehensive_data') -> tuple:
    """
    Full feature engineering pipeline — single entry point.

    Loads raw CSVs → adds event flags → fixes rolling std →
    merges demand + supply → drops warmup rows.

    Args:
        data_dir: path to folder with the 4 raw CSVs

    Returns:
        tuple: (df, events)
            df     — model-ready DataFrame with all FEATURE_COLS present
            events — events calendar DataFrame
    """
    print("=" * 50)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    data   = load_raw_data(data_dir)
    demand = add_event_type_flags(data['demand'], data['events'])
    demand = fix_rolling_std(demand)
    df     = merge_and_build(demand, data['supply'])

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print(f"\nAll {len(FEATURE_COLS)} feature columns present ✓")
    print("=" * 50)
    return df, data['events']


if __name__ == '__main__':
    df, events = build_features()
    print(df[FEATURE_COLS].head())