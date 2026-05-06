"""
Synthetic Data Generator — Dynamic Pricing & Demand Forecasting
================================================================
Generates realistic ride-hailing / e-commerce style data with:
  - Hourly demand with intraday + weekly seasonality
  - Simulated supply (driver availability / stock levels)
  - Base pricing with historical variation
  - Event spikes (holidays, concerts, rain, etc.)
  - Realistic noise

Outputs 4 CSVs into ./data/:
  - demand.csv        → hourly demand counts
  - supply.csv        → hourly supply availability
  - pricing.csv       → historical base prices (with variation)
  - events.csv        → event calendar with types and intensity

Usage:
    python generate_data.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIG — tweak these to change scale/detail
# ─────────────────────────────────────────────
SEED          = 42
START_DATE    = "2023-01-01"
END_DATE      = "2023-12-31"
BASE_DEMAND   = 120          # baseline rides/orders per hour
BASE_PRICE    = 10.0         # base price per unit (₹ or $)
BASE_SUPPLY   = 100          # baseline supply units per hour
ELASTICITY    = -1.2         # price elasticity coefficient
NOISE_STD     = 0.08         # demand noise as % of signal (8%)
OUTPUT_DIR    = "./data"

np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. TIME INDEX — full year, hourly
# ─────────────────────────────────────────────
print("⏱  Building time index...")
timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq="h")
n = len(timestamps)

df = pd.DataFrame({"timestamp": timestamps})
df["hour"]       = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
df["month"]      = df["timestamp"].dt.month
df["week"]       = df["timestamp"].dt.isocalendar().week.astype(int)
df["is_weekend"] = df["day_of_week"] >= 5
df["date"]       = df["timestamp"].dt.date

# ─────────────────────────────────────────────
# 2. INTRADAY DEMAND CURVE
#    Double-peak: morning rush (8–10am) + evening rush (5–8pm)
#    Night trough: 2–5am
# ─────────────────────────────────────────────
print("📈 Generating intraday demand pattern...")

def intraday_multiplier(hour):
    """
    Returns a demand multiplier based on hour of day.
    Models two peak periods typical of ride-hailing / food delivery.
    """
    # Morning peak: 8–10am
    morning = 1.8 * np.exp(-0.5 * ((hour - 9) / 1.2) ** 2)
    # Lunch bump: 12–1pm
    lunch   = 0.6 * np.exp(-0.5 * ((hour - 12.5) / 0.8) ** 2)
    # Evening peak: 5–8pm (wider, stronger)
    evening = 2.2 * np.exp(-0.5 * ((hour - 18.5) / 1.8) ** 2)
    # Late night dip: baseline 0.2 during 2–5am
    base    = 0.25
    return base + morning + lunch + evening

df["intraday_mult"] = df["hour"].apply(intraday_multiplier)

# ─────────────────────────────────────────────
# 3. DAY-OF-WEEK PATTERN
#    Weekends higher for leisure, Mondays lowest
# ─────────────────────────────────────────────
dow_multipliers = {
    0: 0.85,   # Monday   — lowest
    1: 0.90,   # Tuesday
    2: 0.92,   # Wednesday
    3: 0.95,   # Thursday
    4: 1.10,   # Friday   — building up
    5: 1.25,   # Saturday — peak leisure
    6: 1.15,   # Sunday
}
df["dow_mult"] = df["day_of_week"].map(dow_multipliers)

# ─────────────────────────────────────────────
# 4. SEASONAL TREND — monthly
#    Summer and Dec holiday peaks, Feb trough
# ─────────────────────────────────────────────
monthly_multipliers = {
    1:  0.90,   # Jan — post-holiday dip
    2:  0.85,   # Feb — lowest
    3:  0.92,
    4:  0.98,
    5:  1.05,
    6:  1.12,   # Summer begins
    7:  1.18,   # Peak summer
    8:  1.15,
    9:  1.05,
    10: 1.02,
    11: 1.08,   # Pre-holiday
    12: 1.20,   # Holiday surge
}
df["seasonal_mult"] = df["month"].map(monthly_multipliers)

# ─────────────────────────────────────────────
# 5. EVENTS CALENDAR
#    5 types: holiday, concert, sports, rain, festival
#    Each has an intensity and duration
# ─────────────────────────────────────────────
print("🎉 Injecting event spikes...")

events = [
    # (date_str, event_name, type, duration_hours, demand_mult, supply_impact)
    # demand_mult: how much demand spikes (1.0 = no change, 2.0 = doubles)
    # supply_impact: how much supply drops (0.0 = no drop, 0.4 = 40% fewer drivers)

    # Holidays
    ("2023-01-26", "Republic Day",          "holiday",  18, 1.6,  0.30),
    ("2023-08-15", "Independence Day",      "holiday",  18, 1.5,  0.25),
    ("2023-10-02", "Gandhi Jayanti",        "holiday",  15, 1.3,  0.20),
    ("2023-12-25", "Christmas",             "holiday",  24, 1.7,  0.40),
    ("2023-12-31", "New Year's Eve",        "holiday",  18, 2.5,  0.50),
    ("2023-01-01", "New Year's Day",        "holiday",  12, 1.4,  0.45),

    # Concerts / large events
    ("2023-03-11", "Stadium Concert",       "concert",   8, 3.0,  0.15),
    ("2023-07-22", "Music Festival",        "concert",  12, 2.8,  0.10),
    ("2023-09-16", "Sports Finals",         "sports",    6, 2.5,  0.10),
    ("2023-11-05", "EDM Night",             "concert",   6, 2.2,  0.10),

    # Rain events — boost demand, hurt supply
    ("2023-06-15", "Heavy Monsoon Rain",    "weather",   8, 2.0,  0.35),
    ("2023-07-08", "Torrential Rain",       "weather",  12, 2.3,  0.45),
    ("2023-08-03", "Storm",                 "weather",   6, 1.9,  0.40),
    ("2023-09-20", "Heavy Showers",         "weather",   5, 1.7,  0.30),

    # Festivals
    ("2023-03-08", "Holi",                  "festival", 12, 1.8,  0.35),
    ("2023-10-24", "Diwali Eve",            "festival", 18, 2.0,  0.30),
    ("2023-10-25", "Diwali",                "festival", 24, 1.6,  0.50),
    ("2023-11-27", "Local Food Festival",   "festival",  8, 1.4,  0.10),
]

# Build events lookup: timestamp → (demand_mult, supply_impact)
event_demand_boost  = np.ones(n)
event_supply_impact = np.zeros(n)
event_flag          = np.zeros(n, dtype=int)
event_name_col      = [""] * n
event_type_col      = [""] * n

event_records = []

for (date_str, name, etype, duration, dmult, simp) in events:
    event_start = pd.Timestamp(date_str)
    for h in range(duration):
        ts = event_start + timedelta(hours=h)
        # Taper intensity: full for first half, fade in second half
        taper = 1.0 if h < duration // 2 else 1.0 - 0.4 * ((h - duration // 2) / (duration // 2 + 1))
        effective_mult = 1.0 + (dmult - 1.0) * taper
        mask = df["timestamp"] == ts
        idx = df.index[mask]
        if len(idx):
            i = idx[0]
            event_demand_boost[i]  = max(event_demand_boost[i], effective_mult)
            event_supply_impact[i] = max(event_supply_impact[i], simp * taper)
            event_flag[i]         = 1
            event_name_col[i]     = name
            event_type_col[i]     = etype

    event_records.append({
        "date":           date_str,
        "event_name":     name,
        "event_type":     etype,
        "duration_hours": duration,
        "demand_multiplier": dmult,
        "supply_impact":  simp,
    })

df["event_demand_boost"]  = event_demand_boost
df["event_supply_impact"] = event_supply_impact
df["is_event"]            = event_flag
df["event_name"]          = event_name_col
df["event_type"]          = event_type_col

# ─────────────────────────────────────────────
# 6. COMPUTE RAW DEMAND
# ─────────────────────────────────────────────
print("🔢 Computing demand signal...")

# Combine all multipliers multiplicatively
df["demand_signal"] = (
    BASE_DEMAND
    * df["intraday_mult"]
    * df["dow_mult"]
    * df["seasonal_mult"]
    * df["event_demand_boost"]
)

# Add realistic noise (multiplicative log-normal noise preserves non-negativity)
noise = np.random.lognormal(mean=0, sigma=NOISE_STD, size=n)
df["demand"] = np.maximum(0, np.round(df["demand_signal"] * noise).astype(int))

# ─────────────────────────────────────────────
# 7. COMPUTE SUPPLY
#    Supply has its own intraday pattern (fewer drivers at 3am)
#    Events and rain reduce supply (drivers avoid bad conditions)
# ─────────────────────────────────────────────
print("🚗 Generating supply data...")

def supply_intraday(hour):
    """Drivers come online during busy hours, drop off at night."""
    # Peak availability 8am–10pm, low overnight
    daytime = 0.6 + 0.4 * np.clip(
        np.sin(np.pi * (hour - 6) / 14), 0, 1
    )
    return daytime

df["supply_intraday_mult"] = df["hour"].apply(supply_intraday)

# Weekend supply slightly lower (drivers take days off)
df["supply_dow_mult"] = df["is_weekend"].map({True: 0.88, False: 1.0})

# Compute supply with event impact reducing it
supply_signal = (
    BASE_SUPPLY
    * df["supply_intraday_mult"]
    * df["supply_dow_mult"]
    * (1 - df["event_supply_impact"])
)

# Supply noise (drivers randomly go offline)
supply_noise = np.random.lognormal(mean=0, sigma=0.06, size=n)
df["supply"] = np.maximum(1, np.round(supply_signal * supply_noise).astype(int))

# ─────────────────────────────────────────────
# 8. DEMAND-SUPPLY RATIO (DSR)
#    Core signal for pricing engine
# ─────────────────────────────────────────────
df["DSR"] = df["demand"] / (df["supply"] + 1)

# ─────────────────────────────────────────────
# 9. HISTORICAL PRICING WITH VARIATION
#    Price varies based on DSR + some manual adjustments
#    This lets us estimate price elasticity from data
# ─────────────────────────────────────────────
print("💰 Generating historical pricing...")

def historical_price(row, base=BASE_PRICE):
    """
    Simulates what prices WOULD have been under a naive manual pricing policy.
    Includes some randomness to create variation needed for elasticity estimation.
    """
    dsr       = row["DSR"]
    # Rough manual surge: DSR > 1.5 → price goes up, DSR < 0.7 → slight discount
    if dsr > 2.0:
        manual_mult = np.random.uniform(1.4, 1.8)
    elif dsr > 1.5:
        manual_mult = np.random.uniform(1.1, 1.4)
    elif dsr < 0.7:
        manual_mult = np.random.uniform(0.85, 0.95)
    else:
        manual_mult = np.random.uniform(0.95, 1.10)

    # Add random variation (mimics human pricing inconsistency)
    random_noise = np.random.uniform(0.92, 1.08)
    return round(base * manual_mult * random_noise, 2)

df["price"] = df.apply(historical_price, axis=1)

# Adjust observed demand based on price (apply elasticity retrospectively)
# demand_adj = demand * (price / base_price) ^ elasticity
df["demand_price_adjusted"] = np.maximum(
    0, np.round(df["demand"] * (df["price"] / BASE_PRICE) ** ELASTICITY).astype(int)
)

# Revenue = demand_price_adjusted × price
df["revenue"] = df["demand_price_adjusted"] * df["price"]

# ─────────────────────────────────────────────
# 10. LAG FEATURES (pre-computed for convenience)
#     Models can recompute these, but pre-computing
#     them here makes EDA much easier
# ─────────────────────────────────────────────
print("⏳ Adding lag & rolling features...")

df = df.sort_values("timestamp").reset_index(drop=True)

df["demand_lag_1h"]        = df["demand"].shift(1)
df["demand_lag_3h"]        = df["demand"].shift(3)
df["demand_lag_24h"]       = df["demand"].shift(24)
df["demand_lag_7d"]        = df["demand"].shift(24 * 7)

df["demand_rolling_3h"]    = df["demand"].shift(1).rolling(3).mean()
df["demand_rolling_6h"]    = df["demand"].shift(1).rolling(6).mean()
df["demand_rolling_24h"]   = df["demand"].shift(1).rolling(24).mean()
df["demand_rolling_std_3h"]= df["demand"].shift(1).rolling(3).std()

df["supply_lag_1h"]        = df["supply"].shift(1)
df["supply_rolling_3h"]    = df["supply"].shift(1).rolling(3).mean()

df["DSR_lag_1h"]           = df["DSR"].shift(1)
df["DSR_rolling_3h"]       = df["DSR"].shift(1).rolling(3).mean()

# ─────────────────────────────────────────────
# 11. SPLIT INTO FOUR CLEAN CSVs
# ─────────────────────────────────────────────
print("💾 Saving CSVs...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# — demand.csv
demand_cols = [
    "timestamp", "hour", "day_of_week", "month", "week",
    "is_weekend", "is_event", "event_name", "event_type",
    "intraday_mult", "dow_mult", "seasonal_mult", "event_demand_boost",
    "demand", "demand_price_adjusted",
    "demand_lag_1h", "demand_lag_3h", "demand_lag_24h", "demand_lag_7d",
    "demand_rolling_3h", "demand_rolling_6h", "demand_rolling_24h", "demand_rolling_std_3h",
]
df[demand_cols].to_csv(f"{OUTPUT_DIR}/demand.csv", index=False)

# — supply.csv
supply_cols = [
    "timestamp", "hour", "day_of_week", "is_weekend",
    "is_event", "event_supply_impact",
    "supply", "supply_lag_1h", "supply_rolling_3h",
    "DSR", "DSR_lag_1h", "DSR_rolling_3h",
]
df[supply_cols].to_csv(f"{OUTPUT_DIR}/supply.csv", index=False)

# — pricing.csv
pricing_cols = [
    "timestamp", "hour", "day_of_week", "month",
    "is_event", "event_type",
    "demand", "demand_price_adjusted", "supply", "DSR",
    "price", "revenue",
]
df[pricing_cols].to_csv(f"{OUTPUT_DIR}/pricing.csv", index=False)

# — events.csv
events_df = pd.DataFrame(event_records)
events_df.to_csv(f"{OUTPUT_DIR}/events.csv", index=False)

# ─────────────────────────────────────────────
# 12. SUMMARY STATS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅  DATA GENERATION COMPLETE")
print("=" * 55)
print(f"  Period        : {START_DATE} → {END_DATE}")
print(f"  Total rows    : {len(df):,} hourly records")
print(f"  Events        : {len(events)} injected events")
print(f"  Avg demand    : {df['demand'].mean():.1f} units/hour")
print(f"  Peak demand   : {df['demand'].max()} units/hour")
print(f"  Avg supply    : {df['supply'].mean():.1f} units/hour")
print(f"  Avg DSR       : {df['DSR'].mean():.2f}")
print(f"  Peak DSR      : {df['DSR'].max():.2f}  ← pricing opportunity")
print(f"  Price range   : ₹{df['price'].min():.2f} – ₹{df['price'].max():.2f}")
print(f"  Total revenue : ₹{df['revenue'].sum():,.0f}")
print(f"  Event hours   : {df['is_event'].sum()} ({df['is_event'].mean()*100:.1f}% of data)")
print("=" * 55)
print(f"\n  Files saved to: {OUTPUT_DIR}/")
print("    ├── demand.csv")
print("    ├── supply.csv")
print("    ├── pricing.csv")
print("    └── events.csv")
print()

# Quick sanity check
print("📊 Quick sanity — top 5 demand hours:")
top5 = (
    df[["timestamp", "demand", "supply", "DSR", "price", "event_name"]]
    .nlargest(5, "demand")
)
print(top5.to_string(index=False))

print("\n📊 Avg demand by hour of day (peak check):")
hourly = df.groupby("hour")["demand"].mean().round(1)
for h, d in hourly.items():
    bar = "█" * int(d / 10)
    print(f"  {h:02d}h  {d:6.1f}  {bar}")