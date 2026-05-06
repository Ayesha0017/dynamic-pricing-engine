"""
app.py — Dynamic Pricing Dashboard (Polished Version)
Run with: python -m streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('src')

from pricing_engine import (
    adjusted_demand,
    pricing_multiplier,
    get_dynamic_price,
    BASE_PRICE
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Dynamic Pricing Engine")
st.caption("End-to-end pricing system with simulation + real-world validation")

# ─────────────────────────────────────────────
# MODE SELECTOR (DEFAULT = REAL DATA)
# ─────────────────────────────────────────────
mode = st.radio(
    "Select Mode",
    ["Real Data Mode", "Simulation Mode"],
    horizontal=True
)

# =========================================================
# 🔵 REAL DATA MODE (DEFAULT)
# =========================================================
if mode == "Real Data Mode":

    st.subheader("📊 Real Data Pricing Dashboard")

    file_path = "output/phase4_full_simulation.csv"

    if not os.path.exists(file_path):
        st.error("⚠️ Run: python src/simulation.py first")
        st.stop()

    df = pd.read_csv(file_path, parse_dates=['timestamp'])


    index = st.slider("Select Time Step", 0, len(df) - 1, 0)
    row = df.iloc[index]

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Time", str(row['timestamp']))
    col2.metric("DSR", f"{row['DSR']:.2f}")
    col3.metric("Price (₹)", f"{row['smoothed_price']:.2f}")
    col4.metric("Multiplier", f"{row['multiplier']:.2f}×")
    col5.metric("Revenue (₹)", f"{row['dynamic_revenue']:,.0f}")

    st.divider()

    # TRENDS
    st.subheader("📈 Pricing & Demand-Supply Trends Over Time")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['timestamp'], df['smoothed_price'], label="Price (₹)")
    ax.plot(df['timestamp'], df['DSR'], label="DSR")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # REVENUE COMPARISON
    st.subheader("💰 Static vs Dynamic Revenue Over Time")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['timestamp'], df['static_revenue'], label="Static Revenue (₹)")
    ax.plot(df['timestamp'], df['dynamic_revenue'], label="Dynamic Revenue (₹)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Revenue (₹)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # INSIGHTS
    st.subheader("🧠 Business Insights")

    static_total = df['static_revenue'].sum()
    dynamic_total = df['dynamic_revenue'].sum()
    uplift = ((dynamic_total - static_total) / static_total) * 100

    if uplift > 5:
        st.success(f"🚀 Strong uplift: +{uplift:.1f}% revenue increase")
    elif uplift > 0:
        st.info(f"📈 Moderate uplift: +{uplift:.1f}% improvement")
    else:
        st.warning("⚠️ Pricing strategy hurting revenue")

    st.write(f"**Total Static Revenue:** ₹{static_total:,.0f}")
    st.write(f"**Total Dynamic Revenue:** ₹{dynamic_total:,.0f}")

    # ZONE ANALYSIS
    st.subheader("📊 Revenue by Market Condition (DSR Zones)")

    zone_summary = df.groupby('dsr_zone').agg({
        'dynamic_revenue': 'sum',
        'static_revenue': 'sum'
    }).reset_index()

    zone_summary['uplift_%'] = (
        (zone_summary['dynamic_revenue'] - zone_summary['static_revenue'])
        / zone_summary['static_revenue'] * 100
    )

    st.dataframe(zone_summary)


# =========================================================
# 🟢 SIMULATION MODE
# =========================================================
elif mode == "Simulation Mode":

    st.sidebar.header("📊 Market Conditions")

    demand = st.sidebar.slider("Demand (orders)", 20, 500, 150, 10)
    supply = st.sidebar.slider("Supply (drivers/stock)", 10, 200, 80, 5)
    is_event = st.sidebar.toggle("Event active (rain/concert/holiday)", False)

    prev_price = st.sidebar.slider("Previous price (₹)", 5.0, 30.0, 10.0, 0.5)
    base_price = st.sidebar.slider("Base price (₹)", 5.0, 20.0, 10.0, 0.5)

    st.sidebar.divider()

    st.sidebar.header("⚙️ Engine Parameters")

    alpha = st.sidebar.slider("Multiplier sensitivity (α)", 0.1, 1.0, 0.5, 0.05)
    smooth_alpha = st.sidebar.slider("Smoothing factor", 0.1, 1.0, 0.25, 0.05)

    # CORE LOGIC
    DSR = demand / (supply + 1)

    volatility = 0.1 + (0.2 if is_event else 0)
    pred_low = demand * (1 - volatility)
    pred_high = demand * (1 + volatility)

    multiplier = pricing_multiplier(DSR, pred_low, pred_high, demand, alpha=alpha)

    result = get_dynamic_price(
        DSR=DSR,
        pred_low=pred_low,
        pred_high=pred_high,
        pred_mid=demand,
        prev_price=prev_price,
        base_price=base_price,
        smooth_alpha=smooth_alpha
    )

    dynamic_price = result['smoothed_price']
    dsr_zone = result['dsr_zone']

    adj_demand = adjusted_demand(demand, base_price, dynamic_price)

    static_rev = demand * base_price
    dynamic_rev = adj_demand * dynamic_price

    rev_diff = dynamic_rev - static_rev
    rev_uplift_pct = (rev_diff / static_rev * 100) if static_rev != 0 else 0

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("DSR", f"{DSR:.2f}", dsr_zone)
    col2.metric("Multiplier", f"{multiplier:.3f}×")
    col3.metric("Dynamic Price (₹)", f"{dynamic_price:.2f}")
    col4.metric("Revenue (₹)", f"{dynamic_rev:,.0f}", f"{rev_uplift_pct:+.1f}%")
    col5.metric("Demand retained (orders)", f"{adj_demand:.0f}")

    st.divider()

    # VISUALS
    left, right = st.columns(2)

    with left:
        st.subheader("📈 Price Sensitivity Curve (Demand vs Revenue Trade-off)")

        prices = np.linspace(5, 30, 200)
        demands = [adjusted_demand(demand, base_price, p) for p in prices]
        revenues = [d * p for d, p in zip(demands, prices)]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax2 = ax.twinx()

        ax.plot(prices, demands, label="Demand (orders)")
        ax2.plot(prices, revenues, linestyle="--", label="Revenue (₹)")

        ax.axvline(dynamic_price, linestyle="--", linewidth=2, label="Dynamic Price")
        ax.axvline(base_price, linestyle=":", label="Base Price")

        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Demand (orders)")
        ax2.set_ylabel("Revenue (₹)")

        ax.legend()
        st.pyplot(fig)
        plt.close()

    with right:
        st.subheader("📊 Static vs Dynamic Comparison")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Static", "Dynamic"], [static_rev, dynamic_rev])
        ax.set_ylabel("Revenue (₹)")
        st.pyplot(fig)
        plt.close()

    # RECOMMENDATION (NEW)
    st.subheader("🧠 Pricing Recommendation")

    if DSR > 1.2 and rev_uplift_pct > 0:
        st.success("Increase price — strong demand and revenue improving")
    elif DSR < 0.8:
        st.warning("Reduce price — weak demand")
    else:
        st.info("Maintain price — balanced market")

    # BREAKDOWN
    st.subheader("📊 Pricing Decision Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
**Market signal**
- Demand: {demand} orders  
- Supply: {supply} units  
- DSR: {DSR:.2f}  
- Zone: {dsr_zone}  
- Event: {'Yes' if is_event else 'No'}
""")

    with col2:
        st.info(f"""
**Engine output**
- Multiplier: {multiplier:.3f}×  
- Raw price: ₹{base_price * multiplier:.2f}  
- Smoothed price: ₹{dynamic_price:.2f}
""")

    with col3:
        st.success(f"""
**Revenue impact**
- Static: ₹{static_rev:,.0f}  
- Dynamic: ₹{dynamic_rev:,.0f}  
- Change: ₹{rev_diff:+,.0f}  
- Uplift: {rev_uplift_pct:+.1f}%  
- Demand retained: {(adj_demand/demand)*100:.1f}%
""")