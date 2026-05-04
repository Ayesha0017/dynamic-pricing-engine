"""
app.py — Dynamic Pricing Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from pricing_engine import (adjusted_demand, pricing_multiplier,
                             get_dynamic_price, BASE_PRICE)

# ── Page config ──
st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Dynamic Pricing Engine")
st.caption("Ride-hailing / E-commerce · Real-time pricing simulator")

# ── Sidebar controls ──
st.sidebar.header("Market Conditions")
demand   = st.sidebar.slider("Demand (rides/orders)",
                              min_value=20, max_value=500,
                              value=150, step=10)
supply   = st.sidebar.slider("Supply (drivers/stock)",
                              min_value=10, max_value=200,
                              value=80, step=5)
is_event = st.sidebar.toggle("Event active (rain/concert/holiday)", value=False)
prev_price = st.sidebar.slider("Previous hour price (₹)",
                                min_value=8.0, max_value=25.0,
                                value=10.0, step=0.5)

st.sidebar.divider()
st.sidebar.header("Engine Parameters")
alpha       = st.sidebar.slider("Multiplier sensitivity (α)",
                                 0.1, 1.0, 0.5, 0.05)
smooth_alph = st.sidebar.slider("Smoothing factor",
                                 0.1, 1.0, 0.25, 0.05)

# ── Compute ──
DSR      = demand / (supply + 1)
# Simulate uncertainty based on event status
band_w   = demand * (0.3 if is_event else 0.12)
pred_low  = demand * (0.75 if is_event else 0.90)
pred_high = demand * (1.25 if is_event else 1.10)

result = get_dynamic_price(
    DSR=DSR,
    pred_low=pred_low,
    pred_high=pred_high,
    pred_mid=demand,
    prev_price=prev_price,
    smooth_alpha=smooth_alph
)

dynamic_price  = result['smoothed_price']
multiplier     = result['multiplier']
dsr_zone       = result['dsr_zone']
adj_demand     = adjusted_demand(demand, BASE_PRICE, dynamic_price)
static_rev     = demand * BASE_PRICE
dynamic_rev    = adj_demand * dynamic_price
rev_diff       = dynamic_rev - static_rev
rev_uplift_pct = (dynamic_rev - static_rev) / static_rev * 100

# ── Main metrics row ──
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("DSR", f"{DSR:.2f}", dsr_zone)
col2.metric("Multiplier", f"{multiplier:.3f}×")
col3.metric("Dynamic Price", f"₹{dynamic_price:.2f}",
            f"{(dynamic_price/BASE_PRICE - 1)*100:+.1f}% vs base")
col4.metric("Revenue (dynamic)", f"₹{dynamic_rev:,.0f}",
            f"{rev_uplift_pct:+.1f}% vs static")
col5.metric("Demand retained",
            f"{adj_demand:.0f}",
            f"{(adj_demand/demand - 1)*100:+.1f}%")

st.divider()

# ── Two-column layout ──
left, right = st.columns(2)

with left:
    st.subheader("Price vs demand trade-off")
    prices  = np.linspace(5, 30, 200)
    demands = [adjusted_demand(demand, BASE_PRICE, p) for p in prices]
    revs    = [d * p for d, p in zip(demands, prices)]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax2 = ax.twinx()
    ax.plot(prices, demands, color='steelblue', lw=2, label='Demand')
    ax2.plot(prices, revs,   color='tomato',    lw=2,
             label='Revenue', ls='--')
    ax.axvline(dynamic_price, color='green', ls=':', lw=1.5,
               label=f'Dynamic ₹{dynamic_price:.1f}')
    ax.axvline(BASE_PRICE,    color='gray',  ls=':', lw=1,
               label=f'Base ₹{BASE_PRICE}')
    ax.set_xlabel('Price (₹)'); ax.set_ylabel('Demand', color='steelblue')
    ax2.set_ylabel('Revenue (₹)', color='tomato')
    ax.set_title('Demand & revenue curve')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with right:
    st.subheader("Static vs dynamic comparison")
    categories = ['Revenue', 'Demand']
    static_vals  = [static_rev,  demand]
    dynamic_vals = [dynamic_rev, adj_demand]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.5))
    for i, (ax, cat, sv, dv) in enumerate(
            zip(axes, categories, static_vals, dynamic_vals)):
        bars = ax.bar(['Static', 'Dynamic'], [sv, dv],
                      color=['steelblue', 'tomato'], alpha=0.85)
        ax.set_title(cat, fontsize=10)
        for bar, val in zip(bars, [sv, dv]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.01,
                    f'{val:,.0f}', ha='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Decision explanation ──
st.divider()
st.subheader("Pricing decision breakdown")
exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    st.info(f"""
**Market signal**
- Demand: {demand} units
- Supply: {supply} units
- DSR: {DSR:.2f}
- Zone: {dsr_zone}
- Event: {'Yes ⚡' if is_event else 'No'}
""")

with exp_col2:
    st.info(f"""
**Engine output**
- Base multiplier: {1 + alpha*(min(DSR,5.0)-1):.3f}×
- After uncertainty discount: {multiplier:.3f}×
- Raw price: ₹{BASE_PRICE * multiplier:.2f}
- Smoothed price: ₹{dynamic_price:.2f}
""")

with exp_col3:
    color = "success" if rev_diff > 0 else "warning"
    getattr(st, color)(f"""
**Revenue impact**
- Static: ₹{static_rev:,.0f}
- Dynamic: ₹{dynamic_rev:,.0f}
- Difference: ₹{rev_diff:+,.0f}
- Uplift: {rev_uplift_pct:+.1f}%
- Demand retained: {adj_demand/demand*100:.1f}%
""")