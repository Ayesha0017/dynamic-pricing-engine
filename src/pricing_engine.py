"""
Core dynamic pricing logic for the ride-hailing pricing system.
Imports cleanly into notebooks, scripts, and the Streamlit dashboard.
"""

import numpy as np
# ── Default constants ──
BASE_PRICE   = 10.0
ELASTICITY   = -0.7
ALPHA        = 0.5
FLOOR_MULT   = 0.8
CEILING_MULT = 2.5
SMOOTH_ALPHA = 0.25
DSR_CLIP     = 5.0

def adjusted_demand(base_demand: float,
                    base_price: float,
                    new_price: float,
                    elasticity: float = ELASTICITY) -> float:
    """
    Compute demand at a new price using constant price elasticity.

    Formula: demand_new = demand_base × (price_new / price_base) ^ ε
    Elasticity (ε) is negative — higher price reduces demand.

    Args:
        base_demand : observed demand at base price
        base_price  : reference price point
        new_price   : proposed new price
        elasticity  : price elasticity coefficient (default -0.7)

    Returns:
        float: adjusted demand (non-negative)
    """
    price_ratio = new_price / base_price
    return max(0.0, base_demand * (price_ratio ** elasticity))


def pricing_multiplier(DSR: float,
                       pred_low: float,
                       pred_high: float,
                       pred_mid: float,
                       alpha: float = ALPHA,
                       floor: float = FLOOR_MULT,
                       ceiling: float = CEILING_MULT,
                       dsr_clip: float = DSR_CLIP) -> float:
    """
    Compute price multiplier from demand-supply ratio and forecast uncertainty.

    DSR > 1 : demand exceeds supply → price rises
    DSR < 1 : supply exceeds demand → price held at floor
    Wide confidence band → more conservative multiplier (uncertainty discount)

    Args:
        DSR       : demand / (supply + 1) ratio
        pred_low  : 5th percentile forecast
        pred_high : 95th percentile forecast
        pred_mid  : point forecast (median)
        alpha     : multiplier sensitivity (default 0.5)
        floor     : minimum multiplier (default 0.8)
        ceiling   : maximum multiplier (default 2.5)
        dsr_clip  : cap on DSR before multiplier (default 5.0)

    Returns:
        float: price multiplier in [floor, ceiling]
    """
    dsr_clipped        = min(DSR, dsr_clip)
    base_mult          = 1 + alpha * (dsr_clipped - 1)
    band_width         = pred_high - pred_low
    uncertainty_ratio  = band_width / (pred_mid + 1)
    uncertainty_disc   = 1 - min(0.20, uncertainty_ratio * 0.15)
    final_mult         = base_mult * uncertainty_disc

    # Neutral market guard
    if abs(DSR - 1.0) < 0.05:
        final_mult = max(final_mult, 1.0)

    return round(min(ceiling, max(floor, final_mult)), 4)

def get_dynamic_price(DSR: float,
                      pred_low: float,
                      pred_high: float,
                      pred_mid: float,
                      prev_price: float = BASE_PRICE,
                      base_price: float = BASE_PRICE,
                      smooth_alpha: float = SMOOTH_ALPHA) -> dict:
    """
    Compute the smoothed dynamic price for a single hour.

    Applies EMA smoothing to prevent sudden price jumps.
    Returns a dict with all intermediate values for transparency.

    Args:
        DSR        : demand-supply ratio
        pred_low   : lower bound forecast
        pred_high  : upper bound forecast
        pred_mid   : point forecast
        prev_price : previous hour's smoothed price
        base_price : base price constant
        smooth_alpha: EMA weight for new price (0=no change, 1=instant)

    Returns:
        dict with keys: multiplier, raw_price, smoothed_price, dsr_zone
    """
    mult        = pricing_multiplier(DSR, pred_low, pred_high, pred_mid)
    raw_price   = base_price * mult
    smooth_price = smooth_alpha * raw_price + (1 - smooth_alpha) * prev_price

    # DSR zone label
    if DSR < 0.8:     zone = 'Low'
    elif DSR < 1.2:   zone = 'Balanced'
    elif DSR < 2.0:   zone = 'Mild surge'
    else:             zone = 'Strong surge'

    return {
        'multiplier'    : mult,
        'raw_price'     : round(raw_price,    2),
        'smoothed_price': round(smooth_price, 2),
        'dsr_zone'      : zone,
    }