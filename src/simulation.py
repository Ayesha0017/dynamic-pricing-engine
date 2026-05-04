"""
simulation.py
-------------
Phase 4 impact simulation — loads Phase 3 pricing output and runs
a full static vs dynamic revenue comparison broken down by week,
DSR zone, and event type. Prints the business impact summary.

Usage:
    python src/simulation.py

    Or import:
        from src.simulation import run_simulation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ── Constants ────────────────────────────────────────────────────────────────
BASE_PRICE  = 10.0
OUTPUT_DIR  = 'output'


def load_pricing_output(path: str = 'output/phase3_pricing_output.csv'
                         ) -> pd.DataFrame:
    """
    Load Phase 3 pricing output CSV and add helper columns.

    Adds: revenue_diff, hour, date, week, dsr_zone

    Args:
        path: path to phase3_pricing_output.csv

    Returns:
        enriched DataFrame ready for simulation analysis
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])

    print(f"Loaded {len(df):,} rows")
    print(f"Period : {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")

    df['revenue_diff'] = df['dynamic_revenue'] - df['static_revenue']
    df['hour']         = df['timestamp'].dt.hour
    df['date']         = df['timestamp'].dt.date
    df['week']         = df['timestamp'].dt.isocalendar().week.astype(int)

    def dsr_zone(dsr):
        if dsr < 0.8:   return '1_Low (DSR<0.8)'
        elif dsr < 1.2: return '2_Balanced (0.8-1.2)'
        elif dsr < 2.0: return '3_Mild surge (1.2-2.0)'
        else:           return '4_Strong surge (>2.0)'

    df['dsr_zone'] = df['DSR'].apply(dsr_zone)

    static_rev  = df['static_revenue'].sum()
    dynamic_rev = df['dynamic_revenue'].sum()
    uplift_pct  = (dynamic_rev - static_rev) / static_rev * 100

    print(f"\nKey metrics:")
    print(f"  Static revenue  : ₹{static_rev:,.0f}")
    print(f"  Dynamic revenue : ₹{dynamic_rev:,.0f}")
    print(f"  Revenue uplift  : {uplift_pct:+.1f}%")
    print(f"\nDSR zone distribution:")
    print(df['dsr_zone'].value_counts().sort_index())

    return df


def plot_weekly_comparison(df: pd.DataFrame,
                            output_dir: str = OUTPUT_DIR):
    """
    Plot weekly revenue: static vs dynamic (top panel) +
    weekly uplift % (bottom panel).

    Saves: output/weekly_revenue_comparison.png

    Args:
        df         : enriched simulation DataFrame
        output_dir : folder to save PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    static_rev  = df['static_revenue'].sum()
    dynamic_rev = df['dynamic_revenue'].sum()
    uplift_pct  = (dynamic_rev - static_rev) / static_rev * 100

    weekly = df.groupby('week').agg(
        static_rev  = ('static_revenue',  'sum'),
        dynamic_rev = ('dynamic_revenue', 'sum'),
    ).reset_index()
    weekly['uplift_pct'] = ((weekly['dynamic_rev'] - weekly['static_rev'])
                             / weekly['static_rev'] * 100)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={'height_ratios': [2.5, 1]})
    x, width = np.arange(len(weekly)), 0.38
    ax = axes[0]
    ax.bar(x - width/2, weekly['static_rev'],  width,
           label='Static pricing',  color='steelblue', alpha=0.85)
    ax.bar(x + width/2, weekly['dynamic_rev'], width,
           label='Dynamic pricing', color='tomato',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Wk {w}" for w in weekly['week']], fontsize=8)
    ax.set_ylabel('Revenue (₹)', fontsize=10)
    ax.set_title('Weekly revenue — static vs dynamic pricing', fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f'₹{v:,.0f}'))

    ax2 = axes[1]
    colors = ['tomato' if v > 0 else 'steelblue' for v in weekly['uplift_pct']]
    ax2.bar(x, weekly['uplift_pct'], color=colors, alpha=0.85)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axhline(uplift_pct, color='tomato', ls='--', lw=1,
                label=f'Avg uplift {uplift_pct:.1f}%')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Wk {w}" for w in weekly['week']], fontsize=8)
    ax2.set_ylabel('Revenue uplift %', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f'{v:+.0f}%'))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/weekly_revenue_comparison.png', dpi=150)
    plt.show()

    print("\nTop 3 weeks by revenue uplift:")
    top3 = weekly.nlargest(3, 'uplift_pct')[
        ['week', 'static_rev', 'dynamic_rev', 'uplift_pct']]
    print(top3.to_string(index=False))


def plot_dsr_zone_comparison(df: pd.DataFrame,
                              output_dir: str = OUTPUT_DIR):
    """
    Plot revenue by DSR zone (left) and uplift % per zone (right).

    Shows that uplift scales with DSR intensity — confirming the
    pricing engine performs as designed.

    Saves: output/dsr_zone_comparison.png

    Args:
        df         : enriched simulation DataFrame
        output_dir : folder to save PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    zone_summary = df.groupby('dsr_zone').agg(
        hours             = ('timestamp',       'count'),
        static_rev        = ('static_revenue',  'sum'),
        dynamic_rev       = ('dynamic_revenue', 'sum'),
        avg_dsr           = ('DSR',             'mean'),
        avg_dynamic_price = ('smoothed_price',  'mean'),
        avg_conversion    = ('dynamic_demand',
                             lambda x: (x / df.loc[x.index,
                             'actual_demand']).mean() * 100),
    ).reset_index()

    zone_summary['uplift_pct'] = ((zone_summary['dynamic_rev']
                                   - zone_summary['static_rev'])
                                  / zone_summary['static_rev'] * 100)
    zone_summary['zone_label'] = (zone_summary['dsr_zone']
                                   .str.split('_').str[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x, width  = np.arange(len(zone_summary)), 0.38

    ax = axes[0]
    ax.bar(x - width/2, zone_summary['static_rev'],  width,
           label='Static',  color='steelblue', alpha=0.85)
    ax.bar(x + width/2, zone_summary['dynamic_rev'], width,
           label='Dynamic', color='tomato',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_summary['zone_label'], fontsize=8, rotation=15)
    ax.set_title('Revenue by DSR zone', fontsize=11)
    ax.set_ylabel('Total revenue (₹)')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f'₹{v:,.0f}'))

    ax2 = axes[1]
    colors = ['tomato' if v > 0 else 'steelblue'
              for v in zone_summary['uplift_pct']]
    bars = ax2.bar(x, zone_summary['uplift_pct'], color=colors, alpha=0.85)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(zone_summary['zone_label'], fontsize=8, rotation=15)
    ax2.set_title('Revenue uplift % by DSR zone', fontsize=11)
    ax2.set_ylabel('Uplift %')
    ax2.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f'{v:+.0f}%'))
    for bar, val in zip(bars, zone_summary['uplift_pct']):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'{val:+.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dsr_zone_comparison.png', dpi=150)
    plt.show()

    print("\nDSR zone breakdown:")
    print(zone_summary[['zone_label', 'hours', 'avg_dsr',
                         'avg_dynamic_price', 'avg_conversion',
                         'uplift_pct']].to_string(index=False))


def plot_event_comparison(df: pd.DataFrame,
                           output_dir: str = OUTPUT_DIR):
    """
    Plot event vs normal hours — total revenue, revenue per hour,
    and uplift % — as a 3-panel chart.

    Saves: output/event_vs_normal_comparison.png

    Args:
        df         : enriched simulation DataFrame
        output_dir : folder to save PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    event_summary = df.groupby('is_event').agg(
        hours           = ('timestamp',       'count'),
        static_rev      = ('static_revenue',  'sum'),
        dynamic_rev     = ('dynamic_revenue', 'sum'),
        avg_dsr         = ('DSR',             'mean'),
        avg_price       = ('smoothed_price',  'mean'),
        avg_actual_dem  = ('actual_demand',   'mean'),
        avg_dynamic_dem = ('dynamic_demand',  'mean'),
    ).reset_index()

    event_summary['uplift_pct'] = ((event_summary['dynamic_rev']
                                    - event_summary['static_rev'])
                                   / event_summary['static_rev'] * 100)
    event_summary['rev_per_hr_static']  = (event_summary['static_rev']
                                            / event_summary['hours'])
    event_summary['rev_per_hr_dynamic'] = (event_summary['dynamic_rev']
                                            / event_summary['hours'])
    event_summary['label'] = event_summary['is_event'].map(
        {0: 'Normal hours', 1: 'Event hours'})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x, width  = np.arange(2), 0.38

    # Panel 1: total revenue
    ax = axes[0]
    ax.bar(x - width/2, event_summary['static_rev'],
           width, label='Static',  color='steelblue', alpha=0.85)
    ax.bar(x + width/2, event_summary['dynamic_rev'],
           width, label='Dynamic', color='tomato',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(event_summary['label'], fontsize=9)
    ax.set_title('Total revenue', fontsize=10)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f'₹{v/1e6:.2f}M'))

    # Panel 2: revenue per hour
    ax2 = axes[1]
    ax2.bar(x - width/2, event_summary['rev_per_hr_static'],
            width, label='Static',  color='steelblue', alpha=0.85)
    ax2.bar(x + width/2, event_summary['rev_per_hr_dynamic'],
            width, label='Dynamic', color='tomato',    alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(event_summary['label'], fontsize=9)
    ax2.set_title('Revenue per hour', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f'₹{v:,.0f}'))

    # Panel 3: uplift %
    ax3 = axes[2]
    colors = ['tomato' if v > 0 else 'steelblue'
              for v in event_summary['uplift_pct']]
    bars = ax3.bar(x, event_summary['uplift_pct'],
                   color=colors, alpha=0.85, width=0.5)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(event_summary['label'], fontsize=9)
    ax3.set_title('Revenue uplift %', fontsize=10)
    for bar, val in zip(bars, event_summary['uplift_pct']):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'{val:+.1f}%', ha='center', fontsize=10,
                 fontweight='bold')

    plt.suptitle('Event vs normal hours — static vs dynamic pricing',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/event_vs_normal_comparison.png', dpi=150)
    plt.show()

    print("\nEvent vs normal breakdown:")
    cols = ['label', 'hours', 'avg_dsr', 'avg_price',
            'rev_per_hr_static', 'rev_per_hr_dynamic', 'uplift_pct']
    print(event_summary[cols].to_string(index=False))


def print_business_summary(df: pd.DataFrame):
    """
    Print the full business impact summary table —
    the numbers that go in the README and LinkedIn post.

    Args:
        df: enriched simulation DataFrame
    """
    static_rev  = df['static_revenue'].sum()
    dynamic_rev = df['dynamic_revenue'].sum()
    uplift_pct  = (dynamic_rev - static_rev) / static_rev * 100

    test_days    = (df['timestamp'].max() - df['timestamp'].min()).days + 1
    annual_scale = 365 / test_days
    ann_static   = static_rev  * annual_scale
    ann_dynamic  = dynamic_rev * annual_scale
    ann_uplift   = ann_dynamic - ann_static

    avg_conv    = (df['dynamic_demand'] / df['actual_demand']).mean() * 100
    floor_hours = (df['multiplier'] == 0.8).sum()
    ceil_hours  = (df['multiplier'] == 2.5).sum()

    print(f"\n{'='*55}")
    print(f"  PHASE 4 — BUSINESS IMPACT SUMMARY")
    print(f"{'='*55}")
    print(f"\n  TEST PERIOD ({test_days} days)")
    print(f"  {'Static revenue':<30}: ₹{static_rev:>12,.0f}")
    print(f"  {'Dynamic revenue':<30}: ₹{dynamic_rev:>12,.0f}")
    print(f"  {'Revenue uplift':<30}: {uplift_pct:>+11.1f}%")
    print(f"  {'Additional revenue':<30}: ₹{dynamic_rev-static_rev:>12,.0f}")
    print(f"\n  ANNUALISED PROJECTION")
    print(f"  {'Annual static revenue':<30}: ₹{ann_static:>12,.0f}")
    print(f"  {'Annual dynamic revenue':<30}: ₹{ann_dynamic:>12,.0f}")
    print(f"  {'Annual revenue uplift':<30}: ₹{ann_uplift:>12,.0f}")
    print(f"\n  PRICING STATS")
    print(f"  {'Avg dynamic price':<30}: ₹{df['smoothed_price'].mean():>11.2f}")
    print(f"  {'Price range':<30}: "
          f"₹{df['smoothed_price'].min():.2f} – "
          f"₹{df['smoothed_price'].max():.2f}")
    print(f"  {'Hours at floor':<30}: "
          f"{floor_hours:>8} ({floor_hours/len(df)*100:.1f}%)")
    print(f"  {'Hours at ceiling':<30}: "
          f"{ceil_hours:>8} ({ceil_hours/len(df)*100:.1f}%)")
    print(f"\n  CONVERSION")
    print(f"  {'Avg conversion rate':<30}: {avg_conv:>11.1f}%")
    print(f"  {'Demand retained':<30}: "
          f"{avg_conv:.1f}% of baseline")
    print(f"  {'Demand trade-off':<30}: "
          f"{100-avg_conv:.1f}% lost to price sensitivity")
    print(f"{'='*55}")


def run_simulation(pricing_output_path: str = 'output/phase3_pricing_output.csv',
                   output_dir: str = OUTPUT_DIR):
    """
    Full impact simulation pipeline — single entry point.

    Steps:
        1. Load Phase 3 pricing output
        2. Weekly revenue comparison chart
        3. DSR zone breakdown chart
        4. Event vs normal comparison chart
        5. Business impact summary
        6. Save enriched DataFrame

    Args:
        pricing_output_path : path to phase3_pricing_output.csv
        output_dir          : folder for output charts and CSV
    """
    print("=" * 55)
    print("  PHASE 4 — IMPACT SIMULATION")
    print("=" * 55)

    df = load_pricing_output(pricing_output_path)

    print("\n── Weekly revenue comparison ──")
    plot_weekly_comparison(df, output_dir)

    print("\n── DSR zone breakdown ──")
    plot_dsr_zone_comparison(df, output_dir)

    print("\n── Event vs normal hours ──")
    plot_event_comparison(df, output_dir)

    print_business_summary(df)

    # Save enriched output
    out_path = f'{output_dir}/phase4_full_simulation.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"       {output_dir}/weekly_revenue_comparison.png")
    print(f"       {output_dir}/dsr_zone_comparison.png")
    print(f"       {output_dir}/event_vs_normal_comparison.png")


if __name__ == '__main__':
    run_simulation()