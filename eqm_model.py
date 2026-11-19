"""
Bitcoin Empirical Quantile Model (EQM) Rainbow
Reconstruction of @BTCAnalytica's walk-forward EQM (bullmania indicator).

Algorithm (validated Nov 2026 against ~46,000 band values read from his live
indicator, 2015-2026: median error 0.04%, 96% of values within 0.5%):
  Price   : Bitstamp BTC/USD daily closes, history from EQM_START (2011-09-25).
  For each day t (frozen once computed, never revised):
    1. mean_t, std_t = mean / std of ALL closes from EQM_START to t (expanding, raw price)
    2. z_t = (P_t - mean_t) / std_t
    3. pool = the last EQM_POOL (1230) values of z, including today
    4. Band[t][q] = mean_t + std_t * quantile_q(pool)        (Hazen quantile)
  Risk%      = percentile of today's z within the pool
  Risk asym  = the same percentile evaluated at the daily low (high when above 50%)
  Centering  = fraction of days (since bands start) with price below EQM_50

The full-history OLS power law (a_full, b_full) is kept only as a dotted
reference line on the charts; it plays no part in the EQM bands.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
GENESIS   = date(2009, 1, 3)
QUANTILES = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
QNAMES    = ['EQM_1', 'EQM_10', 'EQM_20', 'EQM_30', 'EQM_40', 'EQM_50',
             'EQM_60', 'EQM_70', 'EQM_80', 'EQM_90', 'EQM_99']

BAND_LABELS = [
    'Fire sale',   #  1–10 %
    'Accumulate',  # 10–20 %
    'Cheap',       # 20–30 %
    'Value',       # 30–40 %
    'Below mid',   # 40–50 %
    'Above mid',   # 50–60 %
    'Warm',        # 60–70 %
    'Hot',         # 70–80 %
    'FOMO',        # 80–90 %
    'Bubble',      # 90–99 %
]

import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
DATA_PATH = _os.path.join(_SCRIPT_DIR, "btc_prices.csv")
OUT_DIR   = _SCRIPT_DIR

# EQM parameters (reconstructed from his live indicator, Nov 2026)
EQM_START    = '2011-09-25'   # start of the price history the indicator uses (Bitstamp)
EQM_POOL     = 1230           # number of past standardized prices in the quantile pool
BITSTAMP_START = '2011-08-18' # first Bitstamp daily candle (rows before this are other sources)

# Legacy names kept for chart labels / backwards compatibility
OLS_START    = EQM_START
RESID_WINDOW = EQM_POOL
MIN_OBS      = EQM_POOL

# Full-history OLS start (matches BTCAnalytica's "OLS PL" reference line)
OLS_FULL_START = None         # None = use all available data

# ── Colours ───────────────────────────────────────────────────────────────────
BAND_COLORS = [
    '#1a237e',  # EQM_1  → deep navy indigo  (fire sale)
    '#1565c0',  # EQM_10 → royal blue        (accumulate)
    '#039be5',  # EQM_20 → sky blue          (cheap)
    '#00acc1',  # EQM_30 → cyan-teal         (value)
    '#43a047',  # EQM_40 → medium green      (below mid)
    '#c0ca33',  # EQM_50 → lime yellow-green (above mid)
    '#ffb300',  # EQM_60 → amber             (warm)
    '#ff6d00',  # EQM_70 → deep orange       (hot)
    '#bf360c',  # EQM_80 → burnt red-orange  (FOMO)
    '#dd0000',  # EQM_90 → vivid red         (bubble)
    '#770000',  # EQM_99 → dark maroon       (bubble top)
]

# Rails — extreme reference lines (0.1% and 99.9%); do NOT affect Risk%
RAIL_QUANTILES = [0.001, 0.999]
RAIL_NAMES     = ['EQM_0_1', 'EQM_99_9']
RAIL_COLORS    = ['#888899', '#bb1100']   # gray / dark-red


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    df['t'] = df['date'].apply(lambda d: (d.date() - GENESIS).days)
    df['x'] = np.log(df['t'].astype(float))
    df['y'] = np.log(df['close'].astype(float))
    return df


# ── Fetch latest prices from Bitstamp ─────────────────────────────────────────
def fetch_latest_prices(df):
    """Fetch completed Bitstamp daily candles newer than the CSV (or missing) and save.

    The EQM is defined on Bitstamp BTC/USD closes, so every row from
    BITSTAMP_START onwards must be a Bitstamp close.
    """
    import urllib.request, json as _json, time as _time
    from datetime import datetime as _dt, timezone as _tz

    last_date = df['date'].max()
    start = int(pd.Timestamp(last_date - pd.Timedelta(days=3), tz='UTC').timestamp())
    now = int(_dt.now(_tz.utc).timestamp())
    rows = []
    try:
        while True:
            url = (f"https://www.bitstamp.net/api/v2/ohlc/btcusd/"
                   f"?step=86400&limit=1000&start={start}")
            req = urllib.request.Request(url, headers={'User-Agent': 'btc-eqm/1.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = _json.loads(resp.read())['data']['ohlc']
            if not data:
                break
            for k in data:
                ts = int(k['timestamp'])
                if ts + 86400 > now:            # candle not yet closed
                    continue
                rows.append({'date': pd.Timestamp(ts, unit='s').normalize(),
                             'close': float(k['close'])})
            last_ts = int(data[-1]['timestamp'])
            if last_ts + 86400 > now or len(data) < 2:
                break
            start = last_ts + 86400
            _time.sleep(0.3)
    except Exception as e:
        print(f"  Warning: Bitstamp fetch failed — {e}")
        return df

    if not rows:
        print("  No new completed candles available")
        return df

    new = pd.DataFrame(rows).drop_duplicates('date')
    base = df[['date', 'close']].set_index('date')['close']
    upd = new.set_index('date')['close']
    merged = upd.combine_first(base).sort_index()
    added = int((~upd.index.isin(base.index)).sum())

    out = merged.rename('close').reset_index().rename(columns={'index': 'date'})
    out['t'] = out['date'].apply(lambda d: (d.date() - GENESIS).days)
    out['x'] = np.log(out['t'].astype(float))
    out['y'] = np.log(out['close'].astype(float))

    out[['date', 'close']].assign(date=out['date'].dt.strftime('%Y-%m-%d')).to_csv(DATA_PATH, index=False)
    print(f"  Bitstamp: {added} new day(s) → data now through {out['date'].max().date()}")
    return out


# ── Risk% via log-linear interpolation between band boundaries ────────────────
def compute_risk_pct(price, bands_dict):
    """
    Returns price's percentile [0, 100] interpolated log-linearly
    between adjacent quantile band boundaries.
    """
    q_vals = [(q, float(bands_dict[name])) for q, name in zip(QUANTILES, QNAMES)]
    price_log = np.log(float(price))
    for i in range(len(q_vals) - 1):
        q_lo, v_lo = q_vals[i]
        q_hi, v_hi = q_vals[i + 1]
        if v_lo <= price <= v_hi:
            frac = (price_log - np.log(v_lo)) / (np.log(v_hi) - np.log(v_lo))
            return (q_lo + frac * (q_hi - q_lo)) * 100
    return 0.0 if price < q_vals[0][1] else 100.0


def band_label(risk_pct):
    edges = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 101]
    for i, e in enumerate(edges[1:]):
        if risk_pct <= e:
            return BAND_LABELS[i] if i < len(BAND_LABELS) else 'Extreme'
    return 'Extreme'


# ── Full-history OLS (BTCAnalytica's "OLS PL" reference line) ────────────────
def compute_ols_full(df):
    """
    Fits OLS on the entire price history (from first available date).
    This matches BTCAnalytica's left-panel 'OLS PL' model:
      b ≈ 5.67, giving fair value ≈ 1.002 × ATH at March 2026.
    Returns (a, b) coefficients.
    """
    xs = df['x'].values
    ys = df['y'].values
    c  = np.polyfit(xs, ys, 1)
    return float(c[1]), float(c[0])   # (a, b)


# ── EQM: expanding standardization + rolling quantile pool ───────────────────
def compute_model_c(df, eqm_start=EQM_START, pool_n=EQM_POOL, **_ignored):
    """
    Walk-forward EQM (reconstruction of @BTCAnalytica's indicator).

    Returns
    -------
    hist_bands : DataFrame (index=dates, columns=QNAMES+RAIL_NAMES), each row frozen on its date
    current    : dict of today's values (bands, risk, centering, ATH metrics, OLS reference)
    """
    sub = df[df['date'] >= eqm_start].reset_index(drop=True)
    p   = sub['close'].astype(float)
    mu  = p.expanding(2).mean().values
    sd  = p.expanding(2).std().values
    z   = (p.values - mu) / sd
    n   = len(sub)

    all_qs    = QUANTILES + RAIL_QUANTILES
    all_names = QNAMES + RAIL_NAMES
    hist = {name: np.full(n, np.nan) for name in all_names}
    risk_hist = np.full(n, np.nan)

    print(f"  Walk-forward EQM: {n} days from {eqm_start} (pool = {pool_n} standardized prices)...")
    for i in range(pool_n - 1, n):
        pool = z[i - pool_n + 1: i + 1]
        if not np.all(np.isfinite(pool)):
            continue
        qv = np.quantile(pool, all_qs, method='hazen')
        for name, v in zip(all_names, qv):
            hist[name][i] = mu[i] + sd[i] * v
        risk_hist[i] = 100.0 * np.searchsorted(np.sort(pool), z[i], side='right') / pool_n

    hist_bands = pd.DataFrame(hist, index=sub['date'])
    hist_bands['RISK'] = risk_hist

    # ── Current state ────────────────────────────────────────────────────────
    pool_now = np.sort(z[-pool_n:])
    mu_now, sd_now = float(mu[-1]), float(sd[-1])
    current_bands = {name: float(hist[name][-1]) for name in all_names}
    current_price = float(p.values[-1])
    risk_pct = float(risk_hist[-1])

    # live-page helpers: pool mapped to today's price level (log), so the page's
    # "share of RESIDUALS below log(price)" equals the EQM percentile exactly
    pool_prices = mu_now + sd_now * pool_now
    resid_live = np.log(np.clip(pool_prices, 1e-9, None))

    # centering / walk-forward fit of the 50% line (log price vs same-day EQM_50)
    ok  = ~np.isnan(hist['EQM_50'])
    lp  = np.log(p.values[ok]); l50 = np.log(hist['EQM_50'][ok])
    centering = float(np.mean(lp < l50) * 100)
    r2 = float(1 - np.sum((lp - l50) ** 2) / np.sum((lp - lp.mean()) ** 2))

    ath_price = float(df['close'].max())
    ath_date  = df.loc[df['close'].idxmax(), 'date'].date()
    eqm_ath   = current_bands['EQM_50'] / ath_price
    drawdown  = (current_price / ath_price - 1) * 100

    a_full, b_full = compute_ols_full(df)
    t_now = (pd.Timestamp(df['date'].values[-1]).date() - GENESIS).days
    x_now = np.log(t_now)
    ols_full_now = np.exp(a_full + b_full * x_now)

    current = dict(
        # a = b = 0 so legacy code computing exp(a + b*x + q_offset) returns the band
        a=0.0, b=0.0,
        q_offsets={q: float(np.log(current_bands[nm])) for q, nm in zip(all_qs, all_names)},
        r2=r2, current_bands=current_bands, n_resid=pool_n, resid_window=pool_n,
        risk_pct=risk_pct, centering=centering,
        mu=mu_now, sd=sd_now, resid_live=resid_live,
        ath_price=ath_price, ath_date=ath_date, eqm_ath=eqm_ath, drawdown_pct=drawdown,
        a_full=a_full, b_full=b_full, ols_full_now=ols_full_now,
        ols_full_ath_ratio=ols_full_now / ath_price,
    )
    return hist_bands, current


def _xs_for_dates(dates_series):
    ts = dates_series.apply(lambda d: (pd.Timestamp(d).date() - GENESIS).days)
    return np.log(ts.astype(float).values)


# ── Static PNG chart ──────────────────────────────────────────────────────────
def generate_png(df, hist_bands, current_c, out_path):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    hist_dates = hist_bands.index
    cb = current_c['current_bands']

    # Band fills
    for lo_name, hi_name, color in zip(QNAMES[:-1], QNAMES[1:], BAND_COLORS[:-1]):
        ax.fill_between(hist_dates,
                        hist_bands[lo_name].values,
                        hist_bands[hi_name].values,
                        color=color, alpha=0.22)

    # Band lines
    for name, color in zip(QNAMES, BAND_COLORS):
        lw = 2.5 if name == 'EQM_50' else 1.4
        ax.plot(hist_dates, hist_bands[name].values, color=color, linewidth=lw)

    # Rails (0.1% and 99.9%) — thin reference lines outside the main rainbow
    for name, color in zip(RAIL_NAMES, RAIL_COLORS):
        ax.plot(hist_dates, hist_bands[name].values,
                color=color, linewidth=1.0, linestyle='--', alpha=0.85)

    # Full-history OLS reference line (BTCAnalytica's "OLS PL")
    a_f, b_f = current_c['a_full'], current_c['b_full']
    ols_full_vals = np.exp(a_f + b_f * df['x'].values)
    ax.plot(df['date'], ols_full_vals,
            color='#cc88ff', linewidth=1.0, linestyle='--', alpha=0.7,
            label=f'OLS PL (full hist, b={b_f:.2f})')

    # BTC price
    ax.plot(df['date'], df['close'], color='white', linewidth=1.2, zorder=10)
    ax.scatter([df['date'].values[-1]], [df['close'].values[-1]],
               color='white', s=40, zorder=11)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.set_xlim(hist_bands.first_valid_index(), df['date'].max())
    ax.set_ylim(200, 2_000_000)
    ax.grid(True, which='major', color='#333', linewidth=0.5)
    ax.grid(True, which='minor', color='#222', linewidth=0.3)

    ax.set_title('Bitcoin EQM Rainbow — Walk-Forward Empirical Quantile Model',
                 color='white', fontsize=15, pad=10)
    ax.set_xlabel('Date', color='#aaa')
    ax.set_ylabel('Price (USD, log scale)', color='#aaa')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    # Stats box
    info = (
        f"EQM: expanding std from {EQM_START}  |  Pool: {current_c['resid_window']}d  |  "
        f"R²={current_c['r2']:.4f}  |  Risk={current_c['risk_pct']:.1f}%  |  "
        f"Centering={current_c['centering']:.1f}%  |  "
        f"EQM₅₀/ATH={current_c['eqm_ath']:.3f}  |  "
        f"Drawdown={current_c['drawdown_pct']:.1f}%  |  "
        f"OLS_PL/ATH={current_c['ols_full_ath_ratio']:.3f}"
    )
    ax.text(0.01, 0.99, info, transform=ax.transAxes,
            color='white', fontsize=8, va='top',
            bbox=dict(facecolor='#111', alpha=0.75, edgecolor='#555'))

    # Legend — bands + rails
    legend_patches = []
    # Top rail
    rail99_p = cb[RAIL_NAMES[1]]
    legend_patches.append(mpatches.Patch(color=RAIL_COLORS[1], alpha=0.9,
        label=f'99.9% rail  ${rail99_p/1000:.0f}K'))
    # Main bands (reversed so highest is on top)
    for i, (label, color) in enumerate(zip(BAND_LABELS, BAND_COLORS[:-1])):
        lo_p = cb[QNAMES[i]]
        hi_p = cb[QNAMES[i + 1]]
        lbl  = (f'{label}  ${lo_p/1000:.0f}K–${hi_p/1000:.0f}K  '
                f'({int(QUANTILES[i]*100)}–{int(QUANTILES[i+1]*100)}%)')
        legend_patches.append(mpatches.Patch(color=color, alpha=0.6, label=lbl))
    # Bottom rail
    rail01_p = cb[RAIL_NAMES[0]]
    legend_patches.append(mpatches.Patch(color=RAIL_COLORS[0], alpha=0.9,
        label=f'0.1% rail  ${rail01_p/1000:.0f}K'))

    leg1 = ax.legend(handles=legend_patches[::-1], loc='lower right',
                     fontsize=7.5, framealpha=0.5, facecolor='#111',
                     edgecolor='#555', labelcolor='white',
                     title='EQM Bands (today)', title_fontsize=8,
                     handlelength=1.5, handleheight=1.2)
    ax.add_artist(leg1)
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.5,
              facecolor='#111', edgecolor='#555', labelcolor='white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close()
    print(f"PNG saved: {out_path}")


# ── Interactive Plotly HTML ───────────────────────────────────────────────────
def generate_html(df, hist_bands, current_c, out_path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed, skipping HTML"); return

    cb = current_c['current_bands']
    hd = [d.strftime('%Y-%m-%d') for d in hist_bands.index]

    fig = go.Figure()

    # EQM band fills
    for i in range(len(QNAMES) - 1):
        lo, hi, col = QNAMES[i], QNAMES[i + 1], BAND_COLORS[i]
        lo_vals = hist_bands[lo].values
        hi_vals = hist_bands[hi].values
        mask = ~(np.isnan(lo_vals) | np.isnan(hi_vals))
        hd_m  = [hd[j] for j in range(len(hd)) if mask[j]]
        lo_m  = lo_vals[mask].tolist()
        hi_m  = hi_vals[mask].tolist()
        fig.add_trace(go.Scatter(
            x=hd_m + hd_m[::-1], y=hi_m + lo_m[::-1],
            fill='toself', fillcolor=col, opacity=0.20,
            line=dict(width=0), showlegend=False, hoverinfo='skip'))

    # EQM band lines
    for name, col in zip(QNAMES, BAND_COLORS):
        lw = 2.5 if name == 'EQM_50' else 1.6
        vals = hist_bands[name].values
        fig.add_trace(go.Scatter(
            x=hd, y=vals, mode='lines', name=name,
            legendgroup='EQM', line=dict(color=col, width=lw),
            hovertemplate='%{y:$,.0f}<extra>' + name + '</extra>'))

    # Rails (0.1% and 99.9%)
    for rname, rcol, rlabel in zip(RAIL_NAMES, RAIL_COLORS,
                                   ['0.1% rail', '99.9% rail']):
        rvals = hist_bands[rname].values
        fig.add_trace(go.Scatter(
            x=hd, y=rvals, mode='lines', name=rlabel,
            legendgroup='rails',
            line=dict(color=rcol, width=1.2, dash='dash'),
            hovertemplate='%{y:$,.0f}<extra>' + rlabel + '</extra>'))

    # Full-history OLS reference (BTCAnalytica's OLS PL)
    a_f, b_f = current_c['a_full'], current_c['b_full']
    ols_vals  = np.exp(a_f + b_f * df['x'].values)
    fig.add_trace(go.Scatter(
        x=df['date'].dt.strftime('%Y-%m-%d').tolist(),
        y=ols_vals.tolist(),
        mode='lines', name=f'OLS PL (full hist, b={b_f:.2f})',
        legendgroup='reference',
        line=dict(color='#cc88ff', width=1.2, dash='dash'),
        hovertemplate='%{y:$,.0f}<extra>OLS PL</extra>'))

    # BTC price
    fig.add_trace(go.Scatter(
        x=df['date'].dt.strftime('%Y-%m-%d').tolist(),
        y=df['close'].tolist(),
        mode='lines', name='BTC Price',
        line=dict(color='white', width=1.5),
        hovertemplate='%{x}<br>$%{y:,.0f}<extra>BTC</extra>'))

    # Band table annotation
    band_lines = ['<b>EQM Bands today</b>']
    band_lines.append(f'99.9% rail: ${cb[RAIL_NAMES[1]]/1000:.0f}K')
    for i, label in enumerate(BAND_LABELS):
        lo_p = cb[QNAMES[i]] / 1000
        hi_p = cb[QNAMES[i + 1]] / 1000
        band_lines.append(
            f'{label}: ${lo_p:.0f}K–${hi_p:.0f}K '
            f'({int(QUANTILES[i]*100)}–{int(QUANTILES[i+1]*100)}%)')
    band_lines.append(f'0.1% rail: ${cb[RAIL_NAMES[0]]/1000:.0f}K')

    ath_d  = current_c['ath_date']
    eqm_ath = current_c['eqm_ath']
    dd      = current_c['drawdown_pct']
    opl_r   = current_c['ols_full_ath_ratio']

    fig.update_layout(
        title=dict(text='Bitcoin EQM Rainbow — Walk-Forward Empirical Quantile Model',
                   font=dict(color='white', size=17)),
        paper_bgcolor='#0d0d0d', plot_bgcolor='#111',
        font=dict(color='#aaa'),
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(type='log', showgrid=True, gridcolor='#333',
                   tickformat='$,.0f', range=[1.7, 6.3]),
        legend=dict(bgcolor='#111', bordercolor='#444',
                    font=dict(color='white'), groupclick='toggleitem'),
        annotations=[
            dict(
                text=(
                    f"EQM: expanding std from {EQM_START} | Pool {current_c['resid_window']}d | "
                    f"R²={current_c['r2']:.4f} | Risk={current_c['risk_pct']:.1f}% | "
                    f"Centering={current_c['centering']:.1f}% | "
                    f"EQM₅₀/ATH={eqm_ath:.3f} | Drawdown={dd:.1f}% | "
                    f"OLS_PL/ATH={opl_r:.3f} | ATH=${current_c['ath_price']:,.0f} ({ath_d})"
                ),
                xref='paper', yref='paper', x=0.01, y=0.99,
                showarrow=False, font=dict(color='white', size=9.5),
                bgcolor='rgba(0,0,0,0.55)', align='left'),
            dict(
                text='<br>'.join(band_lines),
                xref='paper', yref='paper', x=0.99, y=0.01,
                xanchor='right', yanchor='bottom',
                showarrow=False, font=dict(color='white', size=9),
                bgcolor='rgba(0,0,0,0.6)', align='left'),
        ],
        height=700,
    )
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"HTML saved: {out_path}")


# ── CSV output ────────────────────────────────────────────────────────────────
def generate_csv(df, hist_bands, current_c, out_path):
    """
    Outputs walk-forward frozen EQM bands + full-history OLS reference
    for every calendar date in the dataset.
    """
    all_dates    = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    price_lookup = df.set_index('date')['close']

    a_f, b_f = current_c['a_full'], current_c['b_full']

    rows = []
    for d in all_dates:
        price_val = price_lookup.get(d, np.nan)
        try:
            pv = round(float(price_val), 2) if not np.isnan(float(price_val)) else ''
        except (TypeError, ValueError):
            pv = ''

        t_d  = (d.date() - GENESIS).days
        x_d  = np.log(t_d)
        ols_full_d = round(float(np.exp(a_f + b_f * x_d)), 2)

        row = {'date': d.strftime('%Y-%m-%d'), 'price_usd': pv,
               'ols_pl_full': ols_full_d}

        all_col_names = QNAMES + RAIL_NAMES
        if d in hist_bands.index:
            v50 = hist_bands.loc[d, 'EQM_50']
            if not np.isnan(v50):
                for name in all_col_names:
                    row[name] = round(float(hist_bands.loc[d, name]), 2)
            else:
                for name in all_col_names:
                    row[name] = ''
        else:
            for name in all_col_names:
                row[name] = ''

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"CSV saved: {out_path}  ({len(rows)} rows)")


# ── Statistics summary ────────────────────────────────────────────────────────
def generate_stats(df, current_c, out_path):
    today  = df['date'].max().date()
    price  = float(df['close'].values[-1])
    t_now  = (today - GENESIS).days
    x_now  = np.log(t_now)
    a, b   = current_c['a'], current_c['b']

    lines = [
        'Bitcoin EQM Rainbow — Model Statistics',
        f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Data      : {df["date"].min().date()} to {today} ({len(df)} rows)',
        f'Current   : ${price:,.0f}',
        f'ATH       : ${current_c["ath_price"]:,.0f}  ({current_c["ath_date"]})',
        f'Drawdown  : {current_c["drawdown_pct"]:.1f}% from ATH',
        '',
        '=' * 64,
        f'  EQM (expanding standardization from {EQM_START} + {EQM_POOL}-day quantile pool)',
        '=' * 64,
        f'  Mean / std of all closes since {EQM_START}: ${current_c["mu"]:,.0f} / ${current_c["sd"]:,.0f}',
        f'  Walk-fwd R² (log price vs same-day EQM_50): {current_c["r2"]:.4f}',
        f'  Risk (current): {current_c["risk_pct"]:.1f}%  '
        f'→ {band_label(current_c["risk_pct"])}',
        f'  Centering     : {current_c["centering"]:.2f}%  '
        f'(fraction of days with price below EQM_50)',
        f'  EQM₅₀ / ATH  : {current_c["eqm_ath"]:.4f}  '
        f'(EQM_50 / all-time high)',
        f'  Pool size     : {current_c["n_resid"]} standardized prices',
        '',
        '  EQM bands today:',
        f'  {"Zone":<12} {"Name":>8}  {"Price":>10}',
        f'  {"-"*35}',
    ]

    # Top rail
    rail_hi = np.exp(a + b * x_now + current_c['q_offsets'][RAIL_QUANTILES[1]])
    lines.append(f'  {"99.9% rail":<12} {RAIL_NAMES[1]:>8}  ${rail_hi:>9,.0f}')

    for q, name, lbl in zip(QUANTILES, QNAMES, ['—'] + BAND_LABELS):
        band_price = np.exp(a + b * x_now + current_c['q_offsets'][q])
        zone = lbl[:10] if lbl != '—' else ''
        lines.append(f'  {zone:<12} {name:>8}  ${band_price:>9,.0f}')

    # Bottom rail
    rail_lo = np.exp(a + b * x_now + current_c['q_offsets'][RAIL_QUANTILES[0]])
    lines.append(f'  {"0.1% rail":<12} {RAIL_NAMES[0]:>8}  ${rail_lo:>9,.0f}')

    # Full-history OLS reference
    a_f, b_f = current_c['a_full'], current_c['b_full']
    ols_full  = current_c['ols_full_now']
    lines += [
        '',
        '=' * 64,
        '  OLS PL (full-history reference — BTCAnalytica left panel)',
        '=' * 64,
        f'  OLS (full hist): log(P) = {a_f:.4f} + {b_f:.4f} × log(t)',
        f'  Fair value now  : ${ols_full:,.0f}',
        f'  OLS_PL / ATH   : {current_c["ols_full_ath_ratio"]:.4f}  '
        f'(>1.0 = OLS exceeds ATH — first ever in a bear market)',
        '',
        '  Cycle mania compression (top ÷ OLS@cycle-start):',
        '    C1 2013:  39×   C2 2017: 32×   C3 2021: 4.6×   C4 2025: 3.2×',
        '  Compression is captured automatically by the rolling residual window.',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Stats saved: {out_path}")


# ── Regenerate eqm_rainbow_tv.html with fresh embedded data ──────────────────
def _replace_js_var(html, var_name, new_json):
    """Replace const VAR = <value>; handling large multiline arrays/objects."""
    import re
    m = re.search(rf'(?:const|let|var)\s+{re.escape(var_name)}\s*=\s*', html)
    if not m:
        raise ValueError(f"JS variable not found: {var_name}")
    keyword = m.group(0).split()[0]
    val_start = m.end()
    # Walk forward counting brackets to find the matching close
    open_chars  = {'[': ']', '{': '}'}
    close_chars = {']', '}'}
    first = html[val_start]
    if first not in open_chars:
        # scalar — find next semicolon (consume any accumulated ones)
        end = html.index(';', val_start)
        while end < len(html) and html[end] == ';':
            end += 1
    else:
        depth, pos = 0, val_start
        in_str, esc = False, False
        while pos < len(html):
            c = html[pos]
            if esc:
                esc = False
            elif c == '\\' and in_str:
                esc = True
            elif c == '"' and not esc:
                in_str = not in_str
            elif not in_str:
                if c in open_chars:
                    depth += 1
                elif c in close_chars:
                    depth -= 1
                    if depth == 0:
                        break
            pos += 1
        end = pos + 1  # one past closing bracket
        # skip optional trailing semicolon(s)
        while end < len(html) and html[end] == ';':
            end += 1
    return html[:m.start()] + f'{keyword} {var_name} = {new_json};' + html[end:]


def generate_tv_html(df, hist_bands, current_c, tv_path):
    """Read eqm_rainbow_tv.html and replace all embedded data blobs."""
    import json as _json

    if not os.path.exists(tv_path):
        print(f"  TV HTML not found at {tv_path} — skipping")
        return

    with open(tv_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # ── PRICE_DATA ────────────────────────────────────────────────────────────
    price_data = [
        {"time": row['date'].strftime('%Y-%m-%d'), "value": round(float(row['close']), 2)}
        for _, row in df.iterrows()
    ]
    html = _replace_js_var(html, 'PRICE_DATA', _json.dumps(price_data))

    # ── BAND_DATA ─────────────────────────────────────────────────────────────
    all_names = QNAMES + RAIL_NAMES
    band_data = {}
    for name in all_names:
        col   = hist_bands[name].dropna()
        band_data[name] = [
            {"time": d.strftime('%Y-%m-%d'), "value": round(float(v), 2)}
            for d, v in col.items()
        ]
    html = _replace_js_var(html, 'BAND_DATA', _json.dumps(band_data))

    # ── OLS_DATA (full-history OLS dotted line) ────────────────────────────────
    a_f, b_f = current_c['a_full'], current_c['b_full']
    ols_data = [
        {"time": row['date'].strftime('%Y-%m-%d'),
         "value": round(float(np.exp(a_f + b_f * row['x'])), 2)}
        for _, row in df.iterrows()
    ]
    html = _replace_js_var(html, 'OLS_DATA', _json.dumps(ols_data))

    # ── OLS_A, OLS_B (EQM walk-forward params for live risk gauge) ────────────
    html = _replace_js_var(html, 'OLS_A', repr(round(float(current_c['a']), 8)))
    html = _replace_js_var(html, 'OLS_B', repr(round(float(current_c['b']), 8)))

    # ── RESIDUALS (rolling window — for live risk computation) ────────────────
    resid_rolling = current_c['resid_live']
    html = _replace_js_var(html, 'RESIDUALS',
                           _json.dumps([round(float(r), 6) for r in resid_rolling]))

    # ── Stale default live-price values (single combined let declaration) ────
    cur_price = round(float(df['close'].values[-1]))
    cur_risk  = round(current_c['risk_pct'], 1)
    cur_eqm50 = round(float(current_c['current_bands']['EQM_50']))
    import re as _re
    html = _re.sub(
        r'let CURRENT_PRICE\s*=\s*[\d.]+,\s*CURRENT_EQM50\s*=\s*[\d.]+,\s*CURRENT_RISK\s*=\s*[\d.]+;',
        f'let CURRENT_PRICE = {cur_price}, CURRENT_EQM50 = {cur_eqm50}, CURRENT_RISK = {cur_risk};',
        html, count=1
    )

    with open(tv_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"TV HTML saved: {tv_path}")


# ── Mobile HTML ───────────────────────────────────────────────────────────────
def generate_mobile_html(df, hist_bands, current_c, mobile_path):
    """Generate a mobile-optimised standalone HTML chart."""
    import json as _json

    # ── Data prep ─────────────────────────────────────────────────────────────
    price_data = [
        {"time": row['date'].strftime('%Y-%m-%d'), "value": round(float(row['close']), 2)}
        for _, row in df.iterrows()
    ]

    all_names = QNAMES + RAIL_NAMES
    band_data = {}
    for name in all_names:
        col = hist_bands[name].dropna()
        band_data[name] = [
            {"time": d.strftime('%Y-%m-%d'), "value": round(float(v), 2)}
            for d, v in col.items()
        ]

    a_f, b_f = current_c['a_full'], current_c['b_full']
    ols_data = [
        {"time": row['date'].strftime('%Y-%m-%d'),
         "value": round(float(np.exp(a_f + b_f * row['x'])), 2)}
        for _, row in df.iterrows()
    ]

    resid_rolling = current_c['resid_live']
    residuals_json = _json.dumps([round(float(r), 6) for r in resid_rolling])

    cur_price = round(float(df['close'].values[-1]))
    cur_eqm50 = round(float(current_c['current_bands']['EQM_50']))
    cur_risk  = round(current_c['risk_pct'], 1)
    cur_band  = band_label(cur_risk)
    live_labels = _json.dumps(list(BAND_LABELS))
    ath_price = int(round(current_c['ath_price']))
    drawdown  = round(current_c['drawdown_pct'], 1)
    ols_a     = round(current_c['a'], 8)
    ols_b     = round(current_c['b'], 8)

    # ── Band legend HTML ──────────────────────────────────────────────────────
    cb = current_c['current_bands']
    leg_rows = []
    leg_rows.append(
        f'<div class="br"><div class="bs" style="background:#bb1100"></div>'
        f'<span class="bn">99.9% rail</span>'
        f'<span class="bv">${cb["EQM_99_9"]/1000:.0f}K</span></div>')
    for i in range(len(BAND_LABELS) - 1, -1, -1):
        lo_v, hi_v = cb[QNAMES[i]], cb[QNAMES[i + 1]]
        leg_rows.append(
            f'<div class="br"><div class="bs" style="background:{BAND_COLORS[i]}"></div>'
            f'<span class="bn">{BAND_LABELS[i]}</span>'
            f'<span class="bv">${lo_v/1000:.0f}K–${hi_v/1000:.0f}K</span></div>')
    leg_rows.append(
        f'<div class="br"><div class="bs" style="background:#888899"></div>'
        f'<span class="bn">0.1% rail</span>'
        f'<span class="bv">${cb["EQM_0_1"]/1000:.0f}K</span></div>')
    legend_html = '\n'.join(leg_rows)

    # ── Determine band color for stats ────────────────────────────────────────
    risk = cur_risk
    if   risk <= 10:  band_color = '#1565c0'
    elif risk <= 20:  band_color = '#039be5'
    elif risk <= 30:  band_color = '#00acc1'
    elif risk <= 40:  band_color = '#43a047'
    elif risk <= 50:  band_color = '#c0ca33'
    elif risk <= 60:  band_color = '#ffb300'
    elif risk <= 70:  band_color = '#ff6d00'
    elif risk <= 80:  band_color = '#bf360c'
    elif risk <= 90:  band_color = '#dd0000'
    else:             band_color = '#770000'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>BTC EQM</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#ccc;font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif;overflow-y:auto;-webkit-text-size-adjust:100%}}
#hdr{{display:flex;justify-content:space-between;align-items:center;padding:9px 14px;background:#111;border-bottom:1px solid #222;position:sticky;top:0;z-index:20}}
#hdr-title{{font-size:13px;color:#eee;font-weight:500}}
#hdr-upd{{font-size:10px;color:#555}}
#chart-wrap{{position:relative;height:55dvh;min-height:300px;background:#0d0d0d}}
#fill-canvas{{position:absolute;top:0;left:0;z-index:1;pointer-events:none}}
#tv-chart{{position:absolute;top:0;left:0;z-index:2;width:100%;height:100%}}
#loading{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#0d0d0d;z-index:50;font-size:13px;color:#555}}
#stats-sec{{padding:12px 16px;border-bottom:1px solid #1e1e1e}}
.sr{{display:flex;justify-content:space-between;align-items:baseline;padding:4px 0;border-bottom:1px solid #141414}}
.sr:last-child{{border-bottom:none}}
.sl{{font-size:11px;color:#555}}
.sv{{font-size:12px;color:#ddd;font-weight:500}}
#s-price{{font-size:24px;color:#fff;font-weight:700;letter-spacing:-.02em}}
#leg-toggle{{display:flex;justify-content:space-between;align-items:center;padding:13px 16px;cursor:pointer;border-bottom:1px solid #1e1e1e;-webkit-tap-highlight-color:transparent}}
#leg-toggle span{{font-size:12px;color:#777}}
#leg-toggle:active{{background:#141414}}
#leg-panel{{display:none;padding:8px 16px 20px;border-bottom:1px solid #1e1e1e}}
.br{{display:flex;align-items:center;gap:10px;padding:4px 0}}
.bs{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
.bn{{font-size:11px;color:#bbb;flex:1}}
.bv{{font-size:10px;color:#555}}
</style>
</head>
<body>

<div id="hdr">
  <span id="hdr-title">Bitcoin EQM Rainbow</span>
  <span id="hdr-upd"></span>
</div>

<div id="chart-wrap">
  <div id="loading">Loading…</div>
  <canvas id="fill-canvas"></canvas>
  <div id="tv-chart"></div>
</div>

<div id="stats-sec">
  <div class="sr" style="padding-bottom:8px;border-bottom:1px solid #2a2a2a;margin-bottom:4px">
    <span class="sl">Price</span>
    <span id="s-price">${cur_price:,}</span>
  </div>
  <div class="sr">
    <span class="sl">Band</span>
    <span id="s-band" class="sv" style="color:{band_color}">{cur_band}</span>
  </div>
  <div class="sr">
    <span class="sl">EQM 50%</span>
    <span id="s-eqm50" class="sv">${cur_eqm50:,}</span>
  </div>
  <div class="sr">
    <span class="sl">Risk</span>
    <span id="s-risk" class="sv">{cur_risk}%</span>
  </div>
  <div class="sr">
    <span class="sl">ATH</span>
    <span class="sv">${ath_price:,}</span>
  </div>
  <div class="sr">
    <span class="sl">Drawdown</span>
    <span class="sv">{drawdown}%</span>
  </div>
</div>

<div id="leg-toggle" onclick="toggleLeg()">
  <span>EQM Bands (today)</span>
  <span id="leg-arrow">▼</span>
</div>
<div id="leg-panel">
{legend_html}
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────────
const PRICE_DATA = {_json.dumps(price_data)};
const BAND_DATA  = {_json.dumps(band_data)};
const OLS_DATA   = {_json.dumps(ols_data)};
const OLS_A      = {ols_a};
const OLS_B      = {ols_b};
const RESIDUALS  = {residuals_json};
const RESID_SORTED = [...RESIDUALS].sort((a,b)=>a-b);
const GENESIS_TS = 1230940800;
let CURRENT_PRICE = {cur_price}, CURRENT_EQM50 = {cur_eqm50}, CURRENT_RISK = {cur_risk};

// ── Canvas + Chart setup ──────────────────────────────────────────────────────
const canvas = document.getElementById('fill-canvas');
const ctx    = canvas.getContext('2d');
const wrap   = document.getElementById('chart-wrap');
const chartEl = document.getElementById('tv-chart');

const QNAMES      = ["EQM_1","EQM_10","EQM_20","EQM_30","EQM_40","EQM_50","EQM_60","EQM_70","EQM_80","EQM_90","EQM_99"];
const RAIL_NAMES  = ["EQM_0_1","EQM_99_9"];
const COLORS      = ["#1f2878","#2065b4","#1195d6","#0ca2b4","#489a4c","#b7c03c","#edac11","#ed6f11","#b33c17","#cd0f0f","#6f0707"];
const RAIL_COLORS = ["#888899","#bb1100"];

const chart = LightweightCharts.createChart(chartEl, {{
  width:  wrap.offsetWidth,
  height: wrap.offsetHeight,
  layout: {{
    background: {{ type: LightweightCharts.ColorType.Solid, color: 'transparent' }},
    textColor: '#666',
  }},
  rightPriceScale: {{
    mode: LightweightCharts.PriceScaleMode.Logarithmic,
    borderColor: '#2a2a2a',
    scaleMargins: {{ top: 0.04, bottom: 0.04 }},
  }},
  leftPriceScale: {{ visible: false }},
  timeScale: {{ borderColor: '#2a2a2a', timeVisible: true, secondsVisible: false }},
  grid: {{
    vertLines: {{ visible: false }},
    horzLines: {{ visible: false }},
  }},
  crosshair: {{
    mode: LightweightCharts.CrosshairMode.Normal,
    vertLine: {{ color: '#444', labelBackgroundColor: '#2a2a2a', width: 1 }},
    horzLine: {{ color: '#444', labelBackgroundColor: '#2a2a2a', width: 1 }},
  }},
}});

// Band series (TV lines, drawn on top of canvas fills)
const bandSeries = {{}};
for (let i = 0; i < QNAMES.length; i++) {{
  const name = QNAMES[i];
  const s = chart.addLineSeries({{
    color: COLORS[i],
    lineWidth: name === 'EQM_50' ? 2 : 1,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  }});
  s.setData(BAND_DATA[name]);
  bandSeries[name] = s;
}}
for (let i = 0; i < RAIL_NAMES.length; i++) {{
  const name = RAIL_NAMES[i];
  const s = chart.addLineSeries({{
    color: RAIL_COLORS[i], lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  }});
  s.setData(BAND_DATA[name]);
  bandSeries[name] = s;
}}

// OLS reference
const olsSeries = chart.addLineSeries({{
  color: '#9955ee', lineWidth: 1,
  lineStyle: LightweightCharts.LineStyle.Dashed,
  priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
}});
olsSeries.setData(OLS_DATA);

// Price line (topmost)
const priceSeries = chart.addLineSeries({{
  color: '#ffffff', lineWidth: 2.5,
  priceLineVisible: false, lastValueVisible: true,
  crosshairMarkerVisible: true, crosshairMarkerRadius: 5,
  crosshairMarkerBorderColor: '#fff', crosshairMarkerBackgroundColor: '#111',
}});
priceSeries.setData(PRICE_DATA);

// ── Helpers ───────────────────────────────────────────────────────────────────
function hexToRgba(hex, a) {{
  const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

// Pre-build time→value lookup per band
const lookup = {{}};
for (const name of [...QNAMES, ...RAIL_NAMES]) {{
  lookup[name] = {{}};
  for (const pt of BAND_DATA[name]) lookup[name][pt.time] = pt.value;
}}
const ALL_TIMES = BAND_DATA[RAIL_NAMES[0]].map(p => p.time);

const FILL_PAIRS = [
  [RAIL_NAMES[0], QNAMES[0],             COLORS[0]],
  ...QNAMES.slice(0,-1).map((lo,i)=>[lo, QNAMES[i+1], COLORS[i]]),
  [QNAMES[QNAMES.length-1], RAIL_NAMES[1], COLORS[COLORS.length-1]],
];

const HALVING_TS  = [1354060800, 1468022400, 1589155200, 1713571200, 1839542400, 1965945600];
const HALVING_LBL = ["Halving 1  Nov '12", "Halving 2  Jul '16", "Halving 3  May '20", "Halving 4  Apr '24", "Halving 5  ~Apr '28", "Halving 6  ~Apr '32"];

// ── Draw fills on canvas ──────────────────────────────────────────────────────
function drawFills() {{
  const w = wrap.offsetWidth, h = wrap.offsetHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const _lastT = ALL_TIMES[ALL_TIMES.length - 1];
  const _refT  = ALL_TIMES[Math.max(0, ALL_TIMES.length - 91)];
  const _x1    = chart.timeScale().timeToCoordinate(_lastT);
  const _x0    = chart.timeScale().timeToCoordinate(_refT);
  const _msDay = 86400000;
  const _pxDay = (_x1 != null && _x0 != null && _x1 !== _x0)
    ? (_x1 - _x0) / ((new Date(_lastT) - new Date(_refT)) / _msDay) : null;
  function tsToX(ts) {{
    const x = chart.timeScale().timeToCoordinate(ts);
    if (x != null) return x;
    if (_pxDay == null || _x1 == null) return null;
    return _x1 + (ts - new Date(_lastT).getTime()/1000) / 86400 * _pxDay;
  }}

  const visRange = chart.timeScale().getVisibleRange();
  if (!visRange) return;
  const tFrom = visRange.from, tTo = visRange.to;
  const visible = ALL_TIMES.filter(t => t >= tFrom && t <= tTo);
  if (visible.length < 2) return;

  const step = Math.max(1, Math.floor(visible.length / 500));

  for (const [loName, hiName, hexColor] of FILL_PAIRS) {{
    const loLookup = lookup[loName], hiLookup = lookup[hiName];
    const loSeries = bandSeries[loName], hiSeries = bandSeries[hiName];
    const hiPts = [], loPts = [];
    for (let i = 0; i < visible.length; i += step) {{
      const t = visible[i];
      const loV = loLookup[t], hiV = hiLookup[t];
      if (loV == null || hiV == null) continue;
      const x  = chart.timeScale().timeToCoordinate(t);
      const yH = hiSeries.priceToCoordinate(hiV);
      const yL = loSeries.priceToCoordinate(loV);
      if (x == null || yH == null || yL == null) continue;
      hiPts.push([x, yH]);
      loPts.push([x, yL]);
    }}
    if (hiPts.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(hiPts[0][0], hiPts[0][1]);
    for (let j = 1; j < hiPts.length; j++) ctx.lineTo(hiPts[j][0], hiPts[j][1]);
    for (let j = loPts.length-1; j >= 0; j--) ctx.lineTo(loPts[j][0], loPts[j][1]);
    ctx.closePath();
    ctx.fillStyle = hexToRgba(hexColor, 0.70);
    ctx.fill();
  }}

  // Halving lines
  ctx.setLineDash([]);
  for (let i = 0; i < HALVING_TS.length; i++) {{
    const x = tsToX(HALVING_TS[i]);
    if (x == null || x < 0 || x > w) continue;
    const isEst = (i >= 4);
    ctx.beginPath();
    ctx.moveTo(x, 0); ctx.lineTo(x, h);
    ctx.setLineDash(isEst ? [6, 4] : []);
    ctx.strokeStyle = isEst ? 'rgba(255,210,0,0.28)' : 'rgba(255,210,0,0.50)';
    ctx.lineWidth = isEst ? 1.5 : 2;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = 'bold 10px -apple-system,sans-serif';
    ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(0,0,0,0.85)';
    ctx.strokeText(HALVING_LBL[i], x + 4, 17);
    ctx.fillStyle = isEst ? 'rgba(255,215,0,0.55)' : 'rgba(255,215,0,0.95)';
    ctx.fillText(HALVING_LBL[i], x + 4, 17);
  }}

  // Black outline under the white TV price line
  const pricePts = [];
  for (const d of PRICE_DATA) {{
    if (d.time < tFrom || d.time > tTo) continue;
    const x = chart.timeScale().timeToCoordinate(d.time);
    const y = priceSeries.priceToCoordinate(d.value);
    if (x == null || y == null) continue;
    pricePts.push([x, y]);
  }}
  if (pricePts.length >= 2) {{
    ctx.beginPath();
    ctx.moveTo(pricePts[0][0], pricePts[0][1]);
    for (let i = 1; i < pricePts.length; i++) ctx.lineTo(pricePts[i][0], pricePts[i][1]);
    ctx.strokeStyle = 'rgba(0,0,0,0.9)';
    ctx.lineWidth = 6;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
  }}
}}

let _raf = null;
function schedDraw() {{ if(_raf) return; _raf = requestAnimationFrame(()=>{{_raf=null;drawFills();}}); }}
new ResizeObserver(schedDraw).observe(wrap);
chart.timeScale().subscribeVisibleLogicalRangeChange(schedDraw);

// ── Live price ────────────────────────────────────────────────────────────────
function computeEQM50() {{
  const n=RESID_SORTED.length, p=0.5*(n-1), lo=Math.floor(p), hi=Math.ceil(p);
  const med=lo===hi?RESID_SORTED[lo]:RESID_SORTED[lo]+(p-lo)*(RESID_SORTED[hi]-RESID_SORTED[lo]);
  return Math.round(Math.exp(OLS_A+OLS_B*Math.log((Date.now()/1000-GENESIS_TS)/86400)+med));
}}
function fmtP(n) {{ return '$'+Math.round(n).toLocaleString(); }}
const LIVE_EDGES  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 101];
const LIVE_LABELS = {live_labels};
const LIVE_COLS   = ['#1565c0','#039be5','#00acc1','#43a047','#c0ca33','#ffb300','#ff6d00','#bf360c','#dd0000','#770000'];
function fetchPrice() {{
  fetch('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT')
    .then(r=>r.json()).then(d=>{{
      const p = Math.round(parseFloat(d.price)); if (!p) return;
      CURRENT_PRICE = p; CURRENT_EQM50 = computeEQM50();
      // Extend the price line + its black outline to the live price so the chart label matches the Price stat
      const _now = new Date();
      const todayStr = _now.getUTCFullYear() + '-' +
        String(_now.getUTCMonth()+1).padStart(2,'0') + '-' +
        String(_now.getUTCDate()).padStart(2,'0');
      priceSeries.update({{time: todayStr, value: p}});
      const _last = PRICE_DATA[PRICE_DATA.length-1];
      if (_last && _last.time === todayStr) {{ _last.value = p; }}
      else {{ PRICE_DATA.push({{time: todayStr, value: p}}); }}
      schedDraw();
      document.getElementById('s-price').textContent = fmtP(p);
      document.getElementById('s-eqm50').textContent = fmtP(CURRENT_EQM50);
      // Live risk = percentile of the live price within today's pool (same rule as the TV page)
      const lp = Math.log(p) - (OLS_A + OLS_B*Math.log((Date.now()/1000-GENESIS_TS)/86400));
      let below = 0; for (const r of RESIDUALS) if (r <= lp) below++;
      CURRENT_RISK = 100*below/RESIDUALS.length;
      document.getElementById('s-risk').textContent = CURRENT_RISK.toFixed(1) + '%';
      const bi = LIVE_EDGES.findIndex(e => CURRENT_RISK <= e);
      const bEl = document.getElementById('s-band');
      bEl.textContent = bi >= 0 && bi < LIVE_LABELS.length ? LIVE_LABELS[bi] : 'Extreme';
      bEl.style.color = LIVE_COLS[Math.max(0, Math.min(9, Math.ceil(CURRENT_RISK/10) - 1))];
      const now = new Date();
      document.getElementById('hdr-upd').textContent =
        now.toLocaleDateString('en-GB',{{day:'numeric',month:'short'}}) + ' · ' +
        now.toLocaleTimeString('en-GB',{{hour:'2-digit',minute:'2-digit'}});
    }}).catch(()=>{{}});
}}
fetchPrice(); setInterval(fetchPrice, 60000);

// ── Legend toggle ─────────────────────────────────────────────────────────────
function toggleLeg() {{
  const p = document.getElementById('leg-panel');
  const open = p.style.display !== 'none';
  p.style.display = open ? 'none' : 'block';
  document.getElementById('leg-arrow').textContent = open ? '▼' : '▲';
}}

// ── Init ──────────────────────────────────────────────────────────────────────
chart.timeScale().fitContent();
requestAnimationFrame(()=>{{ document.getElementById('loading').style.display='none'; drawFills(); }});
setTimeout(drawFills, 300); setTimeout(drawFills, 800);
</script>
</body>
</html>"""

    with open(mobile_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Mobile HTML saved: {mobile_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    print("\nFetching latest prices...")
    df = fetch_latest_prices(df)
    print(f"  Current price : ${df['close'].values[-1]:,.0f}")
    print(f"  ATH           : ${df['close'].max():,.0f}")

    print(f"\nComputing EQM (expanding standardization from {EQM_START}, pool {EQM_POOL})...")
    hist_bands, current_c = compute_model_c(df)
    print(f"  EQM: mean={current_c['mu']:,.0f}  std={current_c['sd']:,.0f}  50%=${current_c['current_bands']['EQM_50']:,.0f}")
    print(f"  OLS (PL) : a={current_c['a_full']:.4f}, b={current_c['b_full']:.4f}")
    print(f"  R²={current_c['r2']:.6f}  Risk={current_c['risk_pct']:.1f}%  "
          f"Centering={current_c['centering']:.1f}%")
    print(f"  EQM₅₀/ATH={current_c['eqm_ath']:.4f}  "
          f"OLS_PL/ATH={current_c['ols_full_ath_ratio']:.4f}  "
          f"Drawdown={current_c['drawdown_pct']:.1f}%")

    print("\nGenerating outputs...")
    generate_csv(df, hist_bands, current_c,
                 f"{OUT_DIR}/eqm_bands.csv")
    generate_png(df, hist_bands, current_c,
                 f"{OUT_DIR}/eqm_rainbow.png")
    generate_html(df, hist_bands, current_c,
                  f"{OUT_DIR}/eqm_rainbow.html")
    generate_stats(df, current_c,
                   f"{OUT_DIR}/eqm_stats.txt")
    generate_tv_html(df, hist_bands, current_c,
                     f"{OUT_DIR}/eqm_rainbow_tv.html")
    generate_mobile_html(df, hist_bands, current_c,
                         f"{OUT_DIR}/eqm_rainbow_mobile.html")
    print("\nAll outputs generated.")
