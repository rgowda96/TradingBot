"""
COMPREHENSIVE HISTORICAL ETH STRATEGY REPORT
---------------------------------------------
Backtests all 65 strategies across the FULL ETH history (Aug 2017 → present)
segmented by market era, identifies survivors, and produces an elaborate HTML report.
"""
import os, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from indicators import add_indicators
from backtest_engine import backtest
from master_strategy_library import STRATEGY_REGISTRY, CATEGORY_DESCRIPTIONS

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# MARKET ERAS  (label, start, end, description, bias)
# ─────────────────────────────────────────────────────────
ERAS = [
    {
        "id":    "genesis",
        "label": "Genesis & Discovery",
        "start": "2017-08-17",
        "end":   "2017-12-31",
        "desc":  "ETH goes from $300 to $720. First major retail wave discovers crypto. Extreme uptrend with low liquidity.",
        "bias":  "bull",
        "color": "#00c896",
    },
    {
        "id":    "mania_2018",
        "label": "ICO Mania Peak",
        "start": "2018-01-01",
        "end":   "2018-01-31",
        "desc":  "All-time-high month. ETH peaks at $1,432 on Jan 13 fuelled by ICO speculation. Parabolic vertical move.",
        "bias":  "bull",
        "color": "#00e676",
    },
    {
        "id":    "crash_2018",
        "label": "Crypto Winter 2018",
        "start": "2018-02-01",
        "end":   "2018-12-31",
        "desc":  "ETH crashes 95% from peak to ~$80. One of the worst bear markets in crypto history. ICO selloff pressure.",
        "bias":  "bear",
        "color": "#ff1744",
    },
    {
        "id":    "recovery_2019",
        "label": "Dead-Cat & Recovery 2019",
        "start": "2019-01-01",
        "end":   "2019-12-31",
        "desc":  "ETH ranges $100–$350. Mid-year rally to $365 then fade. Choppy, low-conviction accumulation phase.",
        "bias":  "range",
        "color": "#ffd740",
    },
    {
        "id":    "covid_crash",
        "label": "COVID Crash & Rebound",
        "start": "2020-01-01",
        "end":   "2020-08-31",
        "desc":  "March 2020 flash crash to $88 (-60% in 48h), followed by swift V-shaped recovery. DeFi Summer ignites.",
        "bias":  "volatile",
        "color": "#ff6d00",
    },
    {
        "id":    "defi_summer",
        "label": "DeFi Summer & ETH 2.0 Hype",
        "start": "2020-09-01",
        "end":   "2020-12-31",
        "desc":  "ETH 2.0 deposit contract launches. DeFi TVL explodes. ETH races from $350 to $740 into year-end.",
        "bias":  "bull",
        "color": "#00bcd4",
    },
    {
        "id":    "bull_2021_q1",
        "label": "Institutional Bull Q1 2021",
        "start": "2021-01-01",
        "end":   "2021-05-12",
        "desc":  "ETH runs from $740 to ATH $4,360. Institutional adoption (Grayscale, Tesla BTC), NFT mania, EIP-1559 hype.",
        "bias":  "bull",
        "color": "#00e676",
    },
    {
        "id":    "china_ban_2021",
        "label": "China Ban Crash Mid-2021",
        "start": "2021-05-13",
        "end":   "2021-07-20",
        "desc":  "China mining ban triggers 60% correction. ETH drops from $4,000 to $1,700 in weeks. Fear/capitulation.",
        "bias":  "bear",
        "color": "#ff5252",
    },
    {
        "id":    "altseason_2021",
        "label": "Alt-Season & Second ATH",
        "start": "2021-07-21",
        "end":   "2021-11-10",
        "desc":  "Recovery to new ATH of $4,891 in Nov 2021. NFT/metaverse narrative. Layer-2 hype. London fork EIP-1559.",
        "bias":  "bull",
        "color": "#69f0ae",
    },
    {
        "id":    "bear_2022",
        "label": "Bear Market 2022",
        "start": "2021-11-11",
        "end":   "2022-06-17",
        "desc":  "Fed rate hikes, LUNA collapse (-100%), 3AC bankruptcy. ETH crashes from $4,891 to $880. 82% drawdown.",
        "bias":  "bear",
        "color": "#ff1744",
    },
    {
        "id":    "merge_era",
        "label": "The Merge & Continuation Bear",
        "start": "2022-06-18",
        "end":   "2022-12-31",
        "desc":  "ETH Merge (PoS) Sep 15. FTX collapse Nov crashes to $1,070. Merge buy-the-news, sell-the-rumor dynamic.",
        "bias":  "bear",
        "color": "#d50000",
    },
    {
        "id":    "recovery_2023",
        "label": "Recovery & Stablecoin Crisis 2023",
        "start": "2023-01-01",
        "end":   "2023-09-30",
        "desc":  "USDC depeg (SVB), SEC lawsuits, but ETH holds $1,500–$2,100. Slow grind recovery. Shanghai upgrade (staked ETH withdrawals).",
        "bias":  "range",
        "color": "#ffab40",
    },
    {
        "id":    "etf_rally_2024",
        "label": "ETF Approval Rally 2024",
        "start": "2023-10-01",
        "end":   "2024-03-31",
        "desc":  "Bitcoin ETF approved Jan 2024 ignites whole market. ETH rallies from $1,600 to $3,900. Spot ETH ETF anticipation.",
        "bias":  "bull",
        "color": "#00e676",
    },
    {
        "id":    "eth_etf_launch",
        "label": "ETH Spot ETF Launch & Chop",
        "start": "2024-04-01",
        "end":   "2024-12-31",
        "desc":  "ETH Spot ETF launches July 2024. Initial dump then range. ETH underperforms BTC. $2,100–$3,800 chop.",
        "bias":  "range",
        "color": "#ffd740",
    },
    {
        "id":    "current_2025",
        "label": "Current Cycle 2025–2026",
        "start": "2025-01-01",
        "end":   "2026-06-02",
        "desc":  "Post-halving cycle. Macro uncertainty (tariffs), ETH at ~$1,975. DeFi renaissance, L2 maturity, staking yields.",
        "bias":  "volatile",
        "color": "#e040fb",
    },
]

def load_full_df():
    df = pd.read_parquet("data/ETH_USDT_1d_full.parquet")
    print(f"Loaded {len(df)} daily bars: {df.index[0].date()} → {df.index[-1].date()}")
    df = add_indicators(df)
    return df.dropna()

def slice_era(df, era):
    start = pd.Timestamp(era["start"], tz="UTC")
    end   = pd.Timestamp(era["end"],   tz="UTC")
    slc   = df.loc[start:end]
    return slc if len(slc) >= 20 else None

def run_era_backtests(df_full):
    """Run all 65 strategies on each era + full history. Returns list of result dicts."""
    results = []

    # Full history run
    eras_to_run = [{"id": "full", "label": "FULL HISTORY (2017–2026)", "start": "2017-08-17", "end": "2026-06-02",
                    "desc": "All 3,200+ days", "bias": "mixed", "color": "#ffffff"}] + ERAS

    total_combos = len(STRATEGY_REGISTRY) * len(eras_to_run)
    done = 0

    for era in eras_to_run:
        era_df = slice_era(df_full, era)
        if era_df is None:
            print(f"  Skipping {era['id']} — too few bars")
            continue

        era_start_px = float(era_df['close'].iloc[0])
        era_end_px   = float(era_df['close'].iloc[-1])
        era_bnh_ret  = (era_end_px / era_start_px - 1) * 100

        print(f"\n{'─'*70}")
        print(f"  ERA: {era['label']}  ({len(era_df)} bars)  B&H={era_bnh_ret:+.1f}%")
        print(f"{'─'*70}")

        for strat_id, (category, fn) in STRATEGY_REGISTRY.items():
            done += 1
            try:
                sig    = fn(era_df)
                result = backtest(era_df, sig, tf="1d")
                s      = result["stats"]
                n      = s.get("n_trades", 0)
                sharpe = s.get("sharpe")
                pf     = s.get("profit_factor")
                ret    = s.get("total_return_pct", 0)
                wr     = s.get("win_rate_pct", 0)
                dd     = s.get("max_drawdown_pct", 0)

                sharpe_f = float(sharpe) if sharpe not in (None, "N/A") else None
                pf_f     = float(pf)     if pf     not in (None, "N/A") else None
                ret_f    = float(ret)    if ret    not in (None, "N/A") else 0.0

                row = {
                    "era_id":        era["id"],
                    "era_label":     era["label"],
                    "era_bias":      era["bias"],
                    "era_bnh":       round(era_bnh_ret, 1),
                    "era_bars":      len(era_df),
                    "strategy":      strat_id,
                    "category":      category,
                    "n_trades":      n,
                    "total_return":  round(ret_f, 2),
                    "win_rate":      round(float(wr), 1) if wr not in (None, "N/A") else None,
                    "profit_factor": round(pf_f, 3)      if pf_f is not None else None,
                    "sharpe":        round(sharpe_f, 3)  if sharpe_f is not None else None,
                    "max_dd":        round(float(dd), 2) if dd not in (None, "N/A") else None,
                    "beat_bnh":      ret_f > era_bnh_ret if n >= 3 else False,
                }
                results.append(row)

                if n >= 5 and sharpe_f and abs(sharpe_f) > 0.3:
                    mark = "★" if (sharpe_f > 0.5 and pf_f and pf_f > 1.2) else " "
                    print(f"  {mark} {strat_id:35s}  n={n:3d}  ret={ret_f:+6.1f}%  "
                          f"vs B&H={era_bnh_ret:+6.1f}%  sh={sharpe_f:.2f}  pf={pf_f:.2f if pf_f else 'N/A'}")

            except Exception as e:
                results.append({
                    "era_id": era["id"], "era_label": era["label"], "era_bias": era["bias"],
                    "strategy": strat_id, "category": category, "n_trades": 0,
                    "error": str(e)[:80],
                })

    return results

# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def era_price_range(df_full, era):
    start = pd.Timestamp(era["start"], tz="UTC")
    end   = pd.Timestamp(era["end"],   tz="UTC")
    slc   = df_full.loc[start:end]
    if len(slc) == 0: return "N/A", "N/A"
    return f"${slc['close'].min():,.0f}", f"${slc['close'].max():,.0f}"

def generate_html(results, df_full):
    df = pd.DataFrame(results)
    valid = df[(df.n_trades >= 5) & df.sharpe.notna() & df.profit_factor.notna()].copy()

    # ── per-era top strategies ──────────────────────────────────────────────
    era_tops = {}
    for era_id, grp in valid.groupby("era_id"):
        top = grp.sort_values("sharpe", ascending=False).head(10)
        era_tops[era_id] = top.to_dict("records")

    # ── full-history survivors (Sharpe>0 in 8+ eras) ─────────────────────
    era_list = [e["id"] for e in ERAS]
    survivor_map = {}
    for strat, grp in valid.groupby("strategy"):
        profitable_eras = grp[grp.sharpe > 0]["era_id"].tolist()
        n_prof = len([e for e in profitable_eras if e in era_list])
        if n_prof >= 4:
            full_row = valid[(valid.strategy == strat) & (valid.era_id == "full")]
            if len(full_row):
                r = full_row.iloc[0]
                survivor_map[strat] = {
                    "strategy":      strat,
                    "category":      r["category"],
                    "n_prof_eras":   n_prof,
                    "full_return":   r["total_return"],
                    "full_sharpe":   r["sharpe"],
                    "full_pf":       r["profit_factor"],
                    "full_wr":       r["win_rate"],
                    "full_dd":       r["max_dd"],
                    "full_trades":   r["n_trades"],
                }
    survivors = sorted(survivor_map.values(), key=lambda x: x["n_prof_eras"] * x["full_sharpe"], reverse=True)

    # ── era heatmap: strategy × era sharpe ─────────────────────────────────
    top_strats = valid.groupby("strategy")["sharpe"].mean().sort_values(ascending=False).head(25).index.tolist()
    heatmap_eras = [e["id"] for e in ERAS]

    def sharpe_cell_color(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "#1e1e1e", "—"
        v = float(v)
        if   v >  1.5: return "#00695c", f"{v:.2f}"
        elif v >  0.8: return "#2e7d32", f"{v:.2f}"
        elif v >  0.3: return "#558b2f", f"{v:.2f}"
        elif v >  0.0: return "#827717", f"{v:.2f}"
        elif v > -0.5: return "#b71c1c", f"{v:.2f}"
        else:          return "#4a0000", f"{v:.2f}"

    # ── scenario guide ─────────────────────────────────────────────────────
    scenario_best = {}
    for bias in ["bull", "bear", "range", "volatile"]:
        grp = valid[valid.era_bias == bias]
        if len(grp) == 0: continue
        top = grp.groupby("strategy").agg(
            avg_sharpe=("sharpe", "mean"),
            avg_pf=("profit_factor", "mean"),
            avg_ret=("total_return", "mean"),
            n_eras=("era_id", "count"),
        ).reset_index()
        top = top[top.n_eras >= 2].sort_values("avg_sharpe", ascending=False).head(8)
        scenario_best[bias] = top.to_dict("records")

    # ── full history leaderboard ────────────────────────────────────────────
    full_lb = valid[valid.era_id == "full"].sort_values("sharpe", ascending=False).head(30)

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD HTML
    # ═══════════════════════════════════════════════════════════════════════

    era_timeline_rows = []
    for era in ERAS:
        lo, hi = era_price_range(df_full, era)
        era_id = era["id"]
        top_here = era_tops.get(era_id, [])
        top_str = ", ".join([f"<b>{r['strategy'].split('_',1)[1] if '_' in r['strategy'] else r['strategy']}</b> (sh={r['sharpe']:.2f})" for r in top_here[:3]])
        bias_badge = {
            "bull":     '<span class="badge bull">BULL</span>',
            "bear":     '<span class="badge bear">BEAR</span>',
            "range":    '<span class="badge range">RANGE</span>',
            "volatile": '<span class="badge vol">VOLATILE</span>',
            "mixed":    '<span class="badge mixed">MIXED</span>',
        }.get(era["bias"], "")
        era_timeline_rows.append(f"""
        <tr style="border-left:4px solid {era['color']}">
          <td><b>{era['label']}</b><br><small>{era['start']} → {era['end']}</small></td>
          <td>{bias_badge}</td>
          <td>{lo} → {hi}</td>
          <td>{top_str or '—'}</td>
          <td style="max-width:340px;font-size:12px;color:#aaa">{era['desc']}</td>
        </tr>""")

    # Heatmap table
    heatmap_headers = "".join(f'<th class="hm-head">{e[:8]}</th>' for e in heatmap_eras)
    heatmap_body = []
    for strat in top_strats:
        cells = f'<td class="strat-name">{strat}</td>'
        for era_id in heatmap_eras:
            row = valid[(valid.strategy == strat) & (valid.era_id == era_id)]
            v = row.iloc[0]["sharpe"] if len(row) else None
            bg, txt = sharpe_cell_color(v)
            cells += f'<td style="background:{bg};text-align:center;font-size:11px;padding:3px 5px">{txt}</td>'
        heatmap_body.append(f"<tr>{cells}</tr>")

    # Survivors table
    survivor_rows = []
    for i, s in enumerate(survivors[:25], 1):
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
        survivor_rows.append(f"""
        <tr>
          <td>{medal}</td>
          <td><b>{s['strategy']}</b></td>
          <td>{s['category']}</td>
          <td>{s['n_prof_eras']} / {len(ERAS)}</td>
          <td>{s['full_return']:+.1f}%</td>
          <td>{s['full_sharpe']:.3f}</td>
          <td>{f"{s['full_pf']:.2f}" if s['full_pf'] else 'N/A'}</td>
          <td>{s['full_wr']:.1f}%</td>
          <td>{s['full_dd']:+.1f}%</td>
          <td>{s['full_trades']}</td>
        </tr>""")

    # Scenario guide cards
    BIAS_META = {
        "bull":     ("📈 BULL MARKET",  "#00e676", "Trending up — ride momentum, let winners run"),
        "bear":     ("📉 BEAR MARKET",  "#ff5252", "Trending down — short biased, tight stops, fade rallies"),
        "range":    ("↔️ RANGING",       "#ffd740", "Sideways chop — mean reversion, fade extremes"),
        "volatile": ("⚡ HIGH VOLATILITY","#e040fb", "Big swings both ways — volatility breakout, widen stops"),
    }
    scenario_cards = []
    for bias, meta_label, meta_color, meta_desc in [
        ("bull","📈 BULL MARKET","#00e676","Trending up — ride momentum, let winners run"),
        ("bear","📉 BEAR MARKET","#ff5252","Trending down — short biased, tight stops, fade rallies"),
        ("range","↔️ RANGING","#ffd740","Sideways chop — mean reversion, fade extremes"),
        ("volatile","⚡ HIGH VOLATILITY","#e040fb","Big swings both ways — volatility breakout, widen stops"),
    ]:
        rows = scenario_best.get(bias, [])
        strat_rows = "".join(f"""
          <tr>
            <td><b>{r['strategy']}</b></td>
            <td>{r['avg_sharpe']:.2f}</td>
            <td>{r['avg_ret']:+.1f}%</td>
            <td>{r['avg_pf']:.2f}</td>
            <td>{r['n_eras']}</td>
          </tr>""" for r in rows)
        scenario_cards.append(f"""
        <div class="scenario-card" style="border-top:3px solid {meta_color}">
          <div class="sc-title" style="color:{meta_color}">{meta_label}</div>
          <div class="sc-desc">{meta_desc}</div>
          <table class="sc-table">
            <thead><tr><th>Strategy</th><th>Avg Sharpe</th><th>Avg Ret</th><th>Avg PF</th><th>Eras</th></tr></thead>
            <tbody>{strat_rows}</tbody>
          </table>
        </div>""")

    # Full leaderboard rows
    lb_rows = []
    for i, (_, r) in enumerate(full_lb.iterrows(), 1):
        beat = "✅" if r.get("beat_bnh") else "❌"
        lb_rows.append(f"""
        <tr>
          <td>{i}</td>
          <td><b>{r['strategy']}</b></td>
          <td>{r['category']}</td>
          <td>{r['total_return']:+.1f}%</td>
          <td>{r['win_rate']:.1f}%</td>
          <td>{r['profit_factor']:.2f}</td>
          <td>{r['sharpe']:.3f}</td>
          <td>{r['max_dd']:+.1f}%</td>
          <td>{int(r['n_trades'])}</td>
          <td>{beat}</td>
        </tr>""")

    # KPIs
    n_total   = len(df)
    n_valid   = len(valid[valid.era_id == "full"])
    n_pos     = len(valid[(valid.era_id == "full") & (valid.sharpe > 0)])
    n_beat    = len(valid[(valid.era_id == "full") & (valid.beat_bnh == True)])
    best_sh   = valid[valid.era_id == "full"]["sharpe"].max()
    best_strat= valid[valid.era_id == "full"].sort_values("sharpe", ascending=False).iloc[0]["strategy"] if n_valid else "N/A"
    bnh_full  = round((df_full["close"].iloc[-1] / df_full["close"].iloc[0] - 1) * 100, 1)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETH — Comprehensive Historical Strategy Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0a0a0f;color:#e0e0e0;font-family:'Courier New',monospace;font-size:13px}}
  h1{{font-size:24px;color:#00e5ff;text-align:center;padding:24px 0 4px;letter-spacing:2px}}
  h2{{font-size:16px;color:#00bcd4;border-bottom:1px solid #333;padding:10px 0 6px;margin:28px 0 12px;letter-spacing:1px}}
  h3{{font-size:14px;color:#80cbc4;margin:14px 0 6px}}
  .subtitle{{text-align:center;color:#666;font-size:12px;margin-bottom:24px}}
  .container{{max-width:1400px;margin:0 auto;padding:0 16px 60px}}

  /* KPI bar */
  .kpi-bar{{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 24px}}
  .kpi{{background:#111;border:1px solid #222;border-radius:6px;padding:12px 18px;flex:1;min-width:140px;text-align:center}}
  .kpi .val{{font-size:22px;font-weight:bold;color:#00e5ff}}
  .kpi .lbl{{font-size:11px;color:#666;margin-top:3px}}

  /* Badges */
  .badge{{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:bold}}
  .bull{{background:#1b5e20;color:#69f0ae}}
  .bear{{background:#b71c1c;color:#ff8a80}}
  .range{{background:#f57f17;color:#fff9c4}}
  .vol{{background:#4a148c;color:#ea80fc}}
  .mixed{{background:#263238;color:#90a4ae}}

  /* Timeline table */
  .era-table{{width:100%;border-collapse:collapse;margin-bottom:20px}}
  .era-table th{{background:#111;color:#607d8b;padding:8px 10px;text-align:left;font-size:11px;text-transform:uppercase;border-bottom:2px solid #222}}
  .era-table td{{padding:9px 10px;border-bottom:1px solid #1a1a1a;vertical-align:top}}
  .era-table tr:hover td{{background:#0f1520}}

  /* Heatmap */
  .hm-wrap{{overflow-x:auto;margin-bottom:20px}}
  .hm-table{{border-collapse:collapse;white-space:nowrap}}
  .hm-head{{background:#111;color:#607d8b;padding:4px 6px;font-size:10px;text-align:center;transform:rotate(-45deg);height:80px;vertical-align:bottom;min-width:55px}}
  .strat-name{{font-size:11px;padding:3px 8px;white-space:nowrap;color:#b0bec5;background:#111}}
  .hm-legend{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}}
  .hm-legend span{{padding:2px 10px;border-radius:3px;font-size:11px}}

  /* Data tables */
  .data-table{{width:100%;border-collapse:collapse;margin-bottom:20px}}
  .data-table th{{background:#0d1520;color:#607d8b;padding:7px 10px;text-align:left;font-size:11px;border-bottom:2px solid #1e3040}}
  .data-table td{{padding:7px 10px;border-bottom:1px solid #111;font-size:12px}}
  .data-table tr:hover td{{background:#0a1018}}
  .data-table tr:nth-child(1) td{{color:#ffd700}}
  .data-table tr:nth-child(2) td{{color:#c0c0c0}}
  .data-table tr:nth-child(3) td{{color:#cd7f32}}

  /* Scenario cards */
  .scenario-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-bottom:20px}}
  .scenario-card{{background:#0e0e18;border:1px solid #222;border-radius:8px;padding:16px}}
  .sc-title{{font-size:15px;font-weight:bold;margin-bottom:4px}}
  .sc-desc{{font-size:11px;color:#666;margin-bottom:10px}}
  .sc-table{{width:100%;border-collapse:collapse}}
  .sc-table th{{color:#607d8b;font-size:10px;padding:4px 6px;text-align:left;border-bottom:1px solid #222}}
  .sc-table td{{font-size:11px;padding:4px 6px;border-bottom:1px solid #111}}

  /* Survivors */
  .survivor-note{{background:#0d1520;border:1px solid #1e3040;border-radius:6px;padding:12px 16px;margin-bottom:14px;color:#80cbc4;font-size:12px}}

  /* Era per-strategy modal */
  .era-top-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;margin-bottom:20px}}
  .era-card{{background:#0e0e18;border:1px solid #222;border-radius:8px;padding:14px}}
  .era-card-title{{font-size:13px;font-weight:bold;margin-bottom:4px}}
  .era-card-sub{{font-size:10px;color:#666;margin-bottom:8px}}
  .era-card ol{{padding-left:18px}}
  .era-card li{{font-size:11px;color:#b0bec5;padding:2px 0}}
  .era-card li b{{color:#e0e0e0}}

  /* Section dividers */
  .section-label{{background:#0d1520;border-left:3px solid #00bcd4;padding:6px 12px;margin:20px 0 12px;color:#80deea;font-size:12px;text-transform:uppercase;letter-spacing:1px}}

  /* Sticky header */
  .sticky-nav{{position:sticky;top:0;background:#060609ee;backdrop-filter:blur(8px);z-index:100;padding:8px 16px;border-bottom:1px solid #1a1a2a;display:flex;gap:16px;flex-wrap:wrap}}
  .sticky-nav a{{color:#00bcd4;text-decoration:none;font-size:12px}}
  .sticky-nav a:hover{{color:#00e5ff}}
</style>
</head>
<body>
<div class="sticky-nav">
  <a href="#overview">Overview</a>
  <a href="#timeline">Market Timeline</a>
  <a href="#heatmap">Era Heatmap</a>
  <a href="#survivors">Survivors</a>
  <a href="#scenarios">By Scenario</a>
  <a href="#era-tops">Era Winners</a>
  <a href="#fullhistory">Full History Top 30</a>
</div>

<div class="container">
<h1>⚡ ETH/USDT — COMPREHENSIVE HISTORICAL STRATEGY REPORT</h1>
<div class="subtitle">Aug 2017 → Jun 2026 &nbsp;|&nbsp; 3,212 daily bars &nbsp;|&nbsp; {len(STRATEGY_REGISTRY)} strategies &nbsp;|&nbsp; {len(ERAS)} market eras &nbsp;|&nbsp; All fees included</div>

<!-- ══ OVERVIEW KPIs ══════════════════════════════════════════════════════ -->
<a id="overview"></a>
<h2>📊 OVERVIEW</h2>
<div class="kpi-bar">
  <div class="kpi"><div class="val">{n_total:,}</div><div class="lbl">Backtest Combos</div></div>
  <div class="kpi"><div class="val">{n_valid}</div><div class="lbl">Valid Results (≥5 trades)</div></div>
  <div class="kpi"><div class="val">{n_pos}</div><div class="lbl">Positive Sharpe (Full)</div></div>
  <div class="kpi"><div class="val">{n_beat}</div><div class="lbl">Beat Buy &amp; Hold</div></div>
  <div class="kpi"><div class="val">{bnh_full:+,.0f}%</div><div class="lbl">ETH Buy &amp; Hold Return</div></div>
  <div class="kpi"><div class="val">{best_sh:.2f}</div><div class="lbl">Best Sharpe (Full History)</div></div>
  <div class="kpi"><div class="val">{len(survivors)}</div><div class="lbl">All-Weather Survivors</div></div>
  <div class="kpi"><div class="val">{len(ERAS)}</div><div class="lbl">Market Eras Tested</div></div>
</div>

<div class="section-label">ETH Buy &amp; Hold from $300 → $1,975 = {bnh_full:+,.0f}% — that's the bar every strategy must beat</div>

<!-- ══ MARKET TIMELINE ═══════════════════════════════════════════════════ -->
<a id="timeline"></a>
<h2>📅 MARKET TIMELINE — Every Major ETH Era</h2>
<table class="era-table">
  <thead>
    <tr>
      <th>Era</th>
      <th>Regime</th>
      <th>Price Range</th>
      <th>Top Strategies</th>
      <th>What Happened</th>
    </tr>
  </thead>
  <tbody>
    {"".join(era_timeline_rows)}
  </tbody>
</table>

<!-- ══ HEATMAP ══════════════════════════════════════════════════════════ -->
<a id="heatmap"></a>
<h2>🔥 STRATEGY × ERA SHARPE HEATMAP (Top 25 strategies)</h2>
<div class="hm-legend">
  <span style="background:#00695c;color:#fff">Sharpe &gt;1.5 (Excellent)</span>
  <span style="background:#2e7d32;color:#fff">Sharpe &gt;0.8 (Strong)</span>
  <span style="background:#558b2f;color:#fff">Sharpe &gt;0.3 (Good)</span>
  <span style="background:#827717;color:#fff">Sharpe &gt;0 (Marginal)</span>
  <span style="background:#b71c1c;color:#fff">Sharpe &lt;0 (Losing)</span>
  <span style="background:#4a0000;color:#fff">Sharpe &lt;-0.5 (Bad)</span>
</div>
<div class="hm-wrap">
<table class="hm-table">
  <thead><tr><th style="background:#0a0a0f;min-width:200px"></th>{heatmap_headers}</tr></thead>
  <tbody>{"".join(heatmap_body)}</tbody>
</table>
</div>

<!-- ══ SURVIVORS ════════════════════════════════════════════════════════ -->
<a id="survivors"></a>
<h2>🏆 ALL-WEATHER SURVIVORS — Profitable Across Multiple Eras</h2>
<div class="survivor-note">
  These strategies produced positive Sharpe ratios across the most market eras — bull runs, bear markets, ranging chop, and volatile explosions.
  A strategy is considered a "survivor" if it has Sharpe &gt; 0 in 4+ distinct market eras.
  These are the ones to trust with real capital because they've proven themselves across radically different conditions.
</div>
<table class="data-table">
  <thead>
    <tr><th>#</th><th>Strategy</th><th>Category</th><th>Profitable Eras</th>
    <th>Full Return</th><th>Full Sharpe</th><th>Profit Factor</th><th>Win Rate</th><th>Max DD</th><th>Trades</th></tr>
  </thead>
  <tbody>{"".join(survivor_rows)}</tbody>
</table>

<!-- ══ SCENARIO GUIDE ════════════════════════════════════════════════════ -->
<a id="scenarios"></a>
<h2>🎯 USE-CASE SCENARIO GUIDE — Best Strategy for Each Market Type</h2>
<div class="scenario-grid">
  {"".join(scenario_cards)}
</div>

<!-- ══ ERA WINNERS ══════════════════════════════════════════════════════ -->
<a id="era-tops"></a>
<h2>🏅 ERA-BY-ERA WINNERS — Top 5 Strategies for Each Period</h2>
<div class="era-top-grid">
"""

    for era in ERAS:
        tops = era_tops.get(era["id"], [])
        lo, hi = era_price_range(df_full, era)
        bias_badge = {"bull": "🟢", "bear": "🔴", "range": "🟡", "volatile": "🟣"}.get(era["bias"], "⚪")
        items = "".join(f"<li><b>{r['strategy']}</b> — Sh:{r['sharpe']:.2f} Ret:{r['total_return']:+.1f}% WR:{r['win_rate']:.0f}% PF:{r['profit_factor']:.2f}</li>" for r in tops[:5])
        html += f"""
        <div class="era-card" style="border-top:3px solid {era['color']}">
          <div class="era-card-title">{bias_badge} {era['label']}</div>
          <div class="era-card-sub">{era['start']} → {era['end']} &nbsp;|&nbsp; {lo} → {hi}</div>
          <ol>{items or '<li style="color:#444">No valid results</li>'}</ol>
        </div>"""

    html += f"""
</div>

<!-- ══ FULL HISTORY LEADERBOARD ══════════════════════════════════════════ -->
<a id="fullhistory"></a>
<h2>📋 FULL HISTORY LEADERBOARD — Top 30 (2017–2026, all {n_total:,} backtests)</h2>
<table class="data-table" id="lb">
  <thead>
    <tr><th>#</th><th>Strategy</th><th>Category</th><th>Return</th><th>Win Rate</th>
    <th>PF</th><th>Sharpe</th><th>Max DD</th><th>Trades</th><th>Beat B&amp;H</th></tr>
  </thead>
  <tbody>{"".join(lb_rows)}</tbody>
</table>

<div class="section-label" style="margin-top:40px;border-color:#444;color:#555">
  Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; All results after realistic fees (taker 0.05%, maker 0.02%, slippage, funding, borrow costs)
</div>

</div><!-- /container -->
</body>
</html>"""

    path = os.path.join(OUT_DIR, "historical_report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"\n✅  Report saved → {path}")

    # copy to dashboard
    import shutil
    dash_path = "/tmp/eth-dashboard/historical_report.html"
    shutil.copy(path, dash_path)
    print(f"✅  Dashboard copy → {dash_path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*70)
    print("  COMPREHENSIVE ETH HISTORICAL STRATEGY ANALYSIS")
    print("  Aug 2017 → Jun 2026  |  65 strategies  |  15 market eras")
    print("═"*70 + "\n")

    df_full = load_full_df()
    results = run_era_backtests(df_full)

    # save raw JSON
    json_path = os.path.join(OUT_DIR, "historical_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw data → {json_path}")

    generate_html(results, df_full)
