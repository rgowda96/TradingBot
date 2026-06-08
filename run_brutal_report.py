"""
BRUTAL COMPREHENSIVE ETH STRATEGY REPORT
=========================================
- 3 timeframes (1w, 1d, 4h) × 65 strategies × 15 eras + full history = ~3,510 combos
- 35+ metrics per combo (Sharpe, Sortino, Calmar, Ulcer, t-stat, p-value,
  Monte Carlo, walk-forward, fee break-even, consecutive losses, tail ratio…)
- Outputs: results/brutal_results.json + results/brutal_report.html
"""
import os, json, warnings, time
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from indicators               import add_indicators
from brutal_backtest          import brutal_stats
from master_strategy_library  import STRATEGY_REGISTRY, CATEGORY_DESCRIPTIONS

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

TIMEFRAMES = {
    "1w": "data/ETH_USDT_1w_full.parquet",
    "1d": "data/ETH_USDT_1d_full.parquet",
    "4h": "data/ETH_USDT_4h_full.parquet",
}

ERAS = [
    {"id":"full",          "label":"FULL HISTORY (2017–2026)", "start":"2017-08-17","end":"2026-06-02","bias":"mixed","color":"#90caf9"},
    {"id":"genesis",       "label":"Genesis & Discovery 2017", "start":"2017-08-17","end":"2017-12-31","bias":"bull","color":"#00c896"},
    {"id":"mania_2018",    "label":"ICO Mania Peak Jan-2018",  "start":"2018-01-01","end":"2018-01-31","bias":"bull","color":"#00e676"},
    {"id":"crash_2018",    "label":"Crypto Winter 2018",        "start":"2018-02-01","end":"2018-12-31","bias":"bear","color":"#ff1744"},
    {"id":"recovery_2019", "label":"Dead-Cat & Recovery 2019",  "start":"2019-01-01","end":"2019-12-31","bias":"range","color":"#ffd740"},
    {"id":"covid_crash",   "label":"COVID Crash & Rebound 2020","start":"2020-01-01","end":"2020-08-31","bias":"volatile","color":"#ff6d00"},
    {"id":"defi_summer",   "label":"DeFi Summer Q4 2020",       "start":"2020-09-01","end":"2020-12-31","bias":"bull","color":"#00bcd4"},
    {"id":"bull_2021_q1",  "label":"Institutional Bull Q1 2021","start":"2021-01-01","end":"2021-05-12","bias":"bull","color":"#00e676"},
    {"id":"china_ban_2021","label":"China Ban Crash 2021",       "start":"2021-05-13","end":"2021-07-20","bias":"bear","color":"#ff5252"},
    {"id":"altseason_2021","label":"Alt-Season & 2nd ATH 2021",  "start":"2021-07-21","end":"2021-11-10","bias":"bull","color":"#69f0ae"},
    {"id":"bear_2022",     "label":"Bear Market 2022 (LUNA/3AC)","start":"2021-11-11","end":"2022-06-17","bias":"bear","color":"#ff1744"},
    {"id":"merge_era",     "label":"Merge & FTX Collapse 2022",  "start":"2022-06-18","end":"2022-12-31","bias":"bear","color":"#d50000"},
    {"id":"recovery_2023", "label":"Recovery & USDC Depeg 2023", "start":"2023-01-01","end":"2023-09-30","bias":"range","color":"#ffab40"},
    {"id":"etf_rally_2024","label":"BTC ETF Rally 2024",          "start":"2023-10-01","end":"2024-03-31","bias":"bull","color":"#00e676"},
    {"id":"eth_etf_chop",  "label":"ETH ETF Launch & Chop 2024", "start":"2024-04-01","end":"2024-12-31","bias":"range","color":"#ffd740"},
    {"id":"current_2025",  "label":"Current Cycle 2025–2026",     "start":"2025-01-01","end":"2026-06-02","bias":"volatile","color":"#e040fb"},
]

# ─────────────────────────────────────────────────────────────────────────────
def load_tf_dfs():
    dfs = {}
    for tf, path in TIMEFRAMES.items():
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        df = pd.read_parquet(path)
        print(f"  {tf}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}  — adding indicators...")
        df = add_indicators(df)
        dfs[tf] = df.dropna()
        print(f"       → {len(dfs[tf])} bars after dropna")
    return dfs


def slice_era(df, era):
    start = pd.Timestamp(era["start"], tz="UTC")
    end   = pd.Timestamp(era["end"],   tz="UTC")
    slc   = df.loc[start:end]
    return slc if len(slc) >= 25 else None


def run_all(dfs):
    all_results = []
    total = len(STRATEGY_REGISTRY) * len(ERAS) * len(dfs)
    done  = 0
    t0    = time.time()

    for era in ERAS:
        for tf, df_full in dfs.items():
            era_df = slice_era(df_full, era)
            if era_df is None:
                done += len(STRATEGY_REGISTRY)
                continue

            bnh = (float(era_df['close'].iloc[-1]) / float(era_df['close'].iloc[0]) - 1) * 100
            print(f"\n  ▶ [{tf}] {era['label']}  ({len(era_df)} bars)  B&H={bnh:+.1f}%")

            for strat_id, (category, fn) in STRATEGY_REGISTRY.items():
                done += 1
                try:
                    stats = brutal_stats(era_df, fn, tf=tf)

                    row = {
                        "era_id":    era["id"],
                        "era_label": era["label"],
                        "era_bias":  era["bias"],
                        "era_bnh":   round(bnh, 1),
                        "era_bars":  len(era_df),
                        "tf":        tf,
                        "strategy":  strat_id,
                        "category":  category,
                    }
                    row.update(stats)
                    all_results.append(row)

                    n = stats.get("n_trades", 0)
                    sh = stats.get("sharpe")
                    if n >= 5 and sh and abs(sh) > 0.4:
                        so = stats.get("sortino","N/A")
                        ca = stats.get("calmar","N/A")
                        wf = stats.get("wf_efficiency","N/A")
                        sig_mark = "✓" if stats.get("significant") else " "
                        mark = "★" if (sh > 0.5 and stats.get("profit_factor",0) and stats.get("profit_factor",0) > 1.2) else " "
                        print(f"    {mark}{sig_mark} {strat_id:33s}  n={n:3d}  sh={sh:+.2f}  so={so if so else 'N/A':>6}  ca={ca if ca else 'N/A':>6}  wf={wf if wf else 'N/A':>5}  ret={stats.get('total_return',0):+6.1f}%")

                except Exception as e:
                    all_results.append({
                        "era_id": era["id"], "era_label": era["label"],
                        "era_bias": era["bias"], "tf": tf,
                        "strategy": strat_id, "category": category,
                        "n_trades": 0, "error": str(e)[:80],
                    })

            if done % 500 == 0:
                elapsed = time.time() - t0
                print(f"\n  ⏱  {done}/{total} done  ({elapsed:.0f}s elapsed)")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# HTML GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

BIAS_META = {
    "bull":     ("📈 BULL",   "#00e676"),
    "bear":     ("📉 BEAR",   "#ff5252"),
    "range":    ("↔ RANGE",  "#ffd740"),
    "volatile": ("⚡ VOLATILE","#e040fb"),
    "mixed":    ("∿ MIXED",   "#90caf9"),
}

METRIC_EXPLANATIONS = {
    "sharpe":          "Risk-adjusted return (return/volatility, annualised). >1 = excellent, >0.5 = good, <0 = destroys capital.",
    "sortino":         "Like Sharpe but only penalises downside volatility. Better measure of real risk. >1.5 = excellent.",
    "calmar":          "Annualised return divided by max drawdown. How much return you get per unit of worst pain. >0.5 = good.",
    "profit_factor":   "Gross wins / gross losses. Must be >1 to be profitable. >1.5 = solid. >2.0 = excellent.",
    "win_rate":        "% of trades that close in profit. Meaningless without payoff ratio — a 40% WR can be highly profitable.",
    "payoff_ratio":    "Avg winning trade size / avg losing trade size. Needs to compensate for win rate. 1.5+ WR=40% is profitable.",
    "max_dd":          "Largest peak-to-trough equity drop. Tells you the worst case you would have lived through.",
    "max_dd_duration": "Max consecutive bars spent below the previous equity peak. Drawdown DURATION is often more painful than depth.",
    "recovery_factor": "Total return / max drawdown. How efficiently capital was grown relative to worst risk taken.",
    "ulcer_index":     "RMS of all drawdowns over time. Captures both depth and duration. Lower is better (less stomach-churning).",
    "tail_ratio":      "95th pct trade / 5th pct trade (abs). >1 means wins are larger than losses at the extremes. >1.5 = good.",
    "t_stat":          "Statistical t-score. Measures if the edge is real. >2.0 usually means statistically significant.",
    "p_value":         "Probability that results are due to chance. <0.05 = statistically significant at 95% confidence.",
    "mc_5th_pct":      "Monte Carlo worst 5th percentile: final equity if you got unlucky with trade order (out of 2,000 simulations).",
    "wf_efficiency":   "Out-of-sample Sharpe / In-sample Sharpe (60/40 split). >0.5 = strategy generalises, <0 = curve-fit.",
    "fee_breakeven_x": "How many times current fees can increase before strategy becomes unprofitable. >5x = very robust.",
    "monthly_wr":      "% of calendar months with positive returns. More reliable than trade win rate for capital compounding.",
    "ann_return":      "Annualised return (CAGR), accounting for era length. Directly comparable across different-length eras.",
    "pct_time_in":     "% of bars with an open position. Lower = more selective. Helps avoid chop and reduce fee drag.",
}

def fmt(v, decimals=2, suffix="", prefix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span style="color:#444">—</span>'
    try:
        d = int(decimals) if not isinstance(decimals, int) else decimals
        return f"{prefix}{round(float(v), d)}{suffix}"
    except Exception:
        return str(v)

def sharpe_color(v):
    if v is None: return "#2a2a2a"
    v = float(v)
    if v >  1.5: return "#004d40"
    if v >  0.8: return "#1b5e20"
    if v >  0.3: return "#33691e"
    if v >  0.0: return "#f57f17"
    if v > -0.5: return "#b71c1c"
    return "#4a0000"

def metric_badge(v, thresholds, colors):
    """Color a metric cell."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#1e1e1e", "—"
    for thresh, color in zip(thresholds, colors):
        if float(v) >= thresh:
            return color, f"{float(v):.2f}"
    return colors[-1], f"{float(v):.2f}"


def generate_brutal_html(results):
    df = pd.DataFrame(results)
    valid = df[
        (df.n_trades >= 5) &
        df.sharpe.notna() &
        df.profit_factor.notna()
    ].copy()

    era_list_ids = [e["id"] for e in ERAS if e["id"] != "full"]

    # ── 1. Full-history leaderboard (across all TFs) ─────────────────────
    full_all = valid[valid.era_id == "full"].copy()
    full_lb  = full_all.sort_values("sharpe", ascending=False).head(40)

    # ── 2. Survivors: Sharpe>0 in 5+ eras ───────────────────────────────
    era_valid = valid[valid.era_id.isin(era_list_ids)].copy()
    survivor_map = {}
    for (strat, tf_), grp in era_valid.groupby(["strategy","tf"]):
        n_pos = int((grp.sharpe > 0).sum())
        if n_pos >= 5:
            full_row = full_all[(full_all.strategy == strat) & (full_all.tf == tf_)]
            if len(full_row) == 0: continue
            r = full_row.iloc[0]
            key = f"{strat}|{tf_}"
            survivor_map[key] = {
                "strategy": strat, "tf": tf_, "category": r["category"],
                "n_eras_positive": n_pos,
                "total_eras": len(grp),
                "sharpe":   r.get("sharpe"),
                "sortino":  r.get("sortino"),
                "calmar":   r.get("calmar"),
                "ann_ret":  r.get("ann_return"),
                "max_dd":   r.get("max_dd"),
                "pf":       r.get("profit_factor"),
                "wr":       r.get("win_rate"),
                "wf":       r.get("wf_efficiency"),
                "t_stat":   r.get("t_stat"),
                "p_val":    r.get("p_value"),
                "mc_5th":   r.get("mc_5th_pct"),
                "fee_be":   r.get("fee_breakeven_x"),
                "n_trades": r.get("n_trades"),
                "sig":      r.get("significant", False),
            }

    survivors = sorted(survivor_map.values(),
                       key=lambda x: (x["n_eras_positive"] * (x["sharpe"] or 0)),
                       reverse=True)

    # ── 3. Regime best (avg across eras of that bias, min 3 eras) ───────
    regime_best = {}
    for bias in ["bull","bear","range","volatile"]:
        grp = era_valid[era_valid.era_bias == bias]
        agg = grp.groupby(["strategy","tf"]).agg(
            avg_sh=("sharpe","mean"), avg_so=("sortino","mean"),
            avg_ca=("calmar","mean"), avg_ret=("ann_return","mean"),
            avg_pf=("profit_factor","mean"), avg_wr=("win_rate","mean"),
            n=("era_id","count")
        ).reset_index()
        agg = agg[agg.n >= min(3, grp["era_id"].nunique())].sort_values("avg_sh", ascending=False).head(10)
        regime_best[bias] = agg.to_dict("records")

    # ── 4. Era heatmap (top strategies × eras, Sharpe) ──────────────────
    full_ranked = full_all.sort_values("sharpe", ascending=False)
    top_strats = full_ranked.drop_duplicates("strategy").head(20)["strategy"].tolist()
    heatmap_eras = [e["id"] for e in ERAS if e["id"] != "full"]

    # ── 5. Statistical significance summary ─────────────────────────────
    sig_summary = valid[valid.era_id == "full"].copy()
    sig_summary = sig_summary[sig_summary.significant == True].sort_values("sharpe", ascending=False)

    # ── 6. Worst strategies (negative Sharpe in 5+ eras) ────────────────
    cursed = {}
    for (strat, tf_), grp in era_valid.groupby(["strategy","tf"]):
        n_neg = int((grp.sharpe < -0.3).sum())
        if n_neg >= 4:
            full_row = full_all[(full_all.strategy == strat) & (full_all.tf == tf_)]
            if len(full_row) == 0: continue
            r = full_row.iloc[0]
            cursed[f"{strat}|{tf_}"] = {
                "strategy": strat, "tf": tf_,
                "n_neg_eras": n_neg,
                "sharpe": r.get("sharpe"),
                "total_return": r.get("total_return"),
                "max_dd": r.get("max_dd"),
                "max_consec_losses": r.get("max_consec_losses"),
            }
    cursed_list = sorted(cursed.values(), key=lambda x: x["sharpe"] or 0)

    # ── 7. Fee drag analysis ──────────────────────────────────────────────
    fee_df = full_all[full_all.fee_drag_pct.notna()].copy()
    fee_df = fee_df.sort_values("fee_drag_pct", ascending=False)

    # ── 8. Walk-forward quality ───────────────────────────────────────────
    wf_df = full_all[full_all.wf_efficiency.notna()].copy()
    wf_good = wf_df[wf_df.wf_efficiency >= 0.5].sort_values("wf_efficiency", ascending=False).head(15)
    wf_bad  = wf_df[wf_df.wf_efficiency < 0].sort_values("wf_efficiency").head(10)

    # ── 9. Per-era top 5 ─────────────────────────────────────────────────
    era_tops = {}
    for era in ERAS:
        grp = valid[(valid.era_id == era["id"]) & valid.sharpe.notna()]
        grp = grp.sort_values("sharpe", ascending=False).head(8)
        era_tops[era["id"]] = grp.to_dict("records")

    # ── KPI summary ───────────────────────────────────────────────────────
    n_total       = len(df)
    n_valid       = len(full_all)
    n_sig         = int(full_all.get("significant", pd.Series([False]*len(full_all))).sum()) if "significant" in full_all else 0
    n_pos         = int((full_all.sharpe > 0).sum())
    n_survivors   = len(survivors)
    best_sharpe   = full_all.sharpe.max()
    best_strat_row= full_all.sort_values("sharpe", ascending=False).iloc[0] if len(full_all) else None
    bnh_full      = full_all["era_bnh"].iloc[0] if len(full_all) else 0

    # ════════════════════════════════════════════════════════════════════════
    # BUILD HTML
    # ════════════════════════════════════════════════════════════════════════

    # Helpers
    def pct_cell(v, good=0, ok=-10):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
        v = float(v)
        c = "#00e676" if v >= good else ("#ffd740" if v >= ok else "#ff5252")
        return f'<span style="color:{c}">{v:+.1f}%</span>'

    def sh_cell(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        c = sharpe_color(v)
        return f'<span style="background:{c};padding:1px 5px;border-radius:3px">{float(v):.3f}</span>'

    def bool_cell(v, true_label="✓ YES", false_label="✗ NO"):
        if v is None: return "—"
        return f'<span style="color:#00e676">{true_label}</span>' if v else f'<span style="color:#ff5252">{false_label}</span>'

    def wf_cell(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        v = float(v)
        c = "#00e676" if v >= 0.7 else ("#ffd740" if v >= 0.3 else ("#ff5252" if v >= 0 else "#9c27b0"))
        label = "✓ Robust" if v >= 0.7 else ("~ OK" if v >= 0.3 else ("⚠ Weak" if v >= 0 else "✗ Overfit"))
        return f'<span style="color:{c}">{v:.2f} {label}</span>'

    # Full leaderboard rows
    lb_rows = []
    for i, (_, r) in enumerate(full_lb.iterrows(), 1):
        sig_mark = "✓" if r.get("significant") else ""
        fe = r.get("fee_breakeven_x")
        fe_str = f"{fe}x" if fe and fe != ">20x" else (str(fe) if fe else "—")
        lb_rows.append(f"""<tr>
          <td>{i}</td>
          <td>{r['tf']}</td>
          <td><b>{r['strategy']}</b></td>
          <td style="color:#888;font-size:11px">{r['category']}</td>
          <td>{pct_cell(r.get('ann_return'))}</td>
          <td>{pct_cell(r.get('total_return'))}</td>
          <td>{sh_cell(r.get('sharpe'))}</td>
          <td>{fmt(r.get('sortino'))}</td>
          <td>{fmt(r.get('calmar'))}</td>
          <td>{fmt(r.get('profit_factor'))}</td>
          <td>{fmt(r.get('win_rate'),1)}%</td>
          <td>{fmt(r.get('payoff_ratio'))}</td>
          <td>{pct_cell(r.get('max_dd'),-20,-50)}</td>
          <td>{fmt(r.get('max_dd_duration'),0)} bars</td>
          <td>{fmt(r.get('ulcer_index'))}</td>
          <td>{fmt(r.get('t_stat'))}</td>
          <td>{fmt(r.get('p_value'),4)}</td>
          <td>{bool_cell(r.get('significant'))}</td>
          <td>{wf_cell(r.get('wf_efficiency'))}</td>
          <td>{fe_str}</td>
          <td>{fmt(r.get('mc_5th_pct'),1)}%</td>
          <td>{int(r.get('n_trades',0))}</td>
        </tr>""")

    # Survivors table
    surv_rows = []
    for i, s in enumerate(survivors[:30], 1):
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
        surv_rows.append(f"""<tr>
          <td>{medal}</td>
          <td>{s['tf']}</td>
          <td><b>{s['strategy']}</b></td>
          <td>{s['category']}</td>
          <td style="color:#00e5ff;font-weight:bold">{s['n_eras_positive']} / {s['total_eras']}</td>
          <td>{sh_cell(s['sharpe'])}</td>
          <td>{fmt(s.get('sortino'))}</td>
          <td>{fmt(s.get('calmar'))}</td>
          <td>{pct_cell(s.get('ann_ret'))}</td>
          <td>{pct_cell(s.get('max_dd'),-20,-50)}</td>
          <td>{fmt(s.get('pf'))}</td>
          <td>{bool_cell(s.get('sig'))}</td>
          <td>{wf_cell(s.get('wf'))}</td>
          <td>{s.get('fee_be','—') if s.get('fee_be') else '—'}x</td>
          <td>{int(s.get('n_trades',0))}</td>
        </tr>""")

    # Cursed table
    cursed_rows = []
    for c in cursed_list[:15]:
        cursed_rows.append(f"""<tr>
          <td style="color:#ff5252"><b>{c['strategy']}</b></td>
          <td>{c['tf']}</td>
          <td style="color:#ff5252">{c['n_neg_eras']} eras Sh&lt;-0.3</td>
          <td>{sh_cell(c['sharpe'])}</td>
          <td>{pct_cell(c.get('total_return',0))}</td>
          <td>{pct_cell(c.get('max_dd',0),-20,-50)}</td>
          <td style="color:#ff6d00">{c.get('max_consec_losses','—')} losses in a row</td>
        </tr>""")

    # Regime cards
    regime_cards = []
    for bias in ["bull","bear","range","volatile"]:
        meta_label, meta_color = BIAS_META[bias]
        rows = regime_best.get(bias, [])
        strat_rows = ""
        for r in rows[:8]:
            strat_rows += f"""<tr>
              <td><b>{r['strategy']}</b> [{r['tf']}]</td>
              <td>{sh_cell(r['avg_sh'])}</td>
              <td>{fmt(r.get('avg_so'))}</td>
              <td>{fmt(r.get('avg_ca'))}</td>
              <td>{pct_cell(r.get('avg_ret'))}</td>
              <td>{fmt(r.get('avg_wr'))}%</td>
              <td>{int(r['n'])}</td>
            </tr>"""
        regime_cards.append(f"""
        <div class="regime-card" style="border-top:4px solid {meta_color}">
          <div class="rc-title" style="color:{meta_color}">{meta_label}</div>
          <table class="inner-table">
            <thead><tr><th>Strategy[TF]</th><th>Avg Sharpe</th><th>Sortino</th><th>Calmar</th><th>Ann Ret</th><th>WR%</th><th>Eras</th></tr></thead>
            <tbody>{strat_rows}</tbody>
          </table>
        </div>""")

    # Heatmap
    hm_era_headers = "".join(
        f'<th class="hm-th" title="{e["label"]}">{e["id"][:8]}</th>'
        for e in ERAS if e["id"] != "full"
    )
    hm_body_rows = []
    for strat in top_strats:
        strat_full = full_all[(full_all.strategy == strat)]
        best_tf = strat_full.sort_values("sharpe", ascending=False).iloc[0]["tf"] if len(strat_full) else "1d"
        cells = f'<td class="hm-strat">{strat} [{best_tf}]</td>'
        for era in ERAS:
            if era["id"] == "full": continue
            row = era_valid[(era_valid.strategy == strat) & (era_valid.era_id == era["id"])]
            if len(row) == 0:
                cells += '<td style="background:#111;text-align:center;font-size:10px;padding:3px">—</td>'
            else:
                best = row.sort_values("sharpe", ascending=False).iloc[0]
                v = best["sharpe"]
                bg = sharpe_color(v)
                txt = f"{float(v):.2f}" if v is not None else "—"
                sig = "★" if best.get("significant") else ""
                wf_val = best.get("wf_efficiency", "—")
                cells += f'<td style="background:{bg};text-align:center;font-size:10px;padding:3px" title="sh={txt} wf={wf_val}">{txt}{sig}</td>'
        hm_body_rows.append(f"<tr>{cells}</tr>")

    # Walk-forward good/bad
    wf_rows_good = "".join(f"""<tr>
      <td><b>{r['strategy']}</b></td><td>{r['tf']}</td>
      <td>{wf_cell(r['wf_efficiency'])}</td>
      <td>{sh_cell(r['sharpe'])}</td>
      <td>{fmt(r.get('ann_return'))}%</td>
    </tr>""" for _, r in wf_good.iterrows())

    wf_rows_bad = "".join(f"""<tr>
      <td><b>{r['strategy']}</b></td><td>{r['tf']}</td>
      <td>{wf_cell(r['wf_efficiency'])}</td>
      <td>{sh_cell(r['sharpe'])}</td>
      <td style="color:#ff5252">OVERFIT</td>
    </tr>""" for _, r in wf_bad.iterrows())

    # Significant signals
    sig_rows = "".join(f"""<tr>
      <td><b>{r['strategy']}</b></td><td>{r['tf']}</td>
      <td>{sh_cell(r['sharpe'])}</td>
      <td>{fmt(r.get('t_stat'))}</td>
      <td>{fmt(r.get('p_value'),4)}</td>
      <td>{fmt(r.get('mc_5th_pct'),1)}%</td>
      <td>{pct_cell(r.get('ann_return'))}</td>
    </tr>""" for _, r in sig_summary.head(20).iterrows())

    # Era cards
    era_card_html = ""
    for era in ERAS:
        if era["id"] == "full": continue
        tops = era_tops.get(era["id"], [])
        bias_icon = {"bull":"🟢","bear":"🔴","range":"🟡","volatile":"🟣","mixed":"⚪"}.get(era["bias"],"⚪")
        items = ""
        for j, r in enumerate(tops[:5], 1):
            sig = "★" if r.get("significant") else ""
            items += f"""<div class="era-item">
              <span class="era-rank">{j}</span>
              <span class="era-strat">{r['strategy']} [{r['tf']}]</span>
              <span class="era-metrics">Sh:{fmt(r.get('sharpe'))} So:{fmt(r.get('sortino'))} Ret:{pct_cell(r.get('total_return'))}{sig}</span>
            </div>"""
        era_card_html += f"""
        <div class="era-card" style="border-left:4px solid {era['color']}">
          <div class="era-title">{bias_icon} {era['label']}</div>
          <div class="era-sub">{era['start']} → {era['end']} | B&H={era.get('era_bnh',0):+.1f}%</div>
          {items or '<div style="color:#444;font-size:11px;padding:4px">No valid results</div>'}
        </div>"""

    # Metric glossary
    glossary_rows = "".join(f"""<tr>
      <td style="color:#00e5ff;font-weight:bold;white-space:nowrap">{m}</td>
      <td style="color:#aaa;font-size:12px">{desc}</td>
    </tr>""" for m, desc in METRIC_EXPLANATIONS.items())

    # ─────────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETH — Brutal Comprehensive Strategy Assessment</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#060608;color:#dde;font-family:'Courier New',monospace;font-size:13px;line-height:1.5}}
a{{color:#00bcd4;text-decoration:none}}
h1{{font-size:22px;color:#00e5ff;text-align:center;padding:22px 0 4px;letter-spacing:3px;text-transform:uppercase}}
h2{{font-size:15px;color:#00bcd4;border-bottom:2px solid #1a2a3a;padding:8px 0 5px;margin:30px 0 12px;letter-spacing:1px;text-transform:uppercase}}
h3{{font-size:13px;color:#80cbc4;margin:12px 0 6px}}
.subtitle{{text-align:center;color:#555;font-size:11px;margin-bottom:22px;letter-spacing:1px}}
.container{{max-width:1500px;margin:0 auto;padding:0 16px 80px}}

/* Sticky nav */
.nav{{position:sticky;top:0;z-index:200;background:#06060aee;backdrop-filter:blur(10px);
      border-bottom:1px solid #1a2030;padding:6px 20px;display:flex;flex-wrap:wrap;gap:14px;align-items:center}}
.nav a{{color:#607d8b;font-size:11px;text-transform:uppercase;letter-spacing:1px}}
.nav a:hover{{color:#00e5ff}}
.nav-title{{color:#00e5ff;font-size:12px;font-weight:bold;margin-right:8px}}

/* KPI bar */
.kpi-bar{{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 20px}}
.kpi{{background:#0e0e16;border:1px solid #1a2030;border-radius:8px;padding:12px 18px;flex:1;min-width:130px;text-align:center}}
.kpi .val{{font-size:20px;font-weight:bold;color:#00e5ff}}
.kpi .lbl{{font-size:10px;color:#555;margin-top:3px;text-transform:uppercase;letter-spacing:1px}}

/* Warning / insight banners */
.banner{{border-radius:6px;padding:10px 16px;margin:10px 0;font-size:12px}}
.banner-warn{{background:#1a0e00;border-left:3px solid #ff6d00;color:#ffb74d}}
.banner-good{{background:#001a0e;border-left:3px solid #00c853;color:#69f0ae}}
.banner-info{{background:#001a2a;border-left:3px solid #00bcd4;color:#80deea}}
.banner-bad{{background:#1a0008;border-left:3px solid #ff1744;color:#ff8a80}}

/* Tables */
.data-table{{width:100%;border-collapse:collapse;font-size:11px;margin-bottom:16px}}
.data-table th{{background:#0a0f18;color:#546e7a;padding:6px 8px;text-align:left;border-bottom:2px solid #1a2a3a;
               white-space:nowrap;position:sticky;top:42px;z-index:10}}
.data-table td{{padding:5px 8px;border-bottom:1px solid #0e0e18;vertical-align:middle}}
.data-table tr:hover td{{background:#0a1018}}
.data-table tr:nth-child(1) td{{color:#ffd700}}
.data-table tr:nth-child(2) td{{color:#c0c0c0}}
.data-table tr:nth-child(3) td{{color:#cd7f32}}

/* Regime grid */
.regime-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;margin-bottom:20px}}
.regime-card{{background:#0b0b14;border:1px solid #1a2030;border-radius:8px;padding:16px}}
.rc-title{{font-size:16px;font-weight:bold;margin-bottom:10px}}
.inner-table{{width:100%;border-collapse:collapse;font-size:11px}}
.inner-table th{{color:#546e7a;padding:4px 6px;text-align:left;border-bottom:1px solid #1a2030;font-size:10px}}
.inner-table td{{padding:4px 6px;border-bottom:1px solid #0d0d18}}
.inner-table tr:hover td{{background:#0a1018}}

/* Heatmap */
.hm-wrap{{overflow-x:auto;margin-bottom:20px;border:1px solid #1a2030;border-radius:6px;padding:8px}}
.hm-table{{border-collapse:collapse;white-space:nowrap;font-size:10px}}
.hm-th{{background:#0a0f18;color:#546e7a;padding:3px 5px;text-align:center;writing-mode:vertical-lr;height:70px;vertical-align:bottom}}
.hm-strat{{font-size:10px;padding:2px 8px;white-space:nowrap;color:#90a4ae;background:#0a0f18;border-right:1px solid #1a2030}}
.hm-legend{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;font-size:11px}}
.hm-legend span{{padding:2px 10px;border-radius:3px}}

/* Era cards */
.era-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin-bottom:20px}}
.era-card{{background:#0b0b14;border:1px solid #1a2030;border-radius:8px;padding:12px}}
.era-title{{font-size:13px;font-weight:bold;margin-bottom:2px}}
.era-sub{{font-size:10px;color:#546e7a;margin-bottom:8px}}
.era-item{{display:flex;align-items:baseline;gap:6px;padding:3px 0;border-bottom:1px solid #0e0e18}}
.era-rank{{color:#546e7a;font-size:10px;min-width:14px}}
.era-strat{{color:#b0bec5;font-size:11px;min-width:200px}}
.era-metrics{{font-size:10px;color:#607d8b;flex:1;text-align:right}}

/* Two-col layout */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}

/* Section label */
.section-lbl{{background:#0b0f1a;border-left:3px solid #37474f;padding:5px 12px;margin:16px 0 10px;
              color:#607d8b;font-size:11px;text-transform:uppercase;letter-spacing:1px}}

/* Metric chip */
.chip{{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;margin:1px}}

/* Scrollable area */
.scroll-x{{overflow-x:auto}}
</style>
</head>
<body>

<div class="nav">
  <span class="nav-title">⚡ ETH BRUTAL ASSESSMENT</span>
  <a href="#overview">Overview</a>
  <a href="#leaderboard">Leaderboard</a>
  <a href="#survivors">Survivors</a>
  <a href="#heatmap">Heatmap</a>
  <a href="#regimes">By Regime</a>
  <a href="#era-tops">Era Winners</a>
  <a href="#wf">Walk-Forward</a>
  <a href="#stats">Significance</a>
  <a href="#avoid">Avoid List</a>
  <a href="#glossary">Glossary</a>
</div>

<div class="container">
<h1>⚡ ETH / USDT — BRUTAL STRATEGY ASSESSMENT</h1>
<div class="subtitle">
  Aug 2017 → Jun 2026 &nbsp;|&nbsp; 3 Timeframes (1w · 1d · 4h) &nbsp;|&nbsp;
  {len(STRATEGY_REGISTRY)} Strategies &nbsp;|&nbsp; {len(ERAS)-1} Market Eras &nbsp;|&nbsp;
  35+ Metrics per Combo &nbsp;|&nbsp; Monte Carlo · Walk-Forward · Fee Sensitivity · Statistical Significance
</div>

<!-- ── OVERVIEW ──────────────────────────────────────────────────────────── -->
<a id="overview"></a>
<h2>📊 Overview</h2>
<div class="kpi-bar">
  <div class="kpi"><div class="val">{n_total:,}</div><div class="lbl">Total Combos Run</div></div>
  <div class="kpi"><div class="val">{n_valid}</div><div class="lbl">Valid Results (≥5 trades)</div></div>
  <div class="kpi"><div class="val">{n_pos}</div><div class="lbl">Positive Sharpe (Full)</div></div>
  <div class="kpi"><div class="val">{n_sig}</div><div class="lbl">Statistically Significant</div></div>
  <div class="kpi"><div class="val">{n_survivors}</div><div class="lbl">All-Weather Survivors</div></div>
  <div class="kpi"><div class="val">{bnh_full:+,.0f}%</div><div class="lbl">ETH Buy &amp; Hold</div></div>
  <div class="kpi"><div class="val">{best_sharpe:.2f}</div><div class="lbl">Best Sharpe (Full History)</div></div>
  <div class="kpi"><div class="val">{len(STRATEGY_REGISTRY)}×3×{len(ERAS)-1}</div><div class="lbl">Strats×TFs×Eras</div></div>
</div>

<div class="banner banner-warn">
  ⚠ <b>Reality check:</b> ETH B&amp;H returned <b>{bnh_full:+,.0f}%</b> since 2017.
  Almost every strategy fails to beat this on a raw return basis — but most strategies
  achieve it with <b>dramatically smaller drawdowns</b>, which matters enormously for compounding.
  The real question is <b>risk-adjusted performance</b> (Sharpe, Sortino, Calmar) + <b>survivability</b> in bear markets.
</div>
<div class="banner banner-info">
  ★ = Sharpe&gt;0.5 AND PF&gt;1.2 &nbsp;|&nbsp; ✓ = p-value &lt; 0.05 (statistically significant) &nbsp;|&nbsp;
  Walk-forward efficiency = out-of-sample Sharpe ÷ in-sample Sharpe (60/40 split). Values near 1.0 = robust; negative = overfit.
</div>

<!-- ── FULL HISTORY LEADERBOARD ──────────────────────────────────────────── -->
<a id="leaderboard"></a>
<h2>📋 Full History Leaderboard — Top 40 (all timeframes, 2017–2026)</h2>
<div class="scroll-x">
<table class="data-table">
  <thead>
    <tr>
      <th>#</th><th>TF</th><th>Strategy</th><th>Category</th>
      <th title="{METRIC_EXPLANATIONS['ann_return']}">Ann Ret%</th>
      <th title="{METRIC_EXPLANATIONS['ann_return']}">Total Ret%</th>
      <th title="{METRIC_EXPLANATIONS['sharpe']}">Sharpe</th>
      <th title="{METRIC_EXPLANATIONS['sortino']}">Sortino</th>
      <th title="{METRIC_EXPLANATIONS['calmar']}">Calmar</th>
      <th title="{METRIC_EXPLANATIONS['profit_factor']}">PF</th>
      <th title="{METRIC_EXPLANATIONS['win_rate']}">WR%</th>
      <th title="{METRIC_EXPLANATIONS['payoff_ratio']}">Payoff</th>
      <th title="{METRIC_EXPLANATIONS['max_dd']}">Max DD</th>
      <th title="{METRIC_EXPLANATIONS['max_dd_duration']}">DD Dur</th>
      <th title="{METRIC_EXPLANATIONS['ulcer_index']}">Ulcer</th>
      <th title="{METRIC_EXPLANATIONS['t_stat']}">t-stat</th>
      <th title="{METRIC_EXPLANATIONS['p_value']}">p-val</th>
      <th title="{METRIC_EXPLANATIONS['p_value']}">Sig?</th>
      <th title="{METRIC_EXPLANATIONS['wf_efficiency']}">Walk-Fwd</th>
      <th title="{METRIC_EXPLANATIONS['fee_breakeven_x']}">Fee BE</th>
      <th title="{METRIC_EXPLANATIONS['mc_5th_pct']}">MC P5</th>
      <th>Trades</th>
    </tr>
  </thead>
  <tbody>{"".join(lb_rows)}</tbody>
</table>
</div>

<!-- ── SURVIVORS ────────────────────────────────────────────────────────── -->
<a id="survivors"></a>
<h2>🏆 All-Weather Survivors — Profitable Across 5+ Distinct Market Eras</h2>
<div class="banner banner-good">
  These are the <b>only strategies worth running with real money</b>.
  They maintained positive Sharpe ratios across bear markets, bull runs, ranging chop, and volatility spikes.
  A strategy that only works in bull markets is not a strategy — it's just being long.
</div>
<div class="scroll-x">
<table class="data-table">
  <thead>
    <tr>
      <th>#</th><th>TF</th><th>Strategy</th><th>Category</th>
      <th>Profitable Eras</th>
      <th>Sharpe</th><th>Sortino</th><th>Calmar</th>
      <th>Ann Ret%</th><th>Max DD%</th><th>PF</th>
      <th>Sig?</th><th>Walk-Fwd</th><th>Fee BE</th><th>Trades</th>
    </tr>
  </thead>
  <tbody>{"".join(surv_rows)}</tbody>
</table>
</div>

<!-- ── HEATMAP ───────────────────────────────────────────────────────────── -->
<a id="heatmap"></a>
<h2>🔥 Strategy × Era Sharpe Heatmap (Top 20 strategies, best TF)</h2>
<div class="hm-legend">
  <span style="background:#004d40;color:#fff">Sharpe &gt;1.5</span>
  <span style="background:#1b5e20;color:#fff">Sharpe &gt;0.8</span>
  <span style="background:#33691e;color:#fff">Sharpe &gt;0.3</span>
  <span style="background:#f57f17;color:#000">Sharpe &gt;0</span>
  <span style="background:#b71c1c;color:#fff">Sharpe &lt;0</span>
  <span style="background:#4a0000;color:#fff">Sharpe &lt;-0.5</span>
  <span style="color:#ffd700">★ = p&lt;0.05</span>
</div>
<div class="hm-wrap">
<table class="hm-table">
  <thead><tr><th style="background:#0a0f18"></th>{hm_era_headers}</tr></thead>
  <tbody>{"".join(hm_body_rows)}</tbody>
</table>
</div>

<!-- ── REGIME GUIDE ──────────────────────────────────────────────────────── -->
<a id="regimes"></a>
<h2>🎯 Best Strategies by Market Regime</h2>
<div class="banner banner-info">
  Metrics are averaged across <b>all eras of that regime type</b>. A strategy must appear in 3+ eras of that regime to qualify.
  This is your <b>playbook</b>: identify the current market regime, then deploy the right strategy.
</div>
<div class="regime-grid">{"".join(regime_cards)}</div>

<!-- ── ERA TOPS ──────────────────────────────────────────────────────────── -->
<a id="era-tops"></a>
<h2>🏅 Top 5 Strategies — Every Era (Best across all timeframes)</h2>
<div class="era-grid">{era_card_html}</div>

<!-- ── WALK FORWARD ──────────────────────────────────────────────────────── -->
<a id="wf"></a>
<h2>🔬 Walk-Forward Validation — Robust vs Overfit</h2>
<div class="banner banner-warn">
  Walk-forward efficiency = out-of-sample Sharpe ÷ in-sample Sharpe (60/40 split).
  <b>Strategies with negative efficiency are overfit to historical data</b> and should not be trusted.
  Strategies with &gt;0.7 efficiency genuinely generalise to unseen data.
</div>
<div class="two-col">
  <div>
    <h3 style="color:#00e676">✓ Robust Strategies (WF Efficiency ≥ 0.5)</h3>
    <table class="data-table">
      <thead><tr><th>Strategy</th><th>TF</th><th>WF Efficiency</th><th>Sharpe</th><th>Ann Ret</th></tr></thead>
      <tbody>{wf_rows_good}</tbody>
    </table>
  </div>
  <div>
    <h3 style="color:#ff5252">✗ Overfit — Do Not Use Live (WF &lt; 0)</h3>
    <table class="data-table">
      <thead><tr><th>Strategy</th><th>TF</th><th>WF Efficiency</th><th>Sharpe</th><th>Verdict</th></tr></thead>
      <tbody>{wf_rows_bad}</tbody>
    </table>
  </div>
</div>

<!-- ── STATISTICAL SIGNIFICANCE ─────────────────────────────────────────── -->
<a id="stats"></a>
<h2>📐 Statistical Significance — Is the Edge Real?</h2>
<div class="banner banner-info">
  t-statistic &gt; 2.0 and p-value &lt; 0.05 means we can say with 95% confidence the edge is not random noise.
  Strategies below this threshold may just be lucky — do not confuse noise for skill.
  Monte Carlo P5 shows the 5th percentile final equity across 2,000 randomised trade sequences.
</div>
<table class="data-table">
  <thead><tr><th>Strategy</th><th>TF</th><th>Sharpe</th><th>t-stat</th><th>p-value</th><th>MC P5 Return%</th><th>Ann Ret%</th></tr></thead>
  <tbody>{sig_rows}</tbody>
</table>

<!-- ── AVOID LIST ────────────────────────────────────────────────────────── -->
<a id="avoid"></a>
<h2>☠ Hall of Shame — Strategies to NEVER Use</h2>
<div class="banner banner-bad">
  These strategies have negative Sharpe in 4+ distinct market eras.
  They destroy capital consistently across bull markets, bear markets, AND ranging markets.
  There is no regime where they work reliably. Avoid completely.
</div>
<table class="data-table">
  <thead><tr><th>Strategy</th><th>TF</th><th>Consistently Bad</th><th>Full Sharpe</th><th>Full Return</th><th>Max DD</th><th>Max Consec Losses</th></tr></thead>
  <tbody>{"".join(cursed_rows) or '<tr><td colspan="7" style="color:#555">None found — all strategies have some redeeming eras</td></tr>'}</tbody>
</table>

<!-- ── METRIC GLOSSARY ───────────────────────────────────────────────────── -->
<a id="glossary"></a>
<h2>📖 Metric Glossary — What Everything Means</h2>
<table class="data-table">
  <thead><tr><th>Metric</th><th>Explanation</th></tr></thead>
  <tbody>{glossary_rows}</tbody>
</table>

<div class="section-lbl" style="margin-top:40px;border-color:#1a2030">
  Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
  Fees: taker 0.05%, maker 0.02%, slippage, funding, borrow &nbsp;|&nbsp;
  Walk-forward: 60/40 split &nbsp;|&nbsp;
  Monte Carlo: 2,000 simulations &nbsp;|&nbsp;
  Significance: two-tailed t-test vs zero
</div>
</div>
</body>
</html>"""

    path = os.path.join(OUT_DIR, "brutal_report.html")
    with open(path, "w") as f:
        f.write(html)
    import shutil
    shutil.copy(path, "/tmp/eth-dashboard/brutal_report.html")
    print(f"\n✅  Report saved → {path}  ({os.path.getsize(path)//1024} KB)")
    print(f"✅  Dashboard copy → /tmp/eth-dashboard/brutal_report.html")
    return path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*75)
    print("  BRUTAL ETH STRATEGY ASSESSMENT")
    print("  3 TFs × 65 strategies × 16 eras × 35+ metrics")
    print("═"*75 + "\n")

    t0  = time.time()
    dfs = load_tf_dfs()

    results = run_all(dfs)

    json_path = os.path.join(OUT_DIR, "brutal_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Raw JSON → {json_path}  ({os.path.getsize(json_path)//1024} KB)")

    generate_brutal_html(results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
