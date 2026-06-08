"""
MASTER UNIFIED BACKTEST + REPORT
=================================
1. Loads ALL previous result JSONs
2. Runs 145 extended strategies × 4 timeframes × 15 eras = ~8,700 new combos
3. Merges everything into results/master_unified.json
4. Generates results/unified_report.html — the single source of truth
"""
import os, json, warnings, time
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from indicators              import add_indicators
from brutal_backtest         import brutal_stats
from master_strategy_library import STRATEGY_REGISTRY   as ORIG_REGISTRY
from extended_strategy_library import EXTENDED_REGISTRY

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

TIMEFRAMES = {
    "1w": "data/ETH_USDT_1w_full.parquet",
    "1d": "data/ETH_USDT_1d_full.parquet",
    "4h": "data/ETH_USDT_4h_full.parquet",
    "1h": "data/ETH_USDT_1h_full.parquet",
}

ERAS = [
    {"id":"full",          "label":"Full History 2017–2026",     "start":"2017-08-17","end":"2026-06-02","bias":"mixed",   "color":"#90caf9"},
    {"id":"genesis",       "label":"Genesis 2017",                "start":"2017-08-17","end":"2017-12-31","bias":"bull",    "color":"#00c896"},
    {"id":"crash_2018",    "label":"Crypto Winter 2018",          "start":"2018-02-01","end":"2018-12-31","bias":"bear",    "color":"#ff1744"},
    {"id":"recovery_2019", "label":"Recovery 2019",               "start":"2019-01-01","end":"2019-12-31","bias":"range",   "color":"#ffd740"},
    {"id":"covid_crash",   "label":"COVID Crash 2020",            "start":"2020-01-01","end":"2020-08-31","bias":"volatile","color":"#ff6d00"},
    {"id":"defi_summer",   "label":"DeFi Summer Q4 2020",         "start":"2020-09-01","end":"2020-12-31","bias":"bull",    "color":"#00bcd4"},
    {"id":"bull_2021_q1",  "label":"Institutional Bull 2021",     "start":"2021-01-01","end":"2021-05-12","bias":"bull",    "color":"#00e676"},
    {"id":"china_ban_2021","label":"China Ban Crash 2021",        "start":"2021-05-13","end":"2021-07-20","bias":"bear",    "color":"#ff5252"},
    {"id":"altseason_2021","label":"Alt-Season 2nd ATH 2021",     "start":"2021-07-21","end":"2021-11-10","bias":"bull",    "color":"#69f0ae"},
    {"id":"bear_2022",     "label":"Bear 2022 LUNA/3AC",          "start":"2021-11-11","end":"2022-06-17","bias":"bear",    "color":"#ff1744"},
    {"id":"merge_era",     "label":"Merge + FTX Collapse 2022",   "start":"2022-06-18","end":"2022-12-31","bias":"bear",    "color":"#d50000"},
    {"id":"recovery_2023", "label":"Recovery 2023",               "start":"2023-01-01","end":"2023-09-30","bias":"range",   "color":"#ffab40"},
    {"id":"etf_rally_2024","label":"BTC ETF Rally 2024",          "start":"2023-10-01","end":"2024-03-31","bias":"bull",    "color":"#00e676"},
    {"id":"eth_etf_chop",  "label":"ETH ETF Chop 2024",          "start":"2024-04-01","end":"2024-12-31","bias":"range",   "color":"#ffd740"},
    {"id":"current_2025",  "label":"Current Cycle 2025–2026",     "start":"2025-01-01","end":"2026-06-02","bias":"volatile","color":"#e040fb"},
]

MASTER_JSON = os.path.join(OUT_DIR, "master_unified.json")

# ─────────────────────────────────────────────────────────────────────────────
def load_all_previous():
    """Load and normalise all previous result JSONs."""
    all_prev = []
    sources = [
        ("results/brutal_results.json",     "brutal"),
        ("results/historical_results.json", "historical"),
        ("results/master_results.json",     "master_strat"),
    ]
    for path, src in sources:
        if not os.path.exists(path): continue
        with open(path) as f:
            rows = json.load(f)
        # normalise field names across sources
        for r in rows:
            r.setdefault("source", src)
            r.setdefault("tf",  r.get("timeframe", "1d"))
            r.setdefault("era_id",   r.get("era_id",   "full"))
            r.setdefault("era_bias", r.get("era_bias",  "mixed"))
            r.setdefault("era_bnh",  r.get("era_bnh",   None))
            r.setdefault("sharpe",   r.get("sharpe",    None))
            r.setdefault("profit_factor", r.get("profit_factor", None))
            r.setdefault("total_return",  r.get("total_return",  r.get("total_return_pct", None)))
            r.setdefault("win_rate",      r.get("win_rate",      r.get("win_rate_pct",     None)))
            r.setdefault("max_dd",        r.get("max_dd",        r.get("max_drawdown_pct", None)))
            # Extended fields may be absent in older results
            for field in ["sortino","calmar","ulcer_index","t_stat","p_value","significant",
                          "wf_efficiency","mc_5th_pct","fee_breakeven_x","ann_return",
                          "monthly_wr","payoff_ratio","tail_ratio","max_consec_losses",
                          "max_dd_duration","recovery_factor","pct_time_in"]:
                r.setdefault(field, None)
        all_prev.extend(rows)
    print(f"  Loaded {len(all_prev):,} previous results from {len(sources)} files")
    return all_prev


def load_tf_dfs():
    dfs = {}
    for tf, path in TIMEFRAMES.items():
        if not os.path.exists(path):
            print(f"  MISSING {path}"); continue
        raw = pd.read_parquet(path)
        print(f"  Adding indicators: {tf} ({len(raw):,} bars)...", end=" ", flush=True)
        df  = add_indicators(raw).dropna()
        dfs[tf] = df
        print(f"→ {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")
    return dfs


def slice_era(df, era):
    s = pd.Timestamp(era["start"], tz="UTC")
    e = pd.Timestamp(era["end"],   tz="UTC")
    slc = df.loc[s:e]
    min_bars = {"1w":15, "1d":20, "4h":40, "1h":80}.get("1d", 20)
    return slc if len(slc) >= min_bars else None


def run_extended(dfs):
    """Run extended strategy library on all TFs and eras."""
    new_results = []
    registry    = EXTENDED_REGISTRY
    total = len(registry) * sum(len([era for era in ERAS
                                     if slice_era(dfs.get(tf, pd.DataFrame()), era) is not None])
                                for tf in dfs)
    done  = 0
    t0    = time.time()

    for era in ERAS:
        for tf, df_full in dfs.items():
            min_bars = {"1w":15,"1d":20,"4h":40,"1h":80}.get(tf,20)
            s = pd.Timestamp(era["start"], tz="UTC")
            e = pd.Timestamp(era["end"],   tz="UTC")
            era_df = df_full.loc[s:e]
            if len(era_df) < min_bars:
                done += len(registry)
                continue

            bnh = (float(era_df['close'].iloc[-1]) / float(era_df['close'].iloc[0]) - 1) * 100
            print(f"\n  ▶ [{tf}] {era['label']}  ({len(era_df):,} bars)  B&H={bnh:+.1f}%")

            for strat_id, (category, fn) in registry.items():
                done += 1
                try:
                    stats = brutal_stats(era_df, fn, tf=tf)
                    row = {
                        "source":    "extended",
                        "era_id":    era["id"],
                        "era_label": era["label"],
                        "era_bias":  era["bias"],
                        "era_bnh":   round(bnh,1),
                        "era_bars":  len(era_df),
                        "tf":        tf,
                        "strategy":  strat_id,
                        "category":  category,
                    }
                    row.update(stats)
                    new_results.append(row)

                    n  = stats.get("n_trades", 0)
                    sh = stats.get("sharpe")
                    if n >= 5 and sh and sh > 0.5:
                        so = stats.get("sortino")
                        wf = stats.get("wf_efficiency")
                        sig = "✓" if stats.get("significant") else " "
                        print(f"   ★{sig} {strat_id:40s}  sh={sh:+.2f}  so={so:.2f if so else 'N/A'}  "
                              f"wf={wf:.2f if wf else 'N/A'}  ret={stats.get('total_return',0):+.1f}%  n={n}")
                except Exception as ex:
                    new_results.append({
                        "source":"extended","era_id":era["id"],"era_label":era["label"],
                        "era_bias":era["bias"],"tf":tf,"strategy":strat_id,
                        "category":category,"n_trades":0,"error":str(ex)[:80]
                    })

            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            print(f"  ⏱  {done:,} done  |  {elapsed:.0f}s  |  ~{rate:.0f}/s")

    return new_results


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_unified_report(all_results):
    df = pd.DataFrame(all_results)

    # Standardise
    for col in ["sharpe","profit_factor","total_return","win_rate","max_dd",
                "sortino","calmar","ann_return","t_stat","p_value","wf_efficiency",
                "mc_5th_pct","fee_breakeven_x","payoff_ratio","ulcer_index",
                "max_dd_duration","max_consec_losses","monthly_wr","pct_time_in"]:
        if col not in df.columns: df[col] = None
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if "significant" not in df.columns: df["significant"] = False
    df["significant"] = df["significant"].fillna(False).astype(bool)

    valid = df[
        df.n_trades.notna() & (df.n_trades >= 5) &
        df.sharpe.notna()   & df.profit_factor.notna()
    ].copy()

    print(f"\n  Total rows: {len(df):,}  |  Valid: {len(valid):,}")

    era_ids = [e["id"] for e in ERAS if e["id"] != "full"]
    era_valid = valid[valid.era_id.isin(era_ids)].copy()

    # ── 1. Grand leaderboard: best result per (strategy, tf) over full history
    full = valid[valid.era_id == "full"].copy()
    full_lb = full.sort_values("sharpe", ascending=False).drop_duplicates(["strategy","tf"]).head(50)

    # ── 2. Survivors
    surv = []
    for (strat,tf_), grp in era_valid.groupby(["strategy","tf"]):
        n_pos = int((grp.sharpe > 0).sum())
        if n_pos < 5: continue
        best  = full[(full.strategy==strat)&(full.tf==tf_)]
        if not len(best): continue
        r = best.sort_values("sharpe",ascending=False).iloc[0]
        surv.append({
            "strategy":r.strategy,"tf":tf_,"category":r.category,
            "n_pos":n_pos,"total_eras":len(grp),
            "sharpe":r.sharpe,"sortino":r.sortino,"calmar":r.calmar,
            "ann_ret":r.ann_return,"max_dd":r.max_dd,"pf":r.profit_factor,
            "wr":r.win_rate,"wf":r.wf_efficiency,"sig":r.significant,
            "mc":r.mc_5th_pct,"fee_be":r.fee_breakeven_x,
            "trades":r.n_trades,"monthly_wr":r.monthly_wr,
            "t_stat":r.t_stat,"p_val":r.p_value,
            "payoff":r.payoff_ratio,"ulcer":r.ulcer_index,
            "max_cl":r.max_consec_losses,
        })
    surv.sort(key=lambda x: (x["n_pos"]*(x["sharpe"] or 0)), reverse=True)

    # ── 3. Regime best
    regime_best = {}
    for bias in ["bull","bear","range","volatile"]:
        grp = era_valid[era_valid.era_bias==bias]
        agg = grp.groupby(["strategy","tf"]).agg(
            avg_sh=("sharpe","mean"), avg_so=("sortino","mean"),
            avg_ca=("calmar","mean"), avg_ret=("ann_return","mean"),
            avg_pf=("profit_factor","mean"), avg_wr=("win_rate","mean"),
            n_eras=("era_id","count")
        ).reset_index()
        agg = agg[agg.n_eras >= 2].sort_values("avg_sh",ascending=False).head(10)
        regime_best[bias] = agg.to_dict("records")

    # ── 4. Statistical significance
    sig_valid = valid[(valid.era_id=="full") & (valid.significant==True)].sort_values("sharpe",ascending=False)

    # ── 5. Heatmap (top 30 strategies × eras)
    top30 = full.dropna(subset=["strategy"]).sort_values("sharpe",ascending=False).drop_duplicates("strategy").head(30)["strategy"].astype(str).tolist()

    # ── 6. Category performance
    cat_agg = full.groupby("category").agg(
        avg_sh=("sharpe","mean"), best_sh=("sharpe","max"),
        n_pos=("sharpe", lambda x: (x>0).sum()),
        n_total=("sharpe","count"),
        avg_ret=("total_return","mean"), avg_dd=("max_dd","mean"),
    ).reset_index().sort_values("avg_sh",ascending=False)

    # ── 7. Era-by-era winners
    era_tops = {}
    for era in ERAS:
        grp = valid[(valid.era_id==era["id"])&valid.sharpe.notna()]
        era_tops[era["id"]] = grp.sort_values("sharpe",ascending=False).head(8).to_dict("records")

    # ── 8. Never-use list
    shame = []
    for (strat,tf_), grp in era_valid.groupby(["strategy","tf"]):
        n_neg = int((grp.sharpe < -0.3).sum())
        if n_neg < 4: continue
        best = full[(full.strategy==strat)&(full.tf==tf_)]
        if not len(best): continue
        r = best.sort_values("sharpe").iloc[0]
        shame.append({"strategy":strat,"tf":tf_,"n_neg":n_neg,
                      "sharpe":r.sharpe,"total_return":r.total_return,
                      "max_dd":r.max_dd,"max_cl":r.max_consec_losses})
    shame.sort(key=lambda x: x["sharpe"] or 0)

    # ── 9. Walk-forward: robust vs overfit
    wf_df = full[full.wf_efficiency.notna()].copy()
    wf_robust = wf_df[wf_df.wf_efficiency >= 0.6].sort_values("wf_efficiency",ascending=False).head(20)
    wf_bad    = wf_df[wf_df.wf_efficiency < 0].sort_values("wf_efficiency").head(15)

    # ── 10. Fee sensitivity breakdown
    fee_df = full[full.fee_breakeven_x.notna()].copy()
    fee_df["fee_be_num"] = pd.to_numeric(
        fee_df.fee_breakeven_x.astype(str).str.replace(">",""), errors='coerce')
    fee_robust = fee_df[fee_df.fee_be_num >= 5].sort_values("fee_be_num",ascending=False).head(20)

    # ── Summary KPIs
    n_combos   = len(df)
    n_valid_f  = len(full)
    n_pos      = int((full.sharpe>0).sum())
    n_sig      = int(full.significant.sum())
    n_surv     = len(surv)
    best_sh    = full.sharpe.max()
    bnh        = float(valid[valid.era_id=="full"].era_bnh.dropna().iloc[0]) if len(valid[valid.era_id=="full"]) else 0
    n_strats   = df.strategy.nunique()
    n_tfs_used = df.tf.nunique()

    # ════════════════════════════════════════════════════════════════════════
    # RENDER HELPERS
    # ════════════════════════════════════════════════════════════════════════

    def sh_color(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "#111"
        v=float(v)
        if v>1.5: return "#004d40"
        if v>0.8: return "#1b5e20"
        if v>0.3: return "#33691e"
        if v>0.0: return "#5d4037"
        if v>-0.5: return "#b71c1c"
        return "#4a0000"

    def f2(v, dec=2, sfx=""):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        try: return f"{round(float(v),dec)}{sfx}"
        except: return str(v)

    def pct(v, good=0, bad=-15):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        v=float(v)
        c="#00e676" if v>=good else ("#ffd740" if v>=bad else "#ff5252")
        return f'<span style="color:{c}">{v:+.1f}%</span>'

    def shcell(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        return f'<span style="background:{sh_color(v)};padding:1px 6px;border-radius:3px;font-size:11px">{float(v):.3f}</span>'

    def wfcell(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        v=float(v)
        if v>=0.8: lb,c="✓ Robust","#00e676"
        elif v>=0.4: lb,c="~ OK","#ffd740"
        elif v>=0.0: lb,c="⚠ Weak","#ff9800"
        else: lb,c="✗ Overfit","#ff5252"
        return f'<span style="color:{c}">{v:.2f} {lb}</span>'

    def boolcell(v):
        return '<span style="color:#00e676">✓ YES</span>' if v else '<span style="color:#666">—</span>'

    # Grand leaderboard rows
    lb_rows = []
    for i,(_, r) in enumerate(full_lb.iterrows(),1):
        fe = r.fee_breakeven_x
        fe_s = f"{fe}x" if fe and str(fe)!=">20x" else (str(fe) if fe else "—")
        lb_rows.append(f"""<tr>
          <td style="color:#666">{i}</td>
          <td><span style="color:#546e7a;font-size:10px">{r.tf}</span></td>
          <td><b style="color:#e0e0e0">{r.strategy}</b></td>
          <td style="color:#78909c;font-size:10px">{r.category}</td>
          <td>{pct(r.ann_return)}</td>
          <td>{pct(r.total_return)}</td>
          <td>{shcell(r.sharpe)}</td>
          <td>{f2(r.sortino)}</td>
          <td>{f2(r.calmar)}</td>
          <td>{f2(r.profit_factor)}</td>
          <td>{f2(r.win_rate,1)}%</td>
          <td>{f2(r.payoff_ratio)}</td>
          <td>{f2(r.monthly_wr,1)}%</td>
          <td>{pct(r.max_dd,-5,-25)}</td>
          <td style="color:#78909c">{f2(r.max_dd_duration,0)} bars</td>
          <td>{f2(r.ulcer_index,3)}</td>
          <td>{f2(r.t_stat)}</td>
          <td>{f2(r.p_value,4)}</td>
          <td>{boolcell(r.significant)}</td>
          <td>{wfcell(r.wf_efficiency)}</td>
          <td style="color:#ffd740">{fe_s}</td>
          <td>{f2(r.mc_5th_pct,1)}%</td>
          <td style="color:#546e7a">{int(r.n_trades)}</td>
        </tr>""")

    # Survivors
    surv_rows = []
    for i,s in enumerate(surv[:35],1):
        medal=["🥇","🥈","🥉"][i-1] if i<=3 else f"#{i}"
        surv_rows.append(f"""<tr>
          <td>{medal}</td>
          <td><span style="color:#546e7a">{s['tf']}</span></td>
          <td><b>{s['strategy']}</b></td>
          <td style="color:#78909c;font-size:10px">{s['category']}</td>
          <td style="color:#00e5ff;font-weight:bold">{s['n_pos']}/{s['total_eras']}</td>
          <td>{shcell(s['sharpe'])}</td>
          <td>{f2(s.get('sortino'))}</td>
          <td>{f2(s.get('calmar'))}</td>
          <td>{pct(s.get('ann_ret'))}</td>
          <td>{pct(s.get('max_dd'),-5,-25)}</td>
          <td>{f2(s.get('pf'))}</td>
          <td>{f2(s.get('wr'),1)}%</td>
          <td>{f2(s.get('monthly_wr'),1)}%</td>
          <td>{f2(s.get('payoff'))}</td>
          <td>{f2(s.get('ulcer'),3)}</td>
          <td>{f2(s.get('t_stat'))}</td>
          <td>{f2(s.get('p_val'),4)}</td>
          <td>{boolcell(s.get('sig'))}</td>
          <td>{wfcell(s.get('wf'))}</td>
          <td style="color:#ffd740">{s.get('fee_be','—')}x</td>
          <td>{f2(s.get('mc'),1)}%</td>
          <td style="color:#546e7a">{int(s.get('trades',0))}</td>
        </tr>""")

    # Regime cards
    regime_cards = ""
    for bias, meta, color in [
        ("bull","📈 BULL MARKET","#00e676"),
        ("bear","📉 BEAR MARKET","#ff5252"),
        ("range","↔ RANGING","#ffd740"),
        ("volatile","⚡ VOLATILE","#e040fb"),
    ]:
        rows = regime_best.get(bias,[])
        inner = "".join(f"""<tr>
          <td><b>{r['strategy']}</b> [{r['tf']}]</td>
          <td>{shcell(r['avg_sh'])}</td>
          <td>{f2(r.get('avg_so'))}</td>
          <td>{f2(r.get('avg_ca'))}</td>
          <td>{pct(r.get('avg_ret'))}</td>
          <td>{f2(r.get('avg_wr'),1)}%</td>
          <td style="color:#546e7a">{int(r['n_eras'])}</td>
        </tr>""" for r in rows)
        regime_cards += f"""
        <div class="regime-card" style="border-top:4px solid {color}">
          <div style="color:{color};font-size:15px;font-weight:bold;margin-bottom:8px">{meta}</div>
          <table class="inner-t">
            <thead><tr><th>Strategy [TF]</th><th>Avg Sharpe</th><th>Sortino</th><th>Calmar</th><th>Ann Ret</th><th>WR%</th><th>Eras</th></tr></thead>
            <tbody>{inner}</tbody>
          </table>
        </div>"""

    # Heatmap
    hm_heads = "".join(f'<th class="hm-th" title="{e["label"]}">{e["id"][:7]}</th>'
                       for e in ERAS if e["id"]!="full")
    hm_rows = []
    for strat in top30:
        best_tf = full[full.strategy==strat].sort_values("sharpe",ascending=False).iloc[0]["tf"] if len(full[full.strategy==strat]) else "1d"
        cells = f'<td class="hm-strat">{strat[:38]} [{best_tf}]</td>'
        for era in ERAS:
            if era["id"]=="full": continue
            row = era_valid[(era_valid.strategy==strat)&(era_valid.era_id==era["id"])]
            if not len(row):
                cells += '<td style="background:#0a0a0a;text-align:center;padding:3px;font-size:9px">—</td>'
            else:
                v = row.sort_values("sharpe",ascending=False).iloc[0]["sharpe"]
                sig = "★" if row.iloc[0].get("significant") else ""
                cells += f'<td style="background:{sh_color(v)};text-align:center;padding:3px;font-size:9px">{f2(v)}{sig}</td>'
        hm_rows.append(f"<tr>{cells}</tr>")

    # Category table
    cat_rows = "".join(f"""<tr>
      <td><b>{r['category']}</b></td>
      <td>{shcell(r['avg_sh'])}</td>
      <td>{shcell(r['best_sh'])}</td>
      <td style="color:#00e676">{int(r['n_pos'])}</td>
      <td style="color:#546e7a">{int(r['n_total'])}</td>
      <td>{pct(r['avg_ret'])}</td>
      <td>{pct(r['avg_dd'],-5,-25)}</td>
    </tr>""" for _,r in cat_agg.iterrows())

    # Era cards
    era_card_html = ""
    for era in ERAS:
        if era["id"]=="full": continue
        tops = era_tops.get(era["id"],[])
        icon = {"bull":"🟢","bear":"🔴","range":"🟡","volatile":"🟣","mixed":"⚪"}.get(era["bias"],"⚪")
        items = ""
        for j,r in enumerate(tops[:5],1):
            sig = "★" if r.get("significant") else ""
            items += f"""<div class="ei">
              <span class="ern">{j}</span>
              <span class="ers">{r.get('strategy','?')} [{r.get('tf','?')}]</span>
              <span class="erm">Sh:{f2(r.get('sharpe'))} So:{f2(r.get('sortino'))} {pct(r.get('total_return'))}{sig}</span>
            </div>"""
        era_card_html += f"""
        <div class="era-card" style="border-left:4px solid {era['color']}">
          <div class="era-t">{icon} {era['label']}</div>
          <div class="era-s">{era['start']} → {era['end']}</div>
          {items or '<div style="color:#333;font-size:11px">No valid results</div>'}
        </div>"""

    # Significance table
    sig_rows = "".join(f"""<tr>
      <td><b>{r['strategy']}</b></td><td>{r['tf']}</td>
      <td>{shcell(r['sharpe'])}</td>
      <td>{f2(r['t_stat'])}</td>
      <td style="color:#00e676">{f2(r['p_value'],4)}</td>
      <td>{f2(r.get('mc_5th_pct'),1)}%</td>
      <td>{pct(r.get('ann_return'))}</td>
      <td>{f2(r.get('wf_efficiency'))}</td>
    </tr>""" for _,r in sig_valid.head(25).iterrows())

    # Walk-forward
    wf_good_rows = "".join(f"""<tr>
      <td><b>{r['strategy']}</b></td><td>{r['tf']}</td>
      <td>{wfcell(r['wf_efficiency'])}</td>
      <td>{shcell(r['sharpe'])}</td>
      <td>{pct(r.get('ann_return'))}</td>
    </tr>""" for _,r in wf_robust.iterrows())

    wf_bad_rows = "".join(f"""<tr>
      <td><b style="color:#ff5252">{r['strategy']}</b></td><td>{r['tf']}</td>
      <td>{wfcell(r['wf_efficiency'])}</td>
      <td>{shcell(r['sharpe'])}</td>
      <td style="color:#ff5252">OVERFIT</td>
    </tr>""" for _,r in wf_bad.iterrows())

    # Fee robust
    fee_rows = "".join(f"""<tr>
      <td><b>{r['strategy']}</b></td><td>{r['tf']}</td>
      <td style="color:#ffd740;font-weight:bold">{r.get('fee_breakeven_x','—')}x</td>
      <td>{shcell(r['sharpe'])}</td>
      <td>{pct(r.get('ann_return'))}</td>
      <td>{r.category}</td>
    </tr>""" for _,r in fee_robust.iterrows())

    # Shame list
    shame_rows = "".join(f"""<tr>
      <td><b style="color:#ff5252">{s['strategy']}</b></td>
      <td>{s['tf']}</td>
      <td style="color:#ff5252">{s['n_neg']} eras Sh&lt;-0.3</td>
      <td>{shcell(s['sharpe'])}</td>
      <td>{pct(s.get('total_return',0))}</td>
      <td>{pct(s.get('max_dd',0),-5,-25)}</td>
      <td style="color:#ff6d00">{s.get('max_cl','—')} losses in a row</td>
    </tr>""" for s in shame[:20])

    # ════════════════════════════════════════════════════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETH — Master Unified Strategy Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#05050a;color:#dde;font-family:'Courier New',monospace;font-size:12px;line-height:1.5}}
a{{color:#00bcd4;text-decoration:none}}
h1{{font-size:20px;color:#00e5ff;text-align:center;padding:20px 0 3px;letter-spacing:3px;text-transform:uppercase}}
h2{{font-size:14px;color:#00bcd4;border-bottom:1px solid #1a2a3a;padding:7px 0 5px;margin:26px 0 10px;letter-spacing:1px}}
.sub{{text-align:center;color:#444;font-size:11px;margin-bottom:20px;letter-spacing:1px}}
.wrap{{max-width:1600px;margin:0 auto;padding:0 14px 80px}}
.nav{{position:sticky;top:0;z-index:200;background:#05050aee;backdrop-filter:blur(10px);
      border-bottom:1px solid #111;padding:5px 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}}
.nav a{{color:#546e7a;font-size:10px;text-transform:uppercase;letter-spacing:1px}}
.nav a:hover{{color:#00e5ff}}
.nav-lbl{{color:#00e5ff;font-size:11px;font-weight:bold;margin-right:6px}}

.kpi-bar{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px}}
.kpi{{background:#0c0c14;border:1px solid #1a2030;border-radius:6px;padding:10px 16px;flex:1;min-width:120px;text-align:center}}
.kpi .v{{font-size:18px;font-weight:bold;color:#00e5ff}}
.kpi .l{{font-size:9px;color:#444;margin-top:2px;text-transform:uppercase;letter-spacing:1px}}

.banner{{border-radius:5px;padding:9px 14px;margin:8px 0;font-size:11px;line-height:1.6}}
.bw{{background:#1a0e00;border-left:3px solid #ff6d00;color:#ffb74d}}
.bg{{background:#001a0e;border-left:3px solid #00c853;color:#69f0ae}}
.bi{{background:#001020;border-left:3px solid #00bcd4;color:#80deea}}
.br{{background:#1a0008;border-left:3px solid #ff1744;color:#ff8a80}}

.dt{{width:100%;border-collapse:collapse;font-size:11px;margin-bottom:14px}}
.dt th{{background:#080c14;color:#455a64;padding:5px 7px;text-align:left;
        border-bottom:1px solid #1a2a3a;white-space:nowrap;position:sticky;top:40px;z-index:5}}
.dt td{{padding:4px 7px;border-bottom:1px solid #0a0a14;vertical-align:middle}}
.dt tr:hover td{{background:#08101a}}
.dt tr:nth-child(1) td{{color:#ffd700}}
.dt tr:nth-child(2) td{{color:#c0c0c0}}
.dt tr:nth-child(3) td{{color:#cd7f32}}

.rg-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px;margin-bottom:18px}}
.regime-card{{background:#0a0a14;border:1px solid #1a2030;border-radius:7px;padding:14px}}
.inner-t{{width:100%;border-collapse:collapse;font-size:10px}}
.inner-t th{{color:#455a64;padding:3px 5px;text-align:left;border-bottom:1px solid #1a2030;font-size:9px}}
.inner-t td{{padding:3px 5px;border-bottom:1px solid #0c0c14}}

.hm-wrap{{overflow-x:auto;margin-bottom:16px;border:1px solid #1a2030;border-radius:5px;padding:6px}}
.hm-t{{border-collapse:collapse;white-space:nowrap;font-size:9px}}
.hm-th{{background:#080c14;color:#455a64;padding:2px 4px;text-align:center;
        writing-mode:vertical-lr;height:65px;vertical-align:bottom;min-width:50px}}
.hm-strat{{font-size:10px;padding:2px 7px;white-space:nowrap;color:#78909c;
           background:#080c14;border-right:1px solid #1a2030}}
.hm-leg{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;font-size:10px}}
.hm-leg span{{padding:1px 8px;border-radius:3px}}

.era-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:10px;margin-bottom:16px}}
.era-card{{background:#0a0a14;border:1px solid #1a2030;border-radius:7px;padding:11px}}
.era-t{{font-size:12px;font-weight:bold;margin-bottom:2px}}
.era-s{{font-size:9px;color:#455a64;margin-bottom:7px}}
.ei{{display:flex;align-items:baseline;gap:5px;padding:2px 0;border-bottom:1px solid #0c0c14}}
.ern{{color:#455a64;font-size:9px;min-width:13px}}
.ers{{color:#90a4ae;font-size:10px;min-width:180px}}
.erm{{font-size:9px;color:#546e7a;flex:1;text-align:right}}

.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}}
.sx{{overflow-x:auto}}
.lbl{{background:#080c14;border-left:3px solid #37474f;padding:4px 10px;margin:14px 0 8px;
      color:#546e7a;font-size:10px;text-transform:uppercase;letter-spacing:1px}}
</style>
</head><body>

<div class="nav">
  <span class="nav-lbl">⚡ ETH MASTER REPORT</span>
  <a href="#kpis">KPIs</a>
  <a href="#leaderboard">Leaderboard</a>
  <a href="#survivors">Survivors</a>
  <a href="#heatmap">Heatmap</a>
  <a href="#regimes">Regimes</a>
  <a href="#era-tops">Era Winners</a>
  <a href="#categories">Categories</a>
  <a href="#significance">Significance</a>
  <a href="#walkfwd">Walk-Fwd</a>
  <a href="#fees">Fee Robust</a>
  <a href="#shame">Avoid</a>
</div>

<div class="wrap">
<h1>⚡ ETH / USDT — MASTER UNIFIED STRATEGY REPORT</h1>
<div class="sub">
  Aug 2017 → Jun 2026 &nbsp;|&nbsp;
  {n_strats} unique strategies &nbsp;|&nbsp;
  {n_tfs_used} timeframes (1w · 1d · 4h · 1h) &nbsp;|&nbsp;
  {len(ERAS)-1} market eras &nbsp;|&nbsp;
  {n_combos:,} total backtest combos &nbsp;|&nbsp;
  35+ metrics including Monte Carlo · Walk-Forward · Statistical Significance · Fee Break-Even
</div>

<a id="kpis"></a>
<h2>📊 Summary</h2>
<div class="kpi-bar">
  <div class="kpi"><div class="v">{n_combos:,}</div><div class="l">Total Combos</div></div>
  <div class="kpi"><div class="v">{n_valid_f:,}</div><div class="l">Valid (≥5 trades)</div></div>
  <div class="kpi"><div class="v">{n_pos}</div><div class="l">Positive Sharpe</div></div>
  <div class="kpi"><div class="v">{n_sig}</div><div class="l">Statistically Sig</div></div>
  <div class="kpi"><div class="v">{n_surv}</div><div class="l">All-Weather Survivors</div></div>
  <div class="kpi"><div class="v">{bnh:+,.0f}%</div><div class="l">ETH Buy &amp; Hold</div></div>
  <div class="kpi"><div class="v">{best_sh:.2f}</div><div class="l">Best Sharpe</div></div>
  <div class="kpi"><div class="v">{n_strats}</div><div class="l">Unique Strategies</div></div>
</div>
<div class="banner bi">
  <b>How to read this report:</b>
  Sharpe &gt;0.5 = good edge. Sharpe &gt;1.0 = excellent. <b>Sortino</b> penalises only downside volatility — prefer it over Sharpe.
  <b>Calmar</b> = annual return ÷ max drawdown — the higher the better.
  <b>Walk-forward efficiency</b> &gt;0.6 means the strategy works on unseen data, not just in-sample.
  <b>Fee break-even</b> = how many times fees can increase before strategy fails.
  <b>★</b> = p-value &lt;0.05 (statistically significant at 95% confidence).
</div>
<div class="banner bw">
  ⚠ <b>ETH Buy &amp; Hold = {bnh:+,.0f}%</b> since Aug 2017.
  Raw returns don't tell the full story — what matters is <b>risk-adjusted</b> performance and <b>surviving drawdowns</b>.
  Mean reversion strategies in a trending asset fail repeatedly. Trend-following dominates ETH.
</div>

<a id="leaderboard"></a>
<h2>🏅 Grand Leaderboard — Top 50 (all strategies, all timeframes, full history)</h2>
<div class="sx"><table class="dt">
  <thead><tr>
    <th>#</th><th>TF</th><th>Strategy</th><th>Category</th>
    <th title="Annualised CAGR">Ann%</th>
    <th title="Total cumulative return">Total%</th>
    <th title="Risk-adjusted return">Sharpe</th>
    <th title="Downside-only Sharpe">Sortino</th>
    <th title="Ann return / Max DD">Calmar</th>
    <th title="Gross wins / Gross losses">PF</th>
    <th title="% trades profitable">WR%</th>
    <th title="Avg win / Avg loss">Payoff</th>
    <th title="% months profitable">Mon WR</th>
    <th title="Worst peak-to-trough">Max DD</th>
    <th title="Bars below peak">DD Dur</th>
    <th title="RMS of all drawdowns">Ulcer</th>
    <th title="t-statistic vs zero">t-stat</th>
    <th title="Probability due to chance">p-val</th>
    <th title="p &lt; 0.05">Sig?</th>
    <th title="OOS Sharpe / IS Sharpe">Walk-Fwd</th>
    <th title="Fee multiples before failure">Fee BE</th>
    <th title="Monte Carlo 5th pct">MC P5</th>
    <th>Trades</th>
  </tr></thead>
  <tbody>{"".join(lb_rows)}</tbody>
</table></div>

<a id="survivors"></a>
<h2>🏆 All-Weather Survivors — Profitable in 5+ Eras (across all market conditions)</h2>
<div class="banner bg">
  The <b>only strategies to deploy with real capital</b>.
  These maintained positive Sharpe ratios across bull runs, bear crashes, ranging chop, and explosive volatility.
  A strategy that only works in one regime is not an edge — it's just lucky timing.
</div>
<div class="sx"><table class="dt">
  <thead><tr>
    <th>#</th><th>TF</th><th>Strategy</th><th>Category</th>
    <th>Eras+</th>
    <th>Sharpe</th><th>Sortino</th><th>Calmar</th>
    <th>Ann%</th><th>Max DD</th><th>PF</th><th>WR%</th><th>Mon WR</th>
    <th>Payoff</th><th>Ulcer</th>
    <th>t-stat</th><th>p-val</th><th>Sig?</th>
    <th>Walk-Fwd</th><th>Fee BE</th><th>MC P5</th><th>Trades</th>
  </tr></thead>
  <tbody>{"".join(surv_rows)}</tbody>
</table></div>

<a id="heatmap"></a>
<h2>🔥 Strategy × Era Sharpe Heatmap — Top 30 Strategies</h2>
<div class="hm-leg">
  <span style="background:#004d40;color:#fff">Sh&gt;1.5</span>
  <span style="background:#1b5e20;color:#fff">Sh&gt;0.8</span>
  <span style="background:#33691e;color:#fff">Sh&gt;0.3</span>
  <span style="background:#5d4037;color:#fff">Sh&gt;0</span>
  <span style="background:#b71c1c;color:#fff">Sh&lt;0</span>
  <span style="background:#4a0000;color:#fff">Sh&lt;-0.5</span>
  <span style="color:#ffd700">★=p&lt;0.05</span>
</div>
<div class="hm-wrap"><table class="hm-t">
  <thead><tr><th style="background:#080c14"></th>{hm_heads}</tr></thead>
  <tbody>{"".join(hm_rows)}</tbody>
</table></div>

<a id="regimes"></a>
<h2>🎯 Best Strategy by Market Regime — Your Playbook</h2>
<div class="banner bi">
  Match the current market regime to the right strategy family.
  Averaged across all eras of that regime type (minimum 2 eras to qualify).
</div>
<div class="rg-grid">{regime_cards}</div>

<a id="era-tops"></a>
<h2>📅 Era-by-Era Winners — What Worked in Each Historical Period</h2>
<div class="era-grid">{era_card_html}</div>

<a id="categories"></a>
<h2>📁 Performance by Strategy Category</h2>
<div class="sx"><table class="dt">
  <thead><tr>
    <th>Category</th><th>Avg Sharpe</th><th>Best Sharpe</th>
    <th>Profitable</th><th>Total</th><th>Avg Return</th><th>Avg Max DD</th>
  </tr></thead>
  <tbody>{cat_rows}</tbody>
</table></div>

<a id="significance"></a>
<h2>📐 Statistically Significant Edges — p &lt; 0.05 (95% Confidence)</h2>
<div class="banner bi">
  These edges are <b>not random</b> — confirmed by two-tailed t-test.
  Monte Carlo P5 = worst-case final return across 2,000 randomised trade sequences.
</div>
<div class="sx"><table class="dt">
  <thead><tr><th>Strategy</th><th>TF</th><th>Sharpe</th><th>t-stat</th>
  <th>p-value</th><th>MC P5 Return</th><th>Ann Return</th><th>Walk-Fwd</th></tr></thead>
  <tbody>{sig_rows}</tbody>
</table></div>

<a id="walkfwd"></a>
<h2>🔬 Walk-Forward: Robust vs Overfit (60/40 split)</h2>
<div class="banner bw">
  Walk-forward efficiency = out-of-sample Sharpe ÷ in-sample Sharpe.
  <b>Negative = strategy is curve-fit to historical data and will likely fail live.</b>
  Only use strategies with WF ≥ 0.5.
</div>
<div class="two-col">
  <div><h2 style="color:#00e676;margin-top:0">✓ Robust (WF ≥ 0.6)</h2>
    <table class="dt"><thead><tr><th>Strategy</th><th>TF</th><th>WF</th><th>Sharpe</th><th>Ann Ret</th></tr></thead>
    <tbody>{wf_good_rows}</tbody></table>
  </div>
  <div><h2 style="color:#ff5252;margin-top:0">✗ Overfit — Do NOT Trade Live</h2>
    <table class="dt"><thead><tr><th>Strategy</th><th>TF</th><th>WF</th><th>Sharpe</th><th>Verdict</th></tr></thead>
    <tbody>{wf_bad_rows}</tbody></table>
  </div>
</div>

<a id="fees"></a>
<h2>💰 Fee Robustness — Strategies That Survive Even 5x+ Higher Fees</h2>
<div class="banner bi">
  Fee break-even = how many times current fees can increase before strategy becomes unprofitable.
  Essential check for real trading: exchanges change fees, slippage increases with size, markets become more efficient.
</div>
<div class="sx"><table class="dt">
  <thead><tr><th>Strategy</th><th>TF</th><th>Fee Break-Even</th><th>Sharpe</th><th>Ann Ret</th><th>Category</th></tr></thead>
  <tbody>{fee_rows}</tbody>
</table></div>

<a id="shame"></a>
<h2>☠ Hall of Shame — Never Use These</h2>
<div class="banner br">
  These strategies have Sharpe &lt; -0.3 in 4+ distinct market eras.
  They consistently destroy capital across multiple market conditions.
  No amount of parameter tuning will fix a fundamentally broken approach.
</div>
<table class="dt">
  <thead><tr><th>Strategy</th><th>TF</th><th>Consistently Bad</th>
  <th>Full Sharpe</th><th>Full Return</th><th>Max DD</th><th>Max Consecutive Losses</th></tr></thead>
  <tbody>{shame_rows}</tbody>
</table>

<div class="lbl" style="margin-top:40px;border-color:#1a2030">
  Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
  {n_combos:,} combos &nbsp;|&nbsp;
  Fees: taker 0.05%, maker 0.02%, spread, slippage, funding, borrow &nbsp;|&nbsp;
  Walk-fwd: 60/40 split &nbsp;|&nbsp;
  Monte Carlo: 2,000 sims &nbsp;|&nbsp;
  Significance: two-tailed t-test vs 0
</div>
</div></body></html>"""

    path = os.path.join(OUT_DIR, "unified_report.html")
    with open(path, "w") as f:
        f.write(html)
    import shutil
    dash = "/tmp/eth-dashboard/unified_report.html"
    shutil.copy(path, dash)
    sz = os.path.getsize(path)
    print(f"\n✅  Unified report → {path}  ({sz//1024} KB)")
    print(f"✅  Dashboard copy → {dash}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*75)
    print("  MASTER UNIFIED BACKTEST + REPORT")
    print("  145 extended strategies × 4 TFs × 15 eras + merge all history")
    print("═"*75 + "\n")

    t0 = time.time()

    # 1. Load all previous results
    print("── Step 1: Loading all previous results ──────────────────────────")
    prev = load_all_previous()

    # 2. Load & prepare data
    print("\n── Step 2: Loading timeframe data ────────────────────────────────")
    dfs = load_tf_dfs()

    # 3. Run extended strategies
    print("\n── Step 3: Running 145 extended strategies ───────────────────────")
    new_results = run_extended(dfs)
    print(f"\n  New results: {len(new_results):,}")

    # 4. Merge everything
    print("\n── Step 4: Merging all results ───────────────────────────────────")
    all_results = prev + new_results
    print(f"  Total combined: {len(all_results):,} results")

    with open(MASTER_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    sz = os.path.getsize(MASTER_JSON)
    print(f"  Master JSON saved → {MASTER_JSON}  ({sz//1024} KB)")

    # 5. Generate unified report
    print("\n── Step 5: Generating unified HTML report ─────────────────────────")
    generate_unified_report(all_results)

    print(f"\n  ✅  Total time: {(time.time()-t0)/60:.1f} minutes")
    print(f"  ✅  Open at: http://localhost:7788/unified_report.html")
