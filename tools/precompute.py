# tools/precompute.py
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess

DATA_DIR = "data/processed"
OUT_DIR  = "assets/prebuilt"  # served fast; good browser caching

os.makedirs(OUT_DIR, exist_ok=True)

def loess_line(y, x=None, frac=0.5):
    if x is None: x = np.arange(len(y))
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return y.tolist()
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=False)
    ys = y.copy(); ys[mask] = sm
    return ys.tolist()

def line_fig(x_labels, y, y_sm, y_title, is_pct=False, name=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=y, mode="lines+markers", name=name or "value"))
    fig.add_trace(go.Scatter(x=x_labels, y=y_sm, mode="lines", name="LOESS", line=dict(dash="dot")))
    fig.update_layout(
        margin=dict(l=16,r=8,t=6,b=24), yaxis_title=y_title, xaxis_title=None,
        template="plotly_white", legend=dict(orientation="h", y=1.02, x=1, yanchor="bottom", xanchor="right"),
        height=240
    )
    if is_pct: fig.update_yaxes(ticksuffix="%", separatethousands=True)
    else:      fig.update_yaxes(separatethousands=True)
    fig.update_xaxes(tickangle=45)
    return fig.to_plotly_json()

# ---- load source data
qtr        = pd.read_csv(os.path.join(DATA_DIR, "income_medians_qtr.csv"))
lease_qtr  = pd.read_csv(os.path.join(DATA_DIR, "lease_medians_qtr.csv"))
permits    = pd.read_csv(os.path.join(DATA_DIR, "bps_ca_final.csv"))
place_ts   = pd.read_csv(os.path.join(DATA_DIR, "place_timeseries.csv"))
crime      = pd.read_csv(os.path.join(DATA_DIR, "sum_df_master.csv"))
ten_age    = pd.read_csv(os.path.join(DATA_DIR, "final_compare.csv"))
mf_map     = pd.read_csv(os.path.join(DATA_DIR, "income_mf_map.csv"))

# Minimal key derivation matching your app (you can paste your helper here)
def city_key(s):
    s = str(s)
    for a,b in [(", California",""),(" city",""),(" CDP","")]:
        s = s.replace(a,b)
    return s.strip().upper()

for df in [qtr, lease_qtr, permits, place_ts, crime, ten_age, mf_map]:
    if "CITY_KEY" not in df.columns:
        for c in ["City","city","PLACE","place","NAME","name"]:
            if c in df.columns:
                df["CITY_KEY"] = df[c].map(city_key); break
        else:
            df["CITY_KEY"] = ""

CITIES = ["UPLAND","MONTCLAIR","CLAREMONT","REDLANDS","ONTARIO"]

# Precompute derived columns once
if "period_index" not in qtr.columns and {"year","quarter_3m"}.issubset(qtr.columns):
    qtr["period_index"] = qtr["year"]*4 + qtr["quarter_3m"] - 1
qtr = qtr.sort_values(["CITY_KEY","period_index"])
qtr["dollar_per_unit"] = qtr["SP"] / qtr["X..of.Units"].replace(0, np.nan)
qtr["price_diff"] = qtr["LP"] - qtr["SP"]

if "period_index" not in lease_qtr.columns and {"year","quarter_3m"}.issubset(lease_qtr.columns):
    lease_qtr["period_index"] = lease_qtr["year"]*4 + lease_qtr["quarter_3m"] - 1
lease_qtr = lease_qtr.sort_values(["CITY_KEY","period_index"])

def period_label(df):
    if {"year","quarter_3m"}.issubset(df.columns):
        return (df["year"].astype(int).astype(str) + "-Q" + df["quarter_3m"].astype(int).astype(str)).tolist()
    return df.index.astype(str).tolist()

# ---- build & write per-city quarterly bundle
for city in CITIES:
    df  = qtr[qtr["CITY_KEY"] == city].copy()
    dlq = lease_qtr[lease_qtr["CITY_KEY"] == city].copy()
    if df.empty and dlq.empty:
        continue

    x = period_label(df)
    def build(df_, col, title, is_pct=False):
        y = df_[col].astype(float).tolist()
        y_sm = loess_line(y, np.arange(len(y)), 0.5)
        return line_fig(x, y, y_sm, title, is_pct=is_pct, name=col)

    bundle = [
        build(df,  "SP",             "Median Sale Price (SP)"),
        build(df,  "count",          "Sale Count"),
        build(df,  "DOM",            "Median DOM"),
        build(df,  "Cap_Rate_Pct",   "Median Cap Rate (%)", is_pct=True),
        build(df,  "dollar_per_unit","Median $ per Unit"),
        build(df,  "price_diff",     "LP − SP"),
        build(df,  "GRM",            "Median GRM"),
        build(dlq, "median_LP",      "Median Lease Price (LP)") if not dlq.empty else None,
        build(dlq, "median_DOM_lease","Median Lease DOM") if not dlq.empty else None,
    ]

    with open(os.path.join(OUT_DIR, f"quarterly_{city}.json"), "w") as f:
        json.dump(bundle, f)
