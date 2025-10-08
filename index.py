# app.py — SUMMARY as a dropdown "city" with an empty page, Carto tiles, sticky legend, hover HTML
import os
import numpy as np
import json
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import dash_leaflet as dl
from pandas.api.types import CategoricalDtype
from dash import State

DISCLAIMER_1 = "Disclaimer: This report is provided for informational purposes only and is not intended as investment, legal, or tax advice. All data (including income, population, and eviction filings) are derived from third-party sources such as the U.S. Census Bureau, the Anti-Eviction Mapping Project, and the Los Angeles Superior Court, and are believed to be reliable but not guaranteed. Readers should verify all information independently before making any real estate or financial decisions. Past trends do not guarantee future performance."
DISCLAIMER_2 = "Based on information from Vesta Plus MLS as of November 25, 2024. All data, including all measurements and calculations of area, is obtained from various sources and has not been, and will not be, verified by broker or MLS. All information should be independently reviewed and verified for accuracy."
DRE_NUM      = "CA DRE #01942714"
# --- NORMALIZATION + GUARDS -----------------------------------------------
from dash.exceptions import PreventUpdate


def _normalize_place_ts(df):
    """Ensure place_ts has a 'year' column."""
    if df is None or len(df) == 0:
        return df
    cols = {c.lower(): c for c in df.columns}
    if 'year' not in df.columns:
        if 'data_year' in cols:
            df = df.rename(columns={cols['data_year']: 'year'})
        elif 'date' in cols:
            # derive year from a Date column if present
            dname = cols['date']
            df = df.assign(year=pd.to_datetime(df[dname], errors='coerce').dt.year)
        elif 'yr' in cols:
            df = df.rename(columns={cols['yr']: 'year'})
        # else: leave as-is; downstream guard will PreventUpdate
    return df

def _normalize_quarterly(df):
    """
    Make sure the quarterly dataframe has expected names:
    SP, LP, DOM, X..of.Units, dollar_per_sqft, dollar_per_unit, period_label
    """
    if df is None or len(df) == 0:
        return df

    rename_map = {}
    # common alternates -> canonical
    for alt, canon in [
        ('median_SP', 'SP'), ('Median_SP', 'SP'), ('sp', 'SP'),
        ('median_LP', 'LP'), ('lp', 'LP'),
        ('median_DOM', 'DOM'), ('dom', 'DOM'),
        ('Units', 'X..of.Units'), ('units', 'X..of.Units'), ('num_units', 'X..of.Units'),
        ('$/sqft', 'dollar_per_sqft'), ('dollar_sqft', 'dollar_per_sqft'),
        ('$/unit', 'dollar_per_unit'), ('dollar_unit', 'dollar_per_unit'),
        ('period', 'period_label'), ('quarter_label', 'period_label')
    ]:
        if alt in df.columns and canon not in df.columns:
            rename_map[alt] = canon

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

def _require_columns(df, needed, ctx=""):
    """Raise PreventUpdate if dataframe is missing columns."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        # Helpful logging in Render logs
        print(f"[WARN] {ctx}: missing columns {missing}. Available: {list(df.columns)}")
        raise PreventUpdate
# --------------------------------------------------------------------------



DATA_DIR = "data/processed"

def _read(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))

def _derive_city_key(df, candidates=("CITY_KEY","City","city","PLACE","place","NAME","name")):
    for c in candidates:
        if c in df.columns:
            key = (df[c].astype(str)
                    .str.replace(", California","", case=False, regex=False)
                    .str.replace(" city","", case=False, regex=False)
                    .str.replace(" CDP","", case=False, regex=False)
                    .str.strip()
                    .str.upper())
            df = df.copy()
            df["CITY_KEY"] = key
            return df
    df = df.copy()
    df["CITY_KEY"] = ""
    return df

# ---- Load data ----
qtr        = _read("income_medians_qtr.csv")
lease_qtr  = _read("lease_medians_qtr.csv")
permits    = _read("bps_ca_final.csv")
place_ts   = _read("place_timeseries.csv")
crime      = _read("sum_df_master.csv")
ten_age    = _read("final_compare.csv")
mf_map     = _read("income_mf_map.csv")

# ---- Normalize keys ----
qtr       = _derive_city_key(qtr,       ("CITY_KEY","City","city"))
lease_qtr = _derive_city_key(lease_qtr, ("CITY_KEY","City","city"))
permits   = _derive_city_key(permits,   ("CITY_KEY","City","city"))
place_ts  = _derive_city_key(place_ts,  ("CITY_KEY","place","PLACE","City"))
crime     = _derive_city_key(crime,     ("CITY_KEY","City","city"))
ten_age   = _derive_city_key(ten_age,   ("CITY_KEY","NAME","name","City"))
mf_map    = _derive_city_key(mf_map,    ("CITY_KEY","City","city"))

# ---- City list (SUMMARY is a dropdown option) ----
CITY_LEVELS = ["UPLAND", "MONTCLAIR", "CLAREMONT", "REDLANDS", "ONTARIO"]

# Filter datasets to real cities only (not SUMMARY)
real_cities = CITY_LEVELS

qtr        = qtr[qtr["CITY_KEY"].isin(real_cities)].copy()
lease_qtr  = lease_qtr[lease_qtr["CITY_KEY"].isin(real_cities)].copy()
place_ts   = place_ts[place_ts["CITY_KEY"].isin(real_cities)].copy()
crime      = crime[crime["CITY_KEY"].isin(real_cities)].copy()
ten_age    = ten_age[ten_age["CITY_KEY"].isin(real_cities)].copy()
mf_map     = mf_map[mf_map["CITY_KEY"].isin(real_cities)].copy()

# ---- Permits yearly totals ----
if not {"Year","Units_Total"}.issubset(permits.columns):
    if "Units_Total" not in permits.columns:
        unit_cols = [c for c in permits.columns if c.lower() in {"units_total","total_permits"}]
        permits["Units_Total"] = permits[unit_cols[0]] if unit_cols else np.nan
    if "Year" not in permits.columns:
        permits["Year"] = permits["year"] if "year" in permits.columns else np.nan

permits = (permits[["CITY_KEY","Year","Units_Total"]]
           .dropna(subset=["CITY_KEY","Year"])
           .groupby(["CITY_KEY","Year"], as_index=False)
           .agg(total_permits=("Units_Total","sum")))

# ---- Place TS standard columns ----
if "year" not in place_ts.columns and "Year" in place_ts.columns:
    place_ts = place_ts.rename(columns={"Year":"year"})
for need in ("median_hh_income","population","vacancy_rate_pct"):
    if need not in place_ts.columns:
        place_ts[need] = np.nan

# ---- Crime ----
if "Total" not in crime.columns:
    tc = [c for c in crime.columns if c.lower()=="total"]
    crime["Total"] = crime[tc[0]] if tc else np.nan
if "data_year" not in crime.columns:
    crime["data_year"] = crime["year"] if "year" in crime.columns else np.nan
if "change" not in crime.columns:
    crime = crime.sort_values(["CITY_KEY","data_year"])
    crime["change"] = crime.groupby("CITY_KEY")["Total"].pct_change()*100.0

# ---- Quarter order helpers ----
def _ensure_period_order(df):
    if "period_index" not in df.columns:
        if {"year","quarter_3m"}.issubset(df.columns):
            df["period_index"] = df["year"]*4 + df["quarter_3m"] - 1
        else:
            df["period_index"] = np.arange(len(df))
    if "period_label" not in df.columns:
        if {"year","quarter_3m"}.issubset(df.columns):
            df["period"] = df["year"].astype(int).astype(str) + "-Q" + df["quarter_3m"].astype(int).astype(str)
            df = df.sort_values("period_index")
            cats = df["period"].unique().tolist()
            df["period_label"] = pd.Categorical(df["period"], categories=cats, ordered=True)
        else:
            df["period_label"] = pd.Categorical(df.index.astype(str), ordered=True)
    return df

qtr       = _ensure_period_order(qtr)
lease_qtr = _ensure_period_order(lease_qtr)

# ---- Map safety ----
for col in ("lat","lon","SP_num"):
    if col not in mf_map.columns:
        mf_map[col] = np.nan
mf_map = mf_map[np.isfinite(mf_map["lat"]) & np.isfinite(mf_map["lon"]) & np.isfinite(mf_map["SP_num"])]

# ------------------ Plot helpers ------------------
def loess_line(y, x=None, frac=0.5):
    if x is None:
        x = np.arange(len(y))
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return x, y
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=False)
    ys = y.astype(float).copy()
    ys[mask] = sm
    return x, ys

def _is_cat(series):
    return isinstance(series.dtype, CategoricalDtype)

def line_with_loess(df, x_order_col, y_col, y_title, is_pct=False):
    df = df.copy()
    if _is_cat(df[x_order_col]):
        x_labels = df[x_order_col].astype(str).tolist()
    else:
        x_labels = df[x_order_col].astype(str).tolist()
    y = df[y_col].values.astype(float)
    _, y_sm = loess_line(y, np.arange(len(df)), 0.5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=y, mode="lines+markers", name=y_col))
    fig.add_trace(go.Scatter(x=x_labels, y=y_sm, mode="lines", name="LOESS", line=dict(dash="dot")))
    fig.update_layout(
        margin=dict(l=16,r=8,t=6,b=24), yaxis_title=y_title, xaxis_title=None,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=240
    )
    if is_pct:
        fig.update_yaxes(ticksuffix="%", separatethousands=True)
    else:
        fig.update_yaxes(separatethousands=True)
    fig.update_xaxes(tickangle=45)
    return fig

def bar_series(df, x_col, y_col, y_title, is_pct=False):
    fig = go.Figure(go.Bar(x=df[x_col].astype(str), y=df[y_col]))
    fig.update_layout(
        margin=dict(l=16,r=8,t=6,b=24), yaxis_title=y_title, xaxis_title=None,
        template="plotly_white", height=240
    )
    if is_pct:
        fig.update_yaxes(ticksuffix="%", separatethousands=True)
    else:
        fig.update_yaxes(separatethousands=True)
    return fig

def city_tab_note(city_key, tab_value):
    city = city_key.upper()
    placeholders = {
        "UPLAND": {
            "prices":"<ul><li><b>Prices:</b> Upland sales prices have been slowly increasing, with a small surge between 2021–2023.</li><li><b>Economic:</b> Sale counts have risen in 2025; DOM has ticked up. Despite rising lease and sale prices, LP–SP is around or below zero.</li><li><b>Composite:</b> Cap rates are rising gradually (≈4% by early Q4 2025). GRM has been decreasing.</li></ul>",
            "permits":"<ul><li><b>Permits:</b> Upland saw a surge around 2022; ~70 permits in 2024.</li><li><b>Income:</b> Median HH income trending up, near $100k.</li><li><b>Population:</b> ~79k in 2023, slowly rising.</li><li><b>Vacancy:</b> Declined from 2014, then edged up from 2023.</li></ul>",
            "crime":"<b>Crime:</b> Crime per 100k residents decreased in 2024.",
            "demo":"<b>Demographics: </b><ul><li><b>Tenure:</b> Renter share up slightly between 2012–2022.</li><li><b>Age:</b> Older on average; strong family presence.</li></ul>",
            "map":"<b>Map: </b>Downtown shows many sales at lower prices with some larger multi-unit buildings; west/northwest trend higher."
        },
        "MONTCLAIR": {
            "prices":"<ul><li><b>Prices:</b> Rising since 2024, but low sales counts can skew metrics.</li><li><b>Economic:</b> DOM trending up; LP–SP ~0 or negative; leases up—demand appears firm.</li><li><b>Composite:</b> Cap rates fluctuate in a low-activity market; GRM rising.</li></ul>",
            "permits":"<ul><li><b>Permits:</b> 2024 jumped relative to prior years, likely tied to zoning/ADU updates.</li><li><b>Income:</b> Median HH income ~ $75k (2023) and rising.</li><li><b>Population:</b> Slight dip since 2019, then stabilized.</li><li><b>Vacancy:</b> Dropped since 2013; ~2% since 2018.</li></ul>",
            "crime":"<b>Crime: </b>Crime in Montclair dropped slightly in 2024.",
            "demo":"<ul><li><b>Tenure:</b> Gradual shift toward renters since 2012.</li><li><b>Age:</b> Larger 35–64 and 65+ cohorts.</li></ul>",
            "map":"Sales concentrate toward the south; overall activity is lighter."
        },
        "CLAREMONT": {
            "prices":"<ul><li><b>Prices:</b> Generally steady since 2018; notable surge in 2022–2023 (VSSP, Colby Circle, SB 9).</li><li><b>Economic:</b> DOM declined since 2018; LP–SP near zero—healthy demand.</li><li><b>Composite:</b> Cap rates up 2022–2025 with a small Q3-2025 cooldown; GRM up in 2023 then steady.</li></ul>",
            "permits":"<ul><li><b>Permits:</b> Peaks are modest (e.g., 2021 ≈ 8), reflecting a built-out city and large institutional footprint.</li><li><b>Income:</b> Median HH income > $120k (2023), outpacing neighbors.</li><li><b>Population:</b> ~36k, very slow growth.</li><li><b>Vacancy:</b> Declining since 2018.</li></ul>",
            "crime":"<b>Crime:</b> Claremont enjoys a low, stable crime environment.",
            "demo":"<ul><li><b>Tenure:</b> 2012–2022 owner share > 60%.</li><li><b>Age:</b> Even distribution with larger youth/65+ (college town effect).</li></ul>",
            "map":"Sales cluster near the Village; higher prices toward the south."
        },
        "REDLANDS": {
            "prices":"<ul><li><b>Prices:</b> Sharp surge in 2021; leveled in 2023.</li><li><b>Economic:</b> DOM rising since 2021; LP–SP around zero; leases rising.</li><li><b>Composite:</b> Cap rates relatively flat; GRM rising.</li></ul>",
            "permits":"<ul><li><b>Permits:</b> ~2021 spike (≈400 units) tied to downtown/transit momentum and ADUs; 2024 still strong (~250).</li><li><b>Income:</b> Up since 2015; near $100k by 2023.</li><li><b>Population:</b> Peaked ~73k in 2022.</li><li><b>Vacancy:</b> From ~7.5% (2017) down toward ~4% (2023).</li></ul>",
            "crime":"<b>Crime:</b> Crime metrics broadly stable with small YoY variation.",
            "demo":"<ul><li><b>Tenure:</b> ~60% owners, slight recent tilt toward renting.</li><li><b>Age:</b> Larger 65+ cohort; strong family age groups.</li></ul>",
            "map":"Sales cluster downtown/central; Mentone cheaper small MF; Loma Linda around the hospital with low cap rates."
        },
        "ONTARIO": {
            "prices":"<ul><li><b>Prices:</b> Lower base but slowly increasing.</li><li><b>Economic:</b> DOM low but inching up; LP–SP ~0 or negative; leases rising—demand plausible.</li><li><b>Composite:</b> Cap rates low and rising; GRM rising (cooling).</li></ul>",
            "permits":"<ul><li><b>Permits:</b> 2023 surge (Ontario Ranch build-out; ADUs/SB 9). 2024 still high (~500).</li><li><b>Income:</b> Rising since 2015; ~ $80k in 2023.</li><li><b>Population:</b> Growth since 2010 (small 2021 dip); ~176k in 2023.</li><li><b>Vacancy:</b> Declining since 2013; ~3.5% since 2018.</li></ul>",
            "crime":"<b>Crime:</b> Ontario crime time-series not available in this dataset.",
            "demo":"<ul><li><b>Tenure:</b> Owner share > 50%; renter share trending up toward 50%.</li><li><b>Age:</b> Larger 65+ and family cohorts; smaller 18–34 share.</li></ul>",
            "map":"Dense clusters near logistics and new MF builds; south side shows higher pricing nodes."
        }
    }
    return placeholders.get(city, {}).get(tab_value, f"Placeholder summary for {city.title()} — {tab_value.title()}")

# ------------------ App layout ------------------
app = Dash(
    __name__,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
        "https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css",
    ],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "SB Dashboard"

GLOBAL_STYLE = {"fontSize":"13px","--pico-font-size":"13px"}

# Use asset loader for the logo (place logo.jpg in ./assets/)
header = html.Div(
    [html.Img(src=None, id="logo", style={
        "height": "110px",
        "width": "110px",
        "display": "block",
        "margin": "0 auto"
    })],
    style={
        "background": "#000",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "0",
        "margin": "0",
        "height": "auto"
    }
)

footer = html.Div(
    [
        html.Div(
            [
                html.Span(DISCLAIMER_1),
                html.Span(" \u00b7 ", style={"margin": "0 6px"}),  # middot separator
                html.Span(DISCLAIMER_2),
                html.Span(" \u00b7 ", style={"margin": "0 6px"}),
                html.Span(DRE_NUM, style={"fontWeight": "600"})
            ],
            style={
                "textAlign": "center",
                "opacity": "0.9",
                "maxWidth": "1200px",
                "margin": "0 auto",
                "padding": "8px 12px",
                "lineHeight": "1.4",
                "fontSize": "12px"
            }
        )
    ],
    style={
        "background": "#000",
        "color": "#fff",
        "padding": "6px 0"
    }
)


# tiny helper: “card” with a compact title above each graph (Shiny-style)
def titled_graph(id_, title_text):
    return html.Div([
        html.Div(title_text, style={"fontWeight":"600","fontSize":"12px","margin":"2px 0 4px"}),
        dcc.Graph(id=id_, config={"displayModeBar": False}, style={"height":"240px"})
    ])

# Sidebar + Store (store keeps the active tab value even when tabs aren't mounted)
sidebar = html.Div([
    html.Label("City", style={"fontWeight":"600"}),
    dcc.Dropdown(
        id="city",
        options=[{"label": c.title(), "value": c} for c in CITY_LEVELS],
        value=CITY_LEVELS[0],  # <-- pick a real city (e.g., "UPLAND")
        clearable=False,
        placeholder="Select a city…",
        className="city-dd",
        style={"fontSize": "12px"}
    ),
    dcc.Store(id="tab_value", data="prices_tab"),
    html.Div(id="city_note",
             className="city-summary",
             style={"color":"#6c757d","fontSize":"16px","lineHeight":"1.5","marginTop":"14px"})
], style={"minWidth":"260px","maxWidth":"260px","padding":"10px"})

def grid(children):
    return html.Div(children, style={
        "display":"grid",
        "gridTemplateColumns":"repeat(2, minmax(0,1fr))",
        "gap":"8px"
    })

# Tabs (used for real cities only)
tabs_for_cities = dcc.Tabs(id="tabs", value="prices_tab", children=[
    dcc.Tab(label="Quarterly Prices & Ops", value="prices_tab", children=[
        grid([
            titled_graph("p1",   "Median Sale Price (Quarterly)"),
            titled_graph("p1_2", "Sale Counts"),
            titled_graph("p2",   "Median DOM (Quarterly)"),
            titled_graph("p3",   "Median Cap Rate (%) (Quarterly)"),
            titled_graph("p4",   "Median $ per Unit (Quarterly)"),
            titled_graph("p5",   "List – Sale Price Difference (Quarterly)"),
            titled_graph("p6",   "Median GRM (Quarterly)"),
            titled_graph("pL1",  "Lease — Median Price (Quarterly)"),
            titled_graph("pL2",  "Lease — Median DOM (Quarterly)")
        ])
    ]),
    dcc.Tab(label="Permits & ACS", value="permits_tab", children=[
        grid([
            titled_graph("p7",  "Total Permits per Year"),
            titled_graph("p8",  "Median Household Income per Year"),
            titled_graph("p9",  "Population per Year"),
            titled_graph("p10", "Vacancy Rate (%) per Year")
        ])
    ]),
    dcc.Tab(label="Crime", value="crime_tab", children=[
        grid([
            titled_graph("p11", "Crime per Year (per 100k)"),
            titled_graph("p12", "Crime Change (% YoY)")
        ])
    ]),
    dcc.Tab(label="Tenure & Age", value="demographics_tab", children=[
        grid([
            titled_graph("p13", "Tenure Shares (2012 vs 2022)"),
            titled_graph("p14", "Age Distribution — COUNTS (2012 vs 2022)"),
            titled_graph("p15", "Age Distribution — PERCENT (2012 vs 2022)")
        ])
    ]),
    dcc.Tab(label="Map", value="map_tab", children=[
        html.Div("Sale Price Pin Map", style={"fontWeight":"600","fontSize":"12px","margin":"2px 0 4px"}),
        html.Div(id="leaflet_map", style={"height":"520px","width":"100%","marginTop":"6px"})
    ]),
])

# Empty summary page (no tabs)
summary_page = html.Div([], style={"padding":"10px"})

# Main content container switches between summary_page and tabs_for_cities
app.layout = html.Div(
    [
        header,
        html.Div(
            [
                sidebar,
                html.Div(id="main_content", style={"flex": "1 1 auto", "padding": "10px"})
            ],
            style={
                "display": "flex",
                "gap": "0",
                "alignItems": "flex-start",
                "justifyContent": "stretch",
                "flex": "1 1 auto"  # fills remaining height between header and footer
            }
        ),
        footer
    ],
    style={**GLOBAL_STYLE, "display": "flex", "flexDirection": "column", "minHeight": "100vh"}
)


# Make Dash aware of both layouts to avoid "nonexistent object" errors at startup
app.validation_layout = html.Div([app.layout, tabs_for_cities])

# ------------------ Switch main content based on city ------------------
@app.callback(Output("main_content","children"), Input("city","value"))
def show_summary_or_tabs(city):
    if city == "SUMMARY":
        return summary_page
    return tabs_for_cities

# Sync the visible tabs (when present) to the always-present store
@app.callback(Output("tab_value","data"), Input("tabs","value"))
def _sync_tab_store(tab_value):
    return tab_value

# Set logo URL after app init so assets is available
@app.callback(Output("logo", "src"), Input("main_content", "children"))
def _set_logo(_):
    # expects ./assets/logo.jpg
    return app.get_asset_url("logo.jpg")

# ------------------ City note (depends on city + tab_value) ------------------
@app.callback(Output("city_note","children"),
              Input("city","value"),
              Input("tab_value","data"))
def update_city_note(city, tab_value):
    if city == "SUMMARY":
        return ""
    key_map = {
        "prices_tab": "prices",
        "permits_tab": "permits",
        "crime_tab": "crime",
        "demographics_tab": "demo",
        "map_tab": "map",
    }
    key = key_map.get(tab_value, "prices")
    txt = city_tab_note(city, key) or f"Placeholder summary for {city.title()} — {key.title()}"
    return html.Div([
        html.B(f"{city.title()} — {key.title()} Summary"),
        html.Br(),
        dcc.Markdown(txt, dangerously_allow_html=True)
    ])

# ------------------ Chart callbacks (return blanks when SUMMARY) ------------------
def _load_bundle(city, name):
    path = os.path.join("assets", "prebuilt", f"{name}_{city}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _json_to_fig(obj):
    return go.Figure(obj)
@app.callback(
    Output("p1","figure"), Output("p1_2","figure"), Output("p2","figure"),
    Output("p3","figure"), Output("p4","figure"), Output("p5","figure"),
    Output("p6","figure"), Output("pL1","figure"), Output("pL2","figure"),
    Input("city","value"), Input("tabs","value")
)
def update_quarterly(city, tab):
    # only compute when a real city is selected AND the Prices tab is visible
    if city == "SUMMARY" or tab != "prices_tab":
        raise PreventUpdate
    bundle = _load_bundle(city, "quarterly")
    if bundle:
        figs = [(_json_to_fig(j) if j is not None else go.Figure()) for j in bundle]
        # ensure length 9
        while len(figs) < 9:
            figs.append(go.Figure())
        return tuple(figs[:9])
    df = qtr[qtr["CITY_KEY"] == city].copy()  # (sorted once at load; see note below)
    f1   = line_with_loess(df, "period_label", "SP", "Median Sale Price (SP)")
    f1_2 = line_with_loess(df, "period_label", "count", "Sale Count")
    f2   = line_with_loess(df, "period_label", "DOM", "Median DOM")
    f3   = line_with_loess(df, "period_label", "Cap_Rate_Pct", "Median Cap Rate (%)", is_pct=True)

    # if you precompute these at load (recommended), you can drop the if-blocks
    if "dollar_per_unit" not in df.columns:
        df["dollar_per_unit"] = df["SP"] / df.get("X..of.Units", 1)
    f4   = line_with_loess(df, "period_label", "dollar_per_unit", "Median $ per Unit")

    if "price_diff" not in df.columns:
        df["price_diff"] = df["LP"] - df["SP"]
    f5   = line_with_loess(df, "period_label", "price_diff", "LP − SP")

    f6   = line_with_loess(df, "period_label", "GRM", "Median GRM")

    dlq  = lease_qtr[lease_qtr["CITY_KEY"] == city].copy()
    fL1  = line_with_loess(dlq, "period_label", "median_LP", "Median Lease Price (LP)")
    fL2  = line_with_loess(dlq, "period_label", "median_DOM_lease", "Median Lease DOM")
    return f1, f1_2, f2, f3, f4, f5, f6, fL1, fL2

@app.callback(
    Output("p7","figure"), Output("p8","figure"), Output("p9","figure"), Output("p10","figure"),
    Input("city","value")
)

def update_permits_acs(city):
    if city == "SUMMARY":
        def blank():
            fig = go.Figure(); fig.update_layout(template="plotly_white", margin=dict(l=16,r=8,t=6,b=24), height=240)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            return fig
        return blank(), blank(), blank(), blank()
    d7 = permits[permits["CITY_KEY"] == city].sort_values("Year")
    f7 = bar_series(d7, "Year", "total_permits", "Total permits")
    d8 = place_ts[place_ts["CITY_KEY"] == city].sort_values("year")
    f8 = line_with_loess(d8, "year", "median_hh_income", "Median household income")
    f9 = line_with_loess(d8, "year", "population", "Population")
    f10 = line_with_loess(d8, "year", "vacancy_rate_pct", "Vacancy rate (%)", is_pct=True)
    return f7, f8, f9, f10

@app.callback(Output("p11","figure"), Output("p12","figure"), Input("city","value"))
def update_crime(city):
    if city == "SUMMARY":
        def blank():
            fig = go.Figure(); fig.update_layout(template="plotly_white", margin=dict(l=16,r=8,t=6,b=24), height=240)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            return fig
        return blank(), blank()
    d = crime[crime["CITY_KEY"] == city].copy().sort_values("data_year")
    f12 = bar_series(d, "data_year", "change", "Crime change (% YoY)", is_pct=True)
    if city == "ONTARIO":
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, text="No crime data for Ontario", showarrow=False, font=dict(size=16))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white", margin=dict(l=16,r=8,t=6,b=24), height=240)
        return fig, f12
    pop = place_ts[place_ts["CITY_KEY"] == city]
    pop = _normalize_place_ts(pop)
    _require_columns(pop, ["year", "population"], ctx="update_crime/place_ts")
    pop = pop[["year", "population"]].rename(columns={"year": "data_year"})
    d = d.merge(pop, on="data_year", how="left")
    d["value"] = np.where((d["Total"].notna()) & (d["population"].notna()) & (d["population"]>0),
                          100000.0 * d["Total"] / d["population"], np.nan)
    f11 = bar_series(d, "data_year", "value", "Crimes per 100k")
    return f11, f12

@app.callback(Output("p13","figure"), Output("p14","figure"), Output("p15","figure"), Input("city","value"))
def update_demo(city):
    if city == "SUMMARY":
        def blank():
            fig = go.Figure(); fig.update_layout(template="plotly_white", margin=dict(l=16,r=8,t=6,b=24), height=240)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            return fig
        return blank(), blank(), blank()
    d = ten_age[ten_age["CITY_KEY"] == city]
    if d.empty:
        def blank(msg):
            fig = go.Figure(); fig.add_annotation(x=0.5,y=0.5,text=msg,showarrow=False)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            fig.update_layout(template="plotly_white", margin=dict(l=16,r=8,t=6,b=24), height=240)
            return fig
        return blank("No data"), blank("No data"), blank("No data")
    row = d.iloc[0]
    long = pd.DataFrame({
        "year":["2012","2022","2012","2022"],
        "tenure":["owner","owner","renter","renter"],
        "pct":[row["pct_owner_2012"],row["pct_owner_2022"],row["pct_renter_2012"],row["pct_renter_2022"]]
    })
    fig13 = px.bar(long, x="year", y="pct", color="tenure", barmode="group", template="plotly_white")
    fig13.update_layout(margin=dict(l=16,r=8,t=6,b=24), yaxis_title="Share (%)", xaxis_title=None,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=240)
    fig13.update_yaxes(ticksuffix="%", separatethousands=True)

    counts_2012 = [row["under18_2012"], row["age18_34_2012"], row["age35_64_2012"], row["age65plus_2012"]]
    counts_2022 = [row["under18_2022"], row["age18_34_2022"], row["age35_64_2022"], row["age65plus_2022"]]
    def mirrored(age_groups, v12, v22, y_title, is_pct=False):
        y12 = [-abs(v) for v in v12]; y22 = [abs(v) for v in v22]
        ticks = np.linspace(min(y12 + y22), max(y12 + y22), 7)
        ticktext = [f"{abs(int(t))}{'%' if is_pct else ''}" for t in ticks]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=age_groups, y=y12, name="2012"))
        fig.add_trace(go.Bar(x=age_groups, y=y22, name="2022"))
        fig.update_layout(barmode="overlay", template="plotly_white",
                          margin=dict(l=16,r=8,t=6,b=24), yaxis_title=y_title, xaxis_title=None,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          height=240)
        fig.update_yaxes(tickvals=ticks, ticktext=ticktext)
        return fig
    fig14 = mirrored(["Under 18","18–34","35–64","65+"], counts_2012, counts_2022, "Population (counts)")
    pct_2012 = [row["pct_under18_2012"], row["pct_18_34_2012"], row["pct_35_64_2012"], row["pct_65plus_2012"]]
    pct_2022 = [row["pct_under18_2022"], row["pct_18_34_2022"], row["pct_35_64_2022"], row["pct_65plus_2022"]]
    fig15 = mirrored(["Under 18","18–34","35–64","65+"], pct_2012, pct_2022, "Population share (%)", is_pct=True)
    return fig13, fig14, fig15

# ---- Map (Carto tiles, sticky legend, hover HTML) ----
@app.callback(Output("leaflet_map","children"), Input("city","value"))
def update_map(city):
    if city == "SUMMARY":
        return html.Div()

    df = mf_map[mf_map["CITY_KEY"] == city].copy()
    if df.empty:
        return dl.Map(children=[
            dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")
        ], center=(34.10,-117.65), zoom=10, style={"height":"520px","width":"100%"})

    vals = df["SP_num"].values
    q = np.quantile(vals, [0, .2, .4, .6, .8, 1.0])
    colors = ["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026"]

    def col_for(v):
        for i in range(len(q)-1):
            if v <= q[i+1]:
                return colors[i]
        return colors[-1]

    markers = []
    for _, r in df.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        sp = float(r["SP_num"])
        col = col_for(sp)
        popup_html = r.get("popup_html") or f"<div style='font-size:12px'><b>Sale price:</b> ${sp:,.0f}</div>"
        tooltip = dl.Tooltip(
            html.Iframe(srcDoc=popup_html, style={"border":"0","width":"260px","height":"160px"}),
            direction="top", permanent=False, sticky=True, opacity=0.95
        )
        markers.append(
            dl.CircleMarker(center=[lat, lon], radius=5,
                            color=col, fillColor=col, fillOpacity=0.90, stroke=False,
                            children=[tooltip])
        )

    legend_labels = [f"${int(q[i]):,} – ${int(q[i+1]):,}" for i in range(len(q)-1)]
    legend_items = html.Div([
        html.Div([
            html.Span(style={"display":"inline-block","width":"10px","height":"10px","background":colors[i],
                             "marginRight":"6px","borderRadius":"50%","border":"1px solid #888"}),
            html.Span(legend_labels[i])
        ], style={"marginBottom":"2px"}) for i in range(len(legend_labels))
    ], style={
        "position":"absolute","bottom":"10px","right":"10px",
        "background":"rgba(255,255,255,0.95)","padding":"6px 8px",
        "border":"1px solid #ddd","borderRadius":"6px","fontSize":"11px","zIndex":"1000"
    })

    carto = dl.TileLayer(
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
                    '&copy; <a href="https://carto.com/">CARTO</a>'
    )

    return dl.Map(
        center=[df["lat"].mean(), df["lon"].mean()],
        zoom=12,
        children=[carto, dl.LayerGroup(markers), dl.ScaleControl(position="bottomleft"), legend_items],
        style={"height":"520px","width":"100%","position":"relative"}
    )

# ------------------ Main ------------------
if __name__ == "__main__":
    print("Open: http://127.0.0.1:8050")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8050)), debug=True)
