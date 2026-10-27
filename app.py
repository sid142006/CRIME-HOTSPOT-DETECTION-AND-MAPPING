"""Crime Hotspot Detection and Mapping - GIS-enabled analytics prototype (academic, decision-support only)."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import folium
import plotly.express as px
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from scipy.ndimage import convolve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

st.set_page_config(page_title="Crime Hotspot Atlas", page_icon="🗺️", layout="wide")

INK, TEAL, HOT, AMBER, LINE = "#12263A", "#0E6E78", "#D1345B", "#F4A259", "#D5DEE5"
REQUIRED = ["incident_datetime", "crime_category", "latitude", "longitude"]
CATS = ["Theft", "Burglary", "Assault", "Vehicle Theft", "Vandalism", "Cyber Fraud"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color:{INK}; }}
.stApp {{ background:#F2F5F7; }}
h1,h2,h3 {{ font-family:'Space Grotesk',sans-serif; letter-spacing:-.01em; }}
.hero {{ background:linear-gradient(120deg,{INK} 0%,#17405A 60%,{TEAL} 100%); color:#fff; padding:26px 30px;
  border-radius:14px; margin-bottom:18px; }}
.hero h1 {{ color:#fff; margin:0 0 4px 0; font-size:2rem; }}
.hero p {{ margin:0; color:#CFE3EA; max-width:760px; }}
.kpi {{ background:#fff; border:1px solid {LINE}; border-left:5px solid {TEAL}; border-radius:10px; padding:12px 16px; }}
.kpi b {{ font-family:'Space Grotesk'; font-size:1.7rem; display:block; }}
.kpi span {{ color:#5B6F80; font-size:.85rem; }}
.kpi.hot {{ border-left-color:{HOT}; }} .kpi.amb {{ border-left-color:{AMBER}; }}
.note {{ background:#FFF6E8; border:1px solid {AMBER}; border-radius:10px; padding:10px 14px; font-size:.9rem; }}
section[data-testid="stSidebar"] {{ background:#fff; border-right:1px solid {LINE}; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def make_demo(seed=7, n=7000):
    """Clearly SYNTHETIC incidents around Pune. No real crime data is used."""
    rng = np.random.default_rng(seed)
    hubs = [("Shivajinagar", 18.5314, 73.8446, .22, .006, .3), ("Hadapsar", 18.5089, 73.9260, .17, .008, .6),
            ("Kothrud", 18.5074, 73.8077, .13, .007, .4), ("Hinjewadi", 18.5912, 73.7389, .14, .008, .5),
            ("Pimpri", 18.6298, 73.7997, .15, .007, .7), ("Swargate", 18.5018, 73.8636, .19, .005, .5)]
    w = np.array([h[3] for h in hubs]); w /= w.sum()
    start, days = pd.Timestamp("2025-10-28"), 365
    idx = rng.choice(len(hubs), n, p=w)
    rows = []
    for i, h in zip(range(n), idx):
        _, la, lo, _, sp, nb = hubs[h]
        night = rng.random() < nb
        hr = int(rng.normal(23.5, 2.2)) % 24 if night else int(rng.choice([11, 13, 18, 20], p=[.2, .2, .3, .3]) + rng.integers(-1, 2)) % 24
        d = int(rng.integers(0, days))
        if h == 3 and rng.random() < .35: d = days - int(rng.integers(1, 60))  # emerging hotspot
        cat = rng.choice(CATS, p=np.roll([.3, .15, .15, .2, .1, .1], h % 3))
        rows.append((start + pd.Timedelta(days=d, hours=hr, minutes=int(rng.integers(0, 60))), cat,
                     la + rng.normal(0, sp), lo + rng.normal(0, sp), hubs[h][0]))
    bg = int(n * .12)
    for _ in range(bg):
        rows.append((start + pd.Timedelta(days=int(rng.integers(0, days)), hours=int(rng.integers(0, 24))),
                     rng.choice(CATS), rng.uniform(18.45, 18.66), rng.uniform(73.72, 73.95), "Other"))
    df = pd.DataFrame(rows, columns=["incident_datetime", "crime_category", "latitude", "longitude", "police_station_area"])
    df.insert(0, "incident_id", [f"SYN-{i:05d}" for i in range(len(df))])
    return df


def clean(raw):
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return None, {"error": f"Missing required column(s): {', '.join(miss)}"}
    rep = {"Rows received": len(df)}
    df["incident_datetime"] = pd.to_datetime(df["incident_datetime"], errors="coerce")
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rep["Invalid or missing dates"] = int(df.incident_datetime.isna().sum())
    bad = df.latitude.isna() | df.longitude.isna() | ~df.latitude.between(-90, 90) | ~df.longitude.between(-180, 180)
    rep["Invalid coordinates"] = int(bad.sum())
    df = df[~bad & df.incident_datetime.notna()].copy()
    df["crime_category"] = df["crime_category"].astype(str).str.strip().str.title()
    if "police_station_area" not in df.columns:
        df["police_station_area"] = "All"
    df["police_station_area"] = df["police_station_area"].fillna("Unknown").astype(str).str.strip()
    before = len(df)
    df = df.drop_duplicates(subset=["incident_datetime", "crime_category", "latitude", "longitude"])
    rep["Duplicates removed"] = before - len(df)
    rep["Usable records"] = len(df)
    rep["Completeness %"] = round(100 * len(df) / max(rep["Rows received"], 1), 1)
    return df.reset_index(drop=True), rep


# ---------------------------------------------------------------- spatial
def origin(df, s):
    lat0, lon0 = math.floor(df.latitude.min() / s) * s, math.floor(df.longitude.min() / s) * s
    return lat0, lon0, int((df.latitude.max() - lat0) // s) + 1, int((df.longitude.max() - lon0) // s) + 1


def grid_counts(df, s, o):
    lat0, lon0, ny, nx = o
    g = np.zeros((ny, nx))
    iy = ((df.latitude - lat0) // s).astype(int); ix = ((df.longitude - lon0) // s).astype(int)
    np.add.at(g, (iy, ix), 1)
    return g


def gi_star(g):
    """Getis-Ord Gi* z-scores on a grid, queen-contiguity binary weights (self included)."""
    n, xbar = g.size, g.mean()
    S = math.sqrt(max((g ** 2).mean() - xbar ** 2, 1e-12))
    k = np.ones((3, 3))
    W = convolve(np.ones_like(g), k, mode="constant")
    num = convolve(g, k, mode="constant") - xbar * W
    den = S * np.sqrt((n * W - W ** 2) / (n - 1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def hotspot_table(g, z, o, s):
    lat0, lon0, _, _ = o
    iy, ix = np.nonzero((g > 0) & (z >= 1.65))
    t = pd.DataFrame({"iy": iy, "ix": ix, "incidents": g[iy, ix].astype(int), "z": z[iy, ix].round(2)})
    t["confidence"] = np.where(t.z >= 2.58, "99%", np.where(t.z >= 1.96, "95%", "90%"))
    t["lat"] = lat0 + (t.iy + .5) * s; t["lon"] = lon0 + (t.ix + .5) * s
    return t.sort_values("z", ascending=False).reset_index(drop=True)


def base_map(df, zoom=12):
    return folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=zoom, tiles="OpenStreetMap")


def show(m, h=560):
    st_folium(m, height=h, use_container_width=True, returned_objects=[], key=f"m{id(m)}")


def hotspot_map(df, tbl, s, heat=True):
    m = base_map(df)
    colors = {"99%": "#8E0F3B", "95%": HOT, "90%": AMBER}
    fg = folium.FeatureGroup("Gi* hotspot cells")
    for r in tbl.itertuples():
        folium.Rectangle([(r.lat - s / 2, r.lon - s / 2), (r.lat + s / 2, r.lon + s / 2)], color=colors[r.confidence],
                         weight=1, fill=True, fill_opacity=.55,
                         tooltip=f"{r.incidents} incidents | z={r.z} | {r.confidence} confidence").add_to(fg)
    fg.add_to(m)
    if heat:
        HeatMap(df[["latitude", "longitude"]].values.tolist(), radius=14, blur=18, min_opacity=.25,
                name="Density (KDE-style heat)", show=False).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def hav(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1]); dp = p2 - p1
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def route_len(pts):
    return sum(hav(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def nn_route(pts):
    left, route = list(range(1, len(pts))), [0]
    while left:
        nxt = min(left, key=lambda j: hav(pts[route[-1]], pts[j]))
        route.append(nxt); left.remove(nxt)
    return route


# ---------------------------------------------------------------- prediction
@st.cache_data(show_spinner=False)
def predict(df, s, o, thr, K):
    lat0, lon0, _, nx = o
    d = df.copy()
    d["cell"] = ((d.latitude - lat0) // s).astype(int) * nx + ((d.longitude - lon0) // s).astype(int)
    d["week"] = d.incident_datetime.dt.to_period("W").dt.start_time
    P = d.groupby(["week", "cell"]).size().unstack(fill_value=0)
    P = P.reindex(pd.date_range(P.index.min(), P.index.max(), freq="7D"), fill_value=0)
    T, C = P.shape
    if T < 14:
        return None
    f = {"cur": P, "lag1": P.shift(1), "lag2": P.shift(2), "m4": P.rolling(4, min_periods=1).mean(),
         "m8": P.rolling(8, min_periods=1).mean(), "cum": P.expanding().mean()}
    X = pd.DataFrame({k: v.fillna(0).values.ravel() for k, v in f.items()})
    X["wk"] = np.repeat(np.arange(T), C); X["cell"] = np.tile(P.columns.values, T)
    X["clat"] = lat0 + (X.cell // nx + .5) * s; X["clon"] = lon0 + (X.cell % nx + .5) * s
    X["y"] = (P.shift(-1) >= thr).values.ravel().astype(int)
    feats = ["cur", "lag1", "lag2", "m4", "m8", "cum", "clat", "clon"]
    split = int(T * .7)
    tr, te = X[(X.wk >= 4) & (X.wk < split)], X[(X.wk >= split) & (X.wk < T - 1)]
    if tr.y.nunique() < 2 or te.empty:
        return None
    rf = RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(tr[feats], tr.y)
    te = te.assign(p=rf.predict_proba(te[feats])[:, 1])
    def p_at_k(col):
        return float(np.mean([w.sort_values(col, ascending=False).head(K).y.mean() for _, w in te.groupby("wk")]))
    yh = (te.p >= .5).astype(int)
    res = {"weeks": T, "test_weeks": int(te.wk.nunique()), "base_rate": float(te.y.mean()),
           "p_at_k_rf": p_at_k("p"), "p_at_k_base": p_at_k("m8"),
           "precision": precision_score(te.y, yh, zero_division=0), "recall": recall_score(te.y, yh, zero_division=0),
           "f1": f1_score(te.y, yh, zero_division=0)}
    last = X[X.wk == T - 1].copy()
    last["risk"] = rf.predict_proba(last[feats])[:, 1]
    res["forecast"] = last.sort_values("risk", ascending=False).head(K)[["clat", "clon", "risk", "m4"]].reset_index(drop=True)
    res["next_week"] = (P.index[-1] + pd.Timedelta(days=7)).date()
    return res


# ---------------------------------------------------------------- sidebar
st.markdown("""<div class="hero"><h1>Crime Hotspot Atlas</h1>
<p>A GIS-enabled analytics and predictive-policing prototype. It finds where recorded incidents concentrate,
tests whether history can anticipate next week, and drafts advisory patrol routes for human review.</p></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    src = st.radio("Source", ["Synthetic demo (Pune)", "Upload CSV"], label_visibility="collapsed")
    raw = None
    if src.startswith("Upload"):
        up = st.file_uploader("CSV with incident_datetime, crime_category, latitude, longitude", type="csv")
        if up: raw = pd.read_csv(up)
    else:
        raw = make_demo()
        st.download_button("Download demo CSV (template)", raw.to_csv(index=False), "demo_incidents.csv", "text/csv")
    if raw is None:
        st.info("Upload a CSV to begin, or switch to the synthetic demo."); st.stop()
    data, rep = clean(raw)
    if data is None:
        st.error(rep["error"]); st.stop()

    st.header("Filters")
    lo, hi = data.incident_datetime.min().date(), data.incident_datetime.max().date()
    dr = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    cats = st.multiselect("Crime category", sorted(data.crime_category.unique()), default=sorted(data.crime_category.unique()))
    areas = st.multiselect("Station area", sorted(data.police_station_area.unique()), default=sorted(data.police_station_area.unique()))
    cell = st.select_slider("Grid cell size", [0.003, 0.005, 0.008], 0.005, format_func=lambda v: {0.003: "~350 m", 0.005: "~550 m", 0.008: "~900 m"}[v])
    heat = st.checkbox("Add density heat layer", True)

if len(dr) != 2: st.stop()
f = data[data.incident_datetime.dt.date.between(*dr) & data.crime_category.isin(cats) & data.police_station_area.isin(areas)]
if len(f) < 30:
    st.warning("Fewer than 30 incidents match these filters. Widen them to see hotspots."); st.stop()

o = origin(data, cell)
g = grid_counts(f, cell, o); z = gi_star(g); tbl = hotspot_table(g, z, o, cell)

tabs = st.tabs(["Hotspot map", "Patterns over time", "Next-week risk", "Patrol planning", "Data quality & responsible use"])

# ---- Tab 1
with tabs[0]:
    k = st.columns(4)
    top_area = f.police_station_area.value_counts().idxmax()
    for col, (cls, val, lab) in zip(k, [("", f"{len(f):,}", "Incidents analysed"), ("hot", int((tbl.confidence == "99%").sum()), "Hotspot cells at 99% confidence"),
                                        ("amb", int(len(tbl)), "Hotspot cells at 90% or higher"), ("", top_area, "Area with most incidents")]):
        col.markdown(f'<div class="kpi {cls}"><b>{val}</b><span>{lab}</span></div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([2.2, 1])
    with a:
        show(hotspot_map(f, tbl, cell, heat))
        st.caption("Cells are shaded by Getis-Ord Gi* confidence: dark red 99%, red 95%, amber 90%. Toggle the heat layer in the map's layer control.")
    with b:
        st.subheader("Top hotspot cells")
        st.dataframe(tbl[["lat", "lon", "incidents", "z", "confidence"]].head(10).round(4), hide_index=True, width="stretch")
        st.download_button("Export hotspot cells (aggregate only)", tbl.to_csv(index=False), "hotspot_cells.csv", "text/csv")
        st.subheader("Rising-activity watchlist")
        end = f.incident_datetime.max()
        rec = f[f.incident_datetime > end - pd.Timedelta(days=28)].police_station_area.value_counts()
        prv = f[(f.incident_datetime <= end - pd.Timedelta(days=28)) & (f.incident_datetime > end - pd.Timedelta(days=56))].police_station_area.value_counts()
        w = pd.DataFrame({"last 28 days": rec, "previous 28": prv}).fillna(0).astype(int)
        w["change %"] = ((w["last 28 days"] - w["previous 28"]) / w["previous 28"].replace(0, np.nan) * 100).round(0)
        w = w[w["last 28 days"] >= 20].sort_values("change %", ascending=False)
        st.dataframe(w, width="stretch")
        st.caption(f"Data last updated {end:%d %b %Y}. A rise in counts is an indicator for review, not a confirmed trend.")

# ---- Tab 2
with tabs[1]:
    c1, c2 = st.columns(2)
    wk = f.set_index("incident_datetime").groupby([pd.Grouper(freq="W"), "crime_category"]).size().reset_index(name="incidents")
    c1.plotly_chart(px.line(wk, x="incident_datetime", y="incidents", color="crime_category", title="Weekly incidents by category",
                            color_discrete_sequence=px.colors.qualitative.Dark24).update_layout(xaxis_title=None, height=380), width="stretch")
    cc = f.crime_category.value_counts().reset_index()
    c2.plotly_chart(px.bar(cc, x="count", y="crime_category", orientation="h", title="Incidents by category",
                           color_discrete_sequence=[TEAL]).update_layout(yaxis_title=None, height=380), width="stretch")
    hw = f.assign(day=f.incident_datetime.dt.day_name(), hour=f.incident_datetime.dt.hour).groupby(["day", "hour"]).size().unstack(fill_value=0)
    hw = hw.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reindex(columns=range(24), fill_value=0)
    st.plotly_chart(px.imshow(hw, aspect="auto", color_continuous_scale=["#EAF2F4", AMBER, HOT, "#4A0D25"],
                              title="When incidents are recorded: day of week by hour", labels=dict(x="Hour of day", y="", color="Incidents")).update_layout(height=360),
                    width="stretch")

# ---- Tab 3
with tabs[2]:
    st.markdown("**Question tested:** will a grid cell record at least *N* incidents next week? The model trains on the earliest 70% of weeks and is scored on the later 30%, so it never sees the future.")
    c1, c2 = st.columns(2)
    thr = c1.slider("A cell counts as 'high' at N incidents per week", 1, 6, 2)
    K = c2.slider("Cells to flag each week (K)", 5, 30, 10)
    res = predict(f, cell, o, thr, K)
    if res is None:
        st.warning("Not enough weekly history, or only one outcome class, for a fair test. Use a longer date range or a lower threshold.")
    else:
        m = st.columns(4)
        m[0].metric("Precision@K, Random Forest", f"{res['p_at_k_rf']:.0%}")
        m[1].metric("Precision@K, 8-week average baseline", f"{res['p_at_k_base']:.0%}")
        m[2].metric("Random pick (base rate)", f"{res['base_rate']:.0%}")
        m[3].metric("Test weeks", res["test_weeks"])
        st.caption(f"Random Forest at a 0.5 cut-off: precision {res['precision']:.2f}, recall {res['recall']:.2f}, F1 {res['f1']:.2f}. "
                   "Report these with the test design above, and treat a model that does not beat the baseline as a finding, not a failure.")
        fc = res["forecast"]
        st.subheader(f"Highest estimated relative risk, week starting {res['next_week']}")
        mm = base_map(f)
        for r in fc.itertuples():
            folium.CircleMarker([r.clat, r.clon], radius=8 + 22 * r.risk, color=HOT, fill=True, fill_opacity=.5,
                                tooltip=f"Estimated relative risk {r.risk:.0%}").add_to(mm)
        show(mm, 440)
        st.session_state["forecast"] = fc

# ---- Tab 4
with tabs[3]:
    st.markdown('<div class="note"><b>Advisory only.</b> Routes are a planning aid built from straight-line distances between cells. '
                'They do not dispatch anyone, and they ignore roads, traffic, staffing, and local knowledge. Authorized officers decide.</div>', unsafe_allow_html=True)
    n = st.slider("Priority cells to cover", 3, 12, 6)
    src_tbl = st.radio("Prioritise by", ["Current Gi* hotspots", "Next-week forecast"], horizontal=True)
    if src_tbl.startswith("Next") and "forecast" in st.session_state:
        pts_df = st.session_state["forecast"].head(n).rename(columns={"clat": "lat", "clon": "lon"})
    else:
        if src_tbl.startswith("Next"): st.info("Open the Next-week risk tab first. Showing current hotspots meanwhile.")
        pts_df = tbl.head(n)
    pts = list(zip(pts_df.lat, pts_df.lon))
    if len(pts) >= 3:
        order = nn_route(pts); rp = [pts[i] for i in order]
        naive, smart = route_len(pts), route_len(rp)
        a, b, c = st.columns(3)
        a.metric("Priority-order route", f"{naive:.1f} km"); b.metric("Nearest-neighbour route", f"{smart:.1f} km", f"{smart - naive:+.1f} km")
        c.metric("Stops", len(rp))
        mp = base_map(f)
        folium.PolyLine(rp, color=TEAL, weight=4, opacity=.85).add_to(mp)
        for i, p in enumerate(rp, 1):
            folium.Marker(p, icon=folium.DivIcon(html=f'<div style="background:{HOT};color:#fff;border-radius:50%;width:26px;height:26px;text-align:center;line-height:26px;font-weight:700">{i}</div>')).add_to(mp)
        show(mp, 500)
        st.caption("Assumes one unit, starting at the first-ranked cell, visiting each cell once. Stop numbers show the suggested visiting order.")

# ---- Tab 5
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data-quality report")
        st.dataframe(pd.Series(rep, name="value").astype(str), width="stretch")
        if src.startswith("Synthetic"):
            st.warning("This demo uses synthetic data generated for interface testing. It says nothing about real crime in Pune.")
    with c2:
        st.subheader("Responsible use")
        st.markdown("""
- A hotspot is a concentration in **recorded** incidents. It can reflect reporting or enforcement patterns, not only crime.
- Never use these outputs as the sole basis to stop, search, or target a person or community.
- The tool works on aggregate cells. It stores no names, victim details, or personal identifiers.
- Model estimates are relative and uncertain. Check the test period, baseline, and data freshness first.
- Real deployment needs authorization, role-based access, audit logs, privacy review, and field validation. None are implemented here.""")
