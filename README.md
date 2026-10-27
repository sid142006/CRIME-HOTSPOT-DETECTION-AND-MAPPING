# Crime Hotspot Atlas

**Crime Hotspot Detection and Mapping: A GIS-Enabled Data Analytics and Predictive Policing Prototype**

An academic mini-project that turns historical incident records into location-based insight: statistically tested hotspots, time patterns, a next-week risk model checked against a baseline, and advisory patrol routes. It is a decision-support prototype, not an operational policing tool.

> **Data notice.** The built-in demo uses **synthetic** incidents generated around Pune for interface testing. It says nothing about real crime. Use only data you are authorized to use.

## Features

| Module | What it does |
|---|---|
| Data validation | Parses dates, validates coordinates, normalises categories, removes duplicates, reports completeness |
| Hotspot map | Grid aggregation with **Getis-Ord Gi\*** z-scores (90/95/99% confidence), optional density heat layer |
| Filters | Date range, crime category, station area, grid cell size (~350 / 550 / 900 m) |
| Patterns | Weekly trends by category, category totals, day-by-hour heatmap |
| Watchlist | Areas whose last 28 days rose against the prior 28 (flag for review, not a confirmed trend) |
| Next-week risk | Random Forest with a **chronological** 70/30 split, compared with an 8-week-average baseline using precision@K, plus precision, recall, F1 |
| Patrol planning | Nearest-neighbour route over top-priority cells versus priority order. Advisory only |
| Responsible use | Data-quality report, limitations, aggregate-only export |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+. The map uses OpenStreetMap tiles, so you need an internet connection.

## Using your own data

Choose **Upload CSV** in the sidebar. Required columns:

| Column | Example |
|---|---|
| `incident_datetime` | `2026-03-14 22:30` |
| `crime_category` | `Theft` |
| `latitude` | `18.5314` |
| `longitude` | `73.8446` |

Optional: `incident_id` (pseudonymous), `police_station_area`. Remove names, phone numbers, victim details, and exact sensitive addresses before uploading. The sidebar's **Download demo CSV** button gives you a template.

## How it works

1. **Clean:** invalid dates and coordinates dropped and counted, categories title-cased, duplicates removed on time + category + location.
2. **Aggregate:** incidents counted in square grid cells whose origin is fixed from the full dataset, so cells stay stable when you change filters.
3. **Detect:** Gi\* with queen-contiguity binary weights (self included). Cells with z ≥ 1.65 / 1.96 / 2.58 are shown at 90 / 95 / 99%.
4. **Predict:** weekly counts per cell. Features: current, lag-1, lag-2, 4- and 8-week means, running mean, cell centre. Target: next-week count ≥ N. Train on the earliest 70% of weeks, score on the rest, so no future data leaks in.
5. **Plan:** top cells ordered by nearest-neighbour using haversine distance, compared with visiting them in priority order.

## Reporting results honestly

- Record the dataset source, licence or permission, date range, and cell size you used.
- Always quote the test period, target (N and K), and the baseline next to any model score.
- On the demo data the Random Forest roughly **ties** the 8-week baseline, because the synthetic hotspots are stable. A tie is a legitimate result. Do not claim crime reduction without a controlled field study.
- Hotspots reflect *recorded* incidents and may carry reporting or enforcement bias.

## Limitations

Straight-line routes ignore roads and staffing. No live feeds, police-system integration, role-based access, or audit logs are implemented (future scope in the report). Gi\* assumes a regular grid and is sensitive to cell size, so compare several sizes.

## Project structure

```
app.py            Streamlit application (all modules)
requirements.txt  Dependencies
README.md         This file
```

## Report mapping

Chapter 3 stack → Python, Pandas, NumPy, scikit-learn, Folium, Plotly, Streamlit. GeoPandas/Shapely are not needed for the grid method and are left out to keep installation light. Add them if you join incidents to GeoJSON police-station boundaries.

## Smart India Hackathon note

Confirm the official SIH problem-statement ID and wording from your team's assigned statement. The matching public statement cited in the report (KANADSHIELD26_P2_07) is a KANAD S.H.I.E.L.D. identifier, not an SIH ID.
