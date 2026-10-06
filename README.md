# 🌊 CWC Flood Monitoring & Predictive Dashboard

Interactive, web-based **Flood Monitoring & Predictive Dashboard** built with
**Streamlit + Plotly + scikit-learn**, driven by the Central Water Commission
(CWC) telemetry workbook `cwc_telemetry_flood_ready_50000.xlsx` (Cauvery basin,
23 stations, Jan–May 2026 hourly readings).

## Features

| Box | What it does |
|-----|--------------|
| **Top bar** | Station search (type-to-filter), `State` / `Basin` / `River` / **Date Range** filters, and 4 KPI cards: Total Stations · Avg Rainfall · Warning/Danger Stations · Total Flood Events |
| **Box 1** | Station profile: metadata (river, basin, district, lat/lon, zero-gauge RL, MSL…), current status **gauge** (Normal / Warning / Danger), latest WL, warning & danger levels, 1-hour change |
| **Box 2** | **Flood Possibility Calculator** — RandomForest classifier on `Is_Flood`. Metrics (Accuracy / Precision / Recall / ROC-AUC), confusion matrix, ROC curve, sliders for water level / 1h change / rainfall, **probability gauge (0–100 %)**, risk badge (`SAFE < 30 %`, `MODERATE 30–70 %`, `CRITICAL > 70 %`), **local feature-contribution explanation** + global importance |
| **Box 3** | Hourly water-level line chart with **yellow dashed warning**, **red dashed danger** lines and **shaded red flood bands** (`Is_Flood = 1`) |
| **Box 4** | Interactive GIS map (OpenStreetMap, no API token) — markers coloured green/yellow/red by status, hover shows name, level, rainfall & flood probability, **click a marker to make it the active station** |
| **Box 5** | Dual-axis **Rainfall vs. Water-Surge** chart + searchable telemetry log with flood rows highlighted in soft red |

## Project layout

```
flood_dashboard/
├── app.py                     # Streamlit UI (all 5 boxes + top bar)
├── data.py                    # loading, cleaning, de-duplication, feature engineering
├── model.py                   # RandomForest Is_Flood classifier + metrics + explanations
├── charts.py                  # all Plotly figures
├── cwc_telemetry_flood_ready_50000.xlsx   # dataset (copied next to app.py)
├── requirements.txt
├── smoke_test.py              # headless UI test (streamlit.testing.v1)
├── .streamlit/config.toml     # theme + server config
└── README.md
```

## Data preparation (what happens at load time)

1. `Data Acquisition Time` parsed with format `%d-%m-%Y %H:%M`.
2. The source workbook stores the same telemetry **2–3× (50,000 raw rows →
   20,507 unique `(Station, timestamp)` readings)** — duplicates are dropped.
3. Margin features engineered: `Gap_to_Warning_m = WL − Warning`,
   `Gap_to_Danger_m = WL − Danger`.
4. `Is_Flood` is the binary target.

### Known data-quality notes (shown in the app sidebar)

* **`Rainfall_mm` is constant 0** across the entire source workbook. The column
  is kept as a model feature (per spec) but contributes no signal until real
  rain-gauge data is provided — the UI flags this explicitly, and the rainfall
  slider lets you explore hypothetical values.
* In this dataset `Is_Flood == (water level ≥ danger level)` exactly, so the
  model learns the threshold near-perfectly. Metrics are sanity checks rather
  than estimates of real-world generalisation.

## 1) Run locally

```bash
# Python 3.10+ recommended
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Dashboard opens at **http://localhost:8501**.

Optional checks:

```bash
python smoke_test.py     # headless render + interaction test
```

The app finds the workbook in this order: `CWC_DATA_PATH` env var → folder
next to `app.py` → `./data/` → repo root → OneDrive Desktop.

## 2) Deploy to Streamlit Community Cloud (recommended, free)

1. Push this folder to a **GitHub repository** (the xlsx is ~6 MB — fine for GitHub).
2. Go to <https://share.streamlit.io> → **Create app** → *Deploy a public repo
   from GitHub*.
3. Pick the repo, set **Main file path** to `app.py`, click **Deploy**.
4. Streamlit Cloud installs `requirements.txt` automatically; the workbook is
   read from the repo. Done — your app lives at
   `https://<your-app-name>.streamlitapp.com`.

> Tip: keep `.streamlit/config.toml` in the repo so the theme travels with it.

## 3) Deploy to Vercel

**Vercel does not natively host long-running Streamlit servers** (its Python
runtime is serverless-functions-only). Honest options:

* **Recommended:** Streamlit Community Cloud (above) or **Render / Railway /
  Fly.io** free tiers — all support `streamlit run app.py` directly.
* **Vercel + container:** wrap the app in a `Dockerfile` (below) and deploy the
  image to a container platform.
* **Vercel-native path:** port the UI to Next.js + Tailwind + Plotly and call a
  Python FastAPI/scikit-learn endpoint as a Vercel serverless function. The
  `model.py` / `charts.py` logic ports 1:1 — but that is a rewrite, not a
  config change.

### Minimal Dockerfile (works on any container host)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## ML details

* **Algorithm:** RandomForest (300 trees, depth 14, `class_weight=balanced_subsample`)
  — swap to XGBoost by replacing `model.train_model()` if desired.
* **Features:** `River Water Level Telemetry Hourly (meter)`, `Rainfall_mm`,
  `Water_Level_Change_1h`, `Warning_Level_m`, `Danger_Level_m`,
  `Gap_to_Warning_m`, `Gap_to_Danger_m`.
* **Split:** stratified 80/20, `random_state=42`.
* **Local explanation:** **SHAP TreeExplainer** (exact for tree models) —
  base rate + each feature's contribution = predicted probability; with a
  leave-one-feature-at-baseline ablation fallback if SHAP is unavailable.

