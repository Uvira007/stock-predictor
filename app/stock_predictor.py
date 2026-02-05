"""Streamlit UI: predict, chart, retrain (with caution)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Backend API URL: SET API_BASE_URL in streamlit Cloud to your Render API URL
API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
# Longer timeouts for hosted API to support cold starts (e.g. Render free tier)
API_TIMEOUT_GET = int(os.environ.get("API_TIMEOUT_GET", "60"))
API_TIMEOUT_POST = int(os.environ.get("API_TIMEOUT_POST", "120"))


def api_get(path: str):
    r = requests.get(f"{API_BASE}{path}", timeout=API_TIMEOUT_GET)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json: dict):
    r = requests.post(f"{API_BASE}{path}", json=json, timeout=API_TIMEOUT_POST)
    r.raise_for_status()
    return r.json()


def main():
    st.set_page_config(page_title="Stock GRU Predictor", page_icon="📈", layout="centered")
    st.title("Stock GRU Predictor")
    st.caption("Multivariate GRU model · Yahoo Finance · Next N days close")

    # Check API and model status
    try:
        status = api_get("/model/status")
    except requests.RequestException:
        st.error("" \
        "Backend API is not reachable. Locally: run `pthon scripts/run_api.py`. "
        "If deployed: set API_BASE_URL to your render API URL")
        st.stop()
    if not status.get("has_checkpoint"):
        st.warning("No trained model found. Train once: `python scripts/train_model.py` from the stock_predictor folder, then restart the API.")
    else:
        st.success("Model loaded and ready.")

    # Tickers from API
    try:
        tickers_data = api_get("/tickers")
        tickers = tickers_data.get("tickers", [])
    except Exception:
        tickers = []

    if not tickers:
        st.info("No tickers available until a model is trained.")
        _render_retrain_section()
        return

    # Sidebar: ticker, n_days
    st.sidebar.header("Prediction")
    ticker = st.sidebar.selectbox("Ticker", options=tickers, index=0)
    n_days_options = [1, 5, 10, 21]
    n_days = st.sidebar.selectbox("Predict next N days", options=n_days_options, index=3)
    if st.sidebar.button("Predict", type="primary"):
        with st.spinner("Fetching latest data and predicting..."):
            try:
                out = api_post("/predict", json={"ticker": ticker, "n_days": n_days})
                st.session_state["last_prediction"] = out
            except requests.RequestException as e:
                st.sidebar.error(str(e))

    # Main: chart
    if "last_prediction" in st.session_state:
        out = st.session_state["last_prediction"]
        preds = out["predictions"]
        last_date = out["last_date"]
        st.subheader(f"{out['ticker']} — Next {len(preds)} days (from {last_date})")
        df = pd.DataFrame({"Day": range(1, len(preds) + 1), "Predicted Close": preds})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Day"], y=df["Predicted Close"], mode="lines+markers", name="Predicted"))
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Predicted Close (USD)",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Select a ticker and click **Predict** to see forecasts.")

    # Retrain section (with caution)
    _render_retrain_section()


def _render_retrain_section():
    st.sidebar.divider()
    st.sidebar.header("Retrain model")
    st.sidebar.caption("Use cautiously — can take several minutes.")
    if st.sidebar.button("Start retrain (background)"):
        with st.sidebar.spinner("Starting retrain..."):
            try:
                api_post("/retrain", json={"epochs": 50, "period": "5y"})
                st.sidebar.success("Retrain started in background. Model will reload when done.")
            except requests.RequestException as e:
                st.sidebar.error(str(e))


if __name__ == "__main__":
    main()
