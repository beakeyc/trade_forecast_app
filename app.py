# app.py
import io
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Trade Price Forecaster", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def load_data(sheet: pd.DataFrame) -> pd.DataFrame:
    """Take a raw sheet, keep cols B & E, coerce to numeric, quarterly index."""
    df = sheet.iloc[:, [1, 4]].copy()  # B and E (0-indexed)
    df.columns = ["date", "price"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"]).sort_values("date")
    df.set_index("date", inplace=True)

    # Ensure a proper quarterly frequency; keep last observation in each quarter
    df = df.resample("Q").last()

    return df


def auto_arima_like(y: pd.Series, p_max=3, d_max=2, q_max=3):
    """
    Super-light ARIMA (p,d,q) grid search using AIC.
    Returns (fitted_result, best_order) or (None, None) if nothing fits.
    """
    y = y.dropna().astype(float)
    n = len(y)

    # With very short samples, shrink the search space
    if n < 24:
        p_max, d_max, q_max = 2, 1, 2

    best_aic = np.inf
    best_res, best_order = None, None

    for p, d, q in itertools.product(range(p_max + 1),
                                     range(d_max + 1),
                                     range(q_max + 1)):
        try:
            res = SARIMAX(
                y,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_res = res
                best_order = (p, d, q)
        except Exception:
            continue

    return best_res, best_order


def forecast_arima(df: pd.DataFrame, periods: int = 8):
    """
    Try to fit an ARIMA; if it fails, fall back to a naive forecast.
    Returns (model_or_none, order_or_none, forecast_series)
    """
    y = df["price"].astype(float)
    model, order = auto_arima_like(y)

    # fallback to a very simple ARIMA if search failed
    if model is None:
        try:
            model = SARIMAX(
                y, order=(0, 1, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            order = (0, 1, 0)
        except Exception:
            model = None
            order = None

    # build forecast index (next 'periods' quarters)
    forecast_index = pd.date_range(
        df.index[-1] + pd.offsets.QuarterEnd(),
        periods=periods,
        freq="Q"
    )

    if model is None:
        # final fallback: naive (repeat last value)
        st.warning("Could not fit an ARIMA model. Returning a naive forecast.")
        last = y.iloc[-1]
        forecast = pd.Series([last] * periods, index=forecast_index)
        return None, None, forecast

    forecast = model.get_forecast(steps=periods).predicted_mean
    # ensure clean quarterly index for plotting / export
    forecast.index = forecast_index
    return model, order, forecast


def plot_model_fit(df: pd.DataFrame, model, label: str):
    if model is None:
        st.info(f"{label}: not enough data / model didn't fit. Skipping fit plot.")
        return

    history = df["price"]
    recent = history[-8:]  # last 2 years if quarterly
    fitted = model.fittedvalues.reindex(recent.index)

    fig, ax = plt.subplots()
    ax.plot(recent.index, recent.values, label="Actual", marker="o")
    ax.plot(fitted.index, fitted.values, label="Fitted", linestyle="--", marker="x")
    ax.set_title(f"{label} – Model Fit (Last 2 Years)")
    ax.set_ylabel("YoY % Change")
    ax.legend()
    st.pyplot(fig)


def plot_forecast(df: pd.DataFrame, forecast: pd.Series, label: str):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["price"], label="Historical", marker="o")
    ax.plot(forecast.index, forecast.values, label="Forecast", linestyle="--", marker="x")
    ax.set_title(f"{label} – Forecast (Next 2 Years)")
    ax.set_ylabel("YoY % Change")
    ax.legend()
    st.pyplot(fig)


# -----------------------------
# UI
# -----------------------------
st.title("📈 Trade Price Forecasting App")
st.markdown(
    "Upload an Excel file with **'exports'** and **'imports'** sheets. "
    "Each should have Date in column **B** and YoY % Change in column **E**."
)

uploaded_file = st.file_uploader("Upload `input.xlsx`", type=["xlsx"])

if uploaded_file:
    try:
        xl = pd.read_excel(uploaded_file, sheet_name=None)
        forecast_outputs = {}

        for label in ["exports", "imports"]:
            if label not in xl:
                st.warning(f"Missing sheet: `{label}`")
                continue

            st.header(f"🧾 {label.capitalize()} Data")

            df = load_data(xl[label])

            if df.empty or not pd.api.types.is_numeric_dtype(df["price"]):
                st.error(f"{label}: couldn't parse numeric 'price' values.")
                continue

            st.dataframe(df.tail())

            model, order, forecast = forecast_arima(df)

            if order is not None:
                st.caption(f"Selected ARIMA order for {label}: {order}")

            plot_model_fit(df, model, label.capitalize())
            plot_forecast(df, forecast, label.capitalize())

            forecast_df = pd.DataFrame({
                "Date": forecast.index,
                "Forecast YoY % Chg": forecast.values
            }).set_index("Date")
            st.subheader(f"{label.capitalize()} Forecast Table")
            st.dataframe(forecast_df.style.format("{:.2f}"))
            forecast_outputs[label] = forecast_df

        if forecast_outputs:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                for label, df_out in forecast_outputs.items():
                    df_out.to_excel(writer, sheet_name=label.capitalize())

            st.download_button(
                label="📥 Download Forecast Excel",
                data=output.getvalue(),
                file_name="forecast_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
