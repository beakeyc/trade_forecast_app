
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import datetime
import io

st.set_page_config(page_title="Trade Price Forecaster", layout="wide")

def load_data(sheet):
    df = sheet.iloc[:, [1, 4]]  # Columns B and E (0-indexed)
    df.columns = ["date", "price"]
    df.dropna(inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.asfreq("Q")
    return df

def forecast_arima(df, periods=8):
    model = auto_arima(df["price"], seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=periods)
    forecast_index = pd.date_range(df.index[-1] + pd.offsets.QuarterEnd(), periods=periods, freq="Q")
    forecast_series = pd.Series(forecast, index=forecast_index)
    return model, forecast_series

def plot_model_fit(df, model, label):
    history = df["price"]
    recent = history[-8:]
    fitted = pd.Series(model.predict_in_sample()[-8:], index=recent.index)
    fig, ax = plt.subplots()
    ax.plot(recent.index, recent.values, label="Actual", marker='o')
    ax.plot(fitted.index, fitted.values, label="Fitted", linestyle="--", marker='x')
    ax.set_title(f"{label} – Model Fit (Last 2 Years)")
    ax.set_ylabel("YoY % Change")
    ax.legend()
    st.pyplot(fig)

def plot_forecast(df, forecast, label):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["price"], label="Historical", marker='o')
    ax.plot(forecast.index, forecast.values, label="Forecast", linestyle="--", marker='x')
    ax.set_title(f"{label} – Forecast (Next 2 Years)")
    ax.set_ylabel("YoY % Change")
    ax.legend()
    st.pyplot(fig)

st.title("📈 Trade Price Forecasting App")
st.markdown("Upload an Excel file with **'exports'** and **'imports'** sheets. Each should have Date in column B and YoY % Change in column E.")

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
            st.dataframe(df.tail())

            model, forecast = forecast_arima(df)

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
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for label, df_out in forecast_outputs.items():
                    df_out.to_excel(writer, sheet_name=label.capitalize())
            st.download_button(
                label="📥 Download Forecast Excel",
                data=output.getvalue(),
                file_name="forecast_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
