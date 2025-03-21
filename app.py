import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
import datetime
import openai
from scipy.signal import find_peaks

# Load secrets
polygon_api_key = st.secrets["POLYGON_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------------------ Data Fetching ------------------------
@st.cache_data(ttl=900)
def fetch_polygon_data(ticker, timespan="day", limit=200):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/2023-01-01/{datetime.date.today()}?adjusted=true&sort=asc&limit={limit}&apiKey={polygon_api_key}"
    r = requests.get(url)
    data = r.json()
    if "results" not in data:
        st.error("Error fetching data from Polygon.")
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

# ------------------------ Pattern Detection ------------------------
def detect_double_bottom(df, threshold=0.02):
    patterns = []
    closes = df["Close"].values
    lows = df["Low"].values

    peaks, _ = find_peaks(-lows, distance=5)
    for i in range(len(peaks) - 1):
        first = peaks[i]
        second = peaks[i + 1]
        if abs(lows[first] - lows[second]) / closes[second] < threshold:
            trough_mid = df.iloc[first:second]["High"].max()
            if trough_mid > lows[first] * (1 + threshold):
                patterns.append((df["Date"].iloc[second], "Double Bottom"))
    return patterns

def detect_head_shoulders(df):
    patterns = []
    closes = df["Close"].values
    peaks, _ = find_peaks(closes, distance=5)
    for i in range(1, len(peaks)-1):
        L, H, R = peaks[i-1], peaks[i], peaks[i+1]
        if closes[H] > closes[L] and closes[H] > closes[R] and abs(closes[L] - closes[R]) / closes[H] < 0.03:
            patterns.append((df["Date"].iloc[R], "Head & Shoulders"))
    return patterns

# ------------------------ Support/Resistance + Volume Filter ------------------------
def confirm_with_volume_sr(df, patterns):
    confirmed = []
    for date, pattern in patterns:
        idx = df.index[df["Date"] == date]
        if len(idx) == 0:
            continue
        idx = idx[0]
        if idx < 3 or idx + 3 >= len(df):
            continue
        local_volume = df.iloc[idx - 3:idx + 3]["Volume"].mean()
        curr_volume = df.iloc[idx]["Volume"]
        resistance = df["Close"].iloc[:idx].max()
        support = df["Close"].iloc[:idx].min()
        curr_price = df["Close"].iloc[idx]

        # Confirm pattern with volume spike and proximity to support/resistance
        if curr_volume > 1.2 * local_volume and (
            abs(curr_price - support) / support < 0.03 or abs(curr_price - resistance) / resistance < 0.03
        ):
            confirmed.append((date, pattern))
    return confirmed

# ------------------------ OpenAI Interpretation ------------------------
def interpret_pattern(ticker, patterns):
    if not patterns:
        return "No significant chart patterns detected recently."

    prompt = f"Analyze the following technical chart patterns for {ticker}:\n"
    for date, pattern in patterns:
        prompt += f"- {pattern} detected on {date.strftime('%Y-%m-%d')}\n"
    prompt += "\nWhat do these patterns imply about the stock's future movement?"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ------------------------ Plotting ------------------------
def plot_chart(df, patterns):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))

    for date, pattern in patterns:
        fig.add_shape(type="line", x0=date, x1=date,
                      y0=df["Low"].min(), y1=df["High"].max(),
                      line=dict(color="Red", dash="dot"))
        fig.add_annotation(x=date, y=df["High"].max(),
                           text=pattern, showarrow=True, arrowhead=1)
    fig.update_layout(title="Price Chart with Detected Patterns", xaxis_title="Date", yaxis_title="Price")
    return fig

# ------------------------ Streamlit UI ------------------------
st.title("📈 Chart Pattern Detection Dashboard")
ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT)", "AAPL").upper()

if st.button("Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        df = fetch_polygon_data(ticker)
        if df.empty:
            st.warning("No data returned.")
        else:
            double_bottoms = detect_double_bottom(df)
            head_shoulders = detect_head_shoulders(df)
            all_patterns = double_bottoms + head_shoulders
            confirmed_patterns = confirm_with_volume_sr(df, all_patterns)

            st.plotly_chart(plot_chart(df, confirmed_patterns), use_container_width=True)
            interpretation = interpret_pattern(ticker, confirmed_patterns)
            st.subheader("🧠 Interpretation (OpenAI GPT-4)")
            st.markdown(interpretation)

