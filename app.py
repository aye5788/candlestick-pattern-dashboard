import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from chart_pattern_detector import detect_patterns
from openai import OpenAI

# Setup page
st.set_page_config(page_title="📊 AI-Powered Pattern Detector", layout="wide")
st.title("📈 Chart Pattern Detector & Explainer (Live w/ Polygon + OpenAI)")

# --- Load secrets
polygon_key = st.secrets["POLYGON_API_KEY"]
openai_key = st.secrets["OPENAI_API_KEY"]

# --- User inputs
symbol = st.text_input("Enter Stock Symbol", value="AAPL")
days = st.slider("Days of Historical Data", min_value=30, max_value=365, value=90)

if st.button("🔍 Fetch & Analyze"):
    with st.spinner("Fetching data from Polygon..."):
        end = datetime.today()
        start = end - timedelta(days=days)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}?adjusted=true&sort=asc&limit=5000&apiKey={polygon_key}"
        res = requests.get(url)
        data = res.json().get("results", [])

        if not data:
            st.error("No data returned. Check symbol or date range.")
        else:
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # --- Chart
            st.subheader(f"📉 {symbol.upper()} Price Chart")
            fig = go.Figure(data=[go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name="Candlestick"
            )])
            st.plotly_chart(fig, use_container_width=True)

            # --- Pattern Detection
            st.subheader("🔍 Detected Patterns")
            df_patterns = detect_patterns(df)
            pattern_rows = df_patterns[df_patterns['Pattern'].notnull()]
            st.dataframe(pattern_rows[['Date', 'Pattern']], use_container_width=True)

            # --- AI Interpretation
            if not pattern_rows.empty:
                st.subheader("🧠 AI Interpretation")
                last_patterns = pattern_rows.tail(3).to_dict(orient='records')
                pattern_descriptions = "\n".join([f"{r['Date'].date()}: {r['Pattern']}" for r in last_patterns])

                # Prompt OpenAI
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst skilled at interpreting chart patterns."},
                        {"role": "user", "content": f"Explain the potential meaning and implications of these recent chart patterns:\n{pattern_descriptions}"}
                    ]
                )
                interpretation = response.choices[0].message.content
                st.success(interpretation)
            else:
                st.info("No chart patterns detected in this timeframe.")
