import streamlit as st
import pandas as pd
import requests
from datetime import date
from openai import OpenAI

# Load API keys
polygon_api_key = st.secrets["POLYGON_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Full ticker list from your dataset
TICKERS = [
    'AAPL', 'ABBV', 'ACIW', 'ADMA', 'AMD', 'AMZN', 'APD', 'AVGO', 'AXP', 'AXTI',
    'BA', 'BAC', 'BCPC', 'BHP', 'BMY', 'BZH', 'C', 'CADE', 'CALX', 'CAT', 'COP',
    'CORT', 'COST', 'CRM', 'CRUS', 'CSX', 'CVX', 'D', 'DE', 'DUK', 'ELV', 'ENB',
    'ENSG', 'EOG', 'EXC', 'FCX', 'FDX', 'FWRD', 'GE', 'GEF', 'GOOGL', 'GS',
    'GSHD', 'HAL', 'HD', 'HON', 'ICFI', 'ISRG', 'JNJ', 'JPM', 'KO', 'KOP',
    'LANC', 'LLY', 'LMT', 'LOW', 'MA', 'MAR', 'MATX', 'MCD', 'META', 'MPC',
    'MRNA', 'MS', 'MSFT', 'NEE', 'NEM', 'NKE', 'NOC', 'NVDA', 'ORCL', 'OSIS',
    'OXY', 'PEP', 'PFE', 'PG', 'PRDO', 'RCL', 'RIO', 'SAIA', 'SBUX', 'SCHW',
    'SLB', 'SO', 'STAA', 'TGT', 'TMO', 'TSLA', 'UNH', 'UPS', 'V', 'VCEL', 'VLO',
    'WFC', 'WIRE', 'WMT', 'XOM'
]

# Simple double bottom detector
def detect_double_bottom(df):
    recent = df.tail(20)
    lows = recent['l'].values
    troughs = [i for i in range(1, len(lows)-1) if lows[i] < lows[i-1] and lows[i] < lows[i+1]]
    if len(troughs) >= 2:
        t1, t2 = lows[troughs[-2]], lows[troughs[-1]]
        return abs(t1 - t2) / t1 < 0.03
    return False

# Fetch data from Polygon
def fetch_eod_data(ticker):
    end = str(date.today())
    start = "2024-01-01"
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=120&apiKey={polygon_api_key}"
    res = requests.get(url)
    if res.status_code == 200 and "results" in res.json():
        return pd.DataFrame(res.json()["results"])
    return pd.DataFrame()

# Interpret via OpenAI
def interpret_summary(findings):
    prompt = (
        f"Here are today's double bottom pattern scan results:\n\n{findings}\n\n"
        "Write a concise summary report. Highlight notable patterns, recurring sectors or tickers, and whether the market may be experiencing a potential shift based on the patterns. Keep it practical and insight-driven."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("📈 EOD Chart Pattern Scanner")
if st.button("🚨 Scan Now"):
    st.info("Scanning all tickers... this may take a minute ⏳")
    results = []
    for ticker in TICKERS:
        df = fetch_eod_data(ticker)
        if not df.empty and detect_double_bottom(df):
            results.append(ticker)

    if results:
        joined = ", ".join(results)
        st.success(f"Patterns detected in: {joined}")
        interpretation = interpret_summary(joined)
        st.markdown("### 🧠 AI Summary")
        st.write(interpretation)
    else:
        st.warning("No double bottom patterns found today.")

