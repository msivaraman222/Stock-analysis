import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("combined_all_files.csv")
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    return df

@st.cache_data
def load_sector_data():
    sectors_df = pd.read_csv("Sectors.csv")
    # Create a mapping from symbol to sector
    # Extract the symbol part from the Symbol column (everything after the colon)
    sectors_df['symbol'] = sectors_df['Symbol'].apply(lambda x: x.split(': ')[1] if ': ' in x else x)
    return sectors_df[['symbol', 'Sector']].rename(columns={'symbol': 'stock', 'Sector': 'sector'})

df = load_data()
sectors_df = load_sector_data()

st.title("📊 Stock Market Data Analysis & Visualization")

# -----------------------------
# Debug Section
# -----------------------------
st.subheader("🔎 Dataset Preview")
st.write("Columns in your dataset:", df.columns.tolist())
st.dataframe(df.head())

# -----------------------------
# Column Mapping (adjust here after we know your actual names)
# -----------------------------
rename_map = {}
for col in df.columns:
    if "stock" in col or "ticker" in col or "symbol" in col:
        rename_map[col] = "stock"
    if "date" in col or "time" in col:
        rename_map[col] = "date"
    if "close" in col and "adj" not in col:
        rename_map[col] = "close"
    if "adj_close" in col or "adj close" in col:
        rename_map[col] = "close"
    if "volume" in col:
        rename_map[col] = "volume"

df = df.rename(columns=rename_map)

# -----------------------------
# Merge sector data with main dataframe
# -----------------------------
df = df.merge(sectors_df, on='stock', how='left')

# -----------------------------
# Proceed only if columns exist
# -----------------------------
if all(col in df.columns for col in ["stock", "close", "date"]):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=["stock", "date"])

    # -----------------------------
    # 1. Key Metrics
    # -----------------------------
    st.header("1️⃣ Key Metrics Dashboard")

    yearly_return = df.groupby("stock")["close"].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
    df = df.merge(yearly_return.rename("yearly_return"), on="stock")

    top10_green = df.drop_duplicates("stock").nlargest(10, "yearly_return")
    top10_loss = df.drop_duplicates("stock").nsmallest(10, "yearly_return")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Green Stocks")
        st.dataframe(top10_green[["stock", "sector", "yearly_return"]])
    with col2:
        st.subheader("Top 10 Loss Stocks")
        st.dataframe(top10_loss[["stock", "sector", "yearly_return"]])

    summary = {
        "Total Green Stocks": (df.drop_duplicates("stock")["yearly_return"] > 0).sum(),
        "Total Red Stocks": (df.drop_duplicates("stock")["yearly_return"] < 0).sum(),
        "Average Price": df["close"].mean(),
        "Average Volume": df["volume"].mean() if "volume" in df.columns else "N/A"
    }
    st.subheader("📌 Market Summary")
    st.json(summary)

    # -----------------------------
    # 2. Volatility Analysis
    # -----------------------------
    st.header("2️⃣ Volatility Analysis")
    df["daily_return"] = df.groupby("stock")["close"].pct_change()
    volatility = df.groupby("stock")["daily_return"].std().nlargest(10)

    fig, ax = plt.subplots(figsize=(10,5))
    volatility.plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Most Volatile Stocks")
    ax.set_ylabel("Volatility (Std Dev of Daily Returns)")
    st.pyplot(fig)

    # -----------------------------
    # 3. Cumulative Return Over Time
    # -----------------------------
    st.header("3️⃣ Cumulative Return Over Time")
    df["cumulative_return"] = (1 + df["daily_return"]).groupby(df["stock"]).cumprod()
    final_cum_return = df.groupby("stock")["cumulative_return"].last().nlargest(5)
    top5_stocks = final_cum_return.index

    fig, ax = plt.subplots(figsize=(10,5))
    for stock in top5_stocks:
        df[df["stock"]==stock].set_index("date")["cumulative_return"].plot(ax=ax, label=stock)
    ax.legend()
    ax.set_title("Cumulative Return of Top 5 Stocks")
    st.pyplot(fig)

    # -----------------------------
    # 4. Sector-wise Performance
    # -----------------------------
    st.header("4️⃣ Sector-wise Performance")
    if "sector" in df.columns:
        sector_perf = df.drop_duplicates("stock").groupby("sector")["yearly_return"].mean()
        fig, ax = plt.subplots(figsize=(10,5))
        sector_perf.plot(kind="bar", ax=ax)
        ax.set_ylabel("Avg Yearly Return")
        ax.set_title("Average Yearly Return by Sector")
        st.pyplot(fig)
        
        # Additional sector analysis
        st.subheader("Sector Distribution")
        sector_counts = df.drop_duplicates("stock")["sector"].value_counts()
        fig, ax = plt.subplots(figsize=(10,5))
        sector_counts.plot(kind="pie", autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        ax.set_title("Number of Stocks by Sector")
        st.pyplot(fig)
    else:
        st.info("ℹ️ No sector column found. Please add sector data.")

    # -----------------------------
    # 5. Stock Price Correlation
    # -----------------------------
    st.header("5️⃣ Stock Price Correlation Heatmap")
    pivot_df = df.pivot(index="date", columns="stock", values="close")
    corr_matrix = pivot_df.corr()

    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax)
    ax.set_title("Stock Price Correlation Heatmap")
    st.pyplot(fig)

    # -----------------------------
    # 6. Top 5 Gainers & Losers (Month-wise)
    # -----------------------------
    st.header("6️⃣ Top 5 Gainers and Losers (Month-wise)")
    df["month"] = df["date"].dt.to_period("M")

    monthly_returns = df.groupby(["stock","month"])["close"].apply(lambda x: (x.iloc[-1]-x.iloc[0])/x.iloc[0]).reset_index(name="monthly_return")
    months = monthly_returns["month"].unique()

    for month in months[-3:]:  # Show only last 3 months to avoid overcrowding
        st.subheader(f"📅 {month}")
        data = monthly_returns[monthly_returns["month"]==month]
        top5 = data.nlargest(5, "monthly_return")
        bottom5 = data.nsmallest(5, "monthly_return")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(top5["stock"], top5["monthly_return"], color="green")
            ax.set_title(f"Top 5 Gainers - {month}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.bar(bottom5["stock"], bottom5["monthly_return"], color="red")
            ax.set_title(f"Top 5 Losers - {month}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
else:
    st.error("❌ Could not find columns for stock, close, date. Please check dataset.")