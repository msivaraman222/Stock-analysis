import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📊 Stock Market Analysis Dashboard")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("combined_all_files.csv")
    df.columns = df.columns.str.lower()  # lowercase
    
    # Rename ticker → stock for consistency
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "stock"})
    
    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["stock", "date"])
    return df

df = load_data()

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = [
    "📌 Key Metrics",
    "📈 Volatility Analysis",
    "📉 Cumulative Returns",
    "🏭 Sector Performance",
    "🔗 Correlation Heatmap",
    "📅 Monthly Gainers & Losers"
]
choice = st.sidebar.radio("Navigate", menu)

# -------------------------------
# 1. Key Metrics
# -------------------------------
if choice == "📌 Key Metrics":
    st.header("📌 Key Metrics")

    # Yearly return = (last_close - first_close) / first_close
    yearly_return = (
        df.groupby("stock")["close"].agg(["first", "last"])
    )
    yearly_return["yearly_return"] = (yearly_return["last"] - yearly_return["first"]) / yearly_return["first"]

    top10 = yearly_return.sort_values("yearly_return", ascending=False).head(10)
    bottom10 = yearly_return.sort_values("yearly_return", ascending=True).head(10)

    st.subheader("🏆 Top 10 Gainers")
    st.dataframe(top10[["yearly_return"]])

    st.subheader("📉 Top 10 Losers")
    st.dataframe(bottom10[["yearly_return"]])

    # Market Summary
    green = (yearly_return["yearly_return"] > 0).sum()
    red = (yearly_return["yearly_return"] <= 0).sum()
    avg_price = df["close"].mean()
    avg_volume = df["volume"].mean()

    st.markdown(f"""
    - ✅ Green Stocks: **{green}**
    - ❌ Red Stocks: **{red}**
    - 💰 Average Price: **{avg_price:.2f}**
    - 📦 Average Volume: **{avg_volume:.2f}**
    """)

# -------------------------------
# 2. Volatility Analysis
# -------------------------------
elif choice == "📈 Volatility Analysis":
    st.header("📈 Volatility Analysis")

    df["daily_return"] = df.groupby("stock")["close"].pct_change()
    volatility = df.groupby("stock")["daily_return"].std().dropna().sort_values(ascending=False)
    top10_volatility = volatility.head(10)

    st.subheader("Top 10 Most Volatile Stocks")
    fig, ax = plt.subplots(figsize=(10, 6))
    top10_volatility.plot(kind="bar", ax=ax, color="tomato", edgecolor="black")
    ax.set_ylabel("Volatility (Std Dev of Daily Returns)")
    ax.set_title("Top 10 Most Volatile Stocks")
    st.pyplot(fig)

    st.dataframe(top10_volatility.reset_index().rename(columns={"daily_return": "volatility"}))

# -------------------------------
# 3. Cumulative Return
# -------------------------------
elif choice == "📉 Cumulative Returns":
    st.header("📉 Cumulative Returns")

    df["daily_return"] = df.groupby("stock")["close"].pct_change()
    df["cumulative_return"] = (1 + df["daily_return"]).groupby(df["stock"]).cumprod()

    # Top 5 performing stocks
    last_returns = df.groupby("stock")["cumulative_return"].last().sort_values(ascending=False).head(5)
    top5_stocks = last_returns.index.tolist()

    st.subheader("Cumulative Return of Top 5 Stocks")
    fig, ax = plt.subplots(figsize=(12, 6))
    for stock in top5_stocks:
        stock_data = df[df["stock"] == stock]
        ax.plot(stock_data["date"], stock_data["cumulative_return"], label=stock)
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Top 5 Performing Stocks (Cumulative Return)")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 4. Sector Performance
# -------------------------------
elif choice == "🏭 Sector Performance":
    st.header("🏭 Sector-wise Performance")

    try:
        sector_df = pd.read_csv("Sectors.csv")
        sector_df.columns = sector_df.columns.str.lower()
        if "ticker" in sector_df.columns:
            sector_df = sector_df.rename(columns={"ticker": "stock"})
        if "stock" not in sector_df.columns or "sector" not in sector_df.columns:
            st.error("❌ Sectors.csv must contain 'stock' and 'sector' columns")
            st.stop()

        yearly_return = (
            df.groupby("stock")["close"].agg(["first", "last"])
        )
        yearly_return["yearly_return"] = (yearly_return["last"] - yearly_return["first"]) / yearly_return["first"]

        merged = yearly_return.merge(sector_df, on="stock", how="left")
        sector_perf = merged.groupby("sector")["yearly_return"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sector_perf.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
        ax.set_ylabel("Average Yearly Return")
        ax.set_title("Sector-wise Performance")
        st.pyplot(fig)

        st.dataframe(sector_perf.reset_index())

    except Exception as e:
        st.error(f"⚠️ Could not load sector data: {e}")

# -------------------------------
# 5. Correlation Heatmap
# -------------------------------
elif choice == "🔗 Correlation Heatmap":
    st.header("🔗 Stock Price Correlation Heatmap")

    pivot_df = df.pivot(index="date", columns="stock", values="close")
    corr = pivot_df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Stock Price Correlation Heatmap")
    st.pyplot(fig)

# -------------------------------
# 6. Monthly Gainers & Losers
# -------------------------------
elif choice == "📅 Monthly Gainers & Losers":
    st.header("📅 Top 5 Monthly Gainers and Losers")

    df["monthly_return"] = df.groupby("stock")["close"].pct_change()
    df["month"] = df["date"].dt.to_period("M")

    monthly_perf = df.groupby(["month", "stock"])["monthly_return"].sum().reset_index()

    months = monthly_perf["month"].unique()
    selected_month = st.selectbox("Select Month", months)

    month_data = monthly_perf[monthly_perf["month"] == selected_month]
    top5 = month_data.nlargest(5, "monthly_return")
    bottom5 = month_data.nsmallest(5, "monthly_return")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🏆 Top 5 Gainers - {selected_month}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(top5["stock"], top5["monthly_return"], color="green", edgecolor="black")
        st.pyplot(fig)
        st.dataframe(top5)

    with col2:
        st.subheader(f"📉 Top 5 Losers - {selected_month}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(bottom5["stock"], bottom5["monthly_return"], color="red", edgecolor="black")
        st.pyplot(fig)
        st.dataframe(bottom5)
