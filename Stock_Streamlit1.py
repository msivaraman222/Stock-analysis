import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Read the CSV file
    df = pd.read_csv('combined_all_files.csv')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M')
    
    # Extract year and month for analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    
    return df

# Load the data
df = load_data()

# Add a sector mapping (this would typically come from a separate dataset)
# For demonstration purposes, I'll create a simple sector mapping
sector_mapping = {
    'SBIN': 'Financial Services',
    'BAJFINANCE': 'Financial Services',
    'TITAN': 'Consumer Goods',
    'ITC': 'Consumer Goods',
    'TCS': 'Information Technology',
    'LT': 'Construction',
    'TATACONSUM': 'Consumer Goods',
    'RELIANCE': 'Conglomerate',
    'HCLTECH': 'Information Technology',
    'JSWSTEEL': 'Metals & Mining',
    'ULTRACEMCO': 'Construction Materials',
    'POWERGRID': 'Utilities',
    'INFY': 'Information Technology',
    'TRENT': 'Retail',
    'BHARTIARTL': 'Telecommunications',
    'TATAMOTORS': 'Automobile',
    'WIPRO': 'Information Technology',
    'TECHM': 'Information Technology',
    'NTPC': 'Utilities',
    'HINDUNILVR': 'Consumer Goods',
    'APOLLOHOSP': 'Healthcare',
    'M&M': 'Automobile',
    'GRASIM': 'Construction Materials',
    'ICICIBANK': 'Financial Services',
    'ADANIENT': 'Conglomerate',
    'ADANIPORTS': 'Infrastructure',
    'BEL': 'Defense',
    'BAJAJFINSV': 'Financial Services',
    'EICHERMOT': 'Automobile',
    'COALINDIA': 'Metals & Mining',
    'MARUTI': 'Automobile',
    'INDUSINDBK': 'Financial Services',
    'ASIANPAINT': 'Consumer Goods',
    'TATASTEEL': 'Metals & Mining',
    'HDFCLIFE': 'Financial Services',
    'DRREDDY': 'Healthcare',
    'SUNPHARMA': 'Healthcare',
    'KOTAKBANK': 'Financial Services',
    'SHRIRAMFIN': 'Financial Services',
    'NESTLEIND': 'Consumer Goods',
    'ONGC': 'Energy',
    'CIPLA': 'Healthcare',
    'BPCL': 'Energy',
    'BRITANNIA': 'Consumer Goods',
    'SBILIFE': 'Financial Services',
    'HINDALCO': 'Metals & Mining',
    'HEROMOTOCO': 'Automobile',
    'AXISBANK': 'Financial Services',
    'HDFCBANK': 'Financial Services',
    'BAJAJ-AUTO': 'Automobile'
}

# Add sector information to the dataframe
df['sector'] = df['Ticker'].map(sector_mapping)

# Calculate daily returns
df = df.sort_values(['Ticker', 'date'])
df['daily_return'] = df.groupby('Ticker')['close'].pct_change()

# Calculate cumulative returns
df['cumulative_return'] = df.groupby('Ticker')['daily_return'].apply(lambda x: (1 + x).cumprod() - 1)

# Calculate volatility (standard deviation of daily returns)
volatility = df.groupby('Ticker')['daily_return'].std().reset_index()
volatility.columns = ['Ticker', 'volatility']
volatility = volatility.sort_values('volatility', ascending=False)

# Calculate overall returns for each stock
overall_returns = df.groupby('Ticker').apply(
    lambda x: (x['close'].iloc[-1] - x['close'].iloc[0]) / x['close'].iloc[0]
).reset_index()
overall_returns.columns = ['Ticker', 'overall_return']

# Calculate monthly returns
df['month_year'] = df['date'].dt.to_period('M')
monthly_returns = df.groupby(['Ticker', 'month_year']).apply(
    lambda x: (x['close'].iloc[-1] - x['close'].iloc[0]) / x['close'].iloc[0]
).reset_index()
monthly_returns.columns = ['Ticker', 'month_year', 'monthly_return']

# Calculate sector performance
sector_performance = df.groupby(['sector', 'month_year']).apply(
    lambda x: (x['close'].iloc[-1] - x['close'].iloc[0]) / x['close'].iloc[0]
).reset_index()
sector_performance.columns = ['sector', 'month_year', 'sector_return']

# Calculate correlation matrix
pivot_df = df.pivot(index='date', columns='Ticker', values='close')
correlation_matrix = pivot_df.corr()

# Streamlit app layout
st.title("Stock Market Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of stock performance including volatility, cumulative returns, 
sector-wise performance, correlation analysis, and monthly gainers/losers.
""")

# Sidebar for filters
st.sidebar.header("Filters")
selected_tickers = st.sidebar.multiselect(
    "Select Tickers to Focus On",
    options=df['Ticker'].unique(),
    default=['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'HINDUNILVR']
)

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Volatility Analysis", 
    "Cumulative Returns", 
    "Sector Performance",
    "Correlation Analysis",
    "Monthly Gainers & Losers"
])

# Tab 1: Volatility Analysis
with tab1:
    st.header("Volatility Analysis (Standard Deviation of Daily Returns)")
    
    top_n = st.slider("Select number of stocks to display", 5, 30, 10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    top_volatility = volatility.head(top_n)
    ax.bar(top_volatility['Ticker'], top_volatility['volatility'])
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Volatility')
    ax.set_title(f'Top {top_n} Most Volatile Stocks')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Volatility Explanation")
    st.markdown("""
    Volatility measures how much a stock's price fluctuates over time. Higher volatility indicates higher risk, 
    as the price can change dramatically in a short period. Lower volatility suggests more stable price movements.
    """)

# Tab 2: Cumulative Returns
with tab2:
    st.header("Cumulative Returns Over Time")
    
    # Get top 5 performing stocks based on overall return
    top_performers = overall_returns.nlargest(5, 'overall_return')['Ticker'].tolist()
    
    # Filter data for top performers
    cumulative_data = df[df['Ticker'].isin(top_performers)]
    
    # Plot cumulative returns
    fig = px.line(cumulative_data, x='date', y='cumulative_return', color='Ticker',
                  title='Cumulative Returns for Top 5 Performing Stocks')
    fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Return', legend_title='Ticker')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 5 Performing Stocks Details")
    top_performers_table = overall_returns.nlargest(5, 'overall_return')
    st.dataframe(top_performers_table.style.format({'overall_return': '{:.2%}'}))

# Tab 3: Sector Performance
with tab3:
    st.header("Sector-wise Performance Analysis")
    
    # Calculate average yearly return by sector
    sector_yearly_return = sector_performance.groupby('sector')['sector_return'].mean().reset_index()
    sector_yearly_return = sector_yearly_return.sort_values('sector_return', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(sector_yearly_return['sector'], sector_yearly_return['sector_return'])
    ax.set_xlabel('Sector')
    ax.set_ylabel('Average Yearly Return')
    ax.set_title('Average Yearly Return by Sector')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Sector performance over time
    st.subheader("Sector Performance Over Time")
    selected_sectors = st.multiselect(
        "Select Sectors to Compare",
        options=sector_performance['sector'].unique(),
        default=['Information Technology', 'Financial Services', 'Healthcare']
    )
    
    if selected_sectors:
        sector_time_data = sector_performance[sector_performance['sector'].isin(selected_sectors)]
        sector_time_data['month_year'] = sector_time_data['month_year'].astype(str)
        
        fig = px.line(sector_time_data, x='month_year', y='sector_return', color='sector',
                      title='Sector Performance Over Time')
        fig.update_layout(xaxis_title='Month-Year', yaxis_title='Return', legend_title='Sector')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Correlation Analysis
with tab4:
    st.header("Stock Price Correlation Analysis")
    
    # Select a subset of stocks for clearer visualization
    correlation_tickers = st.multiselect(
        "Select stocks for correlation analysis",
        options=correlation_matrix.columns.tolist(),
        default=selected_tickers
    )
    
    if len(correlation_tickers) > 1:
        subset_corr = correlation_matrix.loc[correlation_tickers, correlation_tickers]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(subset_corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Stock Price Correlation Heatmap')
        st.pyplot(fig)
        
        st.subheader("Correlation Interpretation")
        st.markdown("""
        - Values close to 1 indicate strong positive correlation (stocks tend to move together)
        - Values close to -1 indicate strong negative correlation (stocks tend to move in opposite directions)
        - Values close to 0 indicate little to no correlation
        """)
    else:
        st.warning("Please select at least 2 stocks for correlation analysis.")

# Tab 5: Monthly Gainers and Losers
with tab5:
    st.header("Monthly Top Gainers and Losers")
    
    # Get unique months for selection
    months = sorted(monthly_returns['month_year'].unique())
    selected_month = st.selectbox("Select Month", options=months)
    
    if selected_month:
        month_data = monthly_returns[monthly_returns['month_year'] == selected_month]
        
        # Top 5 gainers
        top_gainers = month_data.nlargest(5, 'monthly_return')
        
        # Top 5 losers
        top_losers = month_data.nsmallest(5, 'monthly_return')
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top 5 Gainers', 'Top 5 Losers'),
            shared_yaxes=True
        )
        
        fig.add_trace(
            go.Bar(x=top_gainers['Ticker'], y=top_gainers['monthly_return'], 
                   marker_color='green', name='Gainers'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=top_losers['Ticker'], y=top_losers['monthly_return'], 
                   marker_color='red', name='Losers'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"Top Gainers and Losers for {selected_month}",
            showlegend=False,
            yaxis_title="Monthly Return"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 5 Gainers")
            st.dataframe(top_gainers.style.format({'monthly_return': '{:.2%}'}))
        
        with col2:
            st.subheader("Top 5 Losers")
            st.dataframe(top_losers.style.format({'monthly_return': '{:.2%}'}))

# Footer
st.markdown("---")
st.markdown("### About This Dashboard")
st.markdown("""
This dashboard provides analytical insights into stock market performance using various metrics:
- **Volatility**: Measures price fluctuation risk
- **Cumulative Returns**: Shows overall performance over time
- **Sector Performance**: Compares different industry sectors
- **Correlation Analysis**: Identifies relationships between stocks
- **Monthly Performance**: Tracks top gainers and losers each month
""")