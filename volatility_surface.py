import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Custom CSS to hide elements and style UI
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAlert > div {
        padding: 0.5rem 0.5rem;
        border-radius: 0.5rem;
    }
    .stSpinner > div {
        text-align: center;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Implied Volatility Surface Analysis')

# Black-Scholes Call Price function
def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Implied Volatility Calculation with bounds checking
def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    
    # Calculate bounds for arbitrage-free option prices
    min_price = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
    max_price = S * np.exp(-q * T)
    
    if price < min_price or price > max_price:
        return np.nan

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
        # Add reasonability check
        if implied_vol > 5 or implied_vol < 0.01:
            return np.nan
        return implied_vol
    except (ValueError, RuntimeError):
        return np.nan

# Tabs for the app: Input Parameters, Volatility Surface, and Analysis
tab_input, tab_surface, tab_analysis = st.tabs(["Input Parameters", "Volatility Surface", "Analysis"])

with tab_input:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Parameters")
        risk_free_rate = st.number_input('Risk-Free Rate (%)', value=1.5, min_value=0.0, max_value=20.0, format="%.2f") / 100
        dividend_yield = st.number_input('Dividend Yield (%)', value=1.3, min_value=0.0, max_value=20.0, format="%.2f") / 100
        
        st.subheader("Ticker Selection")
        ticker_symbol = st.text_input('Enter Ticker Symbol', value='SPY', max_chars=10).upper()
        
    with col2:
        st.subheader("Strike Range Parameters")
        min_strike_pct = st.number_input('Min Strike (% of Spot)', value=80.0, min_value=1.0, max_value=99.0, format="%.1f")
        max_strike_pct = st.number_input('Max Strike (% of Spot)', value=120.0, min_value=101.0, max_value=200.0, format="%.1f")
        
        st.subheader("View Options")
        y_axis_option = st.selectbox('Y-axis:', ('Strike Price ($)', 'Moneyness'))
        interpolation_method = st.selectbox('Interpolation Method:', ('linear', 'cubic', 'nearest'))
        show_scatter = st.checkbox('Show Raw Data Points', value=False)

if min_strike_pct >= max_strike_pct:
    st.error('Minimum strike percentage must be less than maximum strike percentage.')
    st.stop()

# Cache data fetching to improve performance
@st.cache_data(ttl=300)
def fetch_market_data(symbol):
    ticker = yf.Ticker(symbol)
    spot_history = ticker.history(period='5d')
    if spot_history.empty:
        raise ValueError(f'No price data available for {symbol}')
    spot_price = spot_history['Close'].iloc[-1]
    expirations = ticker.options
    return ticker, spot_price, expirations

# Main data processing and surface generation
with tab_surface:
    try:
        with st.spinner('Fetching market data...'):
            ticker, spot_price, expirations = fetch_market_data(ticker_symbol)
            st.success(f"Current price for {ticker_symbol}: ${spot_price:.2f}")
            
        today = pd.Timestamp('today').normalize()
        exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]
        
        if not exp_dates:
            st.error(f'No valid expiration dates found for {ticker_symbol}')
            st.stop()
            
        # Fetch option chains with progress tracking
        option_data = []
        progress_text = "Fetching option chains..."
        progress_bar = st.progress(0, text=progress_text)
        
        for idx, exp_date in enumerate(exp_dates):
            try:
                chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
                calls = chain.calls
                
                # Filter valid options
                calls = calls[
                    (calls['bid'] > 0) & 
                    (calls['ask'] > 0) & 
                    (calls['volume'] > 0) &
                    (calls['strike'] >= spot_price * (min_strike_pct/100)) & 
                    (calls['strike'] <= spot_price * (max_strike_pct/100))
                ]
                
                for _, row in calls.iterrows():
                    option_data.append({
                        'expirationDate': exp_date,
                        'strike': row['strike'],
                        'bid': row['bid'],
                        'ask': row['ask'],
                        'mid': (row['bid'] + row['ask']) / 2,
                        'volume': row['volume'],
                        'openInterest': row['openInterest']
                    })
                
                progress_bar.progress((idx + 1) / len(exp_dates), 
                                   text=f"{progress_text} ({idx + 1}/{len(exp_dates)})")
                
            except Exception as e:
                st.warning(f'Error fetching data for {exp_date.date()}: {str(e)}')
                continue
                
        progress_bar.empty()
        
        if len(option_data) < 4:
            st.error('Insufficient valid option data. Try expanding strike range or choosing a more liquid underlying.')
            st.stop()
            
        # Process option data
        options_df = pd.DataFrame(option_data)
        options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365
        options_df['moneyness'] = options_df['strike'] / spot_price
        
        # Calculate IVs with progress tracking
        with st.spinner('Calculating implied volatilities...'):
            options_df['impliedVolatility'] = options_df.apply(
                lambda row: implied_volatility(
                    row['mid'], spot_price, row['strike'],
                    row['timeToExpiration'], risk_free_rate, dividend_yield
                ), axis=1
            )
        
        # Remove invalid IVs and convert to percentage
        options_df.dropna(subset=['impliedVolatility'], inplace=True)
        if len(options_df) < 4:
            st.error('Insufficient valid implied volatilities. Try adjusting parameters.')
            st.stop()
            
        options_df['impliedVolatility'] *= 100
        
        # Prepare surface data
        Y = options_df['strike'].values if y_axis_option == 'Strike Price ($)' else options_df['moneyness'].values
        X = options_df['timeToExpiration'].values
        Z = options_df['impliedVolatility'].values
        
        # Create surface plot
        num_points = min(50, len(X))
        ti = np.linspace(X.min(), X.max(), num_points)
        ki = np.linspace(Y.min(), Y.max(), num_points)
        T, K = np.meshgrid(ti, ki)
        
        try:
            Zi = griddata((X, Y), Z, (T, K), method=interpolation_method)
            Zi = np.ma.array(Zi, mask=np.isnan(Zi))
            
            fig = go.Figure()
            
            # Add surface
            fig.add_trace(go.Surface(
                x=T, y=K, z=Zi,
                colorscale='Viridis',
                colorbar_title='IV (%)',
                name='IV Surface'
            ))
            
            # Add scatter points if requested
            if show_scatter:
                fig.add_trace(go.Scatter3d(
                    x=X, y=Y, z=Z,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='red',
                        opacity=0.8
                    ),
                    name='Market Data'
                ))
            
            fig.update_layout(
                title=f'Implied Volatility Surface - {ticker_symbol}',
                scene=dict(
                    xaxis_title='Time to Expiration (years)',
                    yaxis_title=y_axis_option,
                    zaxis_title='Implied Volatility (%)'
                ),
                autosize=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f'Error generating surface plot: {str(e)}')
        
    except ValueError as ve:
        st.error(f"Error fetching data for {ticker_symbol}: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Analysis Tab for future use
with tab_analysis:
    st.header("Option Data Summary")
    st.write(f"Displaying data for {ticker_symbol}")
    st.dataframe(options_df[['expirationDate', 'strike', 'impliedVolatility', 'volume', 'openInterest']].sort_values(by='expirationDate'))