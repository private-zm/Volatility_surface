import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime

# Initialize session state variables
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 10000
if 'cash_balance' not in st.session_state:
    st.session_state.cash_balance = 10000
if 'shares' not in st.session_state:
    st.session_state.shares = 0
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# Page configuration
st.set_page_config(
    page_title="Advanced Trading Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with hidden GitHub link and footer
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stAlert { padding: 0.5rem; margin: 0.5rem 0; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .trade-info {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Hide GitHub Icon/Link */
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
        display: none !important;
    }
    /* Hide "Made with Streamlit" */
    .styles_viewerBadge__1yB5_ {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

class TradingBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.last_signal = 0
        self.last_trade_price = None
        self.position_open = False

    def fetch_data(self, symbol, interval="5min"):
        """Fetch stock data with error handling and caching"""
        if not self.api_key:
            st.error("Please enter your Alpha Vantage API key in the sidebar.")
            return None
            
        try:
            @st.cache_data(ttl=60)
            def _fetch_data(symbol, interval, api_key):
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=compact"
                response = requests.get(url)
                data = response.json()
                
                if "Time Series" in str(data.keys()):
                    time_series_key = f"Time Series ({interval})"
                    df = pd.DataFrame(data[time_series_key]).T
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    return df
                else:
                    st.error("API limit reached or invalid symbol")
                    return None
                
            return _fetch_data(symbol, interval, self.api_key)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def calculate_signals(self, df):
        """Calculate trading signals using multiple technical indicators"""
        if df is None or df.empty:
            return None
        
        # Technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal_Line'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Generate trading signals
        df['Signal'] = 0
        
        # Buy signals (multiple conditions)
        buy_conditions = (
            (df['SMA_20'] > df['SMA_50']) &  # Golden cross
            (df['RSI'] < 70) &  # Not overbought
            (df['MACD'] > df['Signal_Line']) &  # MACD crossover
            (df['Close'] > df['SMA_20'])  # Price above short-term trend
        )
        
        # Sell signals (multiple conditions)
        sell_conditions = (
            (df['SMA_20'] < df['SMA_50']) |  # Death cross
            (df['RSI'] > 80) |  # Overbought
            (df['MACD'] < df['Signal_Line']) |  # MACD crossover
            (df['Close'] < df['SMA_20'])  # Price below short-term trend
        )
        
        df.loc[buy_conditions, 'Signal'] = 1
        df.loc[sell_conditions, 'Signal'] = -1
        
        return df

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal Line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def execute_trade(self, signal, price, timestamp):
        """Execute trades based on signals and manage positions"""
        if signal == self.last_signal:
            return
            
        quantity = self.calculate_position_size(price)
        
        if signal == 1 and not self.position_open and st.session_state.cash_balance >= price * quantity:
            # BUY Signal
            cost = price * quantity
            st.session_state.cash_balance -= cost
            st.session_state.shares += quantity
            st.session_state.trades.append({
                'Timestamp': timestamp,
                'Action': 'BUY',
                'Price': price,
                'Quantity': quantity,
                'Total': cost,
                'Balance': st.session_state.cash_balance
            })
            self.position_open = True
            self.last_trade_price = price
            st.success(f"🔵 BUY: {quantity} shares at ${price:.2f}")
            
        elif signal == -1 and self.position_open and st.session_state.shares > 0:
            # SELL Signal
            revenue = price * st.session_state.shares
            st.session_state.cash_balance += revenue
            profit = revenue - (self.last_trade_price * st.session_state.shares)
            st.session_state.trades.append({
                'Timestamp': timestamp,
                'Action': 'SELL',
                'Price': price,
                'Quantity': st.session_state.shares,
                'Total': revenue,
                'Profit': profit,
                'Balance': st.session_state.cash_balance
            })
            st.session_state.shares = 0
            self.position_open = False
            st.success(f"🔴 SELL: {quantity} shares at ${price:.2f} (Profit: ${profit:.2f})")
        
        self.last_signal = signal
        self.update_portfolio_value(price)

    def calculate_position_size(self, price):
        """Calculate the number of shares to trade based on position sizing rules"""
        risk_per_trade = st.session_state.cash_balance * 0.02  # 2% risk per trade
        position_size = risk_per_trade / price
        return max(1, int(position_size))

    def update_portfolio_value(self, current_price):
        """Update total portfolio value"""
        stock_value = st.session_state.shares * current_price
        st.session_state.portfolio_value = st.session_state.cash_balance + stock_value

    def check_stop_loss_take_profit(self, current_price, stop_loss_pct, take_profit_pct):
        """Check and execute stop loss and take profit orders"""
        if not self.position_open or self.last_trade_price is None:
            return
            
        stop_loss_price = self.last_trade_price * (1 - stop_loss_pct/100)
        take_profit_price = self.last_trade_price * (1 + take_profit_pct/100)
        
        if current_price <= stop_loss_price:
            st.warning(f"⚠️ Stop Loss triggered at ${current_price:.2f}")
            self.execute_trade(-1, current_price, pd.Timestamp.now())
            
        elif current_price >= take_profit_price:
            st.success(f"🎯 Take Profit triggered at ${current_price:.2f}")
            self.execute_trade(-1, current_price, pd.Timestamp.now())

def calculate_performance_metrics():
    """Calculate comprehensive trading performance metrics"""
    if not st.session_state.trades:
        return None
        
    trades_df = pd.DataFrame(st.session_state.trades)
    
    # Calculate profits for completed trades
    trades_df['Profit'] = trades_df['Profit'].fillna(0)
    
    total_trades = len(trades_df)
    profitable_trades = len(trades_df[trades_df['Profit'] > 0])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate profit metrics
    total_profit = trades_df['Profit'].sum()
    avg_profit = trades_df['Profit'].mean() if total_trades > 0 else 0
    
    # Calculate drawdown
    trades_df['Balance'] = trades_df['Balance'].fillna(method='ffill')
    trades_df['Peak'] = trades_df['Balance'].expanding().max()
    trades_df['Drawdown'] = (trades_df['Balance'] - trades_df['Peak']) / trades_df['Peak'] * 100
    max_drawdown = abs(trades_df['Drawdown'].min())
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown
    }

def main():
    st.title("📈 Advanced Trading Bot")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Alpha Vantage API Key", 
                               value=st.session_state.api_key if st.session_state.api_key else "",
                               type="password")
        if api_key:
            st.session_state.api_key = api_key
            
        symbol = st.text_input("Stock Symbol", "AAPL").upper()
        interval = st.selectbox("Time Interval", 
                              ["1min", "5min", "15min", "30min", "60min"],
                              index=1)
        
        st.divider()
        
        # Risk Management Settings
        st.subheader("Risk Management")
        stop_loss = st.slider("Stop Loss (%)", 1, 10, 5)
        take_profit = st.slider("Take Profit (%)", 5, 20, 10)
        
        # Auto-trading toggle
        auto_trading = st.toggle("Enable Auto-Trading", value=False)
        
        # Start/Stop Bot
        if st.button("Start Bot", type="primary"):
            st.session_state.bot_running = True
            st.success("Bot started!")
        if st.button("Stop Bot", type="secondary"):
            st.session_state.bot_running = False
            st.info("Bot stopped!")

    # Show API key instructions if not provided
    if not st.session_state.api_key:
        st.info("""
        To use this trading bot, you need an Alpha Vantage API key. 
        1. Get a free API key at: https://www.alphavantage.co/support/#api-key
        2. Enter your API key in the sidebar
        """)
        st.stop()

    # Initialize trading bot
    bot = TradingBot(st.session_state.api_key)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Trading Chart", "📝 Trade Log", "📈 Performance"])
    
    # Portfolio metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")
    with col2:
        st.metric("Cash Balance", f"${st.session_state.cash_balance:,.2f}")
    with col3:
        st.metric("Shares Held", st.session_state.shares)
    
    with tab1:
        # Fetch and process data
        df = bot.fetch_data(symbol, interval)
        if df is not None:
            df = bot.calculate_signals(df)
            
            # Create interactive chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            # Add technical indicators
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                                   name='SMA 20', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                                   name='SMA 50', line=dict(color='red', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], 
                                   name='BB Upper', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], 
                                   name='BB Lower', line=dict(color='gray', dash='dash')))
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Price Chart',
                yaxis_title='Price',
                xaxis_title='Time',
                height=600,
                template='plotly_white',
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Execute trading logic if bot is running
            if st.session_state.bot_running and auto_trading:
                current_price = df['Close'].iloc[-1]
                last_signal = df['Signal'].iloc[-1]
                
                # Check stop loss and take profit
                bot.check_stop_loss_take_profit(current_price, stop_loss, take_profit)
                
                # Execute trades based on signals
                if last_signal != 0:
                    bot.execute_trade(last_signal, current_price, df.index[-1])
                    
                # Update last check timestamp
                st.session_state.last_update = datetime.now()
                
    with tab2:
        st.subheader("Trading History")
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'])
            trades_df = trades_df.sort_values('Timestamp', ascending=False)
            
            # Format the trades dataframe for display
            display_df = trades_df.copy()
            display_df['Price'] = display_df['Price'].map('${:,.2f}'.format)
            display_df['Total'] = display_df['Total'].map('${:,.2f}'.format)
            display_df['Balance'] = display_df['Balance'].map('${:,.2f}'.format)
            display_df['Profit'] = display_df['Profit'].fillna(0).map('${:,.2f}'.format)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No trades executed yet.")
            
    with tab3:
        st.subheader("Performance Metrics")
        metrics = calculate_performance_metrics()
        
        if metrics:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Trades", metrics['total_trades'])
            with col2:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            with col3:
                st.metric("Total Profit", f"${metrics['total_profit']:,.2f}")
            with col4:
                st.metric("Average Profit", f"${metrics['avg_profit']:,.2f}")
            with col5:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
            
            # Add performance charts
            if st.session_state.trades:
                trades_df = pd.DataFrame(st.session_state.trades)
                trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'])
                
                # Equity curve
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=trades_df['Timestamp'],
                    y=trades_df['Balance'],
                    mode='lines',
                    name='Portfolio Value'
                ))
                
                fig_equity.update_layout(
                    title='Portfolio Equity Curve',
                    xaxis_title='Time',
                    yaxis_title='Portfolio Value ($)',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Profit distribution
                if 'Profit' in trades_df.columns:
                    fig_profit = go.Figure()
                    fig_profit.add_trace(go.Histogram(
                        x=trades_df['Profit'],
                        nbinsx=20,
                        name='Profit Distribution'
                    ))
                    
                    fig_profit.update_layout(
                        title='Profit Distribution',
                        xaxis_title='Profit ($)',
                        yaxis_title='Frequency',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_profit, use_container_width=True)
        else:
            st.info("No performance metrics available yet. Start trading to see analytics.")

if __name__ == "__main__":
    main()