import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats
import talib as ta  # Import TA-Lib

# Initialize session state variables
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 100000  # Starting with 100k
if 'cash_balance' not in st.session_state:
    st.session_state.cash_balance = 100000
if 'positions' not in st.session_state:
    st.session_state.positions = {}  # Dictionary to track multiple positions
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

class QuantBot:
    def __init__(self):
        self.positions = st.session_state.positions
        self.last_signals = {}
        
    def fetch_data(self, symbol, interval="5m", period="1d"):
        """Fetch stock data from Yahoo Finance with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period)
            if df.empty:
                st.error(f"No data available for {symbol}")
                return None
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """Calculate advanced technical and statistical indicators using TA-Lib"""
        if df is None or df.empty:
            return None
        
        # Calculate technical indicators using TA-Lib
        df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = ta.SMA(df['Close'], timeperiod=50)
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['Signal_Line'], _ = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['OBV'] = ta.OBV(df['Close'], df['Volume'])
        df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['MOM'] = ta.MOM(df['Close'], timeperiod=10)
        
        # Statistical indicators
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Z_Score'] = stats.zscore(df['Close'].fillna(method='ffill'))
        
        return df

    def generate_signals(self, df):
        """Generate trading signals using multiple factors"""
        if df is None or df.empty:
            return None
            
        df['Signal'] = 0
        
        # Trend following signals
        trend_signal = (
            (df['SMA_20'] > df['SMA_50']) & 
            (df['Close'] > df['SMA_20']) & 
            (df['ADX'] > 25)  # Strong trend
        )
        
        # Mean reversion signals
        mean_reversion_signal = (
            (df['Z_Score'] < -2) &  # Oversold
            (df['RSI'] < 30) & 
            (df['Close'] < df['BB_Lower'])
        )
        
        # Momentum signals
        momentum_signal = (
            (df['MACD'] > df['Signal_Line']) & 
            (df['MOM'] > 0) & 
            (df['OBV'].diff() > 0)  # Rising volume
        )
        
        # Volatility filter
        valid_volatility = (
            (df['Volatility'] < df['Volatility'].quantile(0.8)) &  # Not too volatile
            (df['ATR'] < df['ATR'].rolling(100).mean())  # Below average range
        )
        
        # Combined buy signal
        buy_signal = (
            (trend_signal | mean_reversion_signal | momentum_signal) & 
            valid_volatility
        )
        
        # Sell signals
        sell_signal = (
            (df['Z_Score'] > 2) |  # Overbought
            (df['RSI'] > 70) |
            (df['Close'] < df['SMA_50']) |  # Trend breakdown
            (df['MACD'] < df['Signal_Line'])
        )
        
        df.loc[buy_signal, 'Signal'] = 1
        df.loc[sell_signal, 'Signal'] = -1
        
        return df

    def calculate_position_size(self, price, volatility):
        """Calculate position size using Kelly Criterion and volatility adjustment"""
        available_capital = st.session_state.cash_balance
        
        # Kelly Criterion calculation
        win_rate = len([t for t in st.session_state.trades if t.get('Profit', 0) > 0]) / max(len(st.session_state.trades), 1)
        avg_win = np.mean([t['Profit'] for t in st.session_state.trades if t.get('Profit', 0) > 0]) if st.session_state.trades else 1
        avg_loss = abs(np.mean([t['Profit'] for t in st.session_state.trades if t.get('Profit', 0) < 0])) if st.session_state.trades else 1
        
        if avg_loss == 0:
            kelly_fraction = 0.1  # Conservative default
        else:
            kelly_fraction = max(0.1, min(0.5, (win_rate - ((1 - win_rate) / (avg_win / avg_loss if avg_win else 1)))))
        
        # Volatility adjustment
        vol_adjustment = max(0.2, min(1.0, 1.0 / (volatility * 10))) if volatility > 0 else 0.5
        
        # Calculate final position size
        position_dollars = available_capital * kelly_fraction * vol_adjustment
        return max(1, int(position_dollars / price))

    def execute_trade(self, symbol, signal, price, timestamp, volatility):
        """Execute trades with position sizing and risk management"""
        if symbol not in self.last_signals:
            self.last_signals[symbol] = 0
            
        if signal == self.last_signals[symbol]:
            return
            
        quantity = self.calculate_position_size(price, volatility)
        
        if signal == 1 and symbol not in self.positions and st.session_state.cash_balance >= price * quantity:
            # BUY
            cost = price * quantity
            st.session_state.cash_balance -= cost
            self.positions[symbol] = {'quantity': quantity, 'price': price}
            st.session_state.positions = self.positions
            
            trade = {
                'Timestamp': timestamp,
                'Symbol': symbol,
                'Action': 'BUY',
                'Price': price,
                'Quantity': quantity,
                'Total': cost,
                'Balance': st.session_state.cash_balance
            }
            st.session_state.trades.append(trade)
            st.success(f"🔵 BUY {symbol}: {quantity} shares at ${price:.2f}")
            
        elif signal == -1 and symbol in self.positions:
            # SELL
            position = self.positions[symbol]
            revenue = price * position['quantity']
            profit = revenue - (position['price'] * position['quantity'])
            st.session_state.cash_balance += revenue
            
            trade = {
                'Timestamp': timestamp,
                'Symbol': symbol,
                'Action': 'SELL',
                'Price': price,
                'Quantity': position['quantity'],
                'Total': revenue,
                'Profit': profit,
                'Balance': st.session_state.cash_balance
            }
            st.session_state.trades.append(trade)
            del self.positions[symbol]
            st.session_state.positions = self.positions
            
            profit_color = "🟢" if profit > 0 else "🔴"
            st.success(f"{profit_color} SELL {symbol}: {position['quantity']} shares at ${price:.2f} for a profit of ${profit:.2f}")

        self.last_signals[symbol] = signal

    def run_bot(self, symbol):
        """Main logic to run the trading bot"""
        if not st.session_state.bot_running:
            st.session_state.bot_running = True
            st.success("Trading bot is running...")
        
        data = self.fetch_data(symbol)
        data = self.calculate_indicators(data)
        data = self.generate_signals(data)

        if data is not None and not data.empty:
            latest_data = data.iloc[-1]
            self.execute_trade(symbol, latest_data['Signal'], latest_data['Close'], latest_data.name, latest_data['Volatility'])

# Streamlit user interface
st.title("Quantitative Trading Bot")
symbol_input = st.text_input("Enter a stock symbol (e.g., AAPL):", "AAPL")
if st.button("Start Trading"):
    bot = QuantBot()
    bot.run_bot(symbol_input)