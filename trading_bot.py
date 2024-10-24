import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats

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
        """Calculate advanced technical and statistical indicators using pandas-ta"""
        if df is None or df.empty:
            return None

        # Create custom strategy with multiple indicators
        strategy = ta.Strategy(
            name="Multi_Factor_Strategy",
            ta=[
                {"kind": "sma", "length": 20},
                {"kind": "sma", "length": 50},
                {"kind": "rsi"},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "obv"},
                {"kind": "adx"},
                {"kind": "atr"},
                {"kind": "mom", "length": 10},
            ]
        )
        
        # Calculate technical indicators
        df.ta.strategy(strategy)
        
        # Rename columns for consistency
        df.rename(columns={
            'SMA_20': 'SMA_20',
            'SMA_50': 'SMA_50',
            'RSI_14': 'RSI',
            'MACD_12_26_9': 'MACD',
            'MACDs_12_26_9': 'Signal_Line',
            'BBU_20_2.0': 'BB_Upper',
            'BBM_20_2.0': 'BB_Middle',
            'BBL_20_2.0': 'BB_Lower',
            'OBV': 'OBV',
            'ADX_14': 'ADX',
            'ATRr_14': 'ATR',
            'MOM_10': 'MOM'
        }, inplace=True)
        
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
            kelly_fraction = max(0.1, min(0.5, (win_rate - ((1 - win_rate) / (avg_win/avg_loss if avg_win else 1)))))
        
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
            st.success(f"{profit_color} SELL {symbol}: {position['quantity']} shares at ${price:.2f} (Profit: ${profit:.2f})")
        
        self.last_signals[symbol] = signal
        self.update_portfolio_value(symbol, price)

    def update_portfolio_value(self, symbol, current_price):
        """Update portfolio value including all positions"""
        portfolio_value = st.session_state.cash_balance
        
        for sym, pos in self.positions.items():
            price = current_price if sym == symbol else self.fetch_data(sym, interval="1m", period="1d")['Close'].iloc[-1]
            portfolio_value += pos['quantity'] * price
            
        st.session_state.portfolio_value = portfolio_value

def calculate_performance_metrics():
    """Calculate comprehensive trading performance metrics"""
    if not st.session_state.trades:
        return None
        
    trades_df = pd.DataFrame(st.session_state.trades)
    trades_df['Profit'] = trades_df['Profit'].fillna(0)
    
    # Basic metrics
    total_trades = len(trades_df)
    profitable_trades = len(trades_df[trades_df['Profit'] > 0])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    total_profit = trades_df['Profit'].sum()
    avg_profit = trades_df['Profit'].mean() if total_trades > 0 else 0
    
    # Risk metrics
    trades_df['Balance'] = trades_df['Balance'].fillna(method='ffill')
    trades_df['Peak'] = trades_df['Balance'].expanding().max()
    trades_df['Drawdown'] = (trades_df['Balance'] - trades_df['Peak']) / trades_df['Peak'] * 100
    max_drawdown = abs(trades_df['Drawdown'].min())
    
    # Calculate Sharpe Ratio (annualized)
    if len(trades_df) > 1:
        returns = trades_df['Profit'] / trades_df['Total'].shift(1)
        sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    else:
        sharpe = 0
        
    # Calculate Sortino Ratio (annualized)
    negative_returns = returns[returns < 0] if len(trades_df) > 1 else pd.Series([0])
    sortino = np.sqrt(252) * (returns.mean() / negative_returns.std()) if len(negative_returns) > 1 and negative_returns.std() != 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino
    }

def main():
    st.title("📊 Quantitative Trading Bot")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Trading Configuration")
        symbols = st.text_input("Stock Symbols (comma-separated)", "AAPL,MSFT,GOOGL").upper().split(',')
        symbols = [s.strip() for s in symbols]
        
        interval = st.selectbox("Time Interval", 
                              ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
                              index=2)
        
        period = st.selectbox("Analysis Period",
                            ["1d", "5d", "1mo", "3mo"],
                            index=1)
        
        st.divider()
        
        # Risk Management
        st.subheader("Risk Management")
        max_position_size = st.slider("Max Position Size (%)", 1, 20, 5)
        stop_loss = st.slider("Stop Loss (%)", 1, 10, 5)
        take_profit = st.slider("Take Profit (%)", 5, 20, 10)
        
        # Trading toggle
        auto_trading = st.toggle("Enable Auto-Trading", value=False)
        
        if st.button("Start Bot", type="primary"):
            st.session_state.bot_running = True
            st.success("Bot started!")
        if st.button("Stop Bot", type="secondary"):
            st.session_state.bot_running = False
            st.info("Bot stopped!")

    # Initialize trading bot
    bot = QuantBot()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Trading Dashboard", "📝 Trade Log", "📊 Performance"])
    
    # Portfolio metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")
    with col2:
        st.metric("Cash Balance", f"${st.session_state.cash_balance:,.2f}")
    with col3:
        positions_value = sum(pos['quantity'] * bot.fetch_data(sym, interval="1m", period="1d")['Close'].iloc[-1]
                            for sym, pos in st.session_state.positions.items()) if st.session_state.positions else 0
        st.metric("Positions Value", f"${positions_value:,.2f}")
    
    with tab1:
        for symbol in symbols:
            st.subheader(f"{symbol} Analysis")
            
            # Fetch and process data
            df = bot.fetch_data(symbol, interval, period)
            if df is not None:
                df = bot.calculate_indicators(df)
                df = bot.generate_signals(df)
                
                # Create interactive chart
                                # Create interactive chart
                fig = go.Figure()

                # Add candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Candlesticks',
                    increasing_line_color='green', decreasing_line_color='red'
                ))

                # Add moving averages (SMA)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1.5), name='SMA 20'))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1.5), name='SMA 50'))

                # Buy and Sell Signals
                buy_signals = df[df['Signal'] == 1]
                sell_signals = df[df['Signal'] == -1]

                fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                                         marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))

                fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                                         marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))

                # Add layout options
                fig.update_layout(
                    title=f'{symbol} Stock Price',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    height=600,
                    showlegend=True
                )

                # Display chart
                st.plotly_chart(fig, use_container_width=True)

                # Auto-trading logic
                if st.session_state.bot_running:
                    # Execute trades based on the latest signal
                    latest_signal = df['Signal'].iloc[-1]
                    latest_price = df['Close'].iloc[-1]
                    latest_volatility = df['Volatility'].iloc[-1]
                    timestamp = df.index[-1]

                    bot.execute_trade(symbol, latest_signal, latest_price, timestamp, latest_volatility)

    with tab2:
        # Display trade log
        st.subheader("Trade Log")
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            st.dataframe(trades_df)
        else:
            st.info("No trades executed yet.")
    
    with tab3:
        # Display performance metrics
        st.subheader("Performance Metrics")
        performance_metrics = calculate_performance_metrics()
        if performance_metrics:
            st.write(f"Total Trades: {performance_metrics['total_trades']}")
            st.write(f"Win Rate: {performance_metrics['win_rate']:.2f}%")
            st.write(f"Total Profit: ${performance_metrics['total_profit']:.2f}")
            st.write(f"Average Profit: ${performance_metrics['avg_profit']:.2f}")
            st.write(f"Max Drawdown: {performance_metrics['max_drawdown']:.2f}%")
            st.write(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
            st.write(f"Sortino Ratio: {performance_metrics['sortino_ratio']:.2f}")
        else:
            st.info("No performance data available yet.")

if __name__ == '__main__':
    main()