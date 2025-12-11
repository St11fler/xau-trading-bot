# XAU Trading Bot v4.8.8

AI-powered automated trading system for Gold (XAU/USD) forex trading using machine learning and MetaTrader 5 integration.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ⚠️ CRITICAL SECURITY WARNINGS

### Trading Risk Disclaimer
```
⚠️  AUTOMATED TRADING CARRIES SIGNIFICANT FINANCIAL RISK

- This bot can execute real trades with real money
- Past performance does not guarantee future results
- You can lose your entire trading account
- ALWAYS test thoroughly on a DEMO account first
- Never invest money you cannot afford to lose
- Monitor the bot continuously during live trading
```

### Security Requirements
```
🔒 NEVER share your MetaTrader 5 credentials
🔒 NEVER commit log files to version control (contains account data)
🔒 NEVER run untrusted code modifications without review
🔒 Always use environment-specific configurations
🔒 Review all trades and bot behavior regularly
```

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Training](#model-training)
- [Backtesting](#backtesting)
- [Live Trading](#live-trading)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### Machine Learning Models
- **Bidirectional LSTM** with self-attention for short-term price prediction (M1, M5, M15)
- **Custom Transformer** architecture with multi-head attention (experimental)
- **Long-term BiLSTM** for H1, H4, D1 timeframe predictions
- **RandomForest & LightGBM** ensemble classifiers for buy/sell decision-making
- **TimeSeriesSplit** cross-validation for robust model evaluation

### Trading Features
- **Dual-strategy system:** Scalping (short-term) + Position trading (long-term)
- **Dynamic risk management:** ATR-based stop-loss and take-profit
- **Real-time inference:** Sub-second prediction latency
- **Multi-timeframe analysis:** Simultaneous processing of 6 timeframes
- **Technical indicators:** 200+ indicators per data point (RSI, MACD, Bollinger Bands, etc.)
- **MetaTrader 5 integration:** Direct order execution and position management

### Data Processing
- **51K+ historical records** with automated data collection
- **178 engineered features** from technical analysis
- **Automated pipeline:** Data collection → Training → Prediction → Classification → Backtesting

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    XAU Trading Bot Pipeline                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Data Collection (collect.py)            │
        │  - MT5 API: M1, M5, M15, H1, H4, D1     │
        │  - Technical Indicators (pandas_ta)      │
        │  - 200+ features per timeframe           │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Model Training                          │
        ├──────────────────────────────────────────┤
        │  Short-term (short.py)                   │
        │  - BiLSTM + Attention → M1/M5/M15        │
        │  - Window: 7 bars, Adam optimizer        │
        ├──────────────────────────────────────────┤
        │  Long-term (long.py)                     │
        │  - BiLSTM → H1/H4/D1                     │
        │  - Window: 37 bars                       │
        ├──────────────────────────────────────────┤
        │  Transformer (mutation_1/)               │
        │  - Multi-head attention (4 heads)        │
        │  - Experimental architecture             │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Prediction Generation (prd.py)          │
        │  - Batch inference on dataset            │
        │  - Inverse scaling for interpretability  │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Action Classification (action.py)       │
        │  - RandomForest (200 estimators)         │
        │  - LightGBM (300 estimators)             │
        │  - Binary: Buy (+1) / Sell (-1)          │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Backtesting (backtest.py)               │
        │  - Historical validation                 │
        │  - Metrics: Win rate, drawdown, P&L      │
        └──────────────┬──────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │  Live Trading (bot.py)                   │
        │  - Real-time MT5 execution               │
        │  - 1-second polling cycle                │
        │  - ATR-based risk management             │
        └─────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- MetaTrader 5 terminal installed and logged in
- Windows OS (MT5 Python API requires Windows)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/xaubot.git
cd xaubot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
pip install tensorflow pandas numpy scikit-learn lightgbm
pip install MetaTrader5 pandas-ta matplotlib seaborn joblib
```

### Step 4: MetaTrader 5 Setup
1. Install [MetaTrader 5](https://www.metatrader5.com/en/download)
2. Open MT5 and login to your account (DEMO account recommended)
3. Enable Algo Trading: Tools → Options → Expert Advisors → Allow automated trading

---

## ⚙️ Configuration

### Trading Parameters (bot.py)

```python
CONFIG = {
    "USE_ATR_SL_TP": True,              # Use ATR-based stop-loss/take-profit
    "ATR_MULTIPLIER_SCALPING": 1.0,     # ATR multiplier for scalping SL/TP
    "ATR_MULTIPLIER_LONGTERM": 2.0,     # ATR multiplier for long-term SL/TP

    "FIXED_SL_SCALPING": 10 * 0.1,      # Fixed SL in points (if not using ATR)
    "FIXED_TP_SCALPING": 20 * 0.1,      # Fixed TP in points
    "FIXED_SL_LONGTERM": 50 * 0.1,      # Fixed SL for long-term trades
    "FIXED_TP_LONGTERM": 100 * 0.1,     # Fixed TP for long-term trades

    "LONGTERM_COOLDOWN_HOURS": 24,      # Cooldown between long-term trades
}

# Lot sizes
LOT_SIZE_SCALPING = 0.03   # Lot size for scalping strategy
LOT_SIZE_LONGTERM = 0.02   # Lot size for position trading

# Magic numbers (for position tracking)
MAGIC_NUMBER_SCALPING = 123456
MAGIC_NUMBER_LONGTERM = 654321
```

### Risk Management
- **Initial balance (backtest):** $500
- **Dynamic SL/TP:** Based on ATR (Average True Range)
- **Margin verification:** Checks free margin before each trade
- **Slippage protection:** Maximum 10 points
- **24-hour cooldown:** For long-term positions

---

## 📖 Usage

### Complete Pipeline (Automated)
Run the entire workflow from data collection to model training:

```bash
python run_pipeline.py
```

This executes:
1. Data collection from MT5
2. Short-term model training
3. Long-term model training
4. Prediction generation
5. Action classifier training
6. Backtesting
7. Model artifact saving

### Individual Steps

#### 1. Data Collection
```bash
python collect.py
```
Collects historical data from MT5 for all timeframes and calculates technical indicators.

Output: `XAUUSD_data_multiple_timeframes.csv`

#### 2. Train Short-term Model
```bash
python short.py
```
Trains BiLSTM model for M1, M5, M15 predictions.

Output: `scalping_model.keras`, `scalping_feature_scaler.pkl`, `scalping_target_scalers.pkl`

#### 3. Train Long-term Model
```bash
python long.py
```
Trains BiLSTM model for H1, H4, D1 predictions.

Output: `longterm_model.keras`, `longterm_feature_scaler.pkl`, `longterm_target_scalers.pkl`

#### 4. Generate Predictions
```bash
python prd.py
```
Generates predictions using trained models and writes back to CSV.

#### 5. Train Action Classifier
```bash
python action.py
```
Trains RandomForest classifier for buy/sell decisions.

Output: `action_classifier.pkl`

#### 6. Backtesting
```bash
python backtest.py
```
Runs historical simulation to evaluate strategy performance.

Output: Backtest metrics and logs in `backtest.log`

#### 7. Live Trading (⚠️ USE WITH CAUTION)
```bash
python bot.py
```
Starts live trading bot with real-time MT5 execution.

**WARNING:** Only run on DEMO account first. Monitor continuously.

---

## 🧪 Model Training

### Short-term Model (short.py)

**Architecture:**
```
Input (window=7, features=~90)
    ↓
BiLSTM(128 units, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
BiLSTM(64 units, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
Self-Attention Layer
    ↓
Flatten → Dense(64, relu) → Dropout(0.3)
    ↓
Output(3) → [M1_close, M5_close, M15_close]
```

**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE (Mean Squared Error)
- Metrics: MAE (Mean Absolute Error)
- Validation: 80/20 split
- Early stopping: Monitors validation loss

### Long-term Model (long.py)

Similar architecture with:
- Window size: 37 bars (larger context)
- Targets: H1, H4, D1 close prices

### Transformer Model (mutation_1/short_transformer.py)

**Experimental Architecture:**
```
Input → Dense(64) → MultiHeadAttention(4 heads, key_dim=64)
    ↓
Skip Connection + LayerNorm
    ↓
FeedForward(128→64) + Skip Connection + LayerNorm
    ↓
GlobalAveragePooling1D → Dense(64, relu) → Output(3)
```

---

## 📊 Backtesting

### Metrics Tracked
- Win rate (%)
- Total profit/loss
- Maximum drawdown
- Number of trades
- Average profit per trade
- Longest winning/losing streak

### Backtest Configuration
```python
INITIAL_BALANCE = 500.0
LOT_SIZE_SCALPING = 0.01
LOT_SIZE_LONGTERM = 0.02
```

### Running Backtests
```bash
python backtest.py
```

Check results in `backtest.log`.

---

## 🔴 Live Trading

### Before Starting Live Trading

1. **Test on DEMO account for at least 1 month**
2. **Monitor all trades manually**
3. **Start with minimum lot sizes**
4. **Set up proper risk limits**
5. **Have a stop-loss strategy**
6. **Monitor system resources**

### Start Bot
```bash
python bot.py
```

### Monitoring
- Log file: `trading_bot.log` (updated in real-time)
- Console output shows each cycle
- MT5 terminal displays open positions

### Stopping the Bot
- Press `Ctrl+C` in terminal
- Bot will shutdown gracefully
- Review open positions in MT5

---

## 📁 Project Structure

```
xaubot_v4.8.8/
│
├── bot.py                      # Live trading bot (667 lines)
├── collect.py                  # Data collection from MT5
├── short.py                    # Short-term LSTM training
├── long.py                     # Long-term LSTM training
├── prd.py                      # Prediction generation
├── action.py                   # Action classifier training
├── backtest.py                 # Backtesting framework (421 lines)
├── run_pipeline.py             # Complete pipeline orchestrator
│
├── mutation_1/                 # Experimental Transformer architecture
│   ├── short_transformer.py    # Transformer for short-term
│   ├── long_transformer.py     # Transformer for long-term
│   ├── action_lgbm.py          # LightGBM classifier
│   └── bot_new.py              # Bot using Transformer models
│
├── newmut05/                   # Optimization iteration
│   └── (similar structure)
│
├── scalping_model.keras        # Trained short-term model (4.7 MB)
├── longterm_model.keras        # Trained long-term model (7.5 MB)
├── action_classifier.pkl       # Trained classifier (1.3 MB)
├── *_scaler.pkl                # Feature/target scalers
│
├── XAUUSD_data_multiple_timeframes.csv  # Historical data (124 MB)
├── trading_bot.log             # Trading logs (excluded from git)
│
├── .gitignore                  # Security-first exclusions
├── cleanup_before_git.py       # Security cleanup script
├── SECURITY_AUDIT_REPORT.md    # Security audit documentation
└── README.md                   # This file
```

---

## 🔧 Technical Details

### Libraries & Frameworks
- **TensorFlow/Keras 2.x** - Deep learning
- **scikit-learn** - ML utilities, RandomForest
- **LightGBM** - Gradient boosting
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **MetaTrader5** - Trading API
- **pandas_ta** - Technical analysis
- **matplotlib/seaborn** - Visualization

### Data Processing
- **Dataset:** 51,246 historical records
- **Features:** 178 engineered features
- **Timeframes:** M1, M5, M15, H1, H4, D1
- **Indicators:** RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, CCI, Williams %R, OBV, Momentum, SMAs, EMAs

### Performance
- **Inference latency:** Sub-second
- **Polling interval:** 1 second
- **Model size:** 4.7 MB (scalping) + 7.5 MB (long-term)
- **Training time:** ~5-10 minutes on GPU

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Test thoroughly on DEMO account**
4. **Commit changes:** `git commit -m 'Add amazing feature'`
5. **Push to branch:** `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Code Standards
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update README with new functionality

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚖️ Legal Disclaimer

```
THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

The authors and contributors of this software are not responsible for any
financial losses incurred from using this trading bot. Forex trading carries
a high level of risk and may not be suitable for all investors.

USE AT YOUR OWN RISK.
```

---

## 📞 Support & Contact

For questions, issues, or feature requests:
- Open an issue on GitHub
- Review existing issues before posting
- Provide detailed logs and configuration when reporting bugs

---

## 🙏 Acknowledgments

- MetaTrader 5 for providing trading API
- TensorFlow team for deep learning framework
- pandas-ta for technical analysis library
- Open-source community for inspiration and tools

---

## 📚 Additional Resources

- [MetaTrader 5 Python Documentation](https://www.mql5.com/en/docs/python_metatrader5)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [Technical Analysis Library](https://github.com/twopirllc/pandas-ta)
- [Algorithmic Trading Best Practices](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp)

---

**Remember:** This is an educational project. Always test thoroughly and understand the risks before live trading.

**Happy Trading! 📈**
