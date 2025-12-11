# Quick Start Guide - XAU Trading Bot

This guide gets you from zero to running the bot in 15 minutes.

---

## ⚡ Before You Start (5 minutes)

### Security First!
Before pushing to GitHub, run:
```bash
python cleanup_before_git.py --dry-run
```
This shows sensitive files that will be removed. See `GITHUB_RELEASE_CHECKLIST.md` for full details.

---

## 🚀 Installation (5 minutes)

### 1. Prerequisites
- Python 3.8+
- Windows OS
- MetaTrader 5 installed

### 2. Setup
```bash
# Clone repository (or download ZIP)
git clone https://github.com/yourusername/xau-trading-bot.git
cd xau-trading-bot

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. MetaTrader 5
1. Download from https://www.metatrader5.com/en/download
2. Install and login to DEMO account
3. Enable Algo Trading:
   - Tools → Options → Expert Advisors
   - ✅ Check "Allow automated trading"

---

## 🎯 First Run (5 minutes)

### Option 1: Complete Pipeline (Recommended)
```bash
python run_pipeline.py
```
This runs everything:
- Collects data from MT5
- Trains models
- Generates predictions
- Trains classifier
- Runs backtest

**Expected time:** 10-15 minutes

### Option 2: Individual Steps
```bash
# 1. Collect data
python collect.py

# 2. Train models
python short.py
python long.py

# 3. Generate predictions
python prd.py

# 4. Train classifier
python action.py

# 5. Backtest
python backtest.py
```

---

## 📊 Check Results

### Backtest Results
```bash
# View backtest log
type backtest.log  # Windows
cat backtest.log   # Linux/Mac
```

Look for:
- Win rate
- Total profit/loss
- Maximum drawdown
- Number of trades

### Model Performance
Check console output for:
- Training accuracy
- Validation loss
- MAE (Mean Absolute Error)

---

## ⚠️ Testing Live (DO THIS FIRST!)

### DEMO Account Only!
```bash
python bot.py
```

**Watch for:**
- "Trading is allowed for account ******" message
- Real-time predictions every second
- Order execution messages

**Monitor for 1-2 hours minimum before considering real money.**

---

## 🛑 Stopping the Bot

Press `Ctrl+C` to stop gracefully.

The bot will:
- Complete current cycle
- Shutdown MT5 connection
- Save final log

**Then:** Check MT5 for open positions and close manually if needed.

---

## 🔧 Configuration

Edit `bot.py` to change:

```python
# Risk parameters
LOT_SIZE_SCALPING = 0.03   # Lower for less risk
LOT_SIZE_LONGTERM = 0.02

# Stop-loss/Take-profit
CONFIG = {
    "ATR_MULTIPLIER_SCALPING": 1.0,   # Higher = wider SL/TP
    "ATR_MULTIPLIER_LONGTERM": 2.0,
}
```

---

## 🐛 Troubleshooting

### "MT5 initialization failed"
- Make sure MT5 is running and logged in
- Check MT5 terminal shows "Connected"
- Enable Algo Trading in MT5 settings

### "No data retrieved"
- Check internet connection
- Verify symbol name is "XAUUSD" (or change in code)
- Try wider date range in `collect.py`

### "Module not found"
```bash
# Activate virtual environment
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "Not enough free margin"
- Account balance too low
- Reduce lot sizes
- Use account with more funds

---

## 📈 Next Steps

1. **Monitor DEMO trading for 1-2 weeks**
2. **Analyze performance metrics**
3. **Adjust configuration based on results**
4. **Read full README.md for advanced features**
5. **Consider custom modifications**

---

## ⚠️ Final Warning

**NEVER skip demo testing.**
**NEVER trade with money you can't afford to lose.**
**ALWAYS monitor the bot actively.**

---

## 📚 Full Documentation

- `README.md` - Complete documentation
- `SECURITY_AUDIT_REPORT.md` - Security analysis
- `GITHUB_RELEASE_CHECKLIST.md` - Release guide

---

**Happy Trading! 📊**

Remember: This is an educational project. Use at your own risk.
