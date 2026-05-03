# 🪙 Khadala Crypto - My Crypto Quant Terminal Project

> A crypto trading dashboard I built in order to learn how to make financial model out of mathematical models and implement them.

Hey there!This is my cryptocurrency quantitative analysis terminal. I built it because I wanted to understand how professional traders and quants analyze crypto markets. It's got 50+ cryptocurrencies and a bunch of math/models I learned during my studies.

 What Does It Do?

Think of it as a Bloomberg terminal but just for crypto and totally free! You can:

- 📈 See live crypto prices (BTC, ETH, SOL, and 40+ others)
- 📊 Run complex math models (GARCH, Heston, PCA - don't worry, the code handles it)
- 🤖 Get ML predictions (Random Forest, Q-Learning)
- 📉 Calculate risk (VaR, CVaR)
- 💰 Optimize your portfolio
- ⚡ Backtest trading strategies

# Try It Live!

[👉 Click here to use the live app 👈](https://khadala-crypto.streamlit.app)

No installation needed - just open in your browser!

# What You'll See

The app has 18 different tabs covering:

| Tab | What's Inside |
|-----|----------------|
| 🏦 Market Dashboard | Live prices, top gainers/losers, volume surge alerts |
| 📈 Prices & Returns | Candlestick charts, return distributions |
| 📊 Covariance & Risk | How assets move together, risk contributions |
| 🔢 PCA Analysis | Finding hidden market factors |
| 🎯 Portfolio Optimization | Finding the best portfolio mix |
| 🤖 ML Predictions | Random Forest price forecasts |
| 🏛️ Options Pricing | Black-Scholes model with Greeks |
| 📉 GARCH & Heston | Advanced volatility modeling |
| 🛡️ Risk Management | VaR, CVaR, XVA framework |
| ⚡ Backtesting | Test strategies with real data |

# Cryptos You Can Analyze

Bitcoin (BTC), Ethereum (ETH), Solana (SOL), Binance (BNB), XRP, Cardano (ADA), Dogecoin (DOGE), Avalanche (AVAX), Chainlink (LINK), Polkadot (DOT), Polygon (MATIC), Litecoin (LTC), Uniswap (UNI), Aave (AAVE), and 35+ more!

#How I Built It

Languages & Tools:
- Python (the main language)
- Streamlit (makes web apps from Python)
- Pandas/NumPy (data manipulation)
- Scikit-learn (machine learning)
- Plotly (interactive charts)
- yfinance (free crypto data)

Time spent:A few months of evenings and weekends

Challenges:Getting all the math models to work correctly and making sure the app doesn't crash when loading 50 cryptos at once!

# Run It Yourself (If You Want)

If you're into coding and want to run this locally:

```bash
# 1. Download the code
git clone https://github.com/daniyalaziz432/khadala-crypto.git

# 2. Go into the folder
cd khadala-crypto

# 3. Install what's needed
pip install -r requirements.txt

# 4. Run it!
streamlit run khadala_crypto.py
