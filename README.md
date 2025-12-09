# 🏠 Charlotte Real Estate Investment Intelligence System
### A Multi-Model Approach to Predicting Prices, Market Phases, and Investment Timing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-red)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-yellow)](https://scikit-learn.org/)

Check out a high-level overview of our work through this poster: [Data Mining and Visual Analytics Poster Presentation - Dec 02, 2025](poster.png)

---

## 📋 Table of Contents
- [Introduction](#introduction)
- [About the Data](#about-the-data)
- [Methods](#methods)
- [Evaluation](#evaluation)
- [Storytelling and Conclusion](#storytelling-and-conclusion)
- [Impact Section](#impact-section)
- [Repository Structure and Data Access](#repository-structure-and-data-access)
- [Installation](#installation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🎯 Introduction

### Problem Statement

Real estate investors in rapidly changing markets face a critical challenge: **timing**. When should you buy? When should you sell? Traditional approaches rely on intuition and lagging indicators, often resulting in missed opportunities or poorly timed decisions. The Charlotte metropolitan area experienced dramatic market shifts—especially during the COVID-19 pandemic—making data-driven guidance more essential than ever.

### Research Questions

Our project set out to answer three fundamental investment questions:

1. **📈 Where are prices heading?** — Can we accurately forecast price trajectories across different time horizons (1, 3, 6, 12 months)?

2. **🌡️ Is competition high or low?** — Can we classify market conditions as HOT, COLD, or STABLE to identify buyer's vs. seller's markets?

3. **💰 Is this property fairly priced?** — Can market activity indicators predict fair property values?

4. **🎯 How do we combine insights?** — Can we integrate multiple models into actionable buy/sell/hold recommendations?

### Overall Goal

Build a **three-model investment intelligence system** that combines time series forecasting, market phase classification, and price regression to generate actionable investment recommendations for the Charlotte real estate market.

---

## 📊 About the Data

### Data Source

**Redfin Housing Market Data**  
- **URL:** [https://www.redfin.com/news/data-center/](https://www.redfin.com/news/data-center/)
- **Access:** Publicly available, downloaded as CSV
- **License:** Subject to Redfin's Data License Agreement

### Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Region** | Charlotte Metropolitan Area (NC & SC) |
| **Zip Codes** | 113 unique regions |
| **Time Period** | 2013-2021 (quarterly aggregates) |
| **Total Records** | 11,765 observations |
| **Train Set** | 2013-2019 (9,254 records) |
| **Test Set** | 2020-2021 (2,511 records) |

### Features

| Category | Features | Description |
|----------|----------|-------------|
| **Volume** | `homes_sold`, `inventory` | Number of transactions and available listings |
| **Velocity** | `median_dom`, `off_market_in_two_weeks` | Speed of sales |
| **Price** | `median_sale_price`, `log_price` | Target variable for regression/time series |
| **Momentum** | `homes_sold_yoy`, `inventory_yoy`, `median_dom_yoy` | Year-over-year changes |
| **Temporal** | `year`, `quarter`, `covid_era` | Time indicators |
| **Geographic** | `region`, `state_NC` | Location identifiers |
| **Target (Classification)** | `market_phase` | HOT / COLD / STABLE |

### Exploratory Data Analysis

**Price Distribution:**
- Median sale prices ranged from ~$50K to $2M+ across zip codes
- Right-skewed distribution → log transformation applied for modeling
- Significant variation between low-end and high-end neighborhoods

**Temporal Trends:**
- Steady price appreciation 2013-2019 (~3-5% annually)
- Dramatic acceleration 2020-2021 (+23% surge)
- Market phase shifted from 66% COLD (2013) → 88% HOT (2021)

**COVID-19 Impact (Key Visualization):**

| Metric | Pre-COVID (2019) | Post-COVID (2021) | Change |
|--------|------------------|-------------------|--------|
| Median Price | $263,000 | $323,000 | **+23%** |
| Days on Market | 38 days | 29 days | **-24%** |
| Sold Above List | 19% | 32% | **+68%** |
| Inventory | 2.4 months | 0.9 months | **-63%** |

**Missing Data:**
- Minimal missing values
- Handled via grouped forward-fill → backfill → global median
- 100% empty columns dropped during preprocessing

---

## 🔬 Methods

### Preprocessing Pipeline

**1. Data Cleaning**
```python
# Missing value strategy for time series consistency
df = df.groupby('region').apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df = df.fillna(df.median())  # Global median for remaining
```

**2. Feature Engineering**

*Market Phase Labels:*
- HOT: High demand, fast sales, bidding wars
- COLD: Low demand, slow sales, buyer's market
- STABLE: Balanced conditions

*Temporal Features:*
```python
covid_era = 1 if year >= 2020 else 0
Q_1, Q_2, Q_3, Q_4 = quarter dummy variables
```

**3. Data Leakage Prevention**

We discovered and removed features that would not be available at prediction time:

| Removed Feature | Reason |
|-----------------|--------|
| `median_sale_price_mom` | Computed using target variable |
| `median_sale_price_yoy` | Computed using target variable |
| `avg_sale_to_list` | Same-period sales data |
| `sold_above_list` | Same-period bidding data |

This was a critical lesson—initial classification models achieved 99% accuracy, which was a red flag. After removing leaky features, we got honest 85.5% accuracy.

**4. Train/Test Split**

Strict temporal split to prevent future data leakage:
- **Train:** 2013-2019 (pre-COVID patterns)
- **Test:** 2020-2021 (COVID and post-COVID)

---

### Modeling Approach

We developed three parallel modeling tracks:

#### Track 1: Time Series Forecasting

**Goal:** Predict future price trajectory from historical prices

**Approach:** Direct forecasting with lagged price sequences

**Models Tested:**

| Category | Models |
|----------|--------|
| Baselines | Naive (last value), Moving Average |
| Machine Learning | Random Forest, XGBoost (with lag features) |
| Deep Learning | LSTM (small/medium/large), GRU (small/medium) |

**Experimental Design:**
- Sequence lengths tested: 4, 6, 12 months of history
- Forecast horizons: 1, 3, 6, 12 months ahead
- Total: 11 models × 12 configurations = **132 experiments**

**What Worked:**
- GRU outperformed LSTM (simpler architecture, less overfitting on our data size)
- Longer sequence length (12 months) generally improved performance
- Deep learning excelled at short-term; tree-based models better for long-term

**What Didn't Work:**
- Recursive forecasting (predicting 1 step, feeding back, repeat) — error accumulated too quickly
- LSTM_large — overfit on our ~9K training samples

**Key Insight — Sequences vs. Lag Features:**
```
Random Forest sees:  [lag_1, lag_2, lag_3, lag_4] → 4 independent numbers (order lost)
GRU sees:            [t-4 → t-3 → t-2 → t-1]      → ordered sequence with trend preserved
```

#### Track 2: Market Phase Classification

**Goal:** Classify current market as HOT, COLD, or STABLE

**Features Used:**
```python
features = [
    'homes_sold', 'inventory', 'median_dom',
    'off_market_in_two_weeks', 'homes_sold_yoy',
    'inventory_yoy', 'median_dom_yoy',
    'year', 'state_NC', 'Q_1', 'Q_2', 'Q_3', 'Q_4'
]
```

**Models Tested:**
- Decision Tree (with balanced class weights)
- Random Forest (with balanced class weights)
- SVM (RBF kernel, balanced class weights)

**What Worked:**
- Random Forest achieved best balance across all classes
- Balanced class weights helped address 77% HOT imbalance in test set

**What Didn't Work:**
- SVM collapsed to predicting only majority class (HOT)
- Decision Tree overfit without max_depth constraints

#### Track 3: Price Regression

**Goal:** Estimate fair market value from market activity indicators

**Features Used (Leakage-Free):**
```python
safe_features = [
    'homes_sold', 'inventory', 'median_dom',
    'off_market_in_two_weeks',
    'homes_sold_mom', 'homes_sold_yoy',
    'inventory_mom', 'inventory_yoy',
    'median_dom_mom', 'median_dom_yoy',
    'year', 'state_NC', 'covid_era',
    'Q_1', 'Q_2', 'Q_3', 'Q_4'
]
```

**Models Tested:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

**What We Learned:**
All models struggled (R² = 0.18 at best). This revealed a fundamental insight: market activity features describe **conditions** (how fast, how competitive) but not **valuations** (what price). COVID-19 amplified this gap—the unprecedented price surge couldn't be predicted from activity indicators alone.

---

## 📈 Evaluation

### Time Series Forecasting Results

**Best Model by Horizon:**

| Horizon | Best Model | Seq Length | R² | RMSE | MAE |
|---------|------------|------------|-----|------|-----|
| 1 month | GRU_small | 6 | **0.853** | 0.159 | 0.083 |
| 3 months | GRU_small | 12 | **0.759** | 0.203 | 0.127 |
| 6 months | RF_tuned | 12 | **0.719** | 0.220 | 0.146 |
| 12 months | RF_tuned | 12 | **0.685** | 0.233 | 0.160 |

**Key Finding:** No single "best" model — deep learning excels short-term, Random Forest wins long-term.

### Classification Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Random Forest** | **85.5%** | Best overall performance |
| Decision Tree | 77.3% | Prone to overfitting |
| SVM | 76.7% | Predicted only majority class |

**Per-Class Performance (Random Forest):**

| Phase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| COLD | 63% | 81% | 71% | 279 |
| STABLE | 64% | 63% | 63% | 307 |
| HOT | **94%** | 90% | 92% | 1,925 |

**Key Finding:** HOT markets detected with 94% precision — critical for identifying high-competition periods.

### Regression Results

| Model | Mean R² (CV) |
|-------|--------------|
| **Linear Regression** | **0.176** |
| XGBoost | 0.109 |
| Gradient Boosting | 0.106 |
| Random Forest | 0.049 |

**Key Finding:** Market activity features couldn't predict price levels. Linear Regression "won" because complex models overfit to weak signals.

### Answering Our Research Questions

| Question | Answer |
|----------|--------|
| **Can we forecast prices?** | ✅ Yes — R² = 0.85 (short-term), 0.69 (long-term) |
| **Can we classify market phase?** | ✅ Yes — 85.5% accuracy, 94% precision on HOT |
| **Can we predict fair value?** | ⚠️ Limited — R² = 0.18; features describe conditions, not valuations |
| **Can we combine into recommendations?** | ✅ Yes — integrated decision system produces buy/sell/hold signals |

---

## 📖 Storytelling and Conclusion

### Key Insights

**1. Price Predicts Price**

Time series models using historical prices (R² = 0.85) vastly outperformed regression using market indicators (R² = 0.18). Why? Historical prices implicitly encode location quality, property characteristics, and economic conditions — information that market activity features simply don't capture.

**2. COVID-19 Was a Regime Shift, Not Just Noise**

The pandemic didn't just add volatility — it fundamentally transformed the market. Models trained on 2013-2019 patterns faced a market that no longer followed those rules. Predictions stayed flat (~$200K) while actual prices soared to $330K+.

**3. Model Selection Depends on Use Case**

There's no universal "best" model:
- **Short-term traders (1-3 months):** Use GRU neural network
- **Long-term investors (6-12 months):** Use Random Forest
- **Market timing:** Use Random Forest classification

**4. Data Leakage Is Silent But Deadly**

Our initial 99% classification accuracy was a red flag, not a success. Features derived from the target variable created circular predictions. Catching and fixing this was one of our most valuable learnings.

### Did We Achieve Our Goal?

**Yes, with nuance.** We built a functional three-model system that can:
- Forecast price direction with strong accuracy (R² = 0.85)
- Detect market phases reliably (85.5% accuracy)
- Generate integrated buy/sell/hold recommendations

However, fair value estimation remains challenging — this requires property-level features (sqft, bedrooms, location quality) that our market-level dataset doesn't include.

### Future Work

1. **Evaluate regression on stable periods (2013-2018)** — Isolate COVID impact from fundamental feature limitations

2. **Integrate streaming data** — Real-time predictions as new Redfin data becomes available

3. **Train on stable market periods** — Improve robustness for "normal" conditions

4. **Build browser extension** — On-page price predictions similar to stock trading widgets

5. **Add property-level features** — Sqft, bedrooms, age, school ratings for better valuation

### What We Learned (Course Reflection)

This project reinforced several core data mining principles:
- **Temporal validation matters** — Random splits leak future information in time series
- **Always check for leakage** — Suspicious accuracy deserves investigation
- **Simple baselines first** — Know what you're trying to beat
- **Domain context is crucial** — COVID wasn't a bug in our data, it was the story

---

## ⚖️ Impact Section

### Positive Impacts

**1. Democratizing Investment Intelligence**

Professional real estate investors have access to sophisticated analytics tools. Our system could help level the playing field for individual investors and first-time homebuyers making the largest purchase of their lives.

**2. Reducing Emotional Decision-Making**

Real estate decisions are often driven by fear (missing out) or greed (overpaying in hot markets). Data-driven recommendations provide objectivity during stressful transactions.

**3. Market Transparency**

Identifying market phases helps buyers understand competitive dynamics. Knowing a market is "HOT" sets realistic expectations about bidding wars and timelines.

### Potential Negative Impacts

**1. Algorithmic Herding**

If many investors use similar models, they might all buy/sell at the same time, potentially amplifying market volatility rather than stabilizing it. This is a known risk in algorithmic trading.

**2. Disadvantaging Non-Users**

If data-driven tools become widespread among some investor groups (e.g., institutional buyers) but not others (e.g., first-time homebuyers), it could widen existing inequalities in real estate markets.

**3. Overconfidence in Predictions**

Models trained on historical data can't predict unprecedented events (like COVID-19). Users might place too much trust in predictions, ignoring the inherent uncertainty. Our R² = 0.85 means 15% of variance is unexplained — that's significant for a major purchase.

**4. Gentrification Acceleration**

If models identify "undervalued" neighborhoods with rising price trajectories, they could accelerate investment in those areas, potentially displacing existing residents and changing community character.

**5. Data Privacy Concerns**

While our dataset uses aggregated market data (not individual transactions), more granular implementations could raise privacy concerns about tracking property values tied to specific addresses and owners.

### Ethical Considerations

We recognize that real estate is not just an investment vehicle — it's where people live. Optimizing for investment returns must be balanced against community impacts. Any deployment of such systems should consider:
- Transparency about model limitations
- Avoiding recommendations that exploit information asymmetries
- Considering impacts on housing affordability

---

### Data Access

**Original Data Source:**  
Redfin Housing Market Data ([https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data?select=zip_code_market_tracker.tsv000](https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data?select=zip_code_market_tracker.tsv000))

**Processed Data:**  
All processed CSV files are included in the `data/` folder of this repository for reproducibility.

### Running the Code

```bash
# Clone repository
git clone https://github.com/yourusername/charlotte-real-estate.git
cd charlotte-real-estate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- 8GB+ RAM recommended

### Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 🙏 Acknowledgments

- **Dataset:** Redfin Housing Market Data
- **Course:** ITCS-3162, 001 — Data Mining, UNC Charlotte
- **Advisor:** [Aileen Benedict](https://www.linkedin.com/in/aileen-benedict/), College of Computing and Informatics
- **Libraries:** TensorFlow, scikit-learn, XGBoost, pandas, matplotlib, seaborn


---

## 👥 Team

- Efren Antonio - [Email](mailto:eantonio@charlotte.edu) - [LinkedIn](https://www.linkedin.com/in/eantonio3/) - [GitHub](https://github.com/eantonio3)
- Hoang Bui - [Email](mailto:mbui5@gmail.com) - [LinkedIn](https://www.linkedin.com/in/minh-hoang-bui-539458304/) - [GitHub](https://github.com/Thuvii)
- Divya Ganesh - [Email](mailto:divya.ganesh05@outlook.com) - [LinkedIn](https://www.linkedin.com/in/divyaganesh05/) - [GitHub](https://github.com/dganesh05)
- Ryan Jacobs - [Email](mailto:rjacob16@charlotte.edu) - [LinkedIn](https://www.linkedin.com/in/ryan-jacobs-08027024a/) - [GitHub](https://github.com/RJUNCC)
- Kyle Lakovidis - [Email](mailto:kiakovid@charlotte.edu) - [LinkedIn](https://www.linkedin.com/in/kyle-iakovidis/) - [GitHub](https://github.com/ItsBreathingjet)
- Heidy Marquez - [Email](mailto:hmarque2@charlotte.edu) - [LinkedIn](https://www.linkedin.com/in/heidymarquez/) - [GitHub](https://github.com/heidyjas)


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

**Note:** The Redfin dataset is subject to Redfin's and Kaggle's Data License Agreement and terms of use.

---

## 🌟 If You Found This Useful

- ⭐ Star this repository
- 🐛 Report issues
- 🤝 Submit pull requests
- 💬 Share feedback

---

<p align="center">
  <i>Built with ❤️ for data-driven real estate investing</i>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/dganesh05/Real-Estate-Market-Timing-Investment-Risk-Stratification?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/dganesh05/Real-Estate-Market-Timing-Investment-Risk-Stratification?style=social" alt="GitHub forks">
</p>
