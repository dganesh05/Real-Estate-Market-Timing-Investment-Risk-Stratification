# 🏠 Charlotte Real Estate Investment Intelligence System

### A Multi-Model Approach to Predicting Prices, Market Phases, and Investment Timing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-red)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-yellow)](https://scikit-learn.org/)

Check out a high-level overview of our work through this poster: [Data Mining and Visual Analytics Poster Presentation - Dec 02, 2025](poster.png)

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
  - [Market Phase Classification](#market-phase-classification)
  - [Price Regression](#price-regression)
  - [Time Series Forecasting](#time-series-forecasting)
- [COVID-19 Impact](#covid-19-impact)
- [Investment Recommendation System](#investment-recommendation-system)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project develops a **three-model investment intelligence system** for the Charlotte metropolitan real estate market, analyzing **11,765 records** across **113 zip codes** from 2013-2021. By combining time series forecasting, market phase classification, and price regression, we generate actionable buy/sell/hold recommendations for real estate investors.

**Key Insight:** The COVID-19 pandemic created an unprecedented regime shift that fundamentally transformed the Charlotte housing market—median prices surged +23%, days on market dropped -24%, and bidding wars became the norm. Our analysis reveals that **price history predicts price better than market indicators**, with time series models (R² = 0.85) vastly outperforming regression on market activity features (R² = 0.18).

**The System Answers Three Questions:**

- 📈 **Where are prices heading?** → Time Series Forecasting
- 🌡️ **Is competition high or low?** → Market Phase Classification
- 💰 **Is this property fairly priced?** → Price Regression

---

## 💡 Problem Statement

Real estate investors in rapidly changing markets face a critical challenge: **timing**. When should you buy? When should you sell? Traditional approaches rely on intuition and lagging indicators, often resulting in missed opportunities or poorly timed decisions.

**Research Questions:**

1. Can we accurately forecast price trajectories across different time horizons?
2. Can we classify market conditions to identify buyer's vs. seller's markets?
3. Can market activity indicators predict fair property values?
4. How do we combine multiple models into actionable investment advice?

**Business Value:**

- Data-driven buy/sell/hold recommendations
- Early detection of market phase transitions
- Quantified price trajectory forecasts
- Risk-aware investment timing

---

## 🔍 Key Findings

### Model Performance Summary

| Track                        | Best Model         | Performance    | Use Case               |
| ---------------------------- | ------------------ | -------------- | ---------------------- |
| **Time Series (Short-term)** | GRU Neural Network | R² = 0.85      | 1-3 month forecasts    |
| **Time Series (Long-term)**  | Random Forest      | R² = 0.69      | 6-12 month forecasts   |
| **Classification**           | Random Forest      | 85.5% accuracy | Market phase detection |
| **Regression**               | Linear Regression  | R² = 0.18      | Fair value estimation  |

### Critical Insights

1. **No single "best" model for forecasting** — GRU excels at short-term (1-3 months), Random Forest wins long-term (6-12 months)

2. **Price predicts price** — Time series using historical prices (R² = 0.85) vastly outperformed regression using market indicators (R² = 0.18)

3. **COVID-19 amplified fundamental limitations** — Market activity features describe _conditions_, not _valuations_; the pandemic made this gap undeniable

4. **HOT markets detected with 94% precision** — Enables investors to avoid bidding wars and time market entry

### COVID-19 Market Transformation

| Metric            | Pre-COVID (2019) | Post-COVID (2021) | Change   |
| ----------------- | ---------------- | ----------------- | -------- |
| Median Sale Price | $263,000         | $323,000          | **+23%** |
| Days on Market    | 38 days          | 29 days           | **-24%** |
| Sold Above List   | 19%              | 32%               | **+68%** |
| Market Phase      | 50% HOT          | 88% HOT           | **+76%** |

---

## 📊 Dataset

**Source:** [Redfin Housing Market Data](https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data?select=zip_code_market_tracker.tsv000)

**Coverage:**

- **Region:** Charlotte Metropolitan Area (NC & SC)
- **Zip Codes:** 113 unique regions
- **Time Period:** 2013-2021 (quarterly aggregates)
- **Records:** 11,765 observations after filtering

**Train/Test Split:**

- **Training:** 2013-2019 (9,254 records)
- **Testing:** 2020-2021 (2,511 records)

**Key Features:**

| Category       | Features                                      |
| -------------- | --------------------------------------------- |
| **Volume**     | homes_sold, inventory                         |
| **Velocity**   | median_dom, off_market_in_two_weeks           |
| **Momentum**   | homes_sold_yoy, inventory_yoy, median_dom_yoy |
| **Temporal**   | year, quarter, covid_era                      |
| **Geographic** | region (zip code), state_NC                   |

**Target Variables:**

- `log_price`: Log-transformed median sale price (regression, time series)
- `market_phase`: HOT / COLD / STABLE classification

**Data Quality:**

- Missing values handled via grouped forward-fill → backfill → global median
- 100% empty columns dropped during preprocessing
- Temporal consistency maintained for time series validity

---

## 🔬 Methodology

### 1. Exploratory Data Analysis

- Temporal trend analysis (2013-2021 price evolution)
- Geographic distribution across 113 zip codes
- COVID-19 impact quantification (pre/post comparison)
- Market phase distribution over time
- Feature correlation and multicollinearity assessment

### 2. Feature Engineering

**Market Phase Labels:**

```python
HOT:    High demand, fast sales, bidding wars
COLD:   Low demand, slow sales, buyer's market
STABLE: Balanced conditions, predictable transactions
```

**Temporal Features:**

```python
covid_era = 1 if year >= 2020 else 0
Q_1, Q_2, Q_3, Q_4 = quarter dummy variables
```

**Momentum Indicators:**

```python
homes_sold_yoy = year-over-year change in sales volume
inventory_yoy = year-over-year change in inventory
median_dom_yoy = year-over-year change in days on market
```

### 3. Data Leakage Prevention

**Removed Features** (computed using target variable):

```python
# These would not be available at prediction time
median_sale_price_mom  # Month-over-month price change
median_sale_price_yoy  # Year-over-year price change
avg_sale_to_list       # Same-period sales ratio
sold_above_list        # Same-period bidding data
```

**Validation Approach:**

- Strict temporal split (no future data leakage)
- TimeSeriesSplit for cross-validation within the training set
- Chronological ordering maintained for all experiments

### 4. Model Development

Three parallel modeling tracks:

1. **Time Series:** Predict future price trajectory from historical prices
2. **Classification:** Detect current market phase from activity indicators
3. **Regression:** Estimate fair value from market conditions

---

## 🤖 Models

### Market Phase Classification

**Objective:** Classify current market conditions as HOT, COLD, or STABLE

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

| Model             | Accuracy  | Notes                                |
| ----------------- | --------- | ------------------------------------ |
| **Random Forest** | **85.5%** | Best overall, balanced class weights |
| Decision Tree     | 77.3%     | Prone to overfitting                 |
| SVM               | 76.7%     | Predicted only majority class        |

**Classification Report (Random Forest):**

| Phase  | Precision | Recall | F1-Score | Support |
| ------ | --------- | ------ | -------- | ------- |
| COLD   | 63%       | 81%    | 71%      | 279     |
| STABLE | 64%       | 63%    | 63%      | 307     |
| HOT    | **94%**   | 90%    | 92%      | 1,925   |

**Key Insight:** HOT markets detected with 94% precision—critical for identifying high-competition periods where buyers should exercise caution.

---

### Price Regression

**Objective:** Estimate fair market value from current market indicators

**Features Used:**

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

| Model                 | Mean R² (CV) |
| --------------------- | ------------ |
| **Linear Regression** | **0.176**    |
| XGBoost               | 0.109        |
| Gradient Boosting     | 0.106        |
| Random Forest         | 0.049        |

**Key Finding:** All models struggled because market activity features describe **conditions**, not **valuations**.

**Why Regression Failed:**

| What Features Tell Us | What They Don't Tell Us  |
| --------------------- | ------------------------ |
| How fast homes sell   | What price they sell at  |
| Inventory levels      | Location quality         |
| Market competition    | Property characteristics |
| Activity trends       | Economic fundamentals    |

**The COVID Problem:** Models trained on 2013-2019 patterns couldn't predict the unprecedented 2020-2021 price surge. Predictions stayed flat (~$200K) while actual prices soared to $330K+.

---

### Time Series Forecasting

**Objective:** Forecast price trajectory across multiple horizons (1, 3, 6, 12 months)

**Approach:** Direct forecasting with lagged price sequences

**Models Tested:**

| Category             | Models                                        |
| -------------------- | --------------------------------------------- |
| **Baselines**        | Naive (last value), Moving Average            |
| **Machine Learning** | Random Forest, XGBoost (with lag features)    |
| **Deep Learning**    | LSTM (small/medium/large), GRU (small/medium) |

**Experimental Design:**

- Sequence lengths: 4, 6, 12 months of history
- Forecast horizons: 1, 3, 6, 12 months ahead
- Total experiments: 11 models × 12 configurations = 132 experiments

**Results — Best Model by Horizon:**

| Horizon   | Best Model | Seq Length | R²        | RMSE  | MAE   |
| --------- | ---------- | ---------- | --------- | ----- | ----- |
| 1 month   | GRU_small  | 6          | **0.853** | 0.159 | 0.083 |
| 3 months  | GRU_small  | 12         | **0.759** | 0.203 | 0.127 |
| 6 months  | RF_tuned   | 12         | **0.719** | 0.220 | 0.146 |
| 12 months | RF_tuned   | 12         | **0.685** | 0.233 | 0.160 |

**Key Insight:** Deep learning (GRU) captures short-term patterns effectively, while tree-based models (Random Forest) generalize better for long-term forecasting.

**Why Sequences Beat Lag Features:**

```
Random Forest sees:  [lag_1, lag_2, lag_3, lag_4] → 4 independent numbers
GRU sees:            [t-4 → t-3 → t-2 → t-1]      → ordered sequence with trend
```

GRU processes values step-by-step, building an internal memory of the trend. Tree-based models see lag features as independent columns—order is lost.

## 🦠 COVID-19 Impact

The Charlotte housing market underwent a dramatic transformation during the pandemic:

### Market Metrics Comparison

```
                    2019        2021        Change
Median Price:      $263,000    $323,000    +23% 📈
Days on Market:    38 days     29 days     -24% ⏱️
Sold Above List:   19%         32%         +68% 🔥
Inventory:         2.4 mo      0.9 mo      -63% 📉
```

### Market Phase Evolution

| Year | COLD | STABLE | HOT     |
| ---- | ---- | ------ | ------- |
| 2013 | 66%  | 22%    | 12%     |
| 2017 | 31%  | 35%    | 34%     |
| 2021 | 3%   | 9%     | **88%** |

**Key Insight:** The Charlotte market transformed from predominantly COLD (2013) to overwhelmingly HOT (2021), representing a fundamental regime shift that challenged all models trained on historical patterns.

---

## 🎯 Investment Recommendation System

### Combining Three Models

The system integrates outputs from all three modeling tracks into unified investment advice:

```
┌────────────────────────────────────────┐
│  Region: 28205  │  Period: June 2021   │
├────────────────────────────────────────┤
│  📈 FORECAST:    +9.5% (6 months)      │
│  🌡️ MARKET:      COLD (87% conf)       │
│  💰 VALUATION:   Undervalued (-3.4%)   │
├────────────────────────────────────────┤
│         🎯 STRONG BUY                  │
│                                        │
│  Rising prices + cold market +         │
│  undervalued = ideal entry point       │
└────────────────────────────────────────┘
```

### Decision Logic

| Forecast  | Market Phase | Valuation   | Recommendation |
| --------- | ------------ | ----------- | -------------- |
| ↑ Rising  | COLD         | Undervalued | **STRONG BUY** |
| ↑ Rising  | COLD         | Fair        | **BUY**        |
| ↑ Rising  | HOT          | Any         | **HOLD/SELL**  |
| ↓ Falling | COLD         | Any         | **WAIT**       |
| ↓ Falling | HOT          | Any         | **SELL**       |
| → Stable  | Any          | Undervalued | **BUY**        |

### Business Value

Each model answers a different question:

- **Time Series:** Where is this market heading? (Direction)
- **Classification:** How competitive is this market? (Risk)
- **Regression:** Is this property fairly priced? (Value)

Combined, they provide comprehensive investment intelligence that no single model could deliver.

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- 8GB+ RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/charlotte-real-estate.git
cd charlotte-real-estate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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

## 🚀 Usage

### Quick Start

```bash
# Run the main analysis notebooks
jupyter notebook
```

### Generate Predictions

```python
# Load trained models
from models import load_models
ts_model, clf_model, reg_model = load_models()

# Generate a recommendation for a region
from recommendation import generate_recommendation
result = generate_recommendation(
    region=28205,
    period='2021-06',
    ts_model=ts_model,
    clf_model=clf_model,
    reg_model=reg_model
)
print(result)
```

---

## 🔮 Future Work

### Model Improvements

- [ ] Evaluate regression on stable market periods (2013-2018) to isolate COVID impact from feature limitations
- [ ] Add property-level features (sqft, bedrooms, age) for improved valuation
- [ ] Incorporate external data (interest rates, employment, migration patterns)
- [ ] Test ensemble approaches combining GRU + Random Forest for all horizons

### Deployment

- [ ] Integrate streaming data for real-time predictions
- [ ] Train on more stable market periods for robustness
- [ ] Build browser extension for on-page price predictions (similar to stock trading tools)
- [ ] Create an interactive dashboard for exploring recommendations by region

### Extended Analysis

- [ ] Expand to additional metro areas (Raleigh, Atlanta, Nashville)
- [ ] Implement spatial analysis (neighborhood spillover effects)
- [ ] Add uncertainty quantification (prediction intervals)

---

## 🙏 Acknowledgments

- **Dataset:** Redfin Housing Market Data ([https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data?select=zip_code_market_tracker.tsv000](https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data?select=zip_code_market_tracker.tsv000))
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
