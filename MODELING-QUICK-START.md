# Charlotte Real Estate - Modeling Quick Start Guide

**Before you start:** Make sure you have these files in your working directory:
- `regression_train.csv` / `regression_test.csv`
- `classification_train.csv` / `classification_test.csv`
- `lstm_timeseries.csv`
- `scaler.pkl`
- `model_config.json`

---

## Track 1: Regression (Price Prediction)

**Goal:** Predict `log_price` (log-transformed median sale price)

**Owner:** _______________

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble_import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
train = pd.read_csv('regression_train.csv')
test = pd.read_csv('regression_test.csv')

# 2. Define features and target
features = [
    'homes_sold', 'inventory', 'median_dom', 
    'off_market_in_two_weeks', 'avg_sale_to_list', 'sold_above_list',
    'median_sale_price_mom', 'median_sale_price_yoy',
    'homes_sold_mom', 'homes_sold_yoy',
    'inventory_mom', 'inventory_yoy',
    'median_dom_mom', 'median_dom_yoy',
    'year', 'state_NC', 'covid_era',
    'Q_1', 'Q_2', 'Q_3', 'Q_4'
]

X_train = train[features]
y_train = train['log_price']
X_test = test[features]
y_test = test['log_price']

# 3. Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"\n{name}")
    print(f"  R²:   {r2_score(y_test, preds):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")

# 4. Convert predictions back to actual dollars
y_test_dollars = np.exp(y_test)
preds_dollars = np.exp(preds)
print(f"  MAE (dollars): ${mean_absolute_error(y_test_dollars, preds_dollars):,.0f}")
```

**Metrics to report:** R², RMSE, MAE in dollars

---

## Track 2: Classification (Market Phase)

**Goal:** Predict market phase (0=COLD, 1=STABLE, 2=HOT)

**Owner:** _______________

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
train = pd.read_csv('classification_train.csv')
test = pd.read_csv('classification_test.csv')

# 2. Define features and target
features = [
    'homes_sold', 'inventory', 'median_dom',
    'off_market_in_two_weeks', 'avg_sale_to_list', 'sold_above_list',
    'median_sale_price_mom', 'median_sale_price_yoy',
    'homes_sold_yoy', 'inventory_yoy', 'median_dom_yoy',
    'year', 'state_NC',
    'Q_1', 'Q_2', 'Q_3', 'Q_4'
]

X_train = train[features]
y_train = train['market_phase_encoded']
X_test = test[features]
y_test = test['market_phase_encoded']

# 3. Train model WITH class_weight='balanced' (handles imbalanced data)
model = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced',  # <-- IMPORTANT: don't remove this
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds, target_names=['COLD', 'STABLE', 'HOT']))

# 5. Confusion matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['COLD', 'STABLE', 'HOT'],
            yticklabels=['COLD', 'STABLE', 'HOT'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Market Phase Confusion Matrix')
plt.show()
```

**Metrics to report:** Accuracy, F1-score per class, confusion matrix

---

## Track 3: LSTM (Time Series Forecasting)

**Goal:** Predict next quarter's price using past 4 quarters

**Owner:** _______________

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load data
df = pd.read_csv('lstm_timeseries.csv')
df = df.sort_values(['region', 'period_begin'])

# 2. Create sequences for ONE zip code (start simple)
def create_sequences(data, seq_length=4):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 3. Pick one zip to test
sample_zip = df[df['region'] == df['region'].iloc[0]]
prices = sample_zip['median_sale_price'].values.reshape(-1, 1)

# 4. Scale data (LSTM needs values between 0-1)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# 5. Create sequences
SEQ_LENGTH = 4
X, y = create_sequences(prices_scaled, SEQ_LENGTH)

# 6. Split (keep last 20% for test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 7. Build model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 8. Train
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# 9. Predict and inverse transform
preds_scaled = model.predict(X_test)
preds = scaler.inverse_transform(preds_scaled)
actual = scaler.inverse_transform(y_test)

# 10. Plot
import matplotlib.pyplot as plt
plt.plot(actual, label='Actual')
plt.plot(preds, label='Predicted')
plt.legend()
plt.title('LSTM Price Forecast')
plt.show()
```

**Metrics to report:** RMSE, MAE, forecast plot

---

## What is `scaler.pkl` and when do I need it?

The `scaler.pkl` file contains a saved StandardScaler object that was fit on training data. You need it when making predictions on new data.

```python
import joblib

# Load the scaler
scaler = joblib.load('scaler.pkl')

# If you have new raw data, transform it before predicting
new_data_scaled = scaler.transform(new_data[continuous_features])
prediction = model.predict(new_data_scaled)
```

**When to use it:**
- Regression/Classification: The train/test CSVs are already scaled—you don't need it unless you're predicting on brand new data
- LSTM: Uses its own MinMaxScaler internally, so ignore `scaler.pkl`

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'column_name'` | Typo in feature name | Copy feature list exactly from above |
| `ValueError: shapes don't align` | Wrong number of features | Check you're using all features in the list |
| `ModuleNotFoundError: xgboost` | XGBoost not installed | Run `pip install xgboost` |
| `ResourceWarning: unclosed file` | Ignore | Doesn't affect results |

---

## Submission Checklist

Each track should produce:

- [ ] Trained model(s) with hyperparameters documented
- [ ] Evaluation metrics on test set
- [ ] At least one visualization (predictions vs actual, confusion matrix, etc.)
- [ ] Brief written summary (2-3 sentences on what worked)