# Multi-Strategy AI Trading Engine: Deep Learning & Ensemble Methods

This repository contains an advanced quantitative trading platform that utilizes a multi-model approach to execute **Pairs Trading strategies**. By analyzing the spread between highly correlated banking stocks, the engine generates actionable Buy, Sell, and Hold signals using a combination of traditional time-series analysis and modern artificial intelligence.

## 🚀 Key Features

* **Statistical Foundation**: Automates stationarity testing using the **Augmented Dickey-Fuller (ADF)** test and implements **SARIMA** models with AIC-based hyperparameter tuning.
* **Ensemble Learning (XGBoost)**: A classification engine that integrates technical indicators like **RSI, MACD, and Bollinger Bands** to predict spread movements.
* **Deep Learning (LSTM)**: Utilizes **Long Short-Term Memory** networks to capture sequential dependencies in price spreads for high-accuracy signal classification.
* **Predictive Entry/Exit Module**: A specialized LSTM regression model designed to forecast future **Z-scores**, enabling proactive entry and exit point identification.
* **Interactive Dashboard**: A full-stack **Streamlit** application featuring multi-tab navigation for model comparison, date-range filtering, and real-time visualization.



## 📊 Model Comparison

| Feature | SARIMA Model | XGBoost Classifier | LSTM Model |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Seasonal Time-Series Forecasting | Technical Indicator Classification | Sequence-based Signal Prediction |
| **Input Data** | Historical Spreads | RSI, MACD, Bollinger Bands, Lags | Lagged Z-Scores |
| **Optimization** | AIC-based Grid Search | GridSearchCV (Max Depth, Learning Rate) | Dropout Layers & Adam Optimizer |
| **Output Type** | Predicted Spread Value | Categorical Signal (Buy/Hold/Sell) | Categorical Signal (Buy/Hold/Sell) |

## 🛠️ Technical Implementation

### 1. Data Processing & Feature Engineering
* **Spread Calculation**: Computes the price difference between selected banking pairs (e.g., ICICI vs Axis).
* **Z-Score Normalization**: Implements rolling mean and standard deviation to calculate Z-scores for signal thresholding.
* **Stationarity Handling**: Automatically applies first-order differencing if the ADF test indicates non-stationarity.



### 2. Machine Learning Pipeline
* **XGBoost**: Trains on lagged features and momentum indicators to solve the classification problem of identifying trade opportunities.
* **LSTM Architecture**: Implements 3D input reshaping `[samples, time steps, features]` to process sequences through recurrent layers and dropout for regularization.

### 3. Strategy Evaluation
* **Metrics**: Evaluates performance using **F1-Score, Precision, Recall, and Accuracy** for classification, and **RMSE/MAE** for regression forecasting.
* **Visualizations**: Generates **Confusion Matrices** and **Signal Scatter Plots** to validate the reliability of trading triggers.



## 💻 Installation & Usage

1. **Clone the repository**:
```bash
git clone [https://github.com/yourusername/AI-Trading-Engine.git](https://github.com/yourusername/AI-Trading-Engine.git)
