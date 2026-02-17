<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&pause=1000&color=00D4FF&center=true&vCenter=true&width=800&lines=TransferIQ+%F0%9F%8F%9F%EF%B8%8F;AI-Powered+Football+Transfer+Valuation;Predict.+Analyze.+Dominate+the+Market." alt="Typing SVG" />

<br/>


[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boost-2ECC71?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-9B59B6?style=for-the-badge)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)]()
[![Colab](https://img.shields.io/badge/Google%20Colab-12%20Notebooks-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **A full-scale, production-ready AI system** that predicts football player
> transfer market values by fusing on-pitch performance, social media sentiment,
> injury risk, and market data — built across 12 development modules with deep
> learning, ensemble models, SHAP explainability, and a live Streamlit dashboard.

</div>

---

## Table of Contents

- [Project Highlights](#project-highlights)
- [System Architecture](#system-architecture)
- [Data Sources & Features](#data-sources--features)
- [Tech Stack](#tech-stack)
- [12-Week Development Modules](#12-week-development-modules)
- [Models & Algorithms](#models--algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Outputs & Files](#key-outputs--files)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Future Scope](#future-scope)
- [License](#license)

---

## Project Highlights

TransferIQ is not just a prediction model — it is a **complete football intelligence platform** built from scratch over 12 weeks of structured development:

| Capability | Details |
|---|---|
| **Transfer Value Prediction** | LSTM deep learning + XGBoost/LightGBM ensemble — predicts market value in €M |
| **Multi-Step Forecasting** | Encoder-Decoder LSTM forecasts player value across next 3 transfer windows |
| **NLP Sentiment Analysis** | VADER + TextBlob sentiment scoring from social media player mentions |
| **Transfer Probability** | Binary & multi-class classification — predicts if/when a player will transfer |
| **Player Similarity Engine** | Cosine similarity + PCA clustering to find comparable players and bargains |
| **SHAP Explainability** | Feature importance and decision explanations for every prediction |
| **Interactive Dashboard** | Full Streamlit web application with charts, predictions, and comparisons |
| **REST API** | FastAPI endpoint for programmatic access to model predictions |
| **PDF Reports** | Auto-generated player valuation reports via ReportLab |
| **80+ Engineered Features** | Performance trends, injury risk scores, contract duration, sentiment metrics |

---

## System Architecture
```
                +------------------------------------------------------------------+
                |                        DATA INGESTION LAYER                      |
                |  +------------------+  +-----------------+  +-----------------+  |
                |  |  StatsBomb Open  |  |  Transfermarkt  |  |  Twitter API    |  |
                |  |  Data (Player    |  |  Web Scraping   |  |  (Social Media  |  |
                |  |  Performance)    |  |  (Market Value) |  |   Sentiment)    |  |
                |  +--------+---------+  +--------+--------+  +--------+--------+  |
                |      Injury Records ──────────────────────────────────────┘      |
                +------------------------------------------------------------------+
                                                  |
                +------------------------------------------------------------------+
                |            PREPROCESSING & FEATURE ENGINEERING                   |
                |   Missing Value Handling  |  StandardScaler / MinMaxScaler       |
                |   80+ Engineered Features |  One-Hot Encoding                    |
                |   Performance Trends      |  IsolationForest Outlier Detection   |
                |   Injury Risk Scores      |  Contract Duration Features          |
                |   NLP Sentiment Scores    |  Player Quality Composite Score      |
                +------------------------------------------------------------------+
                                                  |
                             +--------------------+--------------------+
                             |                    |                    |
                        +-----------+  +-------------------+  +------------------+
                        |Univariate |  | Multivariate LSTM |  | Encoder-Decoder  |
                        |   LSTM    |  | (256-128-64-32    |  | LSTM (Seq2Seq    |
                        |(128-64-32)|  |  units + Dropout) |  |  Multi-Window)   |
                        +-----------+  +-------------------+  +------------------+
                             +--------------------+--------------------+
                                                  |
                                 +----------------+----------------+
                                 |                                 |
                          +------+------+                  +-------+------+
                          |   XGBoost   |                  |   LightGBM   |
                          |  (Optuna    |                  |  (50 Trial   |
                          |  Tuned)     |                  |   Bayesian)  |
                          +------+------+                  +-------+------+
                                 |                                 |
                                 +----------------+----------------+
                                                  |
                                     +------------+------------+
                                     |  WEIGHTED ENSEMBLE      |
                                     |  (XGB 0.5 + LGB 0.5)    |
                                     +------------+------------+
                                                  |
                                 +----------------+----------------+
                                 |                                 |
                          +------+------+                  +-------+-------+
                          |  TRANSFER   |                  |    PLAYER     |
                          |   VALUE     |                  |  SIMILARITY   |
                          | PREDICTION  |                  |    ENGINE     |
                          | (€M output) |                  | (Cosine + PCA)|
                          +------+------+                  +-------+-------+
                                 |
                          +------+------+
                          |    SHAP     |
                          | EXPLAINER   |
                          | (Why this   |
                          |  value?)    |
                          +------+------+
                                 |
                          +------+------+                  +---------------+
                          | STREAMLIT   |                  |   FastAPI     |
                          |  DASHBOARD  |                  |  REST API     |
                          | (Web App)   |                  | /predict POST |
                          +-------------+                  +---------------+
```

---

## Data Sources & Features

### Data Sources

| Source | Data Type | Method |
|--------|-----------|--------|
| **StatsBomb Open Data** | Goals, assists, appearances, avg rating, xG, pass completion | Python SDK |
| **Transfermarkt** | Historical player market valuations per transfer window | Web Scraping (BeautifulSoup) |
| **Twitter API** | Player mention volume, sentiment polarity | Tweepy REST API |
| **Injury Records** | Injury type, duration, frequency, days since last injury | Data Compilation |

### Engineered Features (80+)

The feature engineering pipeline creates over 80 predictive features across four categories:

**Performance Features** — total goals, total assists, avg rating, appearances, xG, pass completion rate, goals per game, rating trend, career stage classification, performance momentum score

**Injury Risk Features** — total injuries, avg injury duration, injury frequency, days since last injury, injury severity score, games missed, had ACL/knee/hamstring injury flags, currently injured flag

**Sentiment Features** — compound sentiment score (VADER), positive/negative/neutral ratios, total mention volume, social media popularity score, sentiment trend

**Market & Contract Features** — contract months remaining, age-value curve position, peak value estimate, transfer history total, player quality composite score, market attractiveness index

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow 2.x / Keras (LSTM, Encoder-Decoder, RepeatVector, TimeDistributed) |
| **ML & Ensemble** | XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, MLPRegressor |
| **Hyperparameter Tuning** | Optuna (Bayesian optimization, 50 trials) |
| **NLP & Sentiment** | VADER (vaderSentiment), TextBlob |
| **Model Explainability** | SHAP (TreeExplainer, force plots, dependence plots) |
| **Classification** | Scikit-learn (LogisticRegression, RandomForest, XGBClassifier), SMOTE |
| **Dimensionality Reduction** | PCA (sklearn.decomposition) |
| **Similarity Engine** | Cosine Similarity (sklearn.metrics.pairwise) |
| **Web Framework** | Streamlit (interactive dashboard), FastAPI (REST API) |
| **Visualization** | Matplotlib, Seaborn, Plotly (interactive), ipywidgets |
| **Data Collection** | BeautifulSoup4, Tweepy, StatsBombPy, Requests |
| **Data Processing** | Pandas, NumPy, Scikit-learn (StandardScaler, MinMaxScaler) |
| **Anomaly Detection** | IsolationForest (sklearn.ensemble) |
| **Report Generation** | ReportLab (PDF reports) |
| **Model Persistence** | Joblib (.pkl), Keras (.h5) |
| **Development** | Google Colab (12 notebooks), Jupyter |

---

## 12-Week Development Modules
```
Module  1 (Week 1)  ──►  Multi-Source Data Collection & EDA
Module  2 (Week 2)  ──►  Data Merging, Cleaning & Feature Engineering
Module  3 (Week 3)  ──►  Advanced Feature Engineering (Performance Trends)
Module  4 (Week 4)  ──►  Sentiment Analysis & Dataset Finalization
Module  5 (Week 5)  ──►  LSTM Deep Learning Models (3 architectures)
Module  6 (Week 6)  ──►  XGBoost / LightGBM Ensemble Models
Module  7 (Week 7)  ──►  Optuna Hyperparameter Tuning & Final Model Selection
Module  8 (Week 8)  ──►  Deployment, Interactive Visualizations & PDF Reports
Module  9 (Week 9)  ──►  Player Similarity & Bargain Finder Engine
Module 10 (Week 10) ──►  Transfer Probability Prediction (Classification)
Module 11 (Week 11) ──►  SHAP Model Explainability & Feature Attribution
Module 12 (Week 12) ──►  Streamlit Interactive Web Dashboard
```

<details>
<summary><b>Module 1 — Data Collection & Exploration</b></summary>

**What was built:**
- Player performance data generation from StatsBomb Open Data
- Market value simulation pipeline with realistic transfer window distributions
- Injury record generation with ACL/knee/hamstring/surgery classifications
- Social media sentiment data with VADER polarity scores and mention volumes
- Live data fetcher class using football-data.org API
- Initial EDA: distributions, missing value heatmaps, correlation analysis

**Key outputs:** `transferiq_players.csv`, `transferiq_performance.csv`, `transferiq_injuries.csv`, `transferiq_sentiment.csv`, `transferiq_market.csv`

</details>

<details>
<summary><b>Module 2 — Cleaning, Merging & Preprocessing</b></summary>

**What was built:**
- Multi-dataset merge pipeline (players + performance + injury + sentiment + market → master_df)
- Missing value detection and imputation with median/mode strategies
- IsolationForest anomaly detection to flag unusual player profiles (5% contamination)
- StandardScaler and MinMaxScaler feature scaling pipelines
- One-hot encoding for categorical variables (position, club, nationality)
- Train-test split (80/20) with stratification

**Key outputs:** `transferiq_final_dataset.csv`, `transferiq_encoded_dataset.csv`, `transferiq_scaled_dataset.csv`

</details>

<details>
<summary><b>Module 3 — Advanced Feature Engineering (Performance Trends)</b></summary>

**What was built:**
- Time-based performance trend features (rating trajectory, scoring trend, form momentum)
- Injury impact scoring (severity score = total injury days / 365)
- Contract urgency features (months remaining, negotiation window flag)
- Career stage classification (emerging / peak / declining / veteran)
- Player quality composite score combining rating, goals, and consistency

</details>

<details>
<summary><b>Module 4 — Sentiment Integration & Final Dataset</b></summary>

**What was built:**
- Full NLP pipeline: VADER compound score, positive/negative/neutral ratios
- Social media popularity index correlated with player fame and position
- Sentiment trend features (improving/declining public perception)
- PlayerComparison prototype using cosine similarity
- Final dataset finalization: 80+ features ready for model training

</details>

<details>
<summary><b>Module 5 — LSTM Deep Learning Models</b></summary>

**What was built — 3 model architectures:**

*Univariate LSTM:* Sequential model — 128 → 64 → 32 LSTM units with Dropout(0.2), Dense(16) → Dense(1). Trained with EarlyStopping (patience=15), ReduceLROnPlateau, and ModelCheckpoint.

*Multivariate LSTM:* Deeper architecture — 256 → 128 → 64 → 32 LSTM units with Dropout(0.3/0.2), Dense(32) → Dense(16) → Dense(1). Processes all 80+ features simultaneously.

*Encoder-Decoder LSTM:* Seq2Seq architecture using Keras Functional API — Encoder LSTM(128) → Dropout → RepeatVector → Decoder LSTM(128) → LSTM(64) → TimeDistributed(Dense(32)) → Flatten → Dense(1). Designed for multi-window forecasting.

*MultiStepForecaster class:* Predicts player values at 6, 12, and 18 months ahead in a single inference pass.

*PyTorch Transformer:* PlayerValueTransformer using nn.TransformerEncoder (d_model=128, nhead=8, 3 layers) with positional encoding.

**Evaluation:** RMSE, MAE, R², MAPE with inverse-transformed predictions; loss curve visualization for all 3 models; 9 saved model files (.h5 + .pkl scalers).

</details>

<details>
<summary><b>Module 6 — Ensemble Models & Integration</b></summary>

**What was built:**
- XGBoost regressor with structured feature importance analysis
- LightGBM regressor with categorical feature handling
- RandomForestRegressor baseline
- GradientBoostingRegressor baseline
- **Weighted ensemble:** XGB(0.5) + LGB(0.5) predictions averaged
- **Meta-model stacking:** Train a meta-learner on top of base model predictions
- **AdvancedEnsemble class:** CatBoost + MLP + XGB + LGB with `scipy.optimize` weight optimization
- **TransferProbabilityModel:** RandomForest classifier for binary transfer prediction with CalibratedClassifierCV

**Feature importance:** Bar plots for top features from both XGBoost and LightGBM models.

</details>

<details>
<summary><b>Module 7 — Optuna Hyperparameter Tuning</b></summary>

**What was built:**
- **Optuna** Bayesian optimization with 50 trials for both XGBoost and LightGBM
- Tuned parameters: learning rate, n_estimators, max_depth, subsample, colsample, reg_alpha/lambda, min_child_weight
- `plot_optimization_history` and `plot_param_importances` from Optuna visualization
- 5-Fold cross-validation with KFold
- Final model selection based on validation RMSE
- **QuantileRegressor** for prediction intervals (2.5th, 50th, 97.5th percentiles — 95% confidence intervals)

**Key outputs:** `final_xgboost_optimized.pkl`, `final_lightgbm_optimized.pkl`

</details>

<details>
<summary><b>Module 8 — Deployment, Visualizations & Reports</b></summary>

**What was built:**
- Final model loading and prediction pipeline
- Plotly interactive dashboards: player value trends, prediction scatter plots, residual analysis, feature importance charts
- `transferiq_deployment.py` script for production use
- **ReportLab PDF generator:** Auto-creates per-player valuation report PDFs
- `predict_player_value()` function: takes dict input → returns predicted value + confidence level
- LSTM + ensemble batch prediction for all players in the dataset
- Comprehensive summary statistics and exportable CSVs

</details>

<details>
<summary><b>Module 9 — Player Similarity & Bargain Finder Engine</b></summary>

**What was built:**
- **Cosine similarity engine:** Find top-N most similar players based on scaled feature vectors
- **PCA clustering:** Reduce 80+ features to 2D for visual player grouping
- **Bargain identifier:** Flags players whose predicted value >> current market value (undervalued)
- **Replacement finder:** Given a player, suggest similar-profile alternatives within a budget
- **Side-by-side comparison:** Multi-stat comparison for any two players
- Interactive Plotly scatter plots with PCA clusters and player labels

**Key outputs:** `similar_players_*.csv`, `bargain_players_*.csv`, `*_replacements.csv`

</details>

<details>
<summary><b>Module 10 — Transfer Probability Prediction</b></summary>

**What was built:**
- **Binary classification:** Will a player transfer in the next window? (yes/no)
- **Multi-class classification:** Which transfer window will they move in?
- **SMOTE** oversampling to handle class imbalance
- Models compared: LogisticRegression, RandomForest, GradientBoosting, XGBClassifier, LGBMClassifier
- Metrics: Accuracy, ROC-AUC, F1-Score, Confusion Matrix, Precision-Recall curves
- Transfer risk scoring: Low / Medium / High / Very High categories
- Team-level transfer risk aggregation and reporting
- Contract expiry risk flags and high-value player watchlist

**Key outputs:** `transfer_probability_model.pkl`, `high_risk_transfers.csv`, `team_transfer_risk_summary.csv`

</details>

<details>
<summary><b>Module 11 — SHAP Model Explainability</b></summary>

**What was built:**
- SHAP TreeExplainer for XGBoost and LightGBM models
- Global summary plots (beeswarm) showing feature importance distribution
- Individual force plots: "Why is this player valued at €X million?"
- SHAP dependence scatter plots with interaction feature highlighting
- `create_dependence_plot()` utility for any feature pair
- Exported interactive HTML SHAP visualizations

</details>

<details>
<summary><b>Module 12 — Streamlit Interactive Dashboard</b></summary>

**What was built — Full web application with 8 pages:**

1. **Home Page** — Overview stats, dataset summary, model performance KPIs
2. **Player Lookup** — Search any player, view detailed performance profile
3. **Market Value Predictor** — Form-based input to predict any player's value in real-time
4. **Market Analysis** — Position-wise value trends, age curves, league comparisons
5. **Transfer Risk Assessment** — Player/team-level transfer probability scores
6. **Player Comparison Tool** — Side-by-side multi-stat comparison with radar charts
7. **SHAP Explanations** — Interactive model decision explanations
8. **Team Analysis** — Squad value breakdowns and transfer budget analysis

*Deployed via:* ngrok (Colab), Streamlit Cloud (one-click), or self-hosted (Heroku/Railway)

</details>

---

## Models & Algorithms

### Regression Models (Transfer Value Prediction)

| Model | Architecture | Optimizer |
|-------|-------------|-----------|
| Univariate LSTM | 128→64→32 units, Dropout(0.2) | Adam lr=0.001, EarlyStopping |
| Multivariate LSTM | 256→128→64→32 units, Dropout(0.3) | Adam lr=0.001, ReduceLROnPlateau |
| Encoder-Decoder LSTM | Encoder(128) → RepeatVector → Decoder(128→64) → TimeDistributed | Adam lr=0.001 |
| XGBoost (Tuned) | 50-trial Optuna Bayesian optimization | RMSE objective |
| LightGBM (Tuned) | 50-trial Optuna Bayesian optimization | RMSE objective |
| Weighted Ensemble | XGB(0.5) + LGB(0.5) | Scipy Nelder-Mead weight optimization |
| Meta-Stacking | XGB/LGB/RF base → meta-learner | 5-fold CV stacking |
| PyTorch Transformer | d_model=128, nhead=8, 3 encoder layers | AdamW |

### Classification Models (Transfer Probability)

| Model | Task | Handling |
|-------|------|---------|
| XGBClassifier | Binary: will transfer? | SMOTE balanced |
| LGBMClassifier | Multi-class: which window? | SMOTE balanced |
| RandomForestClassifier | Binary + calibrated probability | CalibratedClassifierCV |
| LogisticRegression | Baseline binary | StandardScaler |

---

## Evaluation Metrics

| Metric | Task | Interpretation |
|--------|------|---------------|
| **RMSE** (€M) | Regression | Penalizes large prediction errors |
| **MAE** (€M) | Regression | Average absolute prediction error |
| **R² Score** | Regression | Variance in market value explained |
| **MAPE** (%) | Regression | Percentage prediction error |
| **95% Prediction Interval** | Regression | Uncertainty range via Quantile Regression |
| **ROC-AUC** | Classification | Transfer probability discrimination |
| **F1-Score** | Classification | Balanced precision/recall |
| **Confusion Matrix** | Classification | Transfer prediction accuracy breakdown |

---

## Key Outputs & Files
```
Saved Models:
  univariate_lstm_model.h5          — Trained Univariate LSTM
  multivariate_lstm_model.h5        — Trained Multivariate LSTM
  encoder_decoder_lstm_model.h5     — Trained Encoder-Decoder LSTM
  final_xgboost_optimized.pkl       — Optuna-tuned XGBoost
  final_lightgbm_optimized.pkl      — Optuna-tuned LightGBM
  meta_ensemble_model.pkl           — Meta-stacking model
  transfer_probability_model.pkl    — Transfer classifier

Datasets:
  transferiq_final_dataset.csv      — Master dataset (80+ features)
  transferiq_encoded_dataset.csv    — ML-ready encoded dataset
  week5_lstm_results.csv            — LSTM prediction results
  week6_ensemble_results.csv        — Ensemble prediction results
  high_risk_transfers.csv           — Players likely to transfer
  bargain_players_*.csv             — Undervalued player lists

Deployment:
  streamlit_app.py                  — Full interactive web dashboard
  transferiq_api.py                 — FastAPI REST API
  transferiq_deployment.py          — Production prediction script
  report_<player_name>.pdf          — Per-player PDF valuation reports
```

---

## Getting Started

### Prerequisites
```bash
pip install tensorflow keras
pip install xgboost lightgbm catboost optuna
pip install scikit-learn imbalanced-learn
pip install pandas numpy matplotlib seaborn plotly
pip install vaderSentiment textblob
pip install streamlit fastapi uvicorn
pip install shap joblib reportlab
pip install statsbombpy tweepy beautifulsoup4 requests
```

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/your-username/TransferIQ.git
cd TransferIQ

# Install all dependencies
pip install -r requirements.txt

# Run the Colab notebooks in order (Week 1 → Week 12)
# OR run the modular scripts:

python src/week1_data_collection.py
python src/week2_preprocessing.py
python src/week3_feature_engineering.py
python src/week4_sentiment_analysis.py
python src/week5_lstm_models.py
python src/week6_ensemble_models.py
python src/week7_hyperparameter_tuning.py
python src/week8_deployment.py
python src/week9_player_similarity.py
python src/week10_transfer_probability.py
python src/week11_shap_explainability.py
python src/week12_streamlit_dashboard.py

# Launch the web dashboard
streamlit run streamlit_app.py

# Or start the REST API
uvicorn transferiq_api:app --reload
```

### Predict a Player's Value (API)
```python
import requests

player = {
    "age": 25,
    "position": "ST",
    "total_goals": 60,
    "total_assists": 20,
    "avg_rating": 7.8
}

response = requests.post("http://localhost:8000/predict", json=player)
print(response.json())
# Output: {"predicted_value": 42.5, "confidence": "High"}
```

---

## Project Structure
```
TransferIQ/
│
├── notebooks/                         # 12 Google Colab notebooks
│   ├── Week01_Data_Collection.ipynb
│   ├── Week02_Preprocessing.ipynb
│   ├── Week03_Feature_Engineering.ipynb
│   ├── Week04_Sentiment_Analysis.ipynb
│   ├── Week05_LSTM_Models.ipynb
│   ├── Week06_Ensemble_Models.ipynb
│   ├── Week07_Hyperparameter_Tuning.ipynb
│   ├── Week08_Deployment_Visualization.ipynb
│   ├── Week09_Player_Similarity.ipynb
│   ├── Week10_Transfer_Probability.ipynb
│   ├── Week11_SHAP_Explainability.ipynb
│   └── Week12_Streamlit_Dashboard.ipynb
│
├── src/                               # Modular Python scripts
│   ├── week1_data_collection.py
│   ├── week2_preprocessing.py
│   ├── week3_feature_engineering.py
│   ├── week4_sentiment_analysis.py
│   ├── week5_lstm_models.py
│   ├── week6_ensemble_models.py
│   ├── week7_hyperparameter_tuning.py
│   ├── week8_deployment.py
│   ├── week9_player_similarity.py
│   ├── week10_transfer_probability.py
│   ├── week11_shap_explainability.py
│   └── week12_streamlit_dashboard.py
│
├── data/
│   ├── raw/                           # Original collected datasets
│   ├── processed/                     # Cleaned & feature-engineered data
│   └── sentiment/                     # NLP sentiment outputs
│
├── models/
│   ├── lstm/                          # .h5 LSTM model files
│   └── ensemble/                      # .pkl XGBoost, LightGBM, ensemble files
│
├── reports/                           # Auto-generated PDF player reports
├── visualizations/                    # Plotly interactive HTML charts
├── streamlit_app.py                   # Full Streamlit web application
├── transferiq_api.py                  # FastAPI REST API
├── transferiq_deployment.py           # Production prediction script
├── requirements.txt
└── README.md
```

---

## API Reference
```
POST /predict
  Input:  { age, position, total_goals, total_assists, avg_rating, ... }
  Output: { predicted_value: float (€M), confidence: "High" | "Medium" }

GET /player/{player_id}
  Output: Full player profile with current and predicted market value

GET /similar/{player_id}?top_n=5
  Output: List of most similar players by feature cosine similarity

GET /transfer_risk/{player_id}
  Output: { transfer_probability: float, risk_level: "Low" | "Medium" | "High" }
```

---

## Future Scope

- **Real League Data Integration** — replace generated data with live StatsBomb, Opta, or Wyscout feeds
- **Multi-League Models** — separate models for Premier League, La Liga, Bundesliga, Serie A
- **Live Dashboard Updates** — connect to live match data APIs for in-season real-time predictions
- **Video Analysis Integration** — use computer vision to extract performance metrics from match footage
- **Agent & Club Negotiation Tool** — build a contract recommendation module for negotiation scenarios
- **Temporal Fusion Transformer** — upgrade time-series modeling with attention-based TFT architecture
- **Mobile App** — React Native wrapper for the Streamlit backend

---

## License

Licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

⭐ **If you found this project useful, please star the repository!**

*Built with passion across 12 weeks using Deep Learning · NLP · Ensemble ML · SHAP · Streamlit*

</div>
