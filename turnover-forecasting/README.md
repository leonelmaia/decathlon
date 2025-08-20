# Turnover Forecasting

This project focuses on forecasting weekly decatlhon stores turnover using historical sales and store features. The goal is to build an end-to-end pipeline for data exploration, modeling, and multi-step forecasting.

## Dataset

train.csv.gz → historical weekly turnover values.

test.csv.gz → same structure as train but without turnover (used for forecasting).

bu_feat.csv.gz → additional store-level features.


## 1. Preliminary Exploration

Trend analysis: explored turnover evolution over time at the department and store level.

Comparisons: checked differences between departments and stores.

Seasonality: identified recurring peaks/dips likely linked to holidays or sports seasons.

### Department guessing:

Department 73 showed smaller peaks during the summer, then It may not be one of the most popular sports, but it tends to be practiced in milder or warmer temperatures. The stores that had the highest turnover were near the coast (with the exception of some in Paris). Therefore, I believe that it should be any kind of water sports, like Surf or Sailing.

Department 117 experiences peak winter peaks, with colder temperatures and regions where ski resorts are less than a three-hour drive away (with the exception of Paris and Lille). One of the busiest shops is even near Mont Blanc. That said, it's likely to be a snow-related sport. Perhaps skiing or snowboarding.

## 2. Multi-Step Forecasting

Forecast horizon: 8 weeks at the store-department level.

Features used: historical turnover, store-level attributes (bu_feat.csv.gz), time-based features (e.g., week number, seasonality).

Models: tested two forecasting approaches (models such as XGBoost and Prophet).

### Pipeline:

Data preprocessing (merging store features, generating time features).

Train-test split using historical data and time split.

Model training and validation.

Multi-step forecasting on test.csv.gz.

## Export predictions.

##Evaluation

Chosen metrics: RMSE and MAE.

RMSE: penalizes large errors, useful for overall accuracy.

MAPE: interpretable as percentage error, useful across departments with different turnover scales.

##  Environment

Experiments were run locally (not on Google Colab).

Implemented in Python, with common data science libraries:

pandas, numpy for data handling.

matplotlib, seaborn and plotly for visualization.

scikit-learn, Prophet, xgboost for modeling.
##  Observations & Challenges

Departments have heterogeneous turnover patterns, making it hard for a single model to generalize.

Seasonality and special events significantly impact turnover.

Data rescaling (for confidentiality) means absolute turnover magnitudes are not interpretable but relative patterns still hold.

## Results

Produced 8-week ahead forecasts at store-department granularity.

Pipeline can be extended to integrate external signals (holidays, weather, promotions) for improved accuracy.