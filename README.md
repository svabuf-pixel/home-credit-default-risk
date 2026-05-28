# Home Credit Default Risk

## Overview
End-to-end credit risk analysis pipeline built on
the Home Credit Default Risk dataset from Kaggle.

Predicts probability of loan default using
industry-standard credit scoring techniques.

## Techniques
- Exploratory Data Analysis (EDA)
- Weight of Evidence (WoE) and Information Value (IV)
- Optimal Binning (optbinning)
- Logistic Regression Scorecard
- XGBoost / LightGBM
- SHAP Explainability
- Model Evaluation: Gini, KS, AUC-ROC, PSI

## Project Structure
data/raw/          ← not tracked (see .gitignore)
data/processed/    ← not tracked
notebooks/         ← analysis notebooks
src/               ← reusable functions
outputs/           ← figures and saved models
requirements.txt   ← dependencies

## Setup
conda create -n home_credit python=3.11 -y
conda activate home_credit
pip install -r requirements.txt

## Dataset
https://www.kaggle.com/competitions/home-credit-default-risk

Download and place CSV files in data/raw/

## Author
Feliciu — Data Science Portfolio
