import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost

def generate_fraud_data():
    """Generates synthetic transaction data."""
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'user_id': np.random.randint(1, 100, n),
        'transaction_amount': np.random.uniform(5.0, 5000.0, n),
        'user_past_24h_spend': np.random.uniform(0.0, 10000.0, n),
        'merchant_risk_score': np.random.uniform(0.0, 1.0, n),
        'is_fraud': np.random.choice([0, 1], n, p=[0.95, 0.05])
    })
    return df

def train_models():
    df = generate_fraud_data()
    features = ['transaction_amount', 'user_past_24h_spend', 'merchant_risk_score']
    X = df[features]
    y = df['is_fraud']

    # Tells MLflow to use a local SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud_detection_system")

    # 1. Train the Champion (Production) Model
    with mlflow.start_run(run_name="champion_random_forest"):
        champion = RandomForestClassifier(n_estimators=50, max_depth=5)
        champion.fit(X, y)
        mlflow.sklearn.log_model(
            sk_model=champion,
            artifact_path="model",
            registered_model_name="Fraud_Champion_RF"
        )
        print("✅ Champion model trained and registered.")

    # 2. Train the Challenger (Shadow) Model
    with mlflow.start_run(run_name="challenger_xgboost"):
        challenger = XGBClassifier(n_estimators=50, max_depth=5, use_label_encoder=False, eval_metric='logloss')
        challenger.fit(X, y)
        mlflow.xgboost.log_model(
            xgb_model=challenger,
            artifact_path="model",
            registered_model_name="Fraud_Shadow_XGB"
        )
        print("✅ Shadow model trained and registered.")

if __name__ == "__main__":
    train_models()