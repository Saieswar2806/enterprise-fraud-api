from fastapi import FastAPI, BackgroundTasks
import mlflow
import mlflow.pyfunc
import redis
import json
import pandas as pd
from contextlib import asynccontextmanager

champion_model = None
shadow_model = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models and connect to Redis on startup."""
    global champion_model, shadow_model, redis_client
    print("Booting up Enterprise Fraud API...")
    
    try:
        # Tell the server where to find the database
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Load models from local MLflow registry
        champion_model = mlflow.pyfunc.load_model("models:/Fraud_Champion_RF/latest")
        shadow_model = mlflow.pyfunc.load_model("models:/Fraud_Shadow_XGB/latest")
        
        # Connect to Feature Store
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ Models loaded and Redis connected.")
    except Exception as e:
        print(f"Startup failed: {e}")
        
    yield
    if redis_client:
        redis_client.close()

app = FastAPI(title="Fraud Detection API (Shadow Deployed)", lifespan=lifespan)

def log_shadow_prediction(transaction_id: str, features: pd.DataFrame, champion_pred: int):
    """Runs asynchronously to prevent API blocking."""
    shadow_pred = int(shadow_model.predict(features)[0])
    print(f"[SHADOW LOG] Txn: {transaction_id} | Champion: {champion_pred} | Shadow: {shadow_pred}")

@app.post("/predict")
def predict_fraud(transaction_id: str, user_id: int, transaction_amount: float, background_tasks: BackgroundTasks):
    # 1. Fetch live features from Redis
    user_data = redis_client.get(f"user:{user_id}")
    if not user_data:
        user_features = {"user_past_24h_spend": 0.0, "merchant_risk_score": 0.5}
    else:
        user_features = json.loads(user_data)
        
    # 2. Construct Model Input
    input_df = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "user_past_24h_spend": user_features["user_past_24h_spend"],
        "merchant_risk_score": user_features["merchant_risk_score"]
    }])
    
    # 3. Champion Prediction (Synchronous)
    champion_pred = int(champion_model.predict(input_df)[0])
    
    # 4. Shadow Prediction (Asynchronous)
    background_tasks.add_task(log_shadow_prediction, transaction_id, input_df, champion_pred)
    
    # 5. Return only the trusted Champion result
    return {
        "transaction_id": transaction_id,
        "fraud_detected": bool(champion_pred),
        "status": "approved" if champion_pred == 0 else "denied"
    }