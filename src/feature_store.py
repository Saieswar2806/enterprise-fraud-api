import redis
import json
import numpy as np

def push_to_redis():
    # Connect to local Redis
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    print("Pushing user profiles to Redis...")
    for user_id in range(1, 101):
        # Simulating batch-computed features from a data warehouse
        user_features = {
            "user_past_24h_spend": round(np.random.uniform(10.0, 5000.0), 2),
            "merchant_risk_score": round(np.random.uniform(0.1, 0.9), 2)
        }
        r.set(f"user:{user_id}", json.dumps(user_features))
        
    print("✅ Successfully loaded features into Redis.")

if __name__ == "__main__":
    push_to_redis()