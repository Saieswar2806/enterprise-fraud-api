from fastapi.testclient import TestClient
from src.serve import app

def test_health_check():
    """Ensure the API boots up successfully."""
    # Using 'with' forces FastAPI to run the lifespan startup/shutdown events!
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200

def test_predict_fraud_success():
    """Ensure the model returns the exact JSON structure we expect."""
    with TestClient(app) as client:
        response = client.post(
            "/predict?transaction_id=test_999&user_id=1&transaction_amount=100.50"
        )
        
        # 1. Check if the server responded with '200 OK'
        assert response.status_code == 200
        
        # 2. Check if the JSON payload is formatted correctly for the frontend
        data = response.json()
        assert "fraud_detected" in data
        assert "status" in data
        assert data["transaction_id"] == "test_999"
        assert data["status"] in ["approved", "denied"]