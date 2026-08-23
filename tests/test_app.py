from fastapi.testclient import TestClient
import pytest

# Asumiendo que se llama app_2.py
from src.serve.app import app, ml_models

# Cliente de pruebas de FastAPI
client = TestClient(app)

# Clase falsa que simula el comportamiento de GradientBoostingClassifier
class MockModel:
    def predict_proba(self, X):
        # Devuelve [prob_clase_0_ROJO, prob_clase_1_AZUL]
        # X[0] es la primera fila introducida
        return [[0.25, 0.75]] 

@pytest.fixture(autouse=True)
def mock_ml_model():
    """Inyecta el modelo falso en el diccionario global antes de cada test."""
    ml_models["model"] = MockModel()
    yield
    ml_models.clear()

def test_health_endpoint():
    """El healthcheck debe responder 200 y que el modelo está cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_predict_endpoint_success():
    """Prueba que el /predict devuelva p_blue y p_red con un JSON válido."""
    payload = {
        "minute": 15,
        "kills_diff": 2,
        "cs_diff": 15,
        "level_diff": 1,
        "tower_diff": 1,
        "inhib_diff": 0,
        "dragon_diff": 1,
        "herald_diff": 1,
        "baron_diff": 0,
        "grub_diff": 3,
        "kills_diff_d5": 0,
        "cs_diff_d5": 5,
        "level_diff_d5": 0
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "p_blue" in data
    assert "p_red" in data
    assert data["p_blue"] == 0.75 # Valor que devuelve nuestro MockModel
    assert data["p_red"] == 0.25  # 1 - p_blue

def test_predict_endpoint_validation_error():
    """Debe dar error 422 si falta algún feature (ej: sin minute)."""
    payload = {
        "kills_diff": 2,
        "cs_diff": 15,
        "level_diff": 1
        # Faltan todos los demás features de train.FEATURES...
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic Validation Error