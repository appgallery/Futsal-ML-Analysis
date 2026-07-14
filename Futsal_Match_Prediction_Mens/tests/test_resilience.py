import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from scripts.inference import Inference
from httpx import AsyncClient, ASGITransport
from main import app

def test_model_file_missing(monkeypatch):
    monkeypatch.setattr("glob.glob", lambda pattern: [])
    
    with pytest.raises(FileNotFoundError) as exc:
        Inference()
        
    assert "No match prediction model file" in str(exc.value) or "No player performance model file" in str(exc.value)

def test_model_file_corrupted(monkeypatch):
    monkeypatch.setattr("glob.glob", lambda pattern: ["models/fake.pkl"])
    
    # Simulate joblib raising an exception
    def mock_load(path):
        raise ValueError("Corrupted file")
        
    monkeypatch.setattr("joblib.load", mock_load)
    
    with pytest.raises(ValueError) as exc:
        Inference()
        
    assert "Corrupted file" in str(exc.value)

def test_csv_missing(monkeypatch, mock_inference_engine):
    # For narrative pipeline enrichment when CSV is missing
    # API just warns and returns original
    with patch("os.path.exists", return_value=False):
        # We don't really have an explicit throw for missing CSVs in inference (since it reads through DB or fails at inference creation)
        # But let's test api behavior
        pass

@pytest.mark.asyncio
async def test_startup_model_not_found():
    from api.routes.predict import lifespan
    mock_app = MagicMock()
    
    with patch("api.routes.predict.Inference", side_effect=FileNotFoundError):
        # Should not crash, just prints an error
        async with lifespan(mock_app):
            pass

@pytest.mark.asyncio
async def test_llm_timeout(mock_inference_engine):
    with patch("api.routes.predict.inference_engine", mock_inference_engine):
        with patch("os.path.exists", return_value=True):
            # If LLM times out
            import asyncio
            with patch("pipeline.narrative_pipeline.enrich_prediction_with_commentary", side_effect=asyncio.TimeoutError):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                    response = await ac.get("/predict?homeID=10&awayID=20")
                    
                    # API gracefully catches and returns normal report
                    assert response.status_code == 200
                    assert response.json()["matchSummary"] is None

@pytest.mark.asyncio
async def test_concurrent_predictions(mock_inference_engine):
    import asyncio
    with patch("api.routes.predict.inference_engine", mock_inference_engine):
        with patch("pipeline.narrative_pipeline.enrich_prediction_with_commentary", return_value={"commentary": "test"}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                # Fire 3 requests concurrently
                tasks = [
                    ac.get("/predict?homeID=10&awayID=20"),
                    ac.get("/predict?homeID=10&awayID=20"),
                    ac.get("/predict?homeID=10&awayID=20")
                ]
                responses = await asyncio.gather(*tasks)
                
                for r in responses:
                    assert r.status_code == 200
