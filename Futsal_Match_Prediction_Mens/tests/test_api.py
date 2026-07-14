import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_app_inference(mock_inference_engine):
    with patch("api.routes.predict.inference_engine", mock_inference_engine):
        yield

@pytest.mark.asyncio
async def test_predict_success(mock_app_inference):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/predict?homeID=10&awayID=20")
        
    assert response.status_code == 200
    data = response.json()
    assert data["homeTeam"] == "Team A"
    assert data["awayTeam"] == "Team B"

@pytest.mark.asyncio
async def test_predict_same_team_ids(mock_app_inference):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/predict?homeID=10&awayID=10")
        
    assert response.status_code == 400
    assert "cannot be the same" in response.json()["detail"]

@pytest.mark.asyncio
async def test_predict_unknown_home_id(mock_app_inference):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/predict?homeID=99&awayID=20")
        
    assert response.status_code == 400
    assert "not present in the dataset" in response.json()["detail"]

@pytest.mark.asyncio
async def test_predict_unknown_away_id(mock_app_inference):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/predict?homeID=10&awayID=99")
        
    assert response.status_code == 400
    assert "not present in the dataset" in response.json()["detail"]

@pytest.mark.asyncio
async def test_predict_no_inference_engine():
    with patch("api.routes.predict.inference_engine", None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/predict?homeID=10&awayID=20")
            
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"]

@pytest.mark.asyncio
async def test_predict_with_match_id(mock_app_inference):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/predict?homeID=10&awayID=20&matchID=1")
        
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_commentary_failure_graceful(mock_app_inference):
    # If enrich_prediction_with_commentary throws an error, it should be caught and logged
    # and the API should still return 200 with the core report.
    with patch("pipeline.narrative_pipeline.enrich_prediction_with_commentary", side_effect=Exception("LLM Failed")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/predict?homeID=10&awayID=20")
            
        assert response.status_code == 200
        data = response.json()
        assert data["homeTeam"] == "Team A"
        assert data["matchSummary"] is None
