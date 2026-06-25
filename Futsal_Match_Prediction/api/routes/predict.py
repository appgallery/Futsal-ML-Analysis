import os
import contextlib
import hashlib
import json
from typing import Optional
from fastapi import APIRouter, FastAPI, HTTPException, Depends
from scripts.inference import Inference, MatchInput, MatchOutput, MatchReportResponse
from pipeline.data_pipeline import DataPipeline
from pipeline.training_pipeline import TrainingPipeline

routes = APIRouter(tags=["Prediction"])

# STATE_FILE = "data/pipeline_state.json"
# DATA_FILE = "data/series-futsal-men-matches.csv"

inference_engine = None

# def get_file_hash(filepath):
#     hasher = hashlib.md5()
#     with open(filepath, 'rb') as f:
#         buf = f.read()
#     hasher.update(buf)
#     return hasher.hexdigest()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up Prediction service...")
    try:
        print("Running full data and training pipeline on startup...")
        # Run Data Pipeline
        # DataPipeline().run_pipeline()
        
        # Run Training Pipeline
        # TrainingPipeline().run_pipeline()

        # Load the best model for inference
        try:
            global inference_engine
            inference_engine = Inference()
            print("Best models loaded successfully.")
        except FileNotFoundError:
            print("Error: Best model files not found. Cannot perform predictions.")
            
    except Exception as e:
        print(f"Error during model initialization: {e}")
        
    yield
    print("Shutting down Prediction service...")


@routes.get("/predict", response_model=MatchReportResponse)
async def predict(homeID: int, awayID: int, matchID: Optional[int] = None):
    if homeID == awayID:
        raise HTTPException(status_code=400, detail="homeID and awayID cannot be the same")
        
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine is not initialized")
        
    if homeID not in inference_engine.team_memory:
        raise HTTPException(status_code=400, detail=f"homeID {homeID} is not present in the dataset.")
        
    if awayID not in inference_engine.team_memory:
        raise HTTPException(status_code=400, detail=f"awayID {awayID} is not present in the dataset.")

    try:
        match_pred = inference_engine.test_match_infer(
            MatchInput(
                matchID=matchID,
                homeID=homeID,
                awayID=awayID
            )
        )
        report = inference_engine.test_player_infer(match_pred)
        
        try:
            report_dict = report.model_dump()
            players_csv = "data/players_ml.csv"
            matches_csv = "data/matches_ml.csv"
            
            if os.path.exists(players_csv) and os.path.exists(matches_csv):
                from pipeline.narrative_pipeline import enrich_prediction_with_commentary
                api_key = os.environ.get("GEMINI_API_KEY")
                enriched = await enrich_prediction_with_commentary(
                    prediction_json=report_dict,
                    players_csv=players_csv,
                    matches_csv=matches_csv,
                    api_key=api_key
                )
                report = MatchReportResponse(**enriched)
            else:
                print(f"Warning: CSVs not found. players_ml.csv: {os.path.exists(players_csv)}, matches_ml.csv: {os.path.exists(matches_csv)}")
        except Exception as narrative_err:
            print(f"Error generating commentary: {narrative_err}")

        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
