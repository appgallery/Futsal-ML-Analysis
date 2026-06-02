import os
import contextlib
import hashlib
import json
from fastapi import APIRouter, FastAPI, HTTPException, Depends
from scripts.inference import Inference, MatchInput, MatchOutput, MatchReportResponse
from pipeline.data_pipeline import DataPipeline
from pipeline.training_pipeline import TrainingPipeline

routes = APIRouter(tags=["Prediction"])

STATE_FILE = "data/pipeline_state.json"
DATA_FILE = "data/series-futsal-men-matches.csv"

inference_engine = None

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
    hasher.update(buf)
    return hasher.hexdigest()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up Prediction service...")
    try:
        # if os.path.exists(DATA_FILE):
        #     current_hash = get_file_hash(DATA_FILE)
        #     previous_hash = None
        #     
        #     if os.path.exists(STATE_FILE):
        #         with open(STATE_FILE, 'r') as f:
        #             state = json.load(f)
        #             previous_hash = state.get('data_hash')
        #     
        #     if current_hash != previous_hash:
        #         print("New data detected. Running full data and training pipeline...")
        #         # Run Data Pipeline
        #         DataPipeline().run_pipeline()
        #         
        #         # Run Training Pipeline
        #         TrainingPipeline().run_pipeline()
        #         
        #         # Update state
        #         with open(STATE_FILE, 'w') as f:
        #             json.dump({'data_hash': current_hash}, f)
        #     else:
        #         print("No new data detected. Skipping pipeline run.")
        # else:
        #     print(f"Data file {DATA_FILE} not found. Skipping training.")

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
def predict(homeID: int, awayID: int):
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
                homeID=homeID,
                awayID=awayID
            )
        )
        report = inference_engine.test_player_infer(match_pred)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
