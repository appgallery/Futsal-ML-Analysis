import os
import contextlib
import hashlib
import json
from typing import Union

from fastapi import APIRouter, Request, FastAPI, HTTPException, Depends
from scripts.inference import Inference, MatchInput, MatchOutputName, MatchOutputID
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
        if os.path.exists(DATA_FILE):
            current_hash = get_file_hash(DATA_FILE)
            previous_hash = None
            
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    previous_hash = state.get('data_hash')
            
            if current_hash != previous_hash:
                print("New data detected. Running full data and training pipeline...")
                # Run Data Pipeline
                DataPipeline().run_pipeline()
                
                # Run Training Pipeline
                TrainingPipeline().run_pipeline()
                
                # Update state
                with open(STATE_FILE, 'w') as f:
                    json.dump({'data_hash': current_hash}, f)
            else:
                print("No new data detected. Skipping pipeline run.")
        else:
            print(f"Data file {DATA_FILE} not found. Skipping training.")

        # Load the best model for inference
        try:
            global inference_engine
            inference_engine = Inference()
            print("Best model loaded successfully.")
        except FileNotFoundError:
            print("Error: Best model not found. Cannot perform predictions.")
            
    except Exception as e:
        print(f"Error during model initialization: {e}")
        
    yield
    print("Shutting down Prediction service...")


@routes.get("/predict", response_model=Union[MatchOutputName, MatchOutputID])
def predict(request: Request):
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine is not initialized")

    try:
        if "homeTeam" in request.query_params and "awayTeam" in request.query_params:
            return inference_engine.test_infer(
                MatchInput(
                    homeTeam=request.query_params.get("homeTeam"),
                    awayTeam=request.query_params.get("awayTeam")
                )
            )
        elif "homeID" in request.query_params and "awayID" in request.query_params:
            return inference_engine.test_infer(
                MatchInput(
                    homeID=int(request.query_params.get("homeID")),
                    awayID=int(request.query_params.get("awayID"))
                )
            )
        else:
            raise HTTPException(status_code=400, detail="Provide either homeTeam/awayTeam or homeID/awayID")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
