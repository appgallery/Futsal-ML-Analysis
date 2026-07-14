import pytest
import pandas as pd
import numpy as np
from scripts.model_trainer import MatchModelTrainer, PlayerModelTrainer
from pipeline.training_pipeline import MatchTrainingPipeline, PlayerTrainingPipeline
import os
import glob

# --- MatchModelTrainer Tests ---
def test_prepare_data_features(sample_matches_df):
    trainer = MatchModelTrainer()
    # Add dummy feature columns to sample_matches_df for it to prepare
    df = sample_matches_df.copy()
    for col in trainer.feature:
        df[col] = 0.5
    df[trainer.target] = 1
    
    X, y, prepared_df = trainer.prepare_data(df)
    
    assert len(X.columns) == len(trainer.feature)
    for col in trainer.feature:
        assert col in X.columns
    assert trainer.target not in X.columns

def test_prepare_data_target(sample_matches_df):
    trainer = MatchModelTrainer()
    df = sample_matches_df.copy()
    for col in trainer.feature:
        df[col] = 0.5
    df[trainer.target] = 1
    
    X, y, prepared_df = trainer.prepare_data(df)
    
    assert y.name == trainer.target

def test_split_ratio():
    trainer = MatchModelTrainer()
    X = pd.DataFrame(np.random.rand(10, 8), columns=trainer.feature)
    y = pd.Series([1, 0] * 5, name=trainer.target)
    
    X_train, X_test, y_train, y_test = trainer.split_train_test(X, y, split_ratio=0.2)
    
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

def test_pipeline_has_scaler():
    from sklearn.ensemble import RandomForestClassifier
    trainer = MatchModelTrainer()
    pipeline = trainer.build_model(RandomForestClassifier)
    
    steps = [name for name, _ in pipeline.steps]
    assert 'scaler' in steps
    assert 'classifier' in steps

def test_train_returns_reports():
    trainer = MatchModelTrainer()
    X = pd.DataFrame(np.random.rand(20, 8), columns=trainer.feature)
    y = pd.Series([1, 0] * 10, name=trainer.target)
    
    reports = trainer.train(X, y)
    
    assert isinstance(reports, list)
    assert len(reports) > 0
    # verify tuple elements (name, accuracy, f1, params)
    for rep in reports:
        assert isinstance(rep[0], str)
        assert isinstance(rep[1], float)
        assert isinstance(rep[2], float)
        assert isinstance(rep[3], dict)

def test_best_model_selection():
    trainer = MatchModelTrainer()
    X = pd.DataFrame(np.random.rand(20, 8), columns=trainer.feature)
    y = pd.Series([1, 0] * 10, name=trainer.target)
    
    trainer.train(X, y)
    best_model, best_name, X_test, y_test = trainer.train_best_model(X, y)
    
    assert best_model is not None
    assert isinstance(best_name, str)

def test_model_save_load(tmp_path):
    trainer = MatchModelTrainer()
    X = pd.DataFrame(np.random.rand(20, 8), columns=trainer.feature)
    y = pd.Series([1, 0] * 10, name=trainer.target)
    
    trainer.train(X, y)
    best_model, _, X_test, _ = trainer.train_best_model(X, y)
    
    model_path = tmp_path / "test_model.pkl"
    trainer.save_model(best_model, str(model_path))
    
    loaded_model = trainer.load_model(str(model_path))
    
    # Check predictions are identical
    pred_orig = best_model.predict(X_test)
    pred_loaded = loaded_model.predict(X_test)
    assert np.array_equal(pred_orig, pred_loaded)

# --- PlayerModelTrainer Tests ---
def test_prepare_data_drops_targets():
    trainer = PlayerModelTrainer()
    df = pd.DataFrame({
        "match_date": ["2023-01-01", "2023-01-02"],
        "goals": [1, 0],
        "assists": [0, 1],
        "perf_score": [5, 3],
        "is_best_player": [1, 0],
        "feature1": [0.5, 0.2]
    })
    
    X, y, prepared_df = trainer.prepare_data(df)
    
    assert 'goals' not in X.columns
    assert 'assists' not in X.columns
    assert 'perf_score' not in X.columns
    assert 'is_best_player' not in X.columns
    assert 'feature1' in X.columns

def test_target_is_binary():
    trainer = PlayerModelTrainer()
    df = pd.DataFrame({
        "match_date": ["2023-01-01", "2023-01-02"],
        "is_best_player": [1, 0],
        "feature1": [0.5, 0.2]
    })
    
    X, y, _ = trainer.prepare_data(df)
    assert set(y.unique()).issubset({0, 1})

def test_feature_list_saved():
    trainer = PlayerModelTrainer()
    df = pd.DataFrame({
        "match_date": ["2023-01-01", "2023-01-02"],
        "is_best_player": [1, 0],
        "feature1": [0.5, 0.2]
    })
    
    trainer.prepare_data(df)
    assert trainer.feature == ["feature1"]

# --- Training Pipeline Tests ---
def test_old_models_deleted(tmp_path, monkeypatch):
    pipeline = MatchTrainingPipeline()
    
    # Change current working directory temporarily to tmp_path to isolate model creation/deletion
    monkeypatch.chdir(tmp_path)
    os.makedirs("models", exist_ok=True)
    os.makedirs("features", exist_ok=True)
    
    # Create fake old model
    old_model_path = "models/old_matches_model.pkl"
    with open(old_model_path, "w") as f:
        f.write("dummy")
    
    assert os.path.exists(old_model_path)
    
    # Save new best model
    from sklearn.linear_model import LogisticRegression
    dummy_model = LogisticRegression()
    pipeline.save_best_model(dummy_model, "SVC")
    
    # Old should be gone, new should be present
    assert not os.path.exists(old_model_path)
    assert os.path.exists("models/SVC_matches_model.pkl")

def test_features_saved_separately(tmp_path, monkeypatch):
    pipeline = MatchTrainingPipeline()
    monkeypatch.chdir(tmp_path)
    os.makedirs("models", exist_ok=True)
    os.makedirs("features", exist_ok=True)
    
    pipeline.save_features("features/test_features.pkl")
    assert os.path.exists("features/test_features.pkl")
