import os
import pandas as pd
from scripts.model_trainer import ModelTrainer


class MatchTrainingPipeline:
    def __init__(self, dataset_path='data/matches_ml.csv'):
        self.dataset_path = dataset_path
        self.trainer = ModelTrainer(mode='match')

    def load_data(self):
        print("Loading match feature data...")
        return pd.read_csv(self.dataset_path)

    def prepare_training_data(self, feature_df):
        return self.trainer.prepare_data(feature_df)

    def train_models(self, X, y):
        print("Training match models...")
        return self.trainer.train(X, y)

    def print_training_report(self, models_report):
        print("\n" + "="*50)
        print("Match Training Report:")
        print("="*50)

        for report in models_report:
            print(f"Model: {report[0]}")
            print(f"Accuracy: {report[1]}")
            print(f"F1-Score: {report[2]}")
            print(f"Best Params: {report[3]}")
            print("-" * 50)

    def train_best_model(self, X, y):
        return self.trainer.train_best_model(X, y)

    def save_best_model(self, model, model_name):
        import glob
        # Clear old match classification models first
        old_models = glob.glob('models/*_matches_model.pkl')
        for f in old_models:
            try:
                os.remove(f)
            except Exception:
                pass
        dynamic_path = f'models/{model_name}_matches_model.pkl'
        os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
        self.trainer.save_model(model, dynamic_path)
        print(f"Best match model (architecture-specific) saved to {dynamic_path}")

    def save_features(self, path='features/match_trained_features.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.trainer.save_model(self.trainer.feature, path)
        print(f"Match features saved to {path}")

    def run_pipeline(self):
        # Step 1: Load dataset
        feature_df = self.load_data()

        # Step 2: Prepare dataset
        X, y, _ = self.prepare_training_data(feature_df)

        # Step 3: Train all models
        models_report = self.train_models(X, y)

        # Step 4: Print report
        self.print_training_report(models_report)

        # Step 5: Train best model
        best_model, best_model_name, X_test, y_test = (
            self.train_best_model(X, y)
        )

        print(f"\nBest match model name: {best_model_name}")
        print(f"Best match model type: {type(best_model)}")

        # Step 6: Save best model dynamically and generic
        self.save_best_model(best_model, best_model_name)

        # Step 7: Save features
        self.save_features()

        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "X_test": X_test,
            "y_test": y_test
        }


class PlayerTrainingPipeline:
    def __init__(self, dataset_path='data/players_ml.csv'):
        self.dataset_path = dataset_path
        self.trainer = ModelTrainer(mode='player')

    def load_data(self):
        print("Loading player feature data...")
        return pd.read_csv(self.dataset_path)

    def prepare_training_data(self, feature_df):
        return self.trainer.prepare_data(feature_df)

    def train_models(self, X, y):
        print("Training player models...")
        return self.trainer.train(X, y)

    def print_training_report(self, models_report):
        print("\n" + "="*50)
        print("Player Training Report:")
        print("="*50)

        for report in models_report:
            print(f"Model: {report[0]}")
            print(f"F1-Score: {report[1]}")
            print(f"Metrics: {report[2]}")
            print(f"Best Params: {report[3]}")
            print("-" * 50)

    def train_best_model(self, X, y):
        return self.trainer.train_best_model(X, y)

    def save_best_model(self, model, model_name):
        import glob
        # Clear old player regression models first
        old_models = glob.glob('models/*_players_model.pkl')
        for f in old_models:
            try:
                os.remove(f)
            except Exception:
                pass
        dynamic_path = f'models/{model_name}_players_model.pkl'
        os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
        self.trainer.save_model(model, dynamic_path)
        print(f"Best player model (architecture-specific) saved to {dynamic_path}")

    def save_features(self, path='features/player_trained_features.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.trainer.save_model(self.trainer.feature, path)
        print(f"Player features saved to {path}")

    def run_pipeline(self):
        # Step 1: Load dataset
        feature_df = self.load_data()

        # Step 2: Prepare dataset
        X, y, _ = self.prepare_training_data(feature_df)

        # Step 3: Train all models
        models_report = self.train_models(X, y)

        # Step 4: Print report
        self.print_training_report(models_report)

        # Step 5: Train best model
        best_model, best_model_name, X_test, y_test = (
            self.train_best_model(X, y)
        )

        print(f"\nBest player model name: {best_model_name}")
        print(f"Best player model type: {type(best_model)}")

        # Step 6: Save best model dynamically and generic
        self.save_best_model(best_model, best_model_name)

        # Step 7: Save features
        self.save_features()

        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "X_test": X_test,
            "y_test": y_test
        }


# Unified training pipeline runner for prediction API
class TrainingPipeline:
    def __init__(self):
        self.match_pipeline = MatchTrainingPipeline()
        self.player_pipeline = PlayerTrainingPipeline()

    def run_pipeline(self):
        print("\n=======================================================")
        print("RUNNING MATCH DATA MODEL TRAINING PIPELINE")
        print("=======================================================")
        match_results = self.match_pipeline.run_pipeline()

        print("\n=======================================================")
        print("RUNNING PLAYER DATA MODEL TRAINING PIPELINE")
        print("=======================================================")
        player_results = self.player_pipeline.run_pipeline()

        return {
            "match": match_results,
            "player": player_results
        }


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()