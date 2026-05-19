import pandas as pd
from pipeline.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self, dataset_path='data/matches_ml.csv'):
        self.dataset_path = dataset_path
        self.trainer = ModelTrainer()

    def load_data(self):
        print("Loading feature data...")
        return pd.read_csv(self.dataset_path)

    def prepare_training_data(self, feature_df):
        return self.trainer.prepare_data(feature_df)

    def train_models(self, X, y):
        print("Training models...")
        return self.trainer.train(X, y)

    def print_training_report(self, models_report):
        print("Training Report:")

        for report in models_report:
            print(f"Model: {report[0]}")
            print(f"Accuracy: {report[1]}")
            print(f"F1-Score: {report[2]}")
            print(f"Best Params: {report[3]}")
            print("-" * 30)

    def train_best_model(self, X, y):
        return self.trainer.train_best_model(X, y)

    def save_best_model(self, model, path='models/best_model.pkl'):
        self.trainer.save_model(model, path)
        print(f"Best model saved to {path}")

    def run_pipeline(self):
        # Step 1: Load dataset
        feature_df = self.load_data()

        # Step 2: Prepare dataset
        X, y, df = self.prepare_training_data(feature_df)

        # Step 3: Train all models
        models_report = self.train_models(X, y)

        # Step 4: Print report
        self.print_training_report(models_report)

        # Step 5: Train best model
        best_model, best_model_name, X_test, y_test = (
            self.train_best_model(X, y)
        )

        print(f"Best model name: {best_model_name}")
        print(f"Best model type: {type(best_model)}")

        # Step 6: Save best model
        self.save_best_model(best_model)

        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "X_test": X_test,
            "y_test": y_test
        }


if __name__ == "__main__":
    trainer_pipeline = TrainingPipeline()
    trainer_pipeline.run_pipeline()