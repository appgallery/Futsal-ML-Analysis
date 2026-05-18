import pandas as pd
from pipeline.model_trainer import ModelTrainer

# Load the ML dataset
print("Loading feature data...")
feature_df = pd.read_csv('data/matches_ml.csv')

# Initialize the trainer
trainer = ModelTrainer()

# Prepare data
X, y, df = trainer.prepare_data(feature_df)

# Train the models
print("Training models...")
models_report = trainer.train(X, y)

print("Training Report:")
for report in models_report:
    print(f"Model: {report[0]}")
    print(f"Accuracy: {report[1]}")
    print(f"Best Params: {report[2]}")
    print("-" * 30)

# Get the best model
best_model, best_model_name, X_test, y_test = trainer.train_best_model(X, y)

print(f"Best model name: {best_model_name}")
print(f"Best model type: {type(best_model)}")

# Save the best model
trainer.save_model(best_model, 'models/best_model.pkl')
print("Best model saved to models/best_model.pkl")
