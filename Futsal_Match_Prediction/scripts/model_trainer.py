import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    f1_score,
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")


class MatchModelTrainer:
    def __init__(self):
        self.feature = [
            'home_win_rate',
            'away_win_rate',
            'diff_goal_diff',
            'diff_form',
            'diff_attack',
            'diff_defense',
            'diff_elo',
            'h2h_home_win_rate'
        ]
        self.params = {
            'classifier': ["RandomForestClassifier", "SVC", "LogisticRegression"],
            "RandomForestClassifier": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "SVC": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10]
            },
            "LogisticRegression": {
                "C": [0.1, 1, 10]
            }
        }
        self.target = 'outcome'
        self.grid_models = []

    def prepare_data(self, feature_df):
        X = feature_df[self.feature].copy()
        y = feature_df[self.target].copy()
        return X, y, feature_df

    def split_train_test(self, X, y, split_ratio=0.2):
        split_index = int(len(X) * (1 - split_ratio))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        return X_train, X_test, y_train, y_test

    def build_model(self, algo):
        pipeline_list = [
            ('scaler', StandardScaler()),
            ('classifier', algo())
        ]
        return Pipeline(pipeline_list)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        self.grid_models = []
        models_report = []

        for classifier_name in self.params['classifier']:
            print(f"Grid searching: {classifier_name}...")
            if classifier_name == "RandomForestClassifier":
                algo = RandomForestClassifier
                params = self.params["RandomForestClassifier"]
                grid_params = {f"classifier__{k}": v for k, v in params.items()}
                pipeline = self.build_model(algo)
                self.grid_models.append((
                    "RandomForestClassifier",
                    GridSearchCV(
                        pipeline,
                        grid_params,
                        cv=3,
                        scoring="accuracy",
                        n_jobs=1
                    )
                ))
            elif classifier_name == "SVC":
                algo = SVC
                params = self.params["SVC"]
                grid_params = {f"classifier__{k}": v for k, v in params.items()}
                pipeline = self.build_model(algo)
                self.grid_models.append((
                    "SVC",
                    GridSearchCV(
                        pipeline,
                        grid_params,
                        cv=3,
                        scoring="accuracy",
                        n_jobs=1
                    )
                ))
            elif classifier_name == "LogisticRegression":
                algo = LogisticRegression
                params = self.params["LogisticRegression"]
                grid_params = {f"classifier__{k}": v for k, v in params.items()}
                pipeline = self.build_model(algo)
                self.grid_models.append((
                    "LogisticRegression",
                    GridSearchCV(
                        pipeline,
                        grid_params,
                        cv=3,
                        scoring="accuracy",
                        n_jobs=1
                    )
                ))

        for name, grid in self.grid_models:
            print(f"Training {name}...")
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            models_report.append((
                name,
                round(acc, 4),
                round(f1, 4),
                grid.best_params_
            ))

        return models_report

    def train_best_model(self, X, y):
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        best_model = None
        best_accuracy = -float('inf')
        best_model_name = ""

        for name, grid in self.grid_models:
            y_pred = grid.best_estimator_.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = grid.best_estimator_
                best_model_name = name

        return best_model, best_model_name, X_test, y_test

    def save_model(self, model, path):
        joblib.dump(model, path)

    def load_model(self, path):
        return joblib.load(path)


class PlayerModelTrainer:
    def __init__(self):
        self.feature = []  # To be dynamically populated in prepare_data
        self.features = self.feature  # Alias for compatibility
        self.target = 'is_best_player'
        self.grid_models = []

        # Classifier param grids
        self.params = {
            "XGBoost": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            "LightGBM": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            "CatBoost": {
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [200, 500]
            }
        }

    def prepare_data(self, feature_df):
        if 'match_date' in feature_df.columns:
            feature_df['match_date'] = pd.to_datetime(
                feature_df['match_date'],
                errors='coerce'
            )
            feature_df = feature_df.sort_values('match_date')

        DROP_COLS = [
            'match_id', 'user_id',
            'player_name', 'first_name', 'last_name',
            'date_of_birth', 'match_date', 'prev_match_date',
            'goals', 'assists', 'fouls', 'yellow_cards', 'red_cards',
            'goal_contribution', 'clutch_goals', 'perf_score',
            'homeTeamName', 'awayTeamName', 'competitionName', 'seasonName',
            'best_match_score', 'is_best_player'
        ]

        X = feature_df.drop(columns=DROP_COLS, errors='ignore').copy()
        
        self.feature = X.columns.tolist()
        self.features = self.feature
        
        y = feature_df[self.target].copy()
        return X, y, feature_df

    def split_train_test(self, X, y, split_ratio=0.2):
        split_index = int(len(X) * (1 - split_ratio))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        self.grid_models = []
        models_report = []

        classifiers = {
            "XGBoost": XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=7.5, random_state=42),
            "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
            "CatBoost": CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=42, allow_writing_files=False)
        }

        for name, model in classifiers.items():
            print("\n" + "="*80)
            print(f"TRAINING {name}")
            print("="*80)

            n_jobs = 1

            grid = GridSearchCV(
                estimator=model,
                param_grid=self.params[name],
                scoring='f1',
                cv=3,
                verbose=1,
                n_jobs=n_jobs
            )

            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            print(f"BEST PARAMETERS FOR {name}: {grid.best_params_}")

            preds = best_model.predict(X_test)
            try:
                probs = best_model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, probs)
            except:
                roc_auc = 0.0

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            print(f"RESULTS FOR {name}: Accuracy: {round(acc, 4)}, F1: {round(f1, 4)}, ROC_AUC: {round(roc_auc, 4)}")

            self.grid_models.append((name, grid))
            models_report.append((
                name,
                round(f1, 4),
                f"Accuracy: {round(acc, 4)} | ROC_AUC: {round(roc_auc, 4)}",
                grid.best_params_
            ))

        return models_report

    def train_best_model(self, X, y):
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        best_model = None
        best_f1 = -float('inf')
        best_model_name = ""

        for name, grid in self.grid_models:
            preds = grid.best_estimator_.predict(X_test)
            f1 = f1_score(y_test, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_model = grid.best_estimator_
                best_model_name = name

        return best_model, best_model_name, X_test, y_test

    def predict_player_match_performance(self, model, player_df, feature_cols):
        probs = model.predict_proba(player_df[feature_cols])[:, 1]
        result = player_df.copy()
        result['best_player_probability'] = probs
        result['best_player_probability'] = (result['best_player_probability'] * 100).round(2)
        result = result.sort_values('best_player_probability', ascending=False)
        return result

    def save_model(self, model, path):
        joblib.dump(model, path)

    def load_model(self, path):
        return joblib.load(path)


class ModelTrainer:
    def __init__(self, mode='match'):
        self.mode = mode
        if mode == 'match':
            self.trainer = MatchModelTrainer()
        elif mode == 'player':
            self.trainer = PlayerModelTrainer()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @property
    def feature(self):
        return self.trainer.feature

    def prepare_data(self, feature_df):
        return self.trainer.prepare_data(feature_df)

    def train(self, X, y):
        return self.trainer.train(X, y)

    def train_best_model(self, X, y):
        return self.trainer.train_best_model(X, y)

    def save_model(self, model, path):
        self.trainer.save_model(model, path)

    def load_model(self, path):
        return self.trainer.load_model(path)