from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class ModelTrainer:
    def __init__(self):
        self.feature = ['home_win_rate','away_win_rate','diff_goal_diff','diff_form','diff_attack','diff_defense','diff_elo','h2h_home_win_rate']
        self.params = {
            'classifier':["RandomForestClassifier", "SVC", "LogisticRegression"],
            "RandomForestClassifier":{
                "n_estimators": [100,200,300],
                "max_depth": [10,20,30],
                "min_samples_split": [2,5,10],
                "min_samples_leaf": [1,2,4]
            },
            "SVC":{
                "kernel": ["linear", "rbf"],
                "C": [0.1,1,10]
            },
            "LogisticRegression":{
                "C": [0.1,1,10]
            }
        }
        self.target = 'outcome'

    def prepare_data(self,feature_df):
        X = feature_df[self.feature].copy()
        y = feature_df[self.target].copy()
        return X,y,feature_df

    def split_train_test(self,X, y, split_ratio=0.2):
        split_index = int(len(X) * (1 - split_ratio))

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        return X_train, X_test, y_train, y_test

    def build_model(self,algo):
        pipeline_list = [
            ('scaler', StandardScaler()),
            ('classifier', algo())
        ]

        return Pipeline(pipeline_list)

    def train(self,X,y):
        X_train,X_test,y_train,y_test = self.split_train_test(X,y)

        self.grid_models = []
        models_report = []

        for classifier_name in self.params['classifier']:
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
                        cv=5,
                        scoring="accuracy",
                        n_jobs=-1
                    )
                ))
            if classifier_name == "SVC":
                algo = SVC
                params = self.params["SVC"]
                grid_params = {f"classifier__{k}": v for k, v in params.items()}
                pipeline = self.build_model(algo)
                self.grid_models.append((
                    "SVC",
                    GridSearchCV(
                        pipeline,
                        grid_params,
                        cv=5,
                        scoring="accuracy",
                        n_jobs=-1
                    )
                ))
            if classifier_name == "LogisticRegression":
                algo = LogisticRegression
                params = self.params["LogisticRegression"]
                grid_params = {f"classifier__{k}": v for k, v in params.items()}
                pipeline = self.build_model(algo)
                self.grid_models.append((
                    "LogisticRegression",
                    GridSearchCV(
                        pipeline,
                        grid_params,
                        cv=5,
                        scoring="accuracy",
                        n_jobs=-1
                    )
                ))
        for name,grid in self.grid_models:
            print(name)
            grid.fit(X_train,y_train)
            y_pred = grid.predict(X_test)
            models_report.append((
                name,
                accuracy_score(y_test,y_pred),
                grid.best_params_,
                classification_report(y_test,y_pred)
            ))
        return models_report
    
    def train_best_model(self,X,y):
        X_train,X_test,y_train,y_test = self.split_train_test(X,y)

        best_model = None
        best_accuracy = 0
        best_model_name = ""

        for name,grid in self.grid_models:
            if grid.best_score_ > best_accuracy:
                best_accuracy = grid.best_score_
                best_model = grid.best_estimator_
                best_model_name = name
        return best_model, best_model_name, X_test, y_test

    def save_model(self,model,path):
        joblib.dump(model,path)
    
    def load_model(self,path):
        return joblib.load(path)
    
            
        

    
    