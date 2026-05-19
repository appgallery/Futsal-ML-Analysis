import joblib
from typing import Optional, Union
from pydantic import BaseModel, model_validator
from components.data_transformation import DataTransformation
import pandas as pd
import numpy as np

class MatchInput(BaseModel):
    homeTeam: Optional[str] = None
    awayTeam: Optional[str] = None
    homeID: Optional[int] = None
    awayID: Optional[int] = None

    @model_validator(mode="after")
    def check_either_pair_exists(self) -> "MatchInput":
        has_names = self.homeTeam is not None and self.awayTeam is not None
        has_ids = self.homeID is not None and self.awayID is not None
        if not (has_names or has_ids):
            raise ValueError("Provide either both (homeTeam AND awayTeam) OR both (homeID AND awayID).")
        return self

class MatchOutputName(BaseModel):
    homeTeam: str
    awayTeam: str
    prediction: str
    confidence: float

class MatchOutputID(BaseModel):
    homeID: int
    awayID: int
    prediction: str
    confidence: float
    
    
class Inference:
    def __init__(self):
        self.model_path = "models/best_model.pkl"
        self.model = self.load_model(self.model_path)
        
        print("Initializing Inference State... (This may take a few seconds)")
        self.dt = DataTransformation()
        self.dt.transform(save_csv=False)
        self.team_memory = self.dt.team_memory
        self.h2h_memory = self.dt.h2h_memory
        
        # Create mappings
        raw_data = pd.read_csv("data/series-futsal-men-matches.csv")
        self.name_to_id = dict(zip(raw_data['homeTeamName'].str.strip(), raw_data['homeTeamId']))
        self.name_to_id.update(dict(zip(raw_data['awayTeamName'].str.strip(), raw_data['awayTeamId'])))
        self.id_to_name = {v: k for k, v in self.name_to_id.items()}
        print("Inference State Ready!")
    
    def load_model(self, model_path: str):
        import joblib
        return joblib.load(model_path)

    def test_infer(self, data: MatchInput) -> Union[MatchOutputName, MatchOutputID]:
        import pandas as pd
        import numpy as np
        
        # Resolve IDs
        home_id = data.homeID
        away_id = data.awayID
        
        if home_id is None or away_id is None:
            home_id = self.name_to_id.get(data.homeTeam.strip() if data.homeTeam else None)
            away_id = self.name_to_id.get(data.awayTeam.strip() if data.awayTeam else None)
            
            if home_id is None or away_id is None:
                missing = []
                if home_id is None: missing.append(f"Home team '{data.homeTeam}'")
                if away_id is None: missing.append(f"Away team '{data.awayTeam}'")
                raise ValueError(f"Could not find IDs for: {', '.join(missing)}")

        home_name = self.id_to_name.get(home_id, f"ID_{home_id}")
        away_name = self.id_to_name.get(away_id, f"ID_{away_id}")
        
        # Ensure memory exists
        self.dt.initialize_team_memory(home_id)
        self.dt.initialize_team_memory(away_id)
        
        hs = self.team_memory[home_id]
        aws = self.team_memory[away_id]
        
        # Compute Features
        home_win_rate = hs['wins'] / hs['matches'] if hs['matches'] > 0 else 0
        away_win_rate = aws['wins'] / aws['matches'] if aws['matches'] > 0 else 0
        
        home_recent = hs['results'][-5:]
        away_recent = aws['results'][-5:]
        
        home_wf = np.average(home_recent, weights=np.exp(np.linspace(0, 1, len(home_recent)))) if len(home_recent) > 0 else 0
        away_wf = np.average(away_recent, weights=np.exp(np.linspace(0, 1, len(away_recent)))) if len(away_recent) > 0 else 0
        diff_form = home_wf - away_wf
        
        home_attack = np.mean(hs['goals_scored'][-5:]) if len(hs['goals_scored']) > 0 else 0
        away_attack = np.mean(aws['goals_scored'][-5:]) if len(aws['goals_scored']) > 0 else 0
        diff_attack = home_attack - away_attack
        
        home_defense = np.mean(hs['goals_conceded'][-5:]) if len(hs['goals_conceded']) > 0 else 0
        away_defense = np.mean(aws['goals_conceded'][-5:]) if len(aws['goals_conceded']) > 0 else 0
        diff_defense = home_defense - away_defense
        
        home_goal_diff = np.mean(hs['goal_difference'][-5:]) if len(hs['goal_difference']) > 0 else 0
        away_goal_diff = np.mean(aws['goal_difference'][-5:]) if len(aws['goal_difference']) > 0 else 0
        diff_goal_diff = home_goal_diff - away_goal_diff
        
        diff_elo = hs['elo'] - aws['elo']
        
        pair = tuple(sorted([home_id, away_id]))
        self.dt.initialize_h2h_memory(pair)
        h2h = self.h2h_memory[pair]
        
        if h2h['matches'] > 0:
            if pair[0] == home_id:
                h2h_home_win_rate = h2h['team1_wins'] / h2h['matches']
            else:
                h2h_home_win_rate = h2h['team2_wins'] / h2h['matches']
        else:
            h2h_home_win_rate = 0.5
            
        feature_dict = {
            'home_win_rate': [home_win_rate],
            'away_win_rate': [away_win_rate],
            'diff_goal_diff': [diff_goal_diff],
            'diff_form': [diff_form],
            'diff_attack': [diff_attack],
            'diff_defense': [diff_defense],
            'diff_elo': [diff_elo],
            'h2h_home_win_rate': [h2h_home_win_rate]
        }
        
        # Note: Must match exactly ModelTrainer.feature order
        feature_cols = ['home_win_rate','away_win_rate','diff_goal_diff','diff_form','diff_attack','diff_defense','diff_elo','h2h_home_win_rate']
        
        X_infer = pd.DataFrame(feature_dict)[feature_cols]
        
        # Predict
        prediction_num = self.model.predict(X_infer)[0]
        
        # Some models support predict_proba, some don't (like SVC without probability=True)
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_infer)[0]
            confidence = float(np.max(probs))
            
        prediction_str = "Home Win" if prediction_num == 1 else "Away Win"

        if data.homeID is not None and data.awayID is not None:
            return MatchOutputID(
                homeID=home_id,
                awayID=away_id,
                prediction=prediction_str,
                confidence=confidence
            )
        else:
            return MatchOutputName(
                homeTeam=home_name,
                awayTeam=away_name,
                prediction=prediction_str,
                confidence=confidence
            )