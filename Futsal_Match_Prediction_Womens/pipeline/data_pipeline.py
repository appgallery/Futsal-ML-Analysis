from components.data_ingestion import DataIngestion
from components.data_transformation import DataMatchTransformation, DataPlayerTransformation

class DataPipeline:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.match_transformer = DataMatchTransformation()
        self.player_transformer = DataPlayerTransformation()

    def match_run_pipeline(self):
        try:
            # Step 1: Ingest Data
            matches = self.ingestion.ingest_match_data()

            # Step 2: Transform Data
            matches = self.match_transformer.transform(matches)

            return matches
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def player_run_pipeline(self):
        try:
            # Step 1: Ingest Data
            matches = self.ingestion.ingest_match_data()
            players = self.ingestion.ingest_player_data()

            # Step 2: Transform Data
            players = self.player_transformer.transform(players, matches)

            return players
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def run_pipeline(self):
        print("--- Running Match Data Pipeline ---")
        matches = self.match_run_pipeline()
        if matches is not None:
            import os
            os.makedirs('data', exist_ok=True)
            matches.to_csv('data/matches_ml.csv', index=False)
            print(f"Saved {len(matches)} matches to data/matches_ml.csv")
        
        print("\n--- Running Player Data Pipeline ---")
        players = self.player_run_pipeline()
        if players is not None:
            players.to_csv('data/players_ml.csv', index=False)
            print(f"Saved {len(players)} player records to data/players_ml.csv")
        
        return matches, players


if __name__ == "__main__":
    pipeline = DataPipeline()
    processed_data = pipeline.run_pipeline()