import pandas as pd
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
import components.data_transformation as dt


class TestDataIngestion(DataIngestion):
    def __init__(self):
        super().__init__()
        self.match_path = "data/series-futsal-men-matches.csv"


class DataPipeline:
    def __init__(self):
        # Monkey patch DataIngestion inside data_transformation module
        dt.DataIngestion = TestDataIngestion

        self.transformer = DataTransformation()
        self.ingestion = TestDataIngestion()

    def run_pipeline(self):
        try:
            # Step 1: Ingest Data
            matches = self.ingestion.ingest_data()
            print("Ingested data shape:", matches.shape)

            # Step 2: Drop unnecessary columns
            matches = self.transformer.drop_columns(matches)

            # Step 3: Remove invalid matches
            matches = self.transformer.remove_invalid_matches(matches)

            # Step 4: Remove non-completed matches
            matches = self.transformer.remove_non_completed_matches(matches)

            # Step 5: Clean date and sort
            matches = self.transformer.clean_date(matches)

            # Step 6: Create outcome column
            self.transformer.create_outcome(matches)

            # Step 7: Create goal-related columns
            self.transformer.create_home_away_goals(matches)

            # Step 8: Print sample output
            print(
                matches[
                    [
                        'winningTeam',
                        'homeTeamId',
                        'awayTeamId',
                        'outcome',
                        'home_goals',
                        'away_goals'
                    ]
                ].head(20)
            )

            # Step 9: Feature creation
            matches_after_create = self.transformer.create_features(matches)

            print(
                "Type of matches after create_features:",
                type(matches_after_create)
            )

            return matches_after_create

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    pipeline = DataPipeline()
    processed_data = pipeline.run_pipeline()