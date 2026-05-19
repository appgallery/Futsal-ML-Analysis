from pipeline.inference import Inference, MatchInput

class InferencePipeline:
    def __init__(self):
        print("Loading inference engine...")
        self.inference_engine = Inference()

    def predict_by_team_names(self, home_team, away_team):
        try:
            data = MatchInput(
                homeTeam=home_team,
                awayTeam=away_team
            )

            result = self.inference_engine.test_infer(data)

            print("\n--- Prediction By Team Names ---")
            print(result.model_dump_json(indent=2))

            return result

        except Exception as e:
            print("Error:", e)
            return None

    def predict_by_team_ids(self, home_id, away_id):
        try:
            data = MatchInput(
                homeID=home_id,
                awayID=away_id
            )

            result = self.inference_engine.test_infer(data)

            print("\n--- Prediction By Team IDs ---")
            print(result.model_dump_json(indent=2))

            return result

        except Exception as e:
            print("Error:", e)
            return None

    def run_pipeline(self):
        # Test using team names
        self.predict_by_team_names(
            home_team="FC Carlton Heart",
            away_team="Fitzroy FC"
        )

        # Test using team IDs
        self.predict_by_team_ids(
            home_id=329,
            away_id=331
        )


if __name__ == "__main__":
    inference_pipeline = InferencePipeline()
    inference_pipeline.run_pipeline()