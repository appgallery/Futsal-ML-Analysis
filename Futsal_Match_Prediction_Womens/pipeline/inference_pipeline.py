from scripts.inference import Inference, MatchInput

class InferencePipeline:
    def __init__(self):
        print("Loading inference engine...")
        self.inference_engine = Inference()

    def predict_by_team_ids(self, home_id, away_id):
        try:
            data = MatchInput(
                homeID=home_id,
                awayID=away_id
            )

            match_output = self.inference_engine.test_match_infer(data)
            result = self.inference_engine.test_player_infer(match_output)
            print("\n--- Prediction By Team IDs ---")
            print(result.model_dump_json(indent=2))

            return result

        except Exception as e:
            print("Error:", e)
            return None

    def run_pipeline(self):
        # Test using team IDs
        self.predict_by_team_ids(
            home_id=28,
            away_id=2600
        )


if __name__ == "__main__":
    inference_pipeline = InferencePipeline()
    inference_pipeline.run_pipeline()