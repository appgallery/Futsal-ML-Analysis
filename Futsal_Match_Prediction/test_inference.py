from pipeline.inference import Inference, MatchInput

print("Loading inference engine...")
inference_engine = Inference()

print("\n--- Test 1: By Team Names ---")
try:
    # "FC Carlton Heart" and "Fitzroy FC" are usually in the dataset
    data_names = MatchInput(homeTeam="FC Carlton Heart", awayTeam="Fitzroy FC")
    result1 = inference_engine.test_infer(data_names)
    print(result1.model_dump_json(indent=2))
except Exception as e:
    print("Error:", e)

print("\n--- Test 2: By Team IDs ---")
try:
    # ID 22 is "FC Carlton Heart", ID 17 is "Fitzroy FC"
    data_ids = MatchInput(homeID=22, awayID=17)
    result2 = inference_engine.test_infer(data_ids)
    print(result2.model_dump_json(indent=2))
except Exception as e:
    print("Error:", e)
