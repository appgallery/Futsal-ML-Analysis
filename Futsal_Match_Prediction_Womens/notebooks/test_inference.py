import json
import pandas as pd

from futsal_best_player_pipeline import predict_best_player

# Using some typical IDs
test_json = {
    "homeID": 28,
    "awayID": 2600,
    "homeWinProbability": 41.68,
    "awayWinProbability": 58.32
}

try:
    res = predict_best_player(test_json)
    print(json.dumps(res, indent=2))
except Exception as e:
    print(e)
