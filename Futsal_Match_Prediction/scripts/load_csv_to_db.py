import pandas as pd
from sqlalchemy import create_engine
import os

# Use env var if available, else fallback to localhost
db_uri = os.environ.get("DATABASE_URI", "mysql+pymysql://root:root@localhost:3307/airflow_db")
print(f"Connecting to {db_uri}...")
engine = create_engine(db_uri)

match_path = "data/series-futsal-men-matches.csv"
player_path = "data/players-futsal-men-scores.csv"

print("Loading matches data...")
matches = pd.read_csv(match_path)
matches.to_sql('futsal_men_matches', con=engine, if_exists='replace', index=False)
print(f"Loaded {len(matches)} matches into 'futsal_men_matches' table.")

print("Loading players data...")
players = pd.read_csv(player_path)
players.to_sql('futsal_men_scores', con=engine, if_exists='replace', index=False)
print(f"Loaded {len(players)} player records into 'futsal_men_scores' table.")

print("Database seeding complete!")
