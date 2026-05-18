import pandas as pd
import numpy as np

class DataIngestion:

    def __init__(self):
        self.match_path = "data/series-futsal-men-matches.csv"

    def read_data(self):
        return pd.read_csv(self.match_path)

    def ingest_data(self):
        matches_raw = self.read_data()
        matches = matches_raw.copy()
        return matches

    
    
    