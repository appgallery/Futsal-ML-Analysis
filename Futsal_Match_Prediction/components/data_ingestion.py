import pandas as pd
from db.db_connection import DB


class DataIngestion:
    def __init__(self):
        self.db = DB()

        self.match_query = """
        SELECT
            m.*,
            ht.name AS homeTeamName,
            at.name AS awayTeamName,
            c.name AS competitionName,
            s.name AS seasonName
        FROM futsaloz.cmp_matches m
        LEFT JOIN futsaloz.teams ht
            ON m.homeTeamId = ht.id
        LEFT JOIN futsaloz.teams at
            ON m.awayTeamId = at.id
        LEFT JOIN futsaloz.competition c
            ON m.cmp_id = c.id
        LEFT JOIN futsaloz.season s
            ON c.season_id = s.id
        WHERE m.cmp_id IN (
            1028, 948, 878, 725, 627,
            527, 430, 312, 238, 130
        );
        """

        self.player_query = """
        SELECT
            lmsc.*,
            u.first_name,
            u.last_name,
            u.date_of_birth
        FROM futsaloz.live_match_score_cards lmsc
        LEFT JOIN futsaloz.users u
            ON lmsc.user_id = u.user_id
        WHERE lmsc.comp_id IN (
            1028, 948, 878, 725, 627,
            527, 430, 312, 238, 130
        )
        AND lmsc.type IN (
            'SCORE',
            'FOULS',
            'YELLOWCARD',
            'REDCARD'
        )
        AND lmsc.user_id <> 3;
        """

        try:
            if not self.db.db_test_connection():
                raise ConnectionError("SSH Tunnel connection failed")
            
            self.engine = self.db.db_create_engine()
            if self.engine is None:
                raise ConnectionError("Failed to create database engine")
                
            print("Database connection established.")
        except Exception as e:
            self.engine = None
            print(f"Database connection failed: {e}")

    def read_match_data(self):
        if self.engine is None:
            raise ConnectionError("Database engine not available.")

        from sqlalchemy import text
        with self.engine.connect() as conn:
            result = conn.execute(text(self.match_query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def read_player_data(self):
        if self.engine is None:
            raise ConnectionError("Database engine not available.")

        from sqlalchemy import text
        with self.engine.connect() as conn:
            result = conn.execute(text(self.player_query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def ingest_match_data(self):
        matches_raw = self.read_match_data()
        matches = matches_raw.copy()
        return matches

    def ingest_player_data(self):
        player_raw = self.read_player_data()
        player = player_raw.copy()
        return player

    def close_connection(self):
        self.db.db_close_connection()