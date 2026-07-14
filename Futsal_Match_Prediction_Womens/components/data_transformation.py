import numpy as np
from components.data_ingestion import DataIngestion
import pandas as pd

class DataMatchTransformation:
    def __init__(self):
        self.drop_cols = ['quaterFinal','semiFinals','final','note','startTime','endTime','startingTime','courtId','courtName','referee_id','homeTeamKits','awayTeamKits','competitionName','last_match','round']
        self.INITIAL_ELO = 1500
        self.K_FACTOR = 32
        self.team_memory = {}
        self.h2h_memory = {}
        self.feature_cols = [
            'home_win_rate',
            'away_win_rate',
            'home_weighted_form',
            'away_weighted_form',
            'home_attack_strength',
            'away_attack_strength',
            'home_defense_strength',
            'away_defense_strength',
            'home_goal_diff_strength',
            'away_goal_diff_strength',
            'home_scoring_consistency',
            'away_scoring_consistency',
            'home_clean_sheet_rate',
            'away_clean_sheet_rate',
            'home_high_scoring_rate',
            'away_high_scoring_rate',
            'home_elo',
            'away_elo',
            'diff_form',
            'diff_attack',
            'diff_defense',
            'diff_goal_diff',
            'diff_elo',
            'h2h_matches',
            'h2h_home_win_rate',
        ]

        self.remove_cols = ['winningTeam','losingTeam','winningTeamGoals','losingTeamGoals','winningTeamPoints','losingTeamPoints','match_id','cmp_id','seasonName','status','is_bye','is_forfeited','is_cancel','isDraw']
    
    def drop_columns(self,matches):
        existing = [c for c in self.drop_cols if c in matches.columns]
        matches = matches.drop(columns=existing)
        return matches

    def remove_invalid_matches(self,matches):
        matches = matches[
            (matches['is_cancel'] != True) &
            (matches['is_forfeited'] != True)
        ]
        return matches

    def remove_non_completed_matches(self,matches):
        matches = matches[
            matches['status'] == 'Completed'
        ]
        return matches
    
    def clean_date(self,matches):
        matches['startDate'] = pd.to_datetime(
            matches['startDate'],
            errors='coerce'
        )

        matches = matches.sort_values(
            'startDate'
        ).reset_index(drop=True)
        return matches

    # def clean_draw(self,matches):
    #     matches['isDraw'] = matches['isDraw'].map({'yes': 1,'no': 0})
    #     return matches

    def create_home_away_goals(self,matches):
        matches['home_goals'] = np.where(
            matches['winningTeam'] == matches['homeTeamId'],
            matches['winningTeamGoals'],
            matches['losingTeamGoals']
        )
        matches['away_goals'] = np.where(
            matches['winningTeam'] == matches['awayTeamId'],
            matches['winningTeamGoals'],
            matches['losingTeamGoals']
        )
        return matches

    def create_outcome(self, matches):
        def _get_outcome(row):
            if str(row.get('isDraw', '')).lower() == 'yes':
                return 0.5
            elif row['winningTeam'] == row['homeTeamId']:
                return 1
            else:
                return 0

        matches['outcome'] = matches.apply(_get_outcome, axis=1)
        return matches
    
    def create_features(self, matches):
        for col in self.feature_cols:
            matches[col] = 0.0
        return matches

    def initialize_team_memory(self,team_id):
        if team_id not in self.team_memory:
            self.team_memory[team_id] = {
                'matches': 0,
                'wins': 0,
                'goals_scored': [],
                'goals_conceded': [],
                'goal_difference': [],
                'results':[],
                'clean_sheets': 0,
                'elo': self.INITIAL_ELO
            }
    def initialize_h2h_memory(self,pair):
        if pair not in self.h2h_memory:
            self.h2h_memory[pair] = {
                'matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
            }
    
    def feature_engineer(self,matches):
        for idx,row in matches.iterrows():
            home = row['homeTeamId']
            away = row['awayTeamId']

            home_goals = row['home_goals']
            away_goals = row['away_goals']

            outcome = row['outcome']

            self.initialize_team_memory(home)
            self.initialize_team_memory(away)

            hs = self.team_memory[home]
            aws = self.team_memory[away]
            
            # Win Rate
            home_wr = (
                hs['wins'] / hs['matches']
                if hs['matches'] > 0 else 0
            )

            away_wr = (
                aws['wins'] / aws['matches']
                if aws['matches'] > 0 else 0
            )

            matches.loc[idx, 'home_win_rate'] = home_wr
            matches.loc[idx, 'away_win_rate'] = away_wr

            # Weighted Form (last 5 matches)
            home_recent = hs['results'][-5:]
            away_recent = aws['results'][-5:]

            home_wf = (
                np.average(home_recent, weights=np.exp(np.linspace(0, 1, len(home_recent))))
                if len(home_recent) > 0 else 0
            )
            away_wf = (
                np.average(away_recent, weights=np.exp(np.linspace(0, 1, len(away_recent))))
                if len(away_recent) > 0 else 0
            )

            matches.loc[idx, 'home_weighted_form'] = home_wf
            matches.loc[idx, 'away_weighted_form'] = away_wf
            
            # Attack Strength (last 5 matches)
            home_attack = (
                np.mean(hs['goals_scored'][-5:])
                if len(hs['goals_scored']) > 0 else 0
            )

            away_attack = (
                np.mean(aws['goals_scored'][-5:])
                if len(aws['goals_scored']) > 0 else 0
            )

            matches.loc[idx, 'home_attack_strength'] = home_attack
            matches.loc[idx, 'away_attack_strength'] = away_attack

            # Defense Strength (last 5 matches)
            # LOWER IS BETTER
            home_conceded = (
                np.mean(hs['goals_conceded'][-5:])
                if len(hs['goals_conceded']) > 0 else 0
            )

            away_conceded = (
                np.mean(aws['goals_conceded'][-5:])
                if len(aws['goals_conceded']) > 0 else 0
            )

            home_defense = 1 + (1 + home_conceded)
            away_defense = 1 + (1 + away_conceded)
            
            matches.loc[idx, 'home_defense_strength'] = home_defense
            matches.loc[idx, 'away_defense_strength'] = away_defense

            # Goal Difference Strength (last 5 matches)
            home_goal_diff = (
                np.mean(hs['goal_difference'][-5:])
                if len(hs['goal_difference']) > 0 else 0
            )

            away_goal_diff = (
                np.mean(aws['goal_difference'][-5:])
                if len(aws['goal_difference']) > 0 else 0
            )

            matches.loc[idx, 'home_goal_diff_strength'] = home_goal_diff
            matches.loc[idx, 'away_goal_diff_strength'] = away_goal_diff

            # Scoring Consistency (last 5 matches)
            # HOW OFTEN THEY SCORE AT LEAST 1 GOAL

            home_consistency = (
                np.mean(hs['goals_scored'][-5:])
                if len(hs['goals_scored']) > 1 else 0
            )

            away_consistency = (
                np.mean(aws['goals_scored'][-5:])
                if len(aws['goals_scored']) > 1 else 0
            )

            matches.loc[idx, 'home_scoring_consistency'] = home_consistency
            matches.loc[idx, 'away_scoring_consistency'] = away_consistency
            
            # CLEAN SHEET RATE
            home_cs = (
                hs['clean_sheets'] / hs['matches']
                if hs['matches'] > 0 else 0
            )

            away_cs = (
                aws['clean_sheets'] / aws['matches']
                if aws['matches'] > 0 else 0
            )

            matches.loc[idx, 'home_clean_sheet_rate'] = home_cs
            matches.loc[idx, 'away_clean_sheet_rate'] = away_cs

            # HIGH SCORING RATE
            home_high_scoring = (
                np.mean(
                    np.array(hs['goals_scored'][-5:]) >= 4
                )
                if len(hs['goals_scored']) > 0 else 0
            )

            away_high_scoring = (
                np.mean(
                    np.array(aws['goals_scored'][-5:]) >= 4
                )
                if len(aws['goals_scored']) > 0 else 0
            )

            matches.loc[idx, 'home_high_scoring_rate'] = home_high_scoring
            matches.loc[idx, 'away_high_scoring_rate'] = away_high_scoring

            # ELO
            home_elo = hs['elo']
            away_elo = aws['elo']

            matches.loc[idx, 'home_elo'] = home_elo
            matches.loc[idx, 'away_elo'] = away_elo

            # DIFFERENTIAL FEATURES
            matches.loc[idx, 'diff_form'] = (
                home_wf - away_wf
            )

            matches.loc[idx, 'diff_attack'] = (
                home_attack - away_attack
            )

            matches.loc[idx, 'diff_defense'] = (
                home_defense - away_defense
            )

            matches.loc[idx, 'diff_goal_diff'] = (
                home_goal_diff - away_goal_diff
            )

            matches.loc[idx, 'diff_elo'] = (
                home_elo - away_elo
            )


            # HEAD TO HEAD
            pair = tuple(sorted([home, away]))

            self.initialize_h2h_memory(pair)
            h2h = self.h2h_memory[pair]

            matches.loc[idx, 'h2h_matches'] = h2h['matches']

            if h2h['matches'] > 0:

              if pair[0] == home:

                home_h2h_winrate = (
                    h2h['team1_wins'] / h2h['matches']
                )

              else:

                home_h2h_winrate = (
                    h2h['team2_wins'] / h2h['matches']
                )

            else:

              home_h2h_winrate = 0.5

            matches.loc[idx, 'h2h_home_win_rate'] = home_h2h_winrate

            # ELO CALCULATION
            expected_home = (
                1 / (1 + 10 ** ((away_elo - home_elo)/400))
            )

            if outcome == 1:

                actual_home = 1

            elif outcome == 0:

                actual_home = 0

            else:
                
                actual_home = 0.5


            new_home_elo = (
                home_elo +
                self.K_FACTOR * (actual_home - expected_home)
            )

            new_away_elo = (
                away_elo +
                self.K_FACTOR * ((1-actual_home) - (1-expected_home))
            )

            # UPDATE TEAM MEMORY
            hs['matches'] += 1
            aws['matches'] += 1

            if outcome == 1:

                hs['wins'] += 1

            elif outcome == 0:

                aws['wins'] += 1


            hs['goals_scored'].append(home_goals)
            hs['goals_conceded'].append(away_goals)

            aws['goals_scored'].append(away_goals)
            aws['goals_conceded'].append(home_goals)


            hs['goal_difference'].append(
                home_goals - away_goals
            )

            aws['goal_difference'].append(
                away_goals - home_goals
            )

            # RESULTS
            if outcome == 1:

                hs['results'].append(1)
                aws['results'].append(0)

            elif outcome == 0:

                hs['results'].append(0)
                aws['results'].append(1)

            else:

                hs['results'].append(0.5)
                aws['results'].append(0.5)


            # CLEAN SHEETS
            if away_goals == 0:
                hs['clean_sheets'] += 1

            if home_goals == 0:
                aws['clean_sheets'] += 1

            # UPDATE ELO
            hs['elo'] = new_home_elo
            aws['elo'] = new_away_elo
            
            # UPDATE H2H
            h2h['matches'] += 1

            if outcome == 1:
                if pair[0] == home:
                    h2h['team1_wins'] += 1
                else:
                    h2h['team2_wins'] += 1

            elif outcome == 0:
                if pair[0] == away:
                    h2h['team1_wins'] += 1
                else:
                    h2h['team2_wins'] += 1

        return matches
            
    
    def transform(self, matches, save_csv=True):
        # Drop columns
        matches = self.drop_columns(matches)

        # Remove Invalid Matches
        matches = self.remove_invalid_matches(matches)

        # Remove Non Completed Matches
        matches = self.remove_non_completed_matches(matches)

        # Convert Date & Sort
        matches = self.clean_date(matches)

        # Create Outcome
        matches = self.create_outcome(matches)

        # Create Home Away Goals
        matches = self.create_home_away_goals(matches)

        # Create Features
        matches = self.create_features(matches)

        # Feature Engineering
        matches = self.feature_engineer(matches)

        matches_ml = matches.copy()

        match_ml = matches_ml.drop(columns=[c for c in self.remove_cols if c in matches_ml.columns])
        
        if save_csv:
            match_ml.to_csv('data/matches_ml.csv', index=False)

        return match_ml


class DataPlayerTransformation:
    def transform(self, players, matches):
        import numpy as np
        
        # 1. Clean dates
        players['created_on'] = pd.to_datetime(players['created_on'], errors='coerce')
        matches['startDate'] = pd.to_datetime(matches['startDate'], errors='coerce')
        
        # 2. Extract Event Dataframes
        players['type'] = players['type'].astype(str).str.upper()
        
        goals_df = (
            players[players['type'] == 'SCORE']
            .groupby(['match_id', 'user_id'])
            .size()
            .reset_index(name='goals')
        )
        
        fouls_df = (
            players[players['type'] == 'FOULS']
            .groupby(['match_id', 'user_id'])
            .size()
            .reset_index(name='fouls')
        )
        
        yellow_df = (
            players[players['type'] == 'YELLOWCARD']
            .groupby(['match_id', 'user_id'])
            .size()
            .reset_index(name='yellow_cards')
        )
        
        red_df = (
            players[players['type'] == 'REDCARD']
            .groupby(['match_id', 'user_id'])
            .size()
            .reset_index(name='red_cards')
        )
        
        assist_df = (
            players[players['assist_player_id'].notna()]
            .groupby(['match_id', 'assist_player_id'])
            .size()
            .reset_index(name='assists')
        )
        assist_df.rename(columns={'assist_player_id': 'user_id'}, inplace=True)
        
        # 3. Merge Event Dataframes
        player_match = pd.concat([
            goals_df[['match_id', 'user_id']],
            fouls_df[['match_id', 'user_id']],
            yellow_df[['match_id', 'user_id']],
            red_df[['match_id', 'user_id']],
            assist_df[['match_id', 'user_id']]
        ]).drop_duplicates()
        
        for df in [goals_df, fouls_df, yellow_df, red_df, assist_df]:
            player_match = player_match.merge(df, on=['match_id', 'user_id'], how='left')
            
        player_match.fillna(0, inplace=True)
        
        # 4. Merge Player Info
        player_info = (
            players[['match_id', 'user_id', 'team_id', 'first_name', 'last_name', 'date_of_birth']]
            .drop_duplicates(['match_id', 'user_id'])
        )
        player_match = player_match.merge(player_info, on=['match_id', 'user_id'], how='left')
        
        # 5. Merge Match Info
        match_cols = [
            'id', 'startDate', 'homeTeamId', 'awayTeamId', 'homeTeamName',
            'awayTeamName', 'competitionName', 'seasonName'
        ]
        match_cols_avail = [c for c in match_cols if c in matches.columns]
        
        player_match = player_match.merge(
            matches[match_cols_avail],
            left_on='match_id', right_on='id', how='left'
        )
        if 'id' in player_match.columns:
            player_match.drop(columns=['id'], inplace=True)
            
        player_match.rename(columns={'startDate': 'match_date'}, inplace=True)
        
        # 6. Basic Features
        player_match['player_name'] = player_match['first_name'].fillna('') + ' ' + player_match['last_name'].fillna('')
        player_match['match_date'] = pd.to_datetime(player_match['match_date'])
        player_match['date_of_birth'] = pd.to_datetime(player_match['date_of_birth'], errors='coerce')
        player_match['age'] = ((player_match['match_date'] - player_match['date_of_birth']).dt.days) / 365.25
        
        player_match = player_match.sort_values(['user_id', 'match_date'])
        
        player_match['goal_contribution'] = player_match['goals'] + player_match['assists']
        player_match['clutch_goals'] = (player_match['goals'] >= 2).astype(int)
        
        # 7. Perf Score
        player_match['perf_score'] = (
            player_match['goals'] * 3
            + player_match['assists'] * 2
            + player_match['clutch_goals'] * 1.5
            - player_match['fouls'] * 0.5
            - player_match['yellow_cards']
            - player_match['red_cards'] * 3
        )
        
        # 8. Career Stats
        player_match['career_matches'] = player_match.groupby('user_id').cumcount()
        player_match['career_goals'] = player_match.groupby('user_id')['goals'].cumsum() - player_match['goals']
        player_match['career_assists'] = player_match.groupby('user_id')['assists'].cumsum() - player_match['assists']
        player_match['career_perf_score'] = player_match.groupby('user_id')['perf_score'].cumsum() - player_match['perf_score']
        
        player_match['career_gpg'] = player_match['career_goals'] / player_match['career_matches'].replace(0, np.nan)
        player_match['career_apg'] = player_match['career_assists'] / player_match['career_matches'].replace(0, np.nan)
        player_match['career_avg_perf'] = player_match['career_perf_score'] / player_match['career_matches'].replace(0, np.nan)
        
        # 9. Rolling Metrics
        for w in [3, 5, 10]:
            player_match[f'roll{w}_goals'] = player_match.groupby('user_id')['goals'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            player_match[f'roll{w}_assists'] = player_match.groupby('user_id')['assists'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            player_match[f'roll{w}_perf_score'] = player_match.groupby('user_id')['perf_score'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                
        player_match['ewm_goals'] = player_match.groupby('user_id')['goals'].transform(
            lambda x: x.shift(1).ewm(span=5).mean())
        player_match['ewm_assists'] = player_match.groupby('user_id')['assists'].transform(
            lambda x: x.shift(1).ewm(span=5).mean())
        player_match['ewm_perf_score'] = player_match.groupby('user_id')['perf_score'].transform(
            lambda x: x.shift(1).ewm(span=5).mean())
            
        player_match['goal_trend'] = player_match['roll5_goals'] - player_match['career_gpg']
        player_match['assist_trend'] = player_match['roll5_assists'] - player_match['career_apg']
        player_match['perf_trend'] = player_match['roll5_perf_score'] - player_match['career_avg_perf']
        
        # 10. Temporal Features
        player_match['prev_match_date'] = player_match.groupby('user_id')['match_date'].shift(1)
        player_match['days_since_last_match'] = (player_match['match_date'] - player_match['prev_match_date']).dt.days
        player_match['days_since_last_match'] = player_match['days_since_last_match'].fillna(-1)
        
        # 11. Experience
        player_match['experience_score'] = np.log1p(player_match['career_matches']) * np.log1p(player_match['career_goals'] + 1)
        player_match['age'] = player_match['age'].fillna(player_match['age'].median())
        
        player_match['log_career_goals'] = np.log1p(player_match['career_goals'])
        player_match['log_career_assists'] = np.log1p(player_match['career_assists'])
        player_match['log_career_matches'] = np.log1p(player_match['career_matches'])
        
        num_cols = player_match.select_dtypes(include=np.number).columns
        player_match[num_cols] = player_match[num_cols].fillna(0)
        
        # 12. Best Player Target
        player_match['best_match_score'] = player_match.groupby('match_id')['perf_score'].transform('max')
        player_match['is_best_player'] = (player_match['perf_score'] == player_match['best_match_score']).astype(int)
        
        # 13. Filter out rows with team_id == 0
        player_match = player_match[player_match['team_id'] != 0].copy()
        
        # 14. Save CSV
        player_match.to_csv('data/players_ml.csv', index=False)
        
        return player_match


# Alias for backwards compatibility
DataTransformation = DataMatchTransformation

class DataPlayerTransformation:
    def __init__(self):
        pass

    def transform(self, players, matches):
        # 1. Clean types
        players['type'] = players['type'].astype(str).str.upper()

        # 2. Pivot events
        player_match = pd.pivot_table(
            players,
            index=['match_id', 'user_id'],
            columns='type',
            aggfunc='size',
            fill_value=0
        ).reset_index()

        # 3. Add player info
        player_info = players[['match_id', 'user_id', 'team_id', 'first_name', 'last_name', 'date_of_birth']].drop_duplicates(subset=['match_id', 'user_id'])
        player_match = player_match.merge(player_info, on=['match_id', 'user_id'], how='left')

        # 4. Add match info
        match_info = matches[['id', 'startDate', 'homeTeamId', 'awayTeamId', 'winningTeam', 'losingTeam', 'winningTeamGoals', 'losingTeamGoals', 'homeTeamName', 'awayTeamName']].copy()
        match_info = match_info.rename(columns={'id': 'match_id'})
        player_match = player_match.merge(match_info, on='match_id', how='left')

        # 5. Authoritative performance score
        player_match['perf_score'] = (
            player_match.get('SCORE', 0) * 3
            - player_match.get('FOULS', 0) * 0.5
            - player_match.get('YELLOWCARD', 0) * 1
            - player_match.get('REDCARD', 0) * 3
        )

        # 6. Rank in match & best player logic (entire match)
        player_match['best_match_score'] = player_match.groupby('match_id')['perf_score'].transform('max')
        player_match['is_best_player'] = (player_match['perf_score'] == player_match['best_match_score']).astype(int)

        player_match = player_match.sort_values(
            ['match_id', 'perf_score', 'SCORE'],
            ascending=[True, False, False]
        )
        player_match['rank_in_match'] = player_match.groupby('match_id').cumcount() + 1
        player_match['is_best_player'] = (player_match['rank_in_match'] == 1).astype(int)

        # 7. Identify winning team
        player_match['is_winning_team'] = (player_match['team_id'] == player_match['winningTeam']).astype(int)

        # 8. Rank inside team
        player_match = player_match.sort_values(
            ['match_id', 'team_id', 'perf_score', 'SCORE', 'FOULS'],
            ascending=[True, True, False, False, True]
        )
        player_match['rank_in_team'] = player_match.groupby(['match_id', 'team_id']).cumcount() + 1
        
        # is_best_player: rank 1 in their team AND their team won
        player_match['is_best_player'] = ((player_match['rank_in_team'] == 1) & (player_match['is_winning_team'] == 1)).astype(int)

        # 9. Format dates and sort chronologically
        player_match['startDate'] = pd.to_datetime(player_match['startDate'])
        player_match = player_match.sort_values(['user_id', 'startDate']).reset_index(drop=True)

        # 10. Temporal / Career Features
        for col in ['SCORE', 'FOULS', 'YELLOWCARD', 'REDCARD']:
            if col not in player_match.columns:
                player_match[col] = 0

        player_match['career_matches'] = player_match.groupby('user_id').cumcount()
        player_match['career_goals'] = player_match.groupby('user_id')['SCORE'].cumsum() - player_match['SCORE']
        player_match['career_avg_goals'] = (player_match['career_goals'] / player_match['career_matches']).fillna(0)

        player_match['career_fouls'] = player_match.groupby('user_id')['FOULS'].cumsum() - player_match['FOULS']
        player_match['career_yellow'] = player_match.groupby('user_id')['YELLOWCARD'].cumsum() - player_match['YELLOWCARD']
        player_match['career_red'] = player_match.groupby('user_id')['REDCARD'].cumsum() - player_match['REDCARD']

        player_match['career_perf'] = player_match.groupby('user_id')['perf_score'].cumsum() - player_match['perf_score']
        player_match['career_avg_perf'] = (player_match['career_perf'] / player_match['career_matches']).fillna(0)

        # Rolling goals
        player_match['roll3_goals'] = player_match.groupby('user_id')['SCORE'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        player_match['roll5_goals'] = player_match.groupby('user_id')['SCORE'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
        player_match['roll10_goals'] = player_match.groupby('user_id')['SCORE'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
        player_match['goal_trend'] = player_match['roll3_goals'] - player_match['roll10_goals']

        # Rolling perf
        player_match['roll3_perf'] = player_match.groupby('user_id')['perf_score'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        player_match['roll5_perf'] = player_match.groupby('user_id')['perf_score'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
        player_match['roll10_perf'] = player_match.groupby('user_id')['perf_score'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
        player_match['perf_trend'] = player_match['roll3_perf'] - player_match['roll10_perf']

        player_match['log_experience'] = np.log1p(player_match['career_matches'])

        # 11. Age Features
        player_match['date_of_birth'] = pd.to_datetime(player_match['date_of_birth'], errors='coerce')
        player_match['age'] = ((player_match['startDate'] - player_match['date_of_birth']).dt.days / 365.25)
        player_match['age'] = player_match['age'].fillna(player_match['age'].median())

        player_match.loc[player_match['age'] < 10, 'age'] = np.nan

        # Rolling Best Player Rates
        player_match = player_match.sort_values(['user_id', 'startDate'])
        player_match['career_best_player_rate'] = player_match.groupby('user_id')['is_best_player'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        
        player_match['prev_best_player'] = player_match.groupby('user_id')['is_best_player'].shift(1).fillna(0)
        player_match['roll5_best_player_rate'] = player_match.groupby('user_id')['prev_best_player'].transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)
        player_match['roll10_best_player_rate'] = player_match.groupby('user_id')['prev_best_player'].transform(lambda x: x.rolling(10, min_periods=1).mean()).fillna(0)

        # 12. Final Age Filtering and Winning Team filter
        df = player_match[(player_match['is_winning_team'] == 1) & (player_match['age'] >= 15)].copy()
        df.loc[(df['age'] < 15) | (df['age'] > 50), 'age'] = np.nan
        df['age'] = df['age'].fillna(df['age'].median())

        return df

