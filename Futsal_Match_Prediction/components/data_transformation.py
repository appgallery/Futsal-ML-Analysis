import numpy as np
from components.data_ingestion import DataIngestion
import pandas as pd

class DataMatchTransformation:
    def __init__(self):
        self.drop_cols = ['quaterFinal','semiFinals','final','note','startTime','endTime','startingTime','courtId','courtName','referee_id','homeTeamKits','awayTeamKits','competitionName','last_match','round','isDraw']
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

        self.remove_cols = ['winningTeam','losingTeam','winningTeamGoals','losingTeamGoals','winningTeamPoints','losingTeamPoints','match_id','cmp_id','seasonName','status','is_bye','is_forfeited','is_cancel']        
    
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
        matches['outcome'] = matches.apply(
            lambda row: 1 if row['winningTeam'] == row['homeTeamId'] else 0,
            axis=1
        )
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

    def remove_cols(self, player):
        player = player.dropna(
            subset=['user_id', 'match_id']
        )
        player = player.drop_duplicates()
        return player

    
    def standardize_player(self, player):
        player['type'] = (
            player['type']
            .astype(str)
            .str.upper()
            .str.strip()
        )
        player['created_on'] = pd.to_datetime(
            player['created_on'],
            errors='coerce'
        )
        player['player_name'] = (
            player['first_name'].fillna('') +
            ' ' +
            player['last_name'].fillna('')
        ).str.strip()
        return player

    def convert_match_time(self, x):
        try:
            mins, secs = str(x).split(':')
            return int(mins) + (int(secs) / 60)
        except:
            return np.nan

    def create_player_features(self, player):
        player['match_time_mins'] = (
            player['match_time']
            .apply(self.convert_match_time)
        )
        
        player['goal'] = np.where(player['type'] == 'SCORE', 1, 0)
        player['foul'] = np.where(player['type'] == 'FOULS', 1, 0)
        player['yellow_card'] = np.where(player['type'] == 'YELLOWCARD', 1, 0)
        player['red_card'] = np.where(player['type'] == 'REDCARD', 1, 0)
        player['penalty_event'] = np.where(player['type'] == 'PENALTY', 1, 0)
        player['assist'] = np.where(player['assist_player_id'].notnull(), 1, 0)
        player['own_goal'] = np.where(player['is_self_goal'] == 1, 1, 0)
        
        player['clutch_goal'] = np.where(
            (player['goal'] == 1) & (player['match_time_mins'] >= 15),
            1,
            0
        )
        return player

    def aggregate_player_match(self, player):
        player = (
            player.groupby(['match_id', 'team_id', 'user_id', 'player_name'])
            .agg(
                goals=('goal', 'sum'),
                assists=('assist', 'sum'),
                fouls=('foul', 'sum'),
                yellow_cards=('yellow_card', 'sum'),
                red_cards=('red_card', 'sum'),
                penalties=('penalty_event', 'sum'),
                own_goals=('own_goal', 'sum'),
                clutch_goals=('clutch_goal', 'sum'),
                total_player=('type', 'count'),
                first_event_time=('created_on', 'min')
            )
            .reset_index()
        )
        return player

    def merge_match_info(self, players, matches):
        match_cols = [
            'match_id',
            'startDate',
            'homeTeamId',
            'awayTeamId',
            'homeTeamName',
            'awayTeamName',
            'winningTeam',
            'winningTeamGoals',
            'losingTeamGoals'
        ]
        match_cols = [c for c in matches.columns if c in match_cols]
        matches_clean = matches[match_cols].copy()
        matches_clean['startDate'] = pd.to_datetime(
            matches_clean['startDate'],
            errors='coerce'
        )
        matches_clean = matches_clean.drop_duplicates(subset=['match_id'])
        
        players = players.merge(
            matches_clean,
            on='match_id',
            how='left'
        )
        
        players['match_date'] = pd.to_datetime(
            players['startDate'],
            errors='coerce'
        )
        players = players.dropna(subset=['match_date'])
        return players

    def remove_low_history_players(self, players):
        matches_per_player = (
            players
            .groupby('user_id')['match_id']
            .nunique()
        )
        valid_players = matches_per_player[
            matches_per_player >= 3
        ].index
        players = players[
            players['user_id'].isin(valid_players)
        ]
        return players

    def create_performance_features(self, players):
        players['player_team_won'] = np.where(
            players['team_id'] == players['winningTeam'],
            1,
            0
        )
        
        players['perf_score'] = (
            players['goals'] * 3.0 +
            players['assists'] * 2.0 +
            players['clutch_goals'] * 1.5 +
            players['penalties'] * 1.0 +
            players['player_team_won'] * 1.5 -
            players['fouls'] * 0.5 -
            players['yellow_cards'] * 1.0 -
            players['red_cards'] * 3.0 -
            players['own_goals'] * 2.0
        )
        
        players['best_match_score'] = (
            players
            .groupby('match_id')['perf_score']
            .transform('max')
        )
        
        players['is_best_player'] = np.where(
            players['perf_score'] == players['best_match_score'],
            1,
            0
        )
        return players

    def create_rolling_features(self, players):
        WINDOW = 5
        players = players.sort_values(['user_id', 'match_date'])
        
        for col in ['goals', 'assists', 'perf_score', 'fouls']:
            players[f'roll5_{col}'] = (
                players
                .groupby('user_id')[col]
                .transform(
                    lambda x: x.shift(1)
                    .rolling(WINDOW, min_periods=1)
                    .mean()
                )
            )
        return players

    def create_career_features(self, players):
        players['career_matches'] = (
            players
            .groupby('user_id')
            .cumcount()
        )
        
        for col in ['goals', 'assists', 'perf_score']:
            players[f'career_{col}'] = (
                players
                .groupby('user_id')[col]
                .cumsum()
                - players[col]
            )
            
        players['career_gpg'] = np.where(
            players['career_matches'] > 0,
            players['career_goals'] / players['career_matches'],
            0
        )
        players['career_apg'] = np.where(
            players['career_matches'] > 0,
            players['career_assists'] / players['career_matches'],
            0
        )
        players['career_avg_perf'] = np.where(
            players['career_matches'] > 0,
            players['career_perf_score'] / players['career_matches'],
            0
        )
        return players

    def create_momentum_features(self, players):
        players['goal_trend'] = (
            players['roll5_goals'] - players['career_gpg']
        )
        players['perf_trend'] = (
            players['roll5_perf_score'] - players['career_avg_perf']
        )
        return players

    def create_age_features(self, players, player):
        dob = (
            player[['user_id', 'date_of_birth']]
            .drop_duplicates('user_id')
        )
        dob['date_of_birth'] = pd.to_datetime(
            dob['date_of_birth'],
            errors='coerce'
        )
        
        players = players.merge(
            dob,
            on='user_id',
            how='left'
        )
        players['age'] = (
            (players['match_date'] - players['date_of_birth'])
            .dt.days / 365.25
        )
        return players

    def create_log_features(self, players):
        players['log_career_goals'] = np.log1p(
            players['career_goals']
        )
        players['log_career_matches'] = np.log1p(
            players['career_matches']
        )
        return players

    def transform(self, players, matches):
        raw_players = players.copy()

        # Step 1: Clean and Standardize Event (Player) Rows
        players = self.remove_cols(players)
        players = self.standardize_player(players)
        
        # Step 2: Create Event-level Features
        players = self.create_player_features(players)
        
        # Step 3: Aggregate to Player-Match Level
        players = self.aggregate_player_match(players)
        
        # Step 4: Merge Match Details and Outcome
        players = self.merge_match_info(players, matches)
        
        # Step 5: Filter out Low History Players
        players = self.remove_low_history_players(players)
        
        # Step 6: Create Team & Performance Features
        players = self.create_performance_features(players)
        
        # Step 7: Create Rolling Metrics (Time-safe)
        players = self.create_rolling_features(players)
        
        # Step 8: Create Career Averages
        players = self.create_career_features(players)
        
        # Step 9: Create Trends/Momentum
        players = self.create_momentum_features(players)
        
        # Step 10: Create Age Metrics
        players = self.create_age_features(players, raw_players)
        
        # Step 11: Create Log Transforms
        players = self.create_log_features(players)
        
        # Step 12: Final Cleaning
        players = players.fillna(0)
        
        players.to_csv('data/players_ml.csv', index=False)
        
        return players


# Alias for backwards compatibility
DataTransformation = DataMatchTransformation