import os
import json
import pandas as pd
import numpy as np
from typing import Optional
import litellm
import json

# LangChain + LiteLLM
# from langchain_litellm import ChatLiteLLM
# from langchain_core.messages import SystemMessage, HumanMessage

LITELLM_MODEL = "gemini/gemini-3.5-flash"   # or gemini/gemini-2.5-flash
MAX_TOKENS    = 60000  # Increased to prevent commentary from getting cut off


# =============================================================================
# STEP 1 — NARRATIVE CONTEXT BUILDER
# Pure Python / Pandas only. No LLM.
# All numbers are locked here before the prompt is built.
# =============================================================================

class NarrativeContextBuilder:
    """
    Aggregates historical stats from feature-engineered CSVs
    into a structured context dict. Every number that ends up
    in the LLM prompt is computed here — the LLM never calculates.
    """

    def __init__(self, players_csv: str, matches_csv: str):
        self.players = pd.read_csv(players_csv)
        self.matches = pd.read_csv(matches_csv)
        self._prepare_players()

    def _prepare_players(self):
        """
        Add own_team_name / opponent_team_name columns.
        Robust against:
          - team_id missing entirely  (KeyError)
          - team_id float vs homeTeamId int  (type mismatch)
          - NaN values in team_id
        """
        p = self.players

        # Strip any accidental whitespace from column names
        p.columns = p.columns.str.strip()
        p["match_date"] = pd.to_datetime(p["match_date"], errors="coerce")

        if "team_id" in p.columns and "homeTeamId" in p.columns:
            # Cast both to numeric so float 22.0 == int 22 compares cleanly
            team_id_s    = pd.to_numeric(p["team_id"],    errors="coerce")
            home_team_id = pd.to_numeric(p["homeTeamId"], errors="coerce")
            is_home = (team_id_s == home_team_id)

            p["own_team_name"]      = np.where(is_home, p["homeTeamName"], p["awayTeamName"])
            p["opponent_team_name"] = np.where(is_home, p["awayTeamName"], p["homeTeamName"])

        elif "homeTeamName" in p.columns and "awayTeamName" in p.columns:
            # Fallback: team_id absent — use homeTeamName as own side
            print(
                "[NarrativeContextBuilder] Warning: team_id column not found. "
                "Falling back to homeTeamName as own_team_name. "
                "Player-vs-opponent stats may be less accurate."
            )
            p["own_team_name"]      = p["homeTeamName"]
            p["opponent_team_name"] = p["awayTeamName"]

        else:
            raise ValueError(
                f"[NarrativeContextBuilder] Missing required columns. "
                f"Found: {p.columns.tolist()}. "
                f"Need: homeTeamName, awayTeamName, match_date, player_name."
            )

        self.players = p

    # ------------------------------------------------------------------
    # TEAM — H2H SUMMARY
    # ------------------------------------------------------------------

    def _h2h_summary(self, home_team: str, away_team: str) -> dict:
        m = self.matches
        h2h = m[
            ((m["homeTeamName"] == home_team) & (m["awayTeamName"] == away_team)) |
            ((m["homeTeamName"] == away_team) & (m["awayTeamName"] == home_team))
        ].copy()

        total = len(h2h)
        if total == 0:
            return {"total": 0, "home_wins": 0, "away_wins": 0, "summary": "No previous meetings."}

        # Wins from home_team's perspective (passed as home)
        home_wins = len(h2h[
            ((h2h["homeTeamName"] == home_team) & (h2h["outcome"] == 1)) |
            ((h2h["awayTeamName"] == home_team) & (h2h["outcome"] == 0))
        ])
        away_wins = total - home_wins

        # Last 5 results
        h2h_sorted = h2h.sort_values("startDate").tail(5)
        results = []
        for _, row in h2h_sorted.iterrows():
            if row["homeTeamName"] == home_team:
                hg, ag = int(row["home_goals"]), int(row["away_goals"])
                winner = home_team if row["outcome"] == 1 else away_team
            else:
                hg, ag = int(row["away_goals"]), int(row["home_goals"])
                winner = home_team if row["outcome"] == 0 else away_team
            results.append(f"{winner} ({hg}-{ag})")

        return {
            "total":    total,
            "home_wins": home_wins,
            "away_wins": away_wins,
            "last5":    results,
            "dominant": home_team if home_wins > away_wins else away_team,
            "summary":  (
                f"{home_team} leads H2H {home_wins}-{away_wins} across {total} matches. "
                f"Last 5: {', '.join(results)}."
            )
        }

    # ------------------------------------------------------------------
    # TEAM — RECENT FORM (last 5 results as W/L string)
    # ------------------------------------------------------------------

    def _team_form(self, team_name: str, last_n: int = 5) -> dict:
        m = self.matches
        team_matches = m[
            (m["homeTeamName"] == team_name) | (m["awayTeamName"] == team_name)
        ].sort_values("startDate").tail(last_n)

        form_chars     = []
        goals_scored   = []
        goals_conceded = []

        for _, row in team_matches.iterrows():
            if row["homeTeamName"] == team_name:
                gs, gc = int(row["home_goals"]), int(row["away_goals"])
                won = row["outcome"] == 1
            else:
                gs, gc = int(row["away_goals"]), int(row["home_goals"])
                won = row["outcome"] == 0
            form_chars.append("W" if won else "L")
            goals_scored.append(gs)
            goals_conceded.append(gc)

        wins         = form_chars.count("W")
        avg_scored   = round(float(np.mean(goals_scored)),   1) if goals_scored   else 0.0
        avg_conceded = round(float(np.mean(goals_conceded)), 1) if goals_conceded else 0.0

        return {
            "form_string":  " ".join(form_chars),
            "wins_last5":   wins,
            "avg_scored":   avg_scored,
            "avg_conceded": avg_conceded,
        }

    # ------------------------------------------------------------------
    # TEAM — STRENGTH (engineered features from matches_ml.csv)
    # ------------------------------------------------------------------

    def _team_strength(self, team_name: str) -> dict:
        """
        Pulls the latest engineered strength features for a team.
        Reads home_* or away_* columns depending on the team's side
        in their most recent match.
        """
        m = self.matches
        rows = m[
            (m["homeTeamName"] == team_name) |
            (m["awayTeamName"] == team_name)
        ]

        if rows.empty:
            return {}

        latest = rows.sort_values("startDate").iloc[-1]

        if latest["homeTeamName"] == team_name:
            prefix = "home"
        else:
            prefix = "away"

        def _get(col, default=0.0):
            full = f"{prefix}_{col}"
            return round(float(latest[full]), 1) if full in latest.index else default

        return {
            "weighted_form":       _get("weighted_form"),
            "attack_strength":     _get("attack_strength"),
            "defense_strength":    _get("defense_strength"),
            "goal_diff_strength":  _get("goal_diff_strength"),
            "scoring_consistency": _get("scoring_consistency"),
            "clean_sheet_rate":    _get("clean_sheet_rate"),
            "high_scoring_rate":   _get("high_scoring_rate"),
            "elo":                 round(float(latest.get(f"{prefix}_elo", 0)), 0),
        }

    # ------------------------------------------------------------------
    # PLAYER — CAREER SNAPSHOT
    # ------------------------------------------------------------------

    def _player_career(self, player_name: str) -> dict:
        p    = self.players
        rows = p[p["player_name"].str.strip() == player_name.strip()]
        if rows.empty:
            return {}

        latest = rows.sort_values("match_date").iloc[-1]
        last5  = rows.sort_values("match_date").tail(5)

        return {
            # Core career
            "career_matches":   int(latest["career_matches"]) if "career_matches" in latest else 0,
            "career_goals":     int(latest["career_goals"]) if "career_goals" in latest else 0,
            "career_assists":   int(latest["career_assists"]) if "career_assists" in latest else 0,
            "career_gpg":       round(float(latest["career_gpg"]),       2) if "career_gpg" in latest else 0.0,
            "career_avg_perf":  round(float(latest["career_avg_perf"]),  1) if "career_avg_perf" in latest else 0.0,

            # Rolling windows
            "roll5_goals":       int(latest["roll5_goals"]) if "roll5_goals" in latest else 0,
            "roll5_perf_score":  round(float(latest["roll5_perf_score"]),  1) if "roll5_perf_score" in latest else 0.0,
            "roll10_goals":      int(latest["roll10_goals"]) if "roll10_goals" in latest else 0,
            "roll10_perf_score": round(float(latest["roll10_perf_score"]), 1) if "roll10_perf_score" in latest else 0.0,

            # Trend / momentum
            "goal_trend":   round(float(latest["goal_trend"]),   1) if "goal_trend" in latest else 0.0,
            "assist_trend": round(float(latest["assist_trend"]), 1) if "assist_trend" in latest else 0.0,
            "perf_trend":   round(float(latest["perf_trend"]),   1) if "perf_trend" in latest else 0.0,

            # Experience & peak
            "experience_score": round(float(latest["experience_score"]), 2) if "experience_score" in latest else 0.0,
            "best_match_score": round(float(latest["best_match_score"]), 1) if "best_match_score" in latest else 0.0,

            # Last-5 raw goals
            "form_last5_goals": int(last5["goals"].sum()) if "goals" in last5 else 0,
        }

    # ------------------------------------------------------------------
    # PLAYER — VS SPECIFIC OPPONENT
    # ------------------------------------------------------------------

    def _player_vs_opponent(self, player_name: str, opponent_team: str) -> dict:
        p    = self.players
        rows = p[
            (p["player_name"].str.strip() == player_name.strip()) &
            (p["opponent_team_name"] == opponent_team)
        ]
        if rows.empty:
            return {
                "matches":          0,
                "goals":            0,
                "assists":          0,
                "avg_perf":         0.0,
                "goal_rate":        0.0,
                "perf_consistency": 0.0,
            }

        return {
            "matches":  int(rows["match_id"].nunique()) if "match_id" in rows else 0,
            "goals":    int(rows["goals"].sum()) if "goals" in rows else 0,
            "assists":  int(rows["assists"].sum()) if "assists" in rows else 0,
            "avg_perf": round(float(rows["perf_score"].mean()), 2) if "perf_score" in rows else 0.0,

            # NEW — lets LLM distinguish "consistent vs one-off"
            "goal_rate":        round(
                rows["goals"].sum() / max(1, rows["match_id"].nunique()), 1
            ) if "goals" in rows and "match_id" in rows else 0.0,
            "perf_consistency": round(float(rows["perf_score"].std()), 1) if "perf_score" in rows else 0.0,
        }

    # ------------------------------------------------------------------
    # MAIN — BUILD FULL CONTEXT
    # ------------------------------------------------------------------

    def build(self, prediction_json: dict) -> dict:
        """
        Takes the ML prediction JSON and enriches it with
        historical context for every player listed.
        Returns a single context dict ready for the prompt builder.
        """
        home_team    = prediction_json["homeTeam"]
        away_team    = prediction_json["awayTeam"]
        winner       = prediction_json["predictedWinner"]
        loser        = away_team if winner == home_team else home_team
        best_player  = prediction_json["predictedBestPlayer"].strip()
        players_list = prediction_json.get("winningTeamPlayers", [])

        # --- Team-level context ---
        h2h           = self._h2h_summary(home_team, away_team)
        home_form     = self._team_form(home_team)
        away_form     = self._team_form(away_team)
        home_strength = self._team_strength(home_team)   # NEW
        away_strength = self._team_strength(away_team)   # NEW

        # --- Player context (top 3) ---
        key_players     = [p["playerName"].strip() for p in players_list[:3]]
        player_contexts = {}
        for name in key_players:
            opponent = away_team if winner == home_team else home_team
            player_contexts[name] = {
                "career":      self._player_career(name),
                "vs_opponent": self._player_vs_opponent(name, opponent),
                "prob":        next(
                    (p["playWellProbability"] for p in players_list
                     if p["playerName"].strip() == name), None
                ),
            }

        return {
            "home_team":        home_team,
            "away_team":        away_team,
            "predicted_winner": winner,
            "predicted_loser":  loser,
            "home_win_prob":    prediction_json["homeWinProbability"],
            "away_win_prob":    prediction_json["awayWinProbability"],
            "best_player":      best_player,
            "best_player_prob": players_list[0]["playWellProbability"] if players_list else None,
            "h2h":              h2h,
            "home_form":        home_form,
            "away_form":        away_form,
            "home_strength":    home_strength,   # NEW
            "away_strength":    away_strength,   # NEW
            "player_contexts":  player_contexts,
            "all_players":      players_list,
        }


# =============================================================================
# STEP 2 — PROMPT BUILDER
# Injects locked context into a template.
# LLM never does math — it only writes.
# =============================================================================

class PromptBuilder:

    SYSTEM = """
You are a professional football analyst.

Your job is to explain the likely outcome of a match using the
historical information provided.

Rules:

- Interpret statistics instead of simply repeating them.
- Focus on momentum, consistency, attacking strength,
  defensive stability, historical matchup trends and player impact.
- Use statistics only when they strengthen the explanation.
- When mentioning numbers, round them appropriately and avoid
  overwhelming the reader with excessive detail.
- Prefer insights over raw statistics.
- Do not mention AI, machine learning, probabilities,
  models, confidence scores, datasets or feature engineering.
- Do not invent facts.
- Keep the explanation concise and easy to understand.
- Write like an analyst, not a commentator.

Examples:

Bad:
'The team scored 6.8 goals per game and conceded 1.6.'

Better:
'The team has been one of the league's strongest attacking sides while remaining defensively solid.'

Bad:
'The player scored 9 goals in his last 5 matches.'

Better:
'The player has been in excellent scoring form recently.'

Good use of numbers:
'The player has scored 9 goals in his last 5 matches, highlighting his exceptional recent form.'

Output format:
Return a valid JSON object with EXACTLY the following three keys:
{
  "matchSummary": "<2-3 sentences>",
  "keyPlayerInsight": "<2-3 sentences>",
  "overallAssessment": "<1-2 sentences>"
}
"""

    def build(self, ctx: dict) -> str:

        home = ctx["home_team"]
        away = ctx["away_team"]

        winner      = ctx["predicted_winner"]
        best_player = ctx["best_player"]

        h2h = ctx["h2h"]
        hf  = ctx["home_form"]
        af  = ctx["away_form"]
        hs  = ctx.get("home_strength", {})
        as_ = ctx.get("away_strength", {})

        # ------------------------------------------------------------------
        # Player sections — now includes richer career + vs-opponent fields
        # ------------------------------------------------------------------
        player_sections = []
        for player_name, pdata in ctx["player_contexts"].items():

            career = pdata.get("career", {})
            vs_opp = pdata.get("vs_opponent", {})

            player_sections.append(f"""
Player: {player_name}

Career:
- Matches:             {career.get("career_matches", 0)}
- Goals:               {career.get("career_goals", 0)}
- Assists:             {career.get("career_assists", 0)}
- Goals Per Game:      {career.get("career_gpg", 0.0)}
- Avg Performance:     {career.get("career_avg_perf", 0.0)}
- Experience Score:    {career.get("experience_score", 0.0)}
- Best Match Score:    {career.get("best_match_score", 0.0)}

Recent Form:
- Last 5 Goals:        {career.get("roll5_goals", 0)}
- Last 5 Perf Score:   {career.get("roll5_perf_score", 0.0)}
- Last 10 Goals:       {career.get("roll10_goals", 0)}
- Last 10 Perf Score:  {career.get("roll10_perf_score", 0.0)}
- Goals (last 5 raw):  {career.get("form_last5_goals", 0)}

Momentum:
- Goal Trend:          {career.get("goal_trend", 0.0)}
- Assist Trend:        {career.get("assist_trend", 0.0)}
- Performance Trend:   {career.get("perf_trend", 0.0)}

Against This Opponent:
- Matches:             {vs_opp.get("matches", 0)}
- Goals:               {vs_opp.get("goals", 0)}
- Assists:             {vs_opp.get("assists", 0)}
- Goal Rate:           {vs_opp.get("goal_rate", 0.0)}
- Avg Performance:     {vs_opp.get("avg_perf", 0.0)}
- Perf Consistency:    {vs_opp.get("perf_consistency", 0.0)}
""")

        players_text = "\n".join(player_sections)

        # ------------------------------------------------------------------
        # Team strength block — only shown if data is available
        # ------------------------------------------------------------------
        def _strength_block(label: str, s: dict) -> str:
            if not s:
                return f"{label} Strength: No data available.\n"
            return f"""
{label} Team Strength:
- Weighted Form:       {s.get("weighted_form", 0.0)}
- Attack Strength:     {s.get("attack_strength", 0.0)}
- Defense Strength:    {s.get("defense_strength", 0.0)}
- Goal Diff Strength:  {s.get("goal_diff_strength", 0.0)}
- Scoring Consistency: {s.get("scoring_consistency", 0.0)}
- Clean Sheet Rate:    {s.get("clean_sheet_rate", 0.0)}
- High Scoring Rate:   {s.get("high_scoring_rate", 0.0)}
- ELO Rating:          {s.get("elo", 0)}
"""

        home_strength_text = _strength_block(home, hs)
        away_strength_text = _strength_block(away, as_)

        return f"""
MATCH INFORMATION

Home Team:  {home}
Away Team:  {away}
Expected Stronger Team: {winner}

Head-To-Head:
{h2h.get("summary", "")}

Recent Form:

{home}
- Form:            {hf.get("form_string")}
- Wins (last 5):   {hf.get("wins_last5")}
- Goals Scored:    {hf.get("avg_scored", 0.0)}
- Goals Conceded:  {hf.get("avg_conceded", 0.0)}

{away}
- Form:            {af.get("form_string")}
- Wins (last 5):   {af.get("wins_last5")}
- Goals Scored:    {af.get("avg_scored", 0.0)}
- Goals Conceded:  {af.get("avg_conceded", 0.0)}

TEAM STRENGTH
{home_strength_text}
{away_strength_text}

Key Player:  {best_player}

PLAYER INFORMATION
{players_text}

Instructions:

Explain:
1. Why the stronger team has an advantage.
2. What recent trends and team strength metrics support that view.
3. Why the key player stands out — consider form, trend, and head-to-head history.
4. Whether the player consistently performs against this opponent or had a one-off.
5. Any notable historical matchup patterns.

Interpret the statistics.
Do not simply repeat them.
"""


# =============================================================================
# STEP 3 — COMMENTARY GENERATOR
# LangChain ChatLiteLLM → routes to Gemini via LiteLLM.
# Swap LITELLM_MODEL in CONFIG to change provider — zero code changes.
# =============================================================================

class CommentaryGenerator:

    def __init__(self, api_key=None):

        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        response = await litellm.acompletion(
            model=LITELLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            fallbacks=["gemini/gemini-3.1-pro-preview", "gemini/gemini-2.5-flash"]
        )
        return json.loads(response.choices[0].message.content.strip())


# =============================================================================
# STEP 4 — MAIN ORCHESTRATOR
# Drop this into your existing prediction pipeline.
# =============================================================================

async def enrich_prediction_with_commentary(
    prediction_json: dict,
    players_csv:     str,
    matches_csv:     str,
    api_key:         Optional[str] = None,
) -> dict:
    """
    Takes your existing ML prediction JSON.
    Returns the same JSON enriched with a 'commentary' field.
    """
    # 1. Build context (pure ML/Pandas — no LLM)
    builder = NarrativeContextBuilder(players_csv, matches_csv)
    context = builder.build(prediction_json)

    # 2. Build prompt
    prompt_builder = PromptBuilder()
    user_prompt    = prompt_builder.build(context)
    system_prompt  = PromptBuilder.SYSTEM

    # 3. Generate commentary
    generator  = CommentaryGenerator(api_key=api_key)
    commentary = await generator.generate(system_prompt, user_prompt)

    # 4. Return enriched JSON (original + commentary + context snapshot)
    enriched = dict(prediction_json)
    enriched["matchSummary"] = commentary
    enriched["narrativeContext"] = {
        "h2h_total":    context["h2h"]["total"],
        "h2h_dominant": context["h2h"].get("dominant"),
        "home_form":    context["home_form"]["form_string"],
        "away_form":    context["away_form"]["form_string"],
    }
    return enriched

if __name__ == "__main__":
    print("narrative_pipeline.py loaded successfully.")
