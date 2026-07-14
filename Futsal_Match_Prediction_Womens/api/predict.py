import os
import sys
import asyncio
import httpx
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

load_dotenv(os.path.join(parent_dir, '.env'))

from scripts.inference import Inference, MatchInput, MatchReportResponse
from db.db_connection import DB
from sqlalchemy import text

async def fetch_and_predict(com_id: str):
    print(f"Loading inference engine...")
    try:
        inference_engine = Inference()
    except FileNotFoundError:
        print("Error: Best model files not found. Cannot perform predictions.")
        return

    base_url = os.environ.get("FUTSALOZ_API_BASE_URL")
    authorization = os.environ.get("EXTERNAL_API_TOKEN")
    
    if not base_url or not authorization:
        print("Error: FUTSALOZ_API_BASE_URL or EXTERNAL_API_TOKEN environment variables are not set.")
        return
        
    url = f"{base_url}/api/v1/competition/getMatchesByCompIdForMatchSummaryPredict"
    
    headers = {
        "authorization": authorization
    }
    data = {
        "com_id": com_id
    }
    
    print(f"Fetching matches for competition {com_id}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, data=data)
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            print(f"Failed to fetch match details: {e}")
            return
            
    if not response_data.get("status"):
        print("External API returned failure status")
        return
        
    matches = response_data.get("matches", [])
    if not matches:
        print("No matches found for this competition.")
        return
        
    print(f"Found {len(matches)} matches. Processing predictions...")
        
    db_save_url = f"{base_url}/api/v1/competition/prediction/storeMatchPrediction"
    api_key = os.environ.get("GEMINI_API_KEY")
    players_csv = os.path.join(parent_dir, "data", "players_ml.csv")
    matches_csv = os.path.join(parent_dir, "data", "matches_ml.csv")

    for match in matches:
        matchID = int(match.get("id", "0"))
        homeID = int(match.get("homeTeamId", "0"))
        awayID = int(match.get("awayTeamId", "0"))
        
        homeTeamName = match.get("actualHomeTeam")
        awayTeamName = match.get("actualAwayTeam")
        
        # Fallback to DB if names are missing
        if not homeTeamName or not awayTeamName:
            try:
                db = DB()
                if db.db_test_connection():
                    engine = db.db_create_engine()
                    if engine:
                        with engine.connect() as conn:
                            query = text('''
                                SELECT ht.name as homeTeam, at.name as awayTeam
                                FROM futsaloz.cmp_matches m
                                LEFT JOIN futsaloz.teams ht ON m.homeTeamId = ht.id
                                LEFT JOIN futsaloz.teams at ON m.awayTeamId = at.id
                                WHERE m.id = :match_id
                            ''')
                            res = conn.execute(query, {"match_id": matchID})
                            row = res.fetchone()
                            if row:
                                homeTeamName = row[0]
                                awayTeamName = row[1]
                db.db_close_connection()
            except Exception as e:
                print(f"Fallback DB query failed for match {matchID}: {e}")

        if homeID == awayID or homeID == 0 or awayID == 0:
            print(f"Skipping match {matchID}: Invalid homeID ({homeID}) or awayID ({awayID})")
            continue

        print(f"Running inference for Match {matchID} ({homeTeamName} vs {awayTeamName})...")
        try:
            match_pred = inference_engine.test_match_infer(
                MatchInput(
                    matchID=matchID,
                    homeID=homeID,
                    awayID=awayID,
                    homeTeamName=homeTeamName,
                    awayTeamName=awayTeamName
                )
            )
            report = inference_engine.test_player_infer(match_pred)
            report.matchID = matchID

            try:
                report_dict = report.model_dump()
                if os.path.exists(players_csv) and os.path.exists(matches_csv) and api_key:
                    from pipeline.narrative_pipeline import enrich_prediction_with_commentary
                    enriched = await enrich_prediction_with_commentary(
                        prediction_json=report_dict,
                        players_csv=players_csv,
                        matches_csv=matches_csv,
                        api_key=api_key
                    )
                    report = MatchReportResponse(**enriched)
                    report.matchID = matchID
            except Exception as narrative_err:
                print(f"Error generating commentary for match {matchID}: {narrative_err}")

            # Save to external DB via API
            async with httpx.AsyncClient() as client:
                save_resp = await client.post(db_save_url, headers=headers, json=report.model_dump())
                save_resp.raise_for_status()
                print(f"Successfully saved prediction for match {matchID}")

        except Exception as e:
            print(f"Error processing match {matchID}: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_and_predict("1083"))
