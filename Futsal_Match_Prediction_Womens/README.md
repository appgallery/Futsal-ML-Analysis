# Futsal Analytics & Predictive Intelligence System ⚽🤖

## 1. System Overview
This project is an advanced, fully automated Machine Learning and Generative AI orchestration system designed exclusively for predicting the outcomes of Women's Futsal matches. 

Moving beyond traditional statistical analysis, this system predicts which team will win, identifies which player will emerge as the "Player of the Match," and utilizes Large Language Models (LLMs) to automatically generate human-like, professional match commentary. This bridges the gap between complex data science and consumable sports journalism.

## 2. Core Business Value
1. **Match Outcome Prediction:** Forecasting home win versus away win probabilities by analyzing historical match data, dynamic team strength, and momentum indicators.
2. **Player Impact Forecasting:** Identifying the single most impactful player for the predicted winning team (MVP) to highlight key talent.
3. **Automated Sports Journalism:** Synthesizing complex ML features into a human-readable match preview via generative AI, providing immediate content for fan engagement and analyst previews.
4. **Data-Driven Team Analytics:** Generating and exposing highly engineered metrics—such as exponentially weighted player performance and defensive strength indices—for strategic dashboarding and team analysis.

## 3. The Analytics & Feature Engineering Engine

The mathematical heart of the system transforms raw data (goals, cards, historical matches) into predictive, time-aware features. It operates on two distinct levels: Teams and Individual Players.

### 3.1 Team-Level Predictive Features
The system processes historical matches chronologically to build a "memory" for each team, strictly preventing future data from influencing past calculations.
*   **Dynamic ELO Rating System:** Teams are assigned a dynamic rating (starting at 1500) updated via a K-Factor of 32. This ensures that a victory over a highly-rated opponent boosts a team's rating significantly more than defeating a weaker opponent.
*   **Exponentially Weighted Form:** Evaluates the last 5 match outcomes, applying an exponential weight so that a match played yesterday impacts the score heavily, while a match from a month ago carries far less weight.
*   **Attack & Defense Strength Indices:** 
    *   **Attack:** Calculated based on average goals scored in recent matches.
    *   **Defense:** Calculated using a formulation where lower raw conceded goals result in a superior defensive score, creating a reliable baseline for defensive solidity.
*   **Scoring Consistency:** Evaluates how reliably a team scores at least one goal, measuring consistency rather than just peak performance.
*   **Head-to-Head Dominance:** Tracks specific historical matchups to determine if one team has a psychological or tactical edge over the other.

### 3.2 Individual Player Analytics
Calculates individual player momentum, consistency, and overall value to the team.
*   **Event Flattening:** Aggregates event-level data (goals, fouls, yellow/red cards, assists) into match-by-match summaries for every player.
*   **Unified Performance Score:** The ground truth metric for player impact. It balances positive contributions against negative actions using the formula: 
    `(Goals * 3) + (Assists * 2) + (Clutch Goals * 1.5) - (Fouls * 0.5) - (Yellow Cards) - (Red Cards * 3)`
*   **Target Identification (MVP):** To be flagged as the best player, a player must rank #1 in their specific team based on the Performance Score, AND their team must win the match.
*   **Momentum & Rolling Windows:** Calculates 3, 5, and 10-match rolling averages for goals, assists, and overall performance. A "Performance Trend" metric evaluates if a player is peaking or slumping by comparing short-term form against their longer-term average.
*   **Exponential Weighted Moving Average (EWMA):** Smoothly tracks momentum over time to identify breakouts.
*   **Experience Index:** Uses logarithmic scaling against career matches and career goals to quantify veteran presence and experience.

## 4. The Machine Learning Core

Because predicting a match winner is structurally different from predicting an individual MVP out of a full roster, the system utilizes two completely separate modeling pipelines.

### 4.1 Match Outcome Predictor
*   **Objective:** Predict the match winner (Home Win vs. Away Win).
*   **Algorithmic Approach:** Evaluates ensembles like Random Forest, alongside Support Vector Classifiers (SVC) and Logistic Regression.
*   **Validation Strategy:** Employs Time-Series Cross-Validation. Standard randomized validation would cause data leakage (e.g., training on a future match to predict a past match). Time-Series splitting ensures the model only ever learns from the past to predict the future.

### 4.2 Player Performance Predictor
*   **Objective:** Identify the MVP (Binary classification: Is Best Player vs. Is Not Best Player).
*   **The Challenge:** Extreme Class Imbalance. In a roster of 12-15 players, only 1 is the MVP. The dataset is heavily skewed (~92% negative class).
*   **Algorithmic Approach:** Utilizes advanced Gradient Boosting frameworks (XGBoost, LightGBM, CatBoost) and Random Forests.
*   **Imbalance Handling:** The models are heavily penalized for missing the MVP class using techniques like positive class scaling and balanced class weights.
*   **Optimization:** The models optimize for the F1-Score (the harmonic mean of precision and recall) rather than raw accuracy, ensuring the model doesn't cheat by simply predicting "no one is the MVP."

## 5. The Live Inference System

When a new match is scheduled, the inference engine orchestrates the live prediction flow:
1.  **State Reconstruction:** It ingests all historical data up to the current second, rebuilding the team memory, ELO ratings, and player momentum statistics instantly.
2.  **Live Feature Calculation:** Calculates current head-to-head stats, current 5-match form, and differential metrics for the upcoming matchup.
3.  **Data Validation:** Utilizes strict data typing and validation protocols to ensure the data moving through the API is clean and formatted correctly.
4.  **Dynamic Player Availability:** Before predicting the MVP, the system analyzes recent activity (last 120 days) and cross-references live match attendance data. It filters out absent or inactive players to ensure the MVP prediction is grounded in reality.

## 6. The Generative AI Match Reporter

Instead of outputting dry statistical probabilities (e.g., "Home Win 67%"), the system utilizes Generative AI to produce a human-readable match preview.
1.  **Strict Separation of Concerns:** LLMs are notoriously poor at mathematics. To solve this, the analytics engine handles all numerical calculations (form strings, ELO, H2H dominance).
2.  **Context Construction:** The system builds a structured profile containing these pre-calculated statistics.
3.  **Generative Output:** An advanced LLM is prompted strictly to act as a professional football analyst. It is forbidden from mentioning AI, ML, or raw probabilities. It reads the pre-calculated profile and drafts a compelling, 2-3 sentence match summary, a key player insight, and a tactical assessment.
4.  **Resiliency:** The generation pipeline includes exponential backoff and retry logic to gracefully handle API rate limits from the LLM provider.

## 7. API Integration & Architecture
*   **Data Fetching:** Operates asynchronously to pull upcoming match schedules from external provider APIs.
*   **Database Fallbacks:** If the external API fails to provide complete information (like team names), the system automatically falls back to internal direct database queries to resolve IDs to proper names.
*   **Continuous Loop:** The system orchestrates the entire flow—fetching matches, running ML inference, generating AI narrative, and posting the final enriched JSON payload back to the central database.

## 8. Core Technologies
*   **Data Science & Engineering:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn.
*   **Machine Learning Algorithms:** XGBoost, LightGBM, CatBoost.
*   **Web Services & APIs:** FastAPI, HTTPX, Pydantic.
*   **Generative AI:** LangChain, LiteLLM (Google Gemini).
*   **Database & Orchestration:** SQLAlchemy, PyMySQL, Asyncpg, Apache Airflow.

## 9. How to Run

### Using Docker Compose (Recommended)
The easiest way to run the entire system (including Airflow, Postgres, and the API) is via Docker Compose.

```bash
# 1. Initialize the Airflow database and create the admin user
docker-compose up airflow-init

# 2. Build and start all services in detached mode
docker-compose up -d --build

# To check the logs of a specific service (e.g., airflow-webserver)
docker-compose logs -f airflow-webserver

# To shut down the services
docker-compose down
```
Once running, the Airflow Web UI will be accessible at `http://localhost:8080`.

### Running Locally (Development)

1. **Set up the virtual environment:**
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate
```

2. **Install dependencies:**
Using standard `pip` or `uv` (recommended for speed):
```bash
pip install -r requirements.txt
# OR
uv pip install -r requirements.txt
```

3. **Run Local Airflow (Optional):**
If you want to run Airflow outside of Docker:
```bash
# Set your Airflow home directory
export AIRFLOW_HOME=$(pwd)/airflow

# Standalone mode initializes the DB, creates an admin user, and starts the scheduler & webserver
airflow standalone
```

4. **Run the Prediction API:**
```bash
python api/predict.py
```

5. **Run Pipelines Manually:**
You can also run specific pipelines directly via the Python scripts in the `scripts` or `pipeline` directory:
```bash
# Example: Run the model training script
python scripts/model_trainer.py

# Example: Run the inference script
python scripts/inference.py
```
