# Futsal Match & MVP Prediction Pipeline (Men's)

## 📌 Project Overview & Business Objective
The primary business objective of this project is to **enhance fan engagement and provide automated, data-driven analytical coverage** for Men's Futsal matches. 

In sports media, generating high-quality pre-match previews, analyzing player forms, and predicting outcomes require significant manual editorial effort. This project automates that entire lifecycle. By predicting match outcomes, forecasting the Most Valuable Player (MVP), and generating a rich, human-like narrative preview using Generative AI, we deliver instant, scalable, and premium content for every upcoming match.

### Key Goals:
* **Automate Match Previews:** Reduce editorial overhead by automatically synthesizing historical data, Head-to-Head (H2H) records, and player stats.
* **Predictive Insights:** Provide fans with purely data-driven predictions on who will win the match and which players are the ones to watch.
* **Elevate Fan Experience:** Transform raw probabilities and statistics into engaging, story-like sports commentary.

---

## 🏗️ System Architecture
We have built an end-to-end Machine Learning and Generative AI pipeline. To accurately capture the complexity of the sport, the architecture treats **Teams** and **Players** as two distinct, parallel tracks:

1. **Data Ingestion & Transformation:** Processes raw historical match and player data through two separate pipelines—one dedicated to Team/Match dynamics and another dedicated to individual Player performances.
2. **Model Training & Evaluation:** Trains multiple classification models for Match Outcome (using Team data) and Player Performance (using Player data).
3. **Inference Engine:** Serves the best-trained models to make real-time predictions on upcoming fixtures for both teams and individuals.
4. **Narrative Pipeline:** Acts as a bridge between the raw numerical predictions and a Large Language Model to generate context-rich pre-match commentary.
5. **API Integration:** Fetches upcoming matches from external platforms, runs the full inference/narrative pipeline, and automatically pushes the final broadcast-ready reports back to the database.

---

## ⚙️ Data Engineering & Analysis (The Engine)
Futsal is a fast-paced game where both overarching team momentum and individual player brilliance dictate the outcome. To capture this accurately, our data transformation logic is split into two specialized tracks.

### 1. Match Dynamics (Team-Level Engineering)
This track focuses on how the collective unit performs over time. 

* **Dynamic ELO Ratings:** We implemented an ELO rating system to objectively measure team strength over time. A team gains more points for beating a highly-rated opponent than a lower-rated one, providing a true reflection of their current standing.
* **Weighted Recent Form:** Instead of a simple win rate, we calculate form over the most recent matches using exponential weights. A win in the most recent match is weighted significantly higher than a win several weeks ago, capturing immediate momentum.
* **Attack & Defense Strengths:** Evaluates average goals scored (Attack) and goals conceded (Defense) over a rolling window to see exactly where a team excels or struggles.
* **Differential Metrics:** We calculate the difference in form, attack, defense, and ELO between the Home and Away teams. Machine learning models often perform much better on these differential features because they directly compare the two competitors.
* **Scoring Consistency & High-Scoring Rates:** Measures how often a team scores at least one goal, and how often they explode for high-scoring games, giving the model deep insight into offensive reliability.

### 2. Individual Brilliance (Player-Level Engineering)
Predicting an MVP requires microscopic attention to individual form. The player pipeline processes entirely different datasets to build a complete profile of every athlete.

* **Career vs. Recent Rolling Windows:** We calculate overarching career averages (goals per game, average performance scores) alongside immediate rolling windows (stats over the last 5 or 10 matches). This allows the model to differentiate between a historically great player and a player who is currently on a hot streak.
* **Trend & Momentum Scores:** Mathematical representations of whether a player's goal-scoring, assists, and overall performance are trending upwards or downwards heading into the match.
* **Player vs. Specific Opponent (H2H):** We isolate a player's historical performance against the *exact team they are about to face*. This uncovers "bogey teams" or scenarios where a specific player notoriously dominates a specific opponent.
* **Experience & Peak Scores:** Metrics that capture how many high-pressure matches a player has participated in and their highest "ceiling" of performance to date.

---

## 🧠 Machine Learning Models (The Brain)
Because we have two separate data tracks, we deployed two completely different modeling pipelines to capture these different aspects of the game.

### A. Match Outcome Prediction
* **Objective:** Predict whether the Home or Away team will win.
* **Feature Strategy:** Relies heavily on the team differential features (differences in goal metrics, form, and ELO) as well as historical head-to-head win rates.
* **Algorithms Utilized:** Evaluates robust classification algorithms, including Random Forests, Support Vector Machines, and Logistic Regression.
* **Validation:** Uses Time-Series Cross-Validation to ensure the model is evaluated strictly on future matches, preventing any unrealistic "future data leakage."

### B. MVP Prediction (Best Player)
* **Objective:** Identify which player **on the predicted winning team** is most likely to be crowned the MVP of the match. (Note: MVP awards in Futsal are nearly always given to a player on the winning side).
* **Feature Strategy:** Looks purely at the Player-Level engineered features (recent goal trends, opponent-specific performance, momentum scores).
* **Algorithms Utilized:** Uses powerful, state-of-the-art gradient boosting frameworks. Because predicting an MVP is a highly imbalanced problem (only one player wins it per game among many), these algorithms are configured with balanced class weights to successfully pinpoint standout performers.

---

## 🎙️ LLM-Powered Narrative Generation (The Voice)
Raw predictions (e.g., "Team A wins with 65% probability, Player X is the MVP") are informative but dry. To solve this, we integrated an advanced Large Language Model.

**How it works:**
1. **Deterministic Context Building:** The system aggregates both the **Team-Level** and **Player-Level** stats purely in code. It grabs Team A's recent W-L-D form *and* Player X's exact goal rate against Team B.
2. **Prompt Injection:** These mathematically locked-in stats are fed into the LLM prompt. The AI is *never* asked to calculate statistics; it is only asked to act as a sports journalist and write the narrative based on the provided facts.
3. **Commentary Generation:** The AI generates a premium, journalistic pre-match preview. It fluidly highlights the predicted winner, the key player to watch, and weaves the separate team and player contexts together (e.g., "Team A comes into this match riding a 3-game win streak, relying heavily on Player X who has historically dominated Team B's defense...").

---

## 🚀 Pipeline Workflow
The entire operation runs completely hands-free:
1. The system receives a signal for a new competition or matchday.
2. It fetches upcoming fixtures via external APIs.
3. The matches are passed to the **Inference Engine** which predicts the Match Winner (using team models).
4. Based on the predicted winner, the engine filters for that team's players and predicts the Best Player (using player models).
5. These separate predictions, alongside the deeply calculated historical team and player context, flow into the **Narrative Pipeline**.
6. The final output is a complete Pre-Match Report—a JSON payload containing both the hard predictive data and the engaging text commentary.
7. The payload is seamlessly posted back to the external database, ready for fan consumption.

---

## 🛠️ How to Run

### 1. Environment Setup
Make sure you have Python installed. You can install the required dependencies using `pip` or `uv`:
```bash
pip install -r requirements.txt
# OR if using uv
uv pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and populate it with the necessary API keys and database credentials:
```env
FUTSALOZ_API_BASE_URL="https://api.example.com"
EXTERNAL_API_TOKEN="your_external_api_token"
GEMINI_API_KEY="your_gemini_api_key"
```

### 3. Model Training (Optional)
If you need to retrain the models on new data, execute the training pipelines. These scripts will fetch the latest data, generate features, run Grid Search Cross-Validation, and save the best models:
```bash
python scripts/model_trainer.py
```

### 4. Running the Inference API
To fetch upcoming matches, generate predictions, build the LLM narrative, and push the results to the database, run the main prediction script:
```bash
python api/predict.py
```
