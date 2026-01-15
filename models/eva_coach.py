# Library import

import os
import json
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from tensorflow.keras.models import load_model

# Class creation for EVA

class EVA:
    """
    EVA — Esports Virtual Assistant
    Data-driven Rocket League coach
    """

# Initialization

    def __init__(self, df):
        self.df = df
        self.target_col = "result"

        # Loading the funnel neural network
        self.model = load_model("models/funnel_model.keras")
        self.scaler = joblib.load("models/scaler.pkl")

        with open("models/feature_names.json", "r") as f:
            self.feature_names = json.load(f)

        # OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.player_name = None
        self.available_matches = []
        self.last_report = None

    # Function for player selection
    
    def select_player(self, player_name):
        self.player_name = player_name
        matches = self.df[self.df["player name"] == player_name]

        if matches.empty:
            print("No matches found for this player.")
            return []

        self.available_matches = matches.index.tolist()

        print(f"Found {len(self.available_matches)} matches for {player_name}:")
        for idx in self.available_matches:
            res = self.df.loc[idx, self.target_col]
            print(f"- Index {idx} | Result: {'Win' if res == 1 else 'Loss'}")

        return self.available_matches

    # Function to analyze the play
    
    def analyze_match(self, match_index):
        if match_index not in self.available_matches:
            print("Invalid match index.")
            return

        x_orig = self.df.loc[match_index, self.feature_names].values.reshape(1, -1)
        x_scaled = self.scaler.transform(x_orig)
        y_real = self.df.loc[match_index, self.target_col]

        y_prob = self.model.predict(x_scaled)[0][0]
        y_pred = int(y_prob > 0.5)

        explainer = shap.KernelExplainer(self.model.predict, x_scaled)
        shap_values = explainer.shap_values(x_scaled)[0]

        shap_df = pd.DataFrame({
            "feature": self.feature_names,
            "shap": shap_values,
            "value": x_orig.flatten()
        }).assign(abs_shap=lambda d: d["shap"].abs()) \
        .sort_values("abs_shap", ascending=False)

        top3 = shap_df.head(3)

        winners = self.df[self.df[self.target_col] == 1]
        winners_avg = winners[self.feature_names].mean()

        improve_df = pd.DataFrame({
            "feature": self.feature_names,
            "player": x_orig.flatten(),
            "winners_avg": winners_avg.values
        }).assign(gap=lambda d: d["winners_avg"] - d["player"]) \
        .sort_values("gap", ascending=False).head(3)

        self.last_report = {
            "prediction": {
                "predicted": "Win" if y_pred else "Loss",
                "prob": float(y_prob),
                "real": "Win" if y_real == 1 else "Loss"
            },
            "top": top3,
            "improve": improve_df
        }

        self._show_plots(top3, improve_df)

        print("Match analyzed successfully.")

    # PLOTS
    
    def _show_plots(self, top3, improve_df):
        plt.figure(figsize=(6,4))
        plt.bar(top3["feature"], top3["value"], color="#FF6F00")
        plt.title("Your key stats")
        plt.show()

        plt.figure(figsize=(6,4))
        plt.bar(improve_df["feature"], improve_df["player"], color="#FF6F00")
        plt.title("Stats to improve")
        plt.show()

    # Function to discuss and chat with EVA
    
    def chat(self, question):
        if self.last_report is None:
            return "Analyze a match first."

        prompt = f"""
You are EVA, a professional Rocket League coach.

Prediction: {self.last_report['prediction']}
Stats to improve:
{self.last_report['improve'].to_string(index=False)}

Question:
{question}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an esports coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content
