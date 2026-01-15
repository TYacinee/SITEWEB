# Library import

import matplotlib
matplotlib.use("Agg")
import os
import json
import pickle
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model

# Class for EVA

class EVABackend:
    """
    Backend EVA : charge dataset + modèle + scaler + features,
    expose des fonctions:
      - get_players()
      - get_matches(player_name)
      - analyze_match(match_index_df)
      - build_llm_prompt(report, question)
    """

    def __init__(
        self,
        data_path="data/Final_Dataset_Mvp.csv",
        model_path="models/funnel_model.keras",
        scaler_path="models/scaler.pkl",
        features_path="models/feature_names.json",
        target_col="result",
        player_col="player name",
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.target_col = target_col
        self.player_col = player_col

        # Loading dataset
        self.df = pd.read_csv(self.data_path, sep=";")
        if self.df[self.target_col].dtype == object:
            self.df[self.target_col] = self.df[self.target_col].map({"winner": 1, "loser": 0})
        self.model = load_model(self.model_path)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.features_path, "r") as f:
            self.feature_names = json.load(f)

        # Use of SHAP
        X_bg = self.df[self.feature_names].values
        X_bg_scaled = self.scaler.transform(X_bg)
        if X_bg_scaled.shape[0] > 120:
            idx = np.random.choice(X_bg_scaled.shape[0], 120, replace=False)
            X_bg_scaled = X_bg_scaled[idx]

        self.shap_explainer = shap.KernelExplainer(self.model.predict, X_bg_scaled)
    def _fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # Function to match players
    def get_players(self, limit=200):
        if self.player_col not in self.df.columns:
            return []

        players = (
            self.df[self.player_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        players.sort()
        return players[:limit]

    def get_matches(self, player_name):
        if self.player_col not in self.df.columns:
            return {"player": player_name, "matches": []}

        sub = self.df[self.df[self.player_col].astype(str) == str(player_name)]
        matches = []
        for idx in sub.index.tolist():
            res = sub.loc[idx, self.target_col]
            matches.append({
                "index": int(idx),
                "result": "Win" if int(res) == 1 else "Loss"
            })
        return {"player": player_name, "matches": matches}

    # Function to analyze
    def analyze_match(self, match_index_df):
        if match_index_df not in self.df.index:
            raise ValueError("Match index not found in dataset.")
        x_orig = self.df.loc[match_index_df, self.feature_names].values.reshape(1, -1)
        x_scaled = self.scaler.transform(x_orig)
        y_real = int(self.df.loc[match_index_df, self.target_col])
        prob = float(self.model.predict(x_scaled)[0][0])
        pred = 1 if prob > 0.5 else 0
        shap_values = self.shap_explainer.shap_values(x_scaled, nsamples=150)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).reshape(-1)
        shap_df = pd.DataFrame({
            "statistics": self.feature_names,
            "shap_value": shap_values,
            "value": x_orig.reshape(-1)
        })
        shap_df["abs_shap"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("abs_shap", ascending=False)

        top3 = shap_df.head(3).copy()
        winners = self.df[self.df[self.target_col] == 1]
        winners_avg = winners[self.feature_names].mean()
        gap_df = pd.DataFrame({
            "statistics": self.feature_names,
            "player_value": x_orig.reshape(-1),
            "winner_avg": winners_avg.values
        })
        gap_df["gap"] = gap_df["winner_avg"] - gap_df["player_value"]
        gap_df = gap_df.sort_values("gap", ascending=False)
        improve_df = gap_df.head(3).copy()
        strengths = shap_df[
            (shap_df["shap_value"] > 0) &
            (~shap_df["statistics"].isin(top3["statistics"]))
        ]["statistics"].tolist()

        # Plots
        def _style_ax(ax, title):
            ax.set_title(title, color="white", pad=12, fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            for spine in ax.spines.values():
                spine.set_color((1, 1, 1, 0.18))
            ax.grid(axis="y", color=(1, 1, 1, 0.10), linestyle="--", linewidth=0.8)
            ax.set_facecolor((0, 0, 0, 0))
            ax.figure.set_facecolor((0, 0, 0, 0))
            
        def _bar_with_labels(ax, x, y, color):
            bars = ax.bar(x, y, color=color, alpha=0.9)
            for b in bars:
                ax.text(
                b.get_x() + b.get_width()/2,
            b.get_height(),
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
            alpha=0.9
        )
                
        # Plot 1 : Player top3 values
        fig1, ax1 = plt.subplots(figsize=(7.4, 4.2))
        _bar_with_labels(ax1, top3["statistics"], top3["value"], color="#FF6F00")
        ax1.set_ylabel("Value", color="white")
        plt.xticks(rotation=18, ha="right")
        _style_ax(ax1, "Your top influential statistics")
        b64_fig_player_top3 = self._fig_to_base64(fig1)

        # Plot 2 : Winners avg on those top3
        fig2, ax2 = plt.subplots(figsize=(7.4, 4.2))
        _bar_with_labels(ax2, top3["statistics"], winners_avg[top3["statistics"]].values, color="#00E5FF")
        ax2.set_ylabel("Value", color="white")
        plt.xticks(rotation=18, ha="right")
        _style_ax(ax2, "Winners average (same statistics)")
        b64_fig_winners_top3 = self._fig_to_base64(fig2)

        # Plot 3 : Player weakest (gap-based)
        fig3, ax3 = plt.subplots(figsize=(7.4, 4.2))
        _bar_with_labels(ax3, improve_df["statistics"], improve_df["player_value"], color="#FF6F00")
        ax3.set_ylabel("Value", color="white")
        plt.xticks(rotation=18, ha="right")
        _style_ax(ax3, "Your weakest stats (vs winners)")
        b64_fig_player_weak = self._fig_to_base64(fig3)

        # Plot 4 : Winners avg on those weak stats
        fig4, ax4 = plt.subplots(figsize=(7.4, 4.2))
        _bar_with_labels(ax4, improve_df["statistics"], improve_df["winner_avg"], color="#00E5FF")
        ax4.set_ylabel("Value", color="white")
        plt.xticks(rotation=18, ha="right")
        _style_ax(ax4, "Winners average (on those weak stats)")
        b64_fig_winners_weak = self._fig_to_base64(fig4)
        
        # Return
        report = {
            "match_index": int(match_index_df),
            "player_name": str(self.df.loc[match_index_df, self.player_col]) if self.player_col in self.df.columns else "",
            "prediction": {
                "predicted": "Win" if pred == 1 else "Loss",
                "probability": prob,
                "real": "Win" if y_real == 1 else "Loss"
            },
            "top_statistics": top3.to_dict(orient="records"),
            "to_improve": improve_df.to_dict(orient="records"),
            "strengths": strengths,
            "plots": {
                "player_top3": b64_fig_player_top3,
                "winners_top3": b64_fig_winners_top3,
                "player_weak": b64_fig_player_weak,
                "winners_weak": b64_fig_winners_weak
            }
        }
        return report

    # PROMPT (LLM)
    def build_llm_prompt(self, report, question):
        improve_txt = "\n".join(
            f"- {x['statistics']}: you={x['player_value']:.2f}, winners_avg={x['winner_avg']:.2f}"
            for x in report.get("to_improve", [])
        )
        strengths_txt = ", ".join(report.get("strengths", [])) if report.get("strengths") else "None detected."

        p = report["prediction"]
        return f"""
You are EVA, an expert Rocket League esports coach.

RULES:
- Only use the numbers in this report.
- Do not invent stats or thresholds.
- Be concrete and actionable.

REPORT:
Match index: {report['match_index']}
Prediction: {p['predicted']} (prob={p['probability']:.2f})
Actual: {p['real']}

TOP FEATURES (SHAP-influential):
{json.dumps(report.get("top_statistics", []), ensure_ascii=False)}

TOP FEATURES TO IMPROVE (vs winners average):
{improve_txt}

PLAYER STRENGTHS:
{strengths_txt}

QUESTION FROM PLAYER:
{question}
""".strip()