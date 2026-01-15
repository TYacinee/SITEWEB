# app.py
from flask import Flask, render_template, jsonify, request, session, send_from_directory, abort
import pandas as pd
import os

from openai import OpenAI
from models.eva_backend import EVABackend

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me") 

# Data loading
DATA_PATH = "data/Final_Dataset_MVP.csv"
df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")


# EVA BACKEND 
eva_backend = EVABackend(
    data_path="data/Final_Dataset_MVP.csv",
    model_path="models/funnel_model.keras",
    scaler_path="models/scaler.pkl",
    features_path="models/feature_names.json",
    target_col="result",
    player_col="player name"
)

# The different routes

@app.route("/")
def Index():
    return render_template("Index.html")

@app.route("/datasets")
def datasets():
    return render_template("datasets.html")

@app.route("/projects")
def projects():
    return render_template("projects.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/eva")
def eva():
    return render_template("eva.html")

@app.route("/data")
def data():
    return df.to_json(orient="records")

@app.route("/api/players")
def api_players():
    players = eva_backend.get_players(limit=300)
    return jsonify(players)

@app.route("/api/matches")
def api_matches():
    player = request.args.get("player", "")
    res = eva_backend.get_matches(player)
    return jsonify(res)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(force=True)
    match_index = int(payload.get("match_index", -1))

    report = eva_backend.analyze_match(match_index)

    session["last_report"] = {
        "match_index": report["match_index"],
        "prediction": report["prediction"],
        "top_statistics": report["top_statistics"],
        "to_improve": report["to_improve"],
        "strengths": report["strengths"],
    }

    return jsonify(report)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        payload = request.get_json(force=True)
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"answer": "Ask me something 🙂"}), 200

        last = session.get("last_report")
        if not last:
            return jsonify({"answer": "Analyze a match first (choose an index)."}), 200

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or api_key.lower().startswith("ta_cle") or api_key == "OPENAI_API_KEY":
            return jsonify({"answer": "OPENAI_API_KEY is missing or invalid on the server. Set it in your environment and restart Flask."}), 200

        client = OpenAI(api_key=api_key)
        prompt = eva_backend.build_llm_prompt(last, question)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are EVA, a professional Rocket League coach. Be concise, concrete, and motivating."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        answer = resp.choices[0].message.content
        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"answer": f"EVA server error: {type(e).__name__} - {str(e)}"}), 200

@app.route("/download/<path:filename>")
def download_notebook(filename):
    notebooks_dir = os.path.join(app.root_path, "notebooks")
    file_path = os.path.join(notebooks_dir, filename)

    if not os.path.isfile(file_path):
        return abort(404)

    return send_from_directory(notebooks_dir, filename, as_attachment=True)

# 404 Page
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# Run the website
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
