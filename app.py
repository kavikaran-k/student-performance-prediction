import os
import sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

app = Flask(__name__)

# ---------- Database setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            study_hours REAL,
            attendance REAL,
            previous_score REAL,
            sleep_hours REAL,
            extra_activities TEXT,
            internet_access TEXT,
            predicted_score REAL,
            predicted_result TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- Load trained model ----------
bundle = None
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html", model_ready=bundle is not None)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if bundle is None:
        return render_template("predict.html", error="Model not trained yet. Run model/train_model.py first.")

    if request.method == "POST":
        name = request.form.get("student_name", "Student")
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_score = float(request.form["previous_score"])
        sleep_hours = float(request.form["sleep_hours"])
        extra_activities = request.form["extra_activities"]
        internet_access = request.form["internet_access"]

        extra_enc = bundle["le_extra"].transform([extra_activities])[0]
        internet_enc = bundle["le_internet"].transform([internet_access])[0]

        X_new = pd.DataFrame([[study_hours, attendance, previous_score, sleep_hours,
                                extra_enc, internet_enc]], columns=bundle["features"])

        predicted_score = float(bundle["reg_model"].predict(X_new)[0])
        predicted_score = max(0, min(100, predicted_score))
        predicted_class = bundle["clf_model"].predict(X_new)[0]
        predicted_result = "Pass" if predicted_class == 1 else "Fail"

        conn = get_db()
        conn.execute("""INSERT INTO predictions
            (student_name, study_hours, attendance, previous_score, sleep_hours,
             extra_activities, internet_access, predicted_score, predicted_result)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (name, study_hours, attendance, previous_score, sleep_hours,
             extra_activities, internet_access, round(predicted_score, 1), predicted_result))
        conn.commit()
        conn.close()

        return render_template("result.html", name=name, score=round(predicted_score, 1),
                                result=predicted_result)

    return render_template("predict.html", error=None)

@app.route("/dashboard")
def dashboard():
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()

    total = len(rows)
    passed = sum(1 for r in rows if r["predicted_result"] == "Pass")
    failed = total - passed
    avg_score = round(sum(r["predicted_score"] for r in rows) / total, 1) if total else 0

    chart_labels = [r["student_name"] for r in rows[:10]][::-1]
    chart_scores = [r["predicted_score"] for r in rows[:10]][::-1]

    metrics = bundle["metrics"] if bundle else {}

    return render_template("dashboard.html", rows=rows, total=total, passed=passed,
                            failed=failed, avg_score=avg_score, chart_labels=chart_labels,
                            chart_scores=chart_scores, metrics=metrics)

@app.route("/delete/<int:pred_id>", methods=["POST"])
def delete(pred_id):
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
