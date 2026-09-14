# 📊 Student Performance Prediction System

A machine learning web app that predicts a student's final score and pass/fail
outcome based on study hours, attendance, sleep, and other factors. Built with
**Flask, scikit-learn, SQLite, and Chart.js**.

---

## 🧠 What it does
- Trains a **Linear Regression** model to predict numeric score (0–100)
- Trains a **Logistic Regression** model to classify Pass/Fail
- Every prediction is saved to a SQLite database
- A live dashboard shows stats, a bar chart, and prediction history

---

## 🛠️ STEP 1 — Set up in VS Code

1. Unzip this folder and open it in VS Code (`File > Open Folder`).
2. Open a terminal in VS Code (`` Ctrl + ` ``).
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
4. Activate it:
   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ STEP 2 — Generate data & train the model

Run these **once**, in order:

```bash
python data/generate_data.py
python model/train_model.py
```

You'll see output like:
```
Regression  -> MAE: 7.99, R2: 0.399
Classifier  -> Accuracy: 74.5%
Model saved -> model/model.pkl
```

This creates `data/students.csv` (1000 synthetic student records) and
`model/model.pkl` (your trained model).

---

## 🛠️ STEP 3 — Run the app locally

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

Try:
- `/` — Home page
- `/predict` — Enter student details, get a prediction
- `/dashboard` — See analytics + all past predictions

---

## 🌍 STEP 4 — Deploy it live (free, for your resume link)

### Option A: Render.com (recommended, easiest)
1. Push this project to a **GitHub repo** (see Step 5 below).
2. Go to [render.com](https://render.com) → Sign up (free) → **New +** → **Web Service**
3. Connect your GitHub repo.
4. Settings:
   - **Build Command:** `pip install -r requirements.txt && python data/generate_data.py && python model/train_model.py`
   - **Start Command:** `gunicorn app:app`
5. Click **Deploy**. In ~2 minutes you'll get a live URL like:
   `https://student-performance-prediction.onrender.com`
6. Put this link in your resume under the project name.

### Option B: PythonAnywhere (alternative, free tier)
1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your project via the **Files** tab (or `git clone` your repo in a Bash console)
3. Go to **Web** tab → **Add a new web app** → Flask → point to `app.py`
4. Reload the web app — you'll get a link like `yourusername.pythonanywhere.com`

> ⚠️ Free tiers sleep after inactivity — the first load after idle may take
> 20–30 seconds. This is normal and fine for a resume demo link.

---

## 🌍 STEP 5 — Push to GitHub (needed for deployment + resume link)

```bash
git init
git add .
git commit -m "Student Performance Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Create the empty repo on
GitHub first (github.com → New repository → don't initialize with README).

---

## 📁 Project Structure
```
student-performance-prediction/
├── app.py                  # Flask application (routes)
├── data/
│   ├── generate_data.py    # creates synthetic dataset
│   └── students.csv        # generated dataset (after step 2)
├── model/
│   ├── train_model.py      # trains & saves the ML model
│   └── model.pkl           # trained model (after step 2)
├── templates/              # HTML pages (Jinja2)
├── static/css/style.css    # styling
├── predictions.db          # SQLite DB (auto-created on first run)
├── requirements.txt
├── Procfile                # for Render/Heroku deployment
└── README.md
```

## 💬 How to explain this in your interview
- "I engineered a synthetic dataset with realistic noise, then trained two
  models — a regression model for the numeric score and a classification
  model for pass/fail — using scikit-learn."
- "Each prediction request goes through the Flask backend, gets encoded and
  passed to the trained model, and the result is persisted in SQLite."
- "The dashboard aggregates historical predictions using SQL queries and
  visualizes them with Chart.js."
- Be ready to explain: what MAE/R²/accuracy mean, why Linear vs Logistic
  Regression, and how you'd improve it (more features, cross-validation,
  a better model like Random Forest).
