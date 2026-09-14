"""
Generates a realistic synthetic dataset of student performance.
Run this once to create data/students.csv
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

study_hours = np.round(np.random.normal(5, 2, N).clip(0, 12), 1)
attendance = np.round(np.random.normal(78, 12, N).clip(30, 100), 1)
previous_score = np.round(np.random.normal(65, 15, N).clip(0, 100), 1)
sleep_hours = np.round(np.random.normal(6.5, 1.2, N).clip(3, 10), 1)
extra_activities = np.random.choice(["Yes", "No"], size=N, p=[0.4, 0.6])
internet_access = np.random.choice(["Yes", "No"], size=N, p=[0.8, 0.2])

# Final score is a weighted function of the above + noise (so the model has real signal to learn)
final_score = (
    study_hours * 3.0
    + attendance * 0.25
    + previous_score * 0.30
    + sleep_hours * 1.0
    + np.where(extra_activities == "Yes", 2, 0)
    + np.where(internet_access == "Yes", 3, -2)
    - 15
    + np.random.normal(0, 10, N)
)
final_score = np.clip(final_score, 0, 100).round(1)

# Pass/Fail label (>=40 is pass) for the classification view
result = np.where(final_score >= 40, "Pass", "Fail")

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "extra_activities": extra_activities,
    "internet_access": internet_access,
    "final_score": final_score,
    "result": result
})

os.makedirs(os.path.dirname(__file__), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "students.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} rows -> {out_path}")
print(df.head())
