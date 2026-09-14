"""
Trains a Linear Regression model to predict final_score,
and a Logistic Regression model to predict Pass/Fail.
Saves both models + metrics to model/model.pkl
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "students.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

df = pd.read_csv(DATA_PATH)

# Encode categorical columns
le_extra = LabelEncoder()
le_internet = LabelEncoder()
df["extra_activities_enc"] = le_extra.fit_transform(df["extra_activities"])
df["internet_access_enc"] = le_internet.fit_transform(df["internet_access"])

features = ["study_hours", "attendance", "previous_score", "sleep_hours",
            "extra_activities_enc", "internet_access_enc"]

X = df[features]
y_reg = df["final_score"]
y_clf = df["result"].map({"Pass": 1, "Fail": 0})

X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
reg_pred = reg_model.predict(X_test)
mae = mean_absolute_error(y_reg_test, reg_pred)
r2 = r2_score(y_reg_test, reg_pred)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train, y_clf_train)
clf_pred = clf_model.predict(X_test)
acc = accuracy_score(y_clf_test, clf_pred)

print(f"Regression  -> MAE: {mae:.2f}, R2: {r2:.3f}")
print(f"Classifier  -> Accuracy: {acc*100:.1f}%")

bundle = {
    "reg_model": reg_model,
    "clf_model": clf_model,
    "le_extra": le_extra,
    "le_internet": le_internet,
    "features": features,
    "metrics": {"mae": round(mae, 2), "r2": round(r2, 3), "accuracy": round(acc * 100, 1)}
}

joblib.dump(bundle, MODEL_PATH)
print(f"Model saved -> {MODEL_PATH}")
