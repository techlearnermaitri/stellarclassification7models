"""Run once locally: python train.py  -> creates model.json"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

FEATURES = ["u", "g", "r", "i", "z", "redshift"]
CLASS_MAP = {"STAR": 0, "GALAXY": 1, "QSO": 2}

df = pd.read_csv("star_classification.csv").dropna()
df["class"] = df["class"].map(CLASS_MAP)
X, y = df[FEATURES], df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    objective="multi:softprob",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=list(CLASS_MAP)))

model.save_model("model.json")  # small file, safe for GitHub
print("Saved model.json")