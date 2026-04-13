import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import matplotlib.pyplot as plt

# read data
df = pd.read_csv("all_shots.csv")

# Drop the identifier column
df = df.drop(columns=["shot_number"])

# change make or miss to 1 or 0
df["result"] = df["result"].map({"make": 1, "miss": 0})


X = df.drop(columns=["result"])
y = df["result"]

feature_names = list(X.columns)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,  stratify=y
)


# train the model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,       
    reg_lambda=1.0,  
    min_child_weight=3,
    gamma=0.1,
    eval_metric="logloss",
)

# Fit with early stopping on validation data
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# evaluate on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

# export the model 
model.save_model("xgboost_shot_model.json")
joblib.dump(feature_names, "xgboost_feature_names.pkl")
print("Model and feature names saved successfully.")

