import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# read data
df = pd.read_csv("all_shots.csv", on_bad_lines="skip")

# Drop the identifier column
df = df.drop(columns=["shot_number"])

# change make or miss to 1 or 0
df["result"] = df["result"].map({"make": 1, "miss": 0})


X = df.drop(columns=["result"])
y = df["result"]

feature_names = list(X.columns)

# split train, validation, and test (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp
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
    random_state=42,
)

# Fit with early stopping on validation data (NOT test data)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# evaluate on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test ROC-AUC:  {test_auc:.4f}")
print(f"CV Accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

os.makedirs("results", exist_ok=True)

# ROC curve image
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"XGBoost (AUC={test_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join("results", "xgboost_roc_curve.png")
plt.savefig(roc_path, dpi=200)
plt.close()

# Confusion matrix image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["miss", "make"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix")
fig.tight_layout()
cm_path = os.path.join("results", "xgboost_confusion_matrix.png")
fig.savefig(cm_path, dpi=200)
plt.close(fig)

# Feature importance image
importance = model.feature_importances_
top_n = min(12, len(feature_names))
sorted_idx = np.argsort(importance)[-top_n:]
top_features = [feature_names[i] for i in sorted_idx]
top_importance = importance[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importance)
plt.xlabel("Importance")
plt.title("Top Feature Importances")
plt.tight_layout()
fi_path = os.path.join("results", "xgboost_feature_importance.png")
plt.savefig(fi_path, dpi=200)
plt.close()

print(f"Saved image: {roc_path}")
print(f"Saved image: {cm_path}")
print(f"Saved image: {fi_path}")

# export the model 
model.save_model("xgboost_shot_model.json")
joblib.dump(feature_names, "xgboost_feature_names.pkl")
print("Model and feature names saved successfully.")

