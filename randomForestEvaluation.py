import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# =========================================
# LOAD DATA
# =========================================

data = pd.read_csv("processed_alerts.csv")

X = data.drop(columns=['pest_alert'])
y = data['pest_alert']

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# LOAD MODEL
# =========================================

model = joblib.load("random_forest_model.pkl")

# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# =========================================
# METRICS
# =========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n=== RANDOM FOREST METRICS ===")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# =========================================
# CROSS VALIDATION
# =========================================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='f1'
)

print("\nCross Validation Scores:")
print(cv_scores)

print("CV Mean:", cv_scores.mean())

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - Random Forest")

plt.xlabel("Predicted")
plt.ylabel("Real")

plt.tight_layout()

plt.savefig("static/random_forest_confusion.png")

# =========================================
# ROC CURVE
# =========================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr)

plt.plot([0,1],[0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")

plt.tight_layout()

plt.savefig("static/random_forest_roc.png")

# =========================================
# METRICS GRAPH
# =========================================

metrics_names = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1 Score'
]

metrics_values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(8,5))

plt.bar(metrics_names, metrics_values)

plt.ylim(0,1)

plt.title("Random Forest Metrics")

plt.tight_layout()

plt.savefig("static/random_forest_metrics.png")

# =========================================
# PREDICTIONS TABLE
# =========================================

predictions_df = pd.DataFrame({
    'Real Value': y_test.values[:10],
    'Predicted Value': y_pred[:10]
})

predictions_df['Status'] = (
    predictions_df['Real Value']
    ==
    predictions_df['Predicted Value']
)

predictions_df['Status'] = predictions_df['Status'].map({
    True: 'Correct Prediction',
    False: 'Prediction Error'
})

predictions_df.to_csv(
    "random_forest_predictions.csv",
    index=False
)

print("\nPredictions saved.")

print(predictions_df)