import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("processed_alerts.csv")

X = data.drop(columns=['pest_alert'])
y = data['pest_alert']

X_train, X_test, y_train, y_test = train_test_split (
    X,y,test_size=0.2, random_state=42
)

print("Training rows: ", X_train.shape[0])
print("Testing rows: ", X_test.shape[0])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

#Predictions

y_pred = model.predict(X_test)

predictions_df = pd.DataFrame({
    "real_value": y_test.values[:10],
    "predicted_value": y_pred[:10]
})

predictions_df["status"] = (
    predictions_df["real_value"] ==
    predictions_df["predicted_value"]
)

predictions_df["status"] = predictions_df["status"].map({
    True: "Correct Prediction",
    False: "Prediction Error"
})

predictions_df.to_csv(
    "predictions.csv",
    index=False
)

print("\nExample Predictions:")

examples = pd.DataFrame({
    "Real Value": y_test[:10].values,
    "Predicted Value": y_pred[:10]
})

print(examples)

importance = model.feature_importances_

features = X.columns

importance_df= pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=True
)

plt.figure(figsize=(10,6))

plt.barh(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Importance")
plt.ylabel("Features")

plt.title("Feature Importance - Random Forest")

plt.tight_layout()

plt.savefig("static/feature_importance.png")

print("\nFeature importance graph saved.")
print(data['pest_alert'].value_counts())

#Save Model

joblib.dump(model, "random_forest_model.pkl")
