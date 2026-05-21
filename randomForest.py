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
print("Testing rows: ", X_train.shape[0])

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

#Predictions

y_pred = model.predict(X_test)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

plt.barh(
    importance['Feature'],
    importance['Importance']
)

plt.xlabel("Importance")
plt.ylabel("Features")

plt.title("Feature Importance - Random Forest")

plt.tight_layout()

plt.savefig("static/feature_importance.png")

print("\nFeature importance graph saved.")

#Save Model

joblib.dump(model, "random_forest_model.pkl")
