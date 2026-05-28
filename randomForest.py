import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv("processed_alerts.csv")

X = data.drop(columns=['pest_alert'])
y = data['pest_alert']

categorical_cols = [
    "department",
    "crop",
    "year",
    "season",
    "pest_name",
    "species"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

model_fitted = pipeline.named_steps["model"]
importance = model_fitted.feature_importances_

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(10, 6))

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

joblib.dump(pipeline, "random_forest_model.pkl")

print("\nModel saved successfully.")