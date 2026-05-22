import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# 1. Load the preprocessed dataset
data = pd.read_csv('processed_alerts.csv')

# Define features and target variable
X = data.drop(columns=['pest_alert'])
y = data['pest_alert']

# Split data (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the Decision Tree Model with defined hyperparameters
dt_model = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)

# Generate predictions
y_pred = dt_model.predict(X_test)

# 3. Evaluate and print metrics to console
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 4. Save the model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)
print("Model 'decision_tree_model.pkl' saved successfully.")

# 5. GENERATE VISUALIZATIONS
os.makedirs('static/images', exist_ok=True)

# Confusion Matrix visualization
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['No Alert', 'Alert'], yticklabels=['No Alert', 'Alert'])
plt.title('Decision Tree - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('static/images/decision_tree_confusion_matrix.png')
plt.close()

# Feature Importance visualization
importances = dt_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Decision Tree - Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('static/images/decision_tree_feature_importance.png')
plt.close()

print("Decision Tree charts saved in static/images/")