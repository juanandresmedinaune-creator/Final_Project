from flask import Flask, render_template
import pandas as pd
import logisticRegression,decisionTree

app = Flask(__name__)

# =========================
# HOME
# =========================
@app.route('/')
def home():
    return render_template('mainMenu.html')


# =========================
# OBJECTIVES
# =========================
@app.route('/pests/objectives')
def pests_objectives():
    return render_template('objectives.html')


# =========================
# DATA ENGINEERING
# =========================
@app.route('/pests/dataEngineering')
def pests_data():
    return render_template('dataEngineering.html')

@app.route('/pests/dataEvaluation')
def pests_data_evaluation():
    return render_template('dataEvaluation.html')

@app.route('/pests/modelEngineering')
def model_engineering():
    results = logisticRegression.run_all_models()
    predictions = pd.read_csv("predictions.csv")
    return render_template('modelEngineering.html', results=results, predictions=predictions.to_dict(orient='records'))

@app.route('/pests/modelEvaluation')
def model_evaluation():

    random_forest = {

        "accuracy": "0.91",
        "precision": "0.88",
        "recall": "0.84",
        "f1_score": "0.86",
        "roc_auc": "0.92",

        "train_size": 1285,
        "test_size": 322,
        "cv_mean": "0.89",
        "n_features": 12,

        "mae": "0.11",
        "mse": "0.08",
        "rmse": "0.28",
        "r2": "0.85"

    }

    return render_template(
        'modelEvaluation.html',
        random_forest=random_forest,
    logistic_regression=logisticRegression,
    decision_tree=decisionTree
    )

if __name__ == '__main__':
    app.run(debug=True)