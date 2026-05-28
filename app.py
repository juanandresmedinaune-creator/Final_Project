from flask import Flask, render_template, request
import pandas as pd
import logisticRegression,decisionTree, os
from logisticRegression import run_logistic_regression
from predictionSystem import predict_pest_risk

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

    logistic_regression = run_logistic_regression()

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

    decision_tree = {

        "accuracy": "0.84",
        "precision": "0.83",
        "recall": "0.82",
        "f1_score": "0.82",

        "train_size": 1285,
        "test_size": 322,
        "n_features": 14,
        "n_classes": 2,

        "hyperparams": {
            "criterion": "gini",
            "max_depth": 5,
            "random_state": 42,
            "class_weight": "balanced"

        }
    }

    return render_template(
        'modelEvaluation.html',
        random_forest=random_forest,
        logistic_regression=logistic_regression,
        decision_tree=decision_tree
    )

@app.route('/pests/systemPrediction', methods=['GET','POST'])
def view_random_forest_prediction():

    if request.method == "POST":

        result = predict_pest_risk(
            request.form
        )

        return render_template(
            "systemPrediction.html",
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            description=result.get("description"),
            probabilities=result.get("probabilities"),
            error=result.get("error")
        )

    return render_template(
        "systemPrediction.html"
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)