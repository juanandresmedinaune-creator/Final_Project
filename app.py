from flask import Flask, render_template
import pandas as pd
import models

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

@app.route('/models/engineering')
def model_engineering():
    results = models.run_all_models()
    return render_template('modelEngineering.html', results=results)

@app.route('/models/development/logistic-regression')
def model_lr():
    result = models.run_logistic_regression()
    print("LOGISTIC RESULT ERROR:", result.get("error"))
    return render_template('modelDevelopment.html', model=result, active='lr')

@app.route('/models/development/random-forest')
def model_rf():
    result = {
        "name": "Random Forest",
        "icon": "🌲",
        "description": "This model has not been integrated yet.",
        "error": "run_random_forest() is not available in models.py yet.",
        "train_size": 0,
        "test_size": 0,
        "accuracy": 0,
        "cv_std": 0,
        "show_metrics": False
    }
    return render_template('modelDevelopment.html', model=result, active='rf')

@app.route('/models/development/gradient-boosting')
def model_gb():
    result = {
        "name": "Gradient Boosting",
        "icon": "🚀",
        "description": "This model has not been integrated yet.",
        "error": "run_gradient_boosting() is not available in models.py yet.",
        "train_size": 0,
        "test_size": 0,
        "accuracy": 0,
        "cv_std": 0,
        "show_metrics": False
    }
    return render_template('modelDevelopment.html', model=result, active='gb')

@app.route('/models/evaluation/logistic-regression')
def model_lr_evaluation():
    result = models.run_logistic_regression()
    print("LOGISTIC RESULT ERROR:", result.get("error"))
    return render_template('modelEvaluation.html', model=result)


# =========================
# APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)