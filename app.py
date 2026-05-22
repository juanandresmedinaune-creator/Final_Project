from flask import Flask, render_template
import pandas as pd
import logisticRegression

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

if __name__ == '__main__':
    app.run(debug=True)