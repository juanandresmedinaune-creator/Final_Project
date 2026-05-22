from flask import Flask, render_template
import pandas as pd

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


# =========================
# MODEL ENGINEERING
# =========================
@app.route('/pests/modelEngineering')
def pests_model():

    # CSV GENERAL
    predictions = pd.read_csv("predictions.csv")

   
    return render_template(
        'modelEngineering.html',
        predictions=predictions.to_dict(orient='records'),
      
    )


# =========================
# APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)
