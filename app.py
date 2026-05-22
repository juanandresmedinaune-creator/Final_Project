from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('mainMenu.html')

@app.route('/pests/objectives')
def pests_objectives():
    return render_template('objectives.html')

@app.route('/pests/dataEngineering')
def pests_data():
    return render_template('dataEngineering.html')

@app.route('/pests/modelEngineering')
def pests_model():
    predictions = pd.read_csv("predictions.csv")
    return render_template('modelEngineering.html',
                           predictions = predictions.to_dict(orient='records'))

@app.route('/models/development/decision-tree')
def decision_tree_development():
    return render_template('decisionTreeDevelopment.html')

if __name__ == '__main__':
    app.run(debug=True)