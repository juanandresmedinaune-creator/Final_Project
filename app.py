from flask import Flask, render_template
import models

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

@app.route('/models/engineering')
def model_engineering():
    results = models.run_all_models()
    return render_template('modelEngineering.html', models=results)

@app.route('/models/development/logistic-regression')
def model_lr():
    result = models.run_logistic_regression()
    if not result:
        return "No se pudo generar el modelo. Revisa processed_alerts.csv y el target.", 500
    return render_template('modelDevelopment.html', model=result, active='lr')

if __name__ == '__main__':
    app.run(debug=True)