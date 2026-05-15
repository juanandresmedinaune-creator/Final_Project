from flask import Flask, render_template

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

if __name__ == '__main__':
    app.run(debug=True)