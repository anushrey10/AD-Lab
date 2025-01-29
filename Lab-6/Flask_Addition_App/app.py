from flask import Flask, render_template, request
from models.calculator import add_numbers, load_computations

app = Flask(__name__)

@app.route('/')
def index():
    # Load all previous computations
    computations = load_computations()
    return render_template('index.html', computations=computations)

@app.route('/add', methods=['POST'])
def add():
    # Get input from the user
    num1 = float(request.form['num1'])
    num2 = float(request.form['num2'])
    
    # Perform the addition and save the computation
    result = add_numbers(num1, num2)
    
    # Load updated computations
    computations = load_computations()
    
    return render_template('index.html', computations=computations)

if __name__ == '__main__':
    app.run(debug=True)