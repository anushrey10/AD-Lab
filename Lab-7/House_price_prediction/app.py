import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# Flask API Setup
app = Flask(__name__,static_folder='static')

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft_living = float(request.form['sqft_living'])
    bedrooms = float(request.form['bedrooms'])
    
    # Scale the input features
    scaled_features = scaler.transform([[sqft_living, bedrooms]])
    prediction = model.predict(scaled_features)
    
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)