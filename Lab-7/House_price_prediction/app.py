from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input features from the form
        features = [
            float(request.form['sqft_living']),
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['floors']),
            float(request.form['sqft_above']),
            float(request.form['sqft_basement']),
            float(request.form['lat']),
            float(request.form['long']),
            float(request.form['house_age']),
            int(request.form['waterfront']),
            int(request.form['view']),
            int(request.form['condition']),
            int(request.form['grade']),
            int(request.form['is_renovated'])
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction:,.2f}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
