from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # Serve the frontend

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if "SqFt" not in data:
        return jsonify({"error": "SqFt value is required"}), 400
    
    try:
        sq_ft = float(data["SqFt"])
        prediction = model.predict(np.array([[sq_ft]]))
        return jsonify({"SqFt": sq_ft, "Predicted Price": round(prediction[0], 2)})
    
    except ValueError:
        return jsonify({"error": "Invalid input. Please provide a numeric value for SqFt."}), 400

if __name__ == "__main__":
    app.run(debug=True)
