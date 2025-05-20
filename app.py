from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("isolation_forest_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features")
    if features is None:
        return jsonify({"error": "Missing features"}), 400

    # Chuyển features thành numpy array, ví dụ:
    X = np.array(features).reshape(1, -1)

    prediction = model.predict(X)[0]
    score = model.decision_function(X)[0]

    return jsonify({"prediction": int(prediction), "score": float(score)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
