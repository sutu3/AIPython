from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import traceback # Moved import traceback to the top for convention

app = Flask(__name__)

# Đường dẫn đến file model
model_path = 'isolation_forest_model.joblib'
model = None
try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at {model_path}. Ensure it's in the correct path and included in your Git repository.")
except Exception as e:
    print(f"❌ ERROR: Could not load model: {e}")

# Danh sách các features mà model đã được huấn luyện và mong đợi ở đầu vào
# Đảm bảo thứ tự này khớp với thứ tự khi huấn luyện model.
required_features = [
    'TOTAL_ADDED_LIQUIDITY', 'TOTAL_REMOVED_LIQUIDITY',
    'NUM_LIQUIDITY_ADDS', 'NUM_LIQUIDITY_REMOVES', 'ADD_TO_REMOVE_RATIO',
    'LAST_POOL_ACTIVITY_TIMESTAMP_hour', 'LAST_POOL_ACTIVITY_TIMESTAMP_day',
    'LAST_POOL_ACTIVITY_TIMESTAMP_weekday',
    'LAST_POOL_ACTIVITY_TIMESTAMP_month',
    'FIRST_POOL_ACTIVITY_TIMESTAMP_hour',
    'FIRST_POOL_ACTIVITY_TIMESTAMP_day',
    'FIRST_POOL_ACTIVITY_TIMESTAMP_weekday',
    'FIRST_POOL_ACTIVITY_TIMESTAMP_month', 'LAST_SWAP_TIMESTAMP_hour',
    'LAST_SWAP_TIMESTAMP_day', 'LAST_SWAP_TIMESTAMP_weekday',
    'LAST_SWAP_TIMESTAMP_month', 'INACTIVITY_STATUS_Active',
    'INACTIVITY_STATUS_Inactive'
]

@app.route('/')
def home():
    if model:
        return "✅ AI Model API is running and model is loaded!"
    else:
        return "⚠️ AI Model API is running BUT THE MODEL FAILED TO LOAD. Check server logs."

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if model is None:
        return jsonify({'error': 'Model not loaded or failed to load. Check server logs.'}), 500

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid JSON data. Expected a dictionary.'}), 400

    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return jsonify({'error': f'Missing features in input: {", ".join(missing_features)}'}), 400

    # Ensure the order of features matches how the model was trained
    try:
        feature_values_in_order = [data[f] for f in required_features]
    except KeyError as e:
        return jsonify({'error': f'Missing feature in input: {str(e)}'}), 400


    input_df = pd.DataFrame([feature_values_in_order], columns=required_features)

    try:
        # Get raw prediction (-1 for anomaly, 1 for normal)
        prediction_raw = model.predict(input_df) # Returns an array, e.g., np.array([-1]) or np.array([1])

        # Determine textual label based on the notebook's logic:
        # "# -1 = Anomaly (bất thường), 1 = Normal (bình thường)"
        # print("Prediction:", "Anomaly" if prediction[0] == -1 else "Normal")
        textual_prediction_label = "Anomaly" if prediction_raw[0] == -1 else "Normal"

        # Determine the binary is_rug_pull flag
        # If textual_prediction_label is "Anomaly", then is_rug_pull is 1. Otherwise, it's 0.
        is_rug_pull = 1 if textual_prediction_label == "Anomaly" else 0

        # Get anomaly score (lower scores are more anomalous for IsolationForest's decision_function)
        # Note: The notebook image uses anomaly_score = best_model.decision_function(sample_df)
        # The decision_function of IsolationForest returns scores where more negative is more anomalous.
        # A common convention is that values close to -0.5 are anomalies, and values close to 0.5 are normal.
        # Your previous print of "Anomaly Score: 0.1310..." with "Prediction: Normal" suggests your
        # interpretation or thresholding might be specific.
        # The `predict` method's -1/1 output is based on the `contamination` parameter.
        # `decision_function` provides the raw scores.

        anomaly_score_value = None # Initialize variable for the score
        if hasattr(model, "decision_function"):
            # .decision_function usually returns an array of scores, one per sample
            anomaly_score_value = model.decision_function(input_df)[0]

        return jsonify({
            'prediction_label': textual_prediction_label, # NEW: Textual label like in the notebook print
            'is_rug_pull_prediction': is_rug_pull,        # Existing: 1 if Anomaly, 0 if Normal
            'anomaly_score': float(anomaly_score_value) if anomaly_score_value is not None else None, # Anomaly score from decision_function
            'raw_model_prediction': int(prediction_raw[0]) # Raw output from model.predict() (-1 or 1)
        })
    except Exception as e:
        traceback.print_exc() # Prints full traceback to server logs for debugging
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # Render will provide PORT via environment variable
    port = int(os.environ.get('PORT', 5000))
    # debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)