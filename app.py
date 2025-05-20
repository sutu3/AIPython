from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import traceback

# from sklearn.ensemble import IsolationForest # Không cần thiết cho việc load

app = Flask(__name__)

# Đường dẫn đến file model
model_path = 'isolation_forest_model.joblib'
model = None  # Biến này sẽ chứa model được tải

try:
    # --- CHỈ TẢI MODEL ---
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at {model_path}. Ensure it's in the correct path (root of your project, alongside app.py) and included in your Git repository.")
except Exception as e:
    print(f"❌ ERROR: Could not load model: {e}")
    traceback.print_exc()

# ... phần còn lại của code (required_features, các route @app.route, v.v...) ...
# (Giữ nguyên như phiên bản đúng ở câu trả lời trước)

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
    if model is not None: # Check if model was successfully loaded
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
        prediction_raw = model.predict(input_df)
        textual_prediction_label = "Anomaly" if prediction_raw[0] == -1 else "Normal"
        is_rug_pull = 1 if textual_prediction_label == "Anomaly" else 0
        anomaly_score_value = None
        if hasattr(model, "decision_function"):
            anomaly_score_value = model.decision_function(input_df)[0]

        return jsonify({
            'prediction_label': textual_prediction_label,
            'is_rug_pull_prediction': is_rug_pull,
            'anomaly_score': float(anomaly_score_value) if anomaly_score_value is not None else None,
            'raw_model_prediction': int(prediction_raw[0])
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))