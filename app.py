from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

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
def predict_endpoint(): # Đổi tên hàm để tránh trùng với biến 'predict'
    global input_df, data
    if model is None:
        return jsonify({'error': 'Model not loaded or failed to load. Check server logs.'}), 500

    try:
        # Lấy dữ liệu JSON từ request
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON data. Expected a dictionary.'}), 400

        # Kiểm tra các feature bị thiếu
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features in input: {", ".join(missing_features)}'}), 400

        # Tạo DataFrame với đúng thứ tự các cột như khi huấn luyện
        # Dữ liệu đầu vào cho DataFrame phải là một list của các list (hoặc dicts)
        # Ở đây, data là một dict, chúng ta lấy các giá trị theo thứ tự của required_features
        try:
            feature_values_in_order = [data[f] for f in required_features]
        except KeyError as e:
            # Điều này không nên xảy ra nếu kiểm tra missing_features ở trên hoạt động đúng
            return jsonify({'error': f'Internal error: Key not found while preparing data: {str(e)}'}), 500

        input_df = pd.DataFrame([feature_values_in_order], columns=required_features)

        # Thực hiện dự đoán
        # model.predict() trả về 1 cho inlier, -1 cho outlier
        prediction_raw = model.predict(input_df)
        # Chuyển đổi: 1 nếu là outlier (-1), 0 nếu là inlier (1)
        is_rug_pull = 1 if prediction_raw[0] == -1 else 0

        # Lấy anomaly score (tùy chọn, nhưng hữu ích)
        # decision_function trả về score, giá trị càng âm, càng có khả năng là outlier
        anomaly_score = model.decision_function(input_df)

        return jsonify({
            'is_rug_pull_prediction': is_rug_pull, # 1 = Rug Pull (outlier), 0 = Not Rug Pull (inlier)
            'anomaly_score': float(anomaly_score[0]),
            'raw_model_prediction': int(prediction_raw[0]) # Giữ lại giá trị gốc của model nếu cần
        })

    except TypeError as e:
        # Thường xảy ra nếu dữ liệu đầu vào không đúng kiểu (ví dụ: string thay vì number)
        print(f"❌ TypeError during prediction: {e}. Input data: {data}")
        return jsonify({'error': f'TypeError during prediction. Check data types. Details: {str(e)}'}), 400
    except ValueError as e:
        # Thường xảy ra nếu có NaN hoặc kiểu dữ liệu không hợp lệ trong input_df
        print(f"❌ ValueError during prediction: {e}. Input DataFrame head:\n{input_df.head().to_string() if 'input_df' in locals() else 'N/A'}")
        return jsonify({'error': f'ValueError during prediction. Check for NaNs or invalid data. Details: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ An unexpected error occurred during prediction: {e}")
        import traceback
        traceback.print_exc() # In đầy đủ stack trace ra log server
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Render sẽ cung cấp PORT qua biến môi trường
    port = int(os.environ.get('PORT', 5000))
    # debug=False cho production
    app.run(host='0.0.0.0', port=port, debug=False)