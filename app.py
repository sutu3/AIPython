from flask import Flask, request, jsonify
import joblib
import pandas as pd # Hoặc numpy, tùy cách bạn xử lý input cho model
import os

app = Flask(__name__)

# Đường dẫn đến model của bạn (điều chỉnh nếu cần)
# Giả sử model_path nằm cùng cấp với app.py, hoặc trong 1 thư mục con
# Ví dụ: model_path = 'RugPullDetectionModel/isolation_forest_model.joblib'
# Hoặc nếu file model nằm cùng cấp app.py:
model_path = 'isolation_forest_model.joblib'

model = None
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {model_path}. Ensure it's in the correct path and included in your Git repository.")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")

@app.route('/')
def home():
    return "AI Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded or failed to load. Check server logs.'}), 500

    try:
        # Lấy dữ liệu JSON từ request
        data = request.get_json(force=True)

        # === XỬ LÝ INPUT DATA ===
        # Dữ liệu 'data' cần được chuyển đổi thành định dạng mà mô hình của bạn mong đợi.
        # Ví dụ: nếu mô hình của bạn được train trên Pandas DataFrame với các cột cụ thể:
        # features_df = pd.DataFrame([data]) # Giả sử 'data' là một dict {'feature1': val1, ...}

        # Quan trọng: Đảm bảo các features đầu vào (tên cột, thứ tự, kiểu dữ liệu)
        # PHẢI GIỐNG HỆT như lúc bạn train model.
        # Ví dụ, nếu bạn train model với các cột sau:
        # ['TOTAL_ADDED_LIQUIDITY', 'TOTAL_REMOVED_LIQUIDITY', ..., 'INACTIVITY_STATUS_Inactive']
        # thì 'data' phải chứa các key này.

        # Ví dụ đơn giản: giả sử data là một list các giá trị feature theo đúng thứ tự
        # feature_values = [data['feature1'], data['feature2'], ...]
        # prediction_input = [feature_values] # model.predict thường nhận 2D array

        # >>> THAY THẾ PHẦN NÀY BẰNG LOGIC XỬ LÝ INPUT CỤ THỂ CỦA BẠN <<<
        # Ví dụ:
        # required_features = ['TOTAL_ADDED_LIQUIDITY', 'TOTAL_REMOVED_LIQUIDITY', ..., 'INACTIVITY_STATUS_Inactive']
        # input_data_for_model = {}
        # for feature_name in required_features:
        #     if feature_name not in data:
        #         return jsonify({'error': f'Missing feature: {feature_name}'}), 400
        #     input_data_for_model[feature_name] = data[feature_name]

        # features_df = pd.DataFrame([input_data_for_model])
        # prediction = model.predict(features_df)
        # proba = model.predict_proba(features_df) # Nếu model hỗ trợ và bạn cần xác suất

        # Giả sử bạn đã có PredictResponse từ Java trả về JSON đúng các feature cần thiết
        features_df = pd.DataFrame([data]) # data là JSON object từ request
        prediction = model.predict(features_df)

        # Chuyển đổi kết quả dự đoán (thường là numpy array) sang list để jsonify
        result = prediction.tolist()

        return jsonify({'prediction': result})

    except KeyError as e:
        return jsonify({'error': f'Missing key in input data: {str(e)}'}), 400
    except Exception as e:
        print(f"Error during prediction: {e}") # Log lỗi ra console của server
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # Render sẽ cung cấp PORT qua biến môi trường
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) # debug=False cho production