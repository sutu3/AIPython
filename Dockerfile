# Chọn base image Python
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# Expose port
EXPOSE 5000

# Chạy Flask app qua gunicorn (production)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4"]
