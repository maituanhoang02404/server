#!/bin/bash

echo "Starting server setup..."

# Cài Python (nếu chưa có, Deepnote thường đã có Python)
python3 -m venv venv
source venv/bin/activate

# Cập nhật pip và cài các thư viện
pip install --upgrade pip
pip install fastapi uvicorn python-multipart moviepy opencv-python Pillow numpy requests

# Tải mã nguồn (giả sử bạn upload thủ công hoặc dùng GitHub)
if [ ! -d "server_files" ]; then
    echo "Extracting server files..."
    unzip server_files.zip -d .
    mv server_files/* .
    rm -rf server_files server_files.zip
fi

# Cài FFmpeg (tùy chọn, cần kiểm tra hỗ trợ của Deepnote)
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Please install manually or skip if not needed."
    # Thêm lệnh cài FFmpeg nếu Deepnote hỗ trợ (ví dụ: apt-get install ffmpeg)
fi

# Chạy server
echo "Starting server..."
uvicorn server:app --host 0.0.0.0 --port 5000 --reload &

echo "Setup completed! Server is running at http://localhost:5000"
