import os
import base64
import json
import requests # Thư viện để gọi API của Ollama
import configparser # Thư viện để đọc file config.ini
import re 

# --- ĐỌC CẤU HÌNH ---
config = configparser.ConfigParser()
config.read('config.ini')

AI_PROVIDER = config.get('AI_Settings', 'provider', fallback='local')
OPENAI_API_KEY = config.get('AI_Settings', 'openai_api_key', fallback='')

# --- CÁC HÀM XỬ LÝ ---
def sort_key(filename):
    """Trích xuất số ở đầu tên file để sắp xếp."""
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def encode_image_to_base64(image_path):
    """Chuyển đổi file ảnh sang định dạng base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_narrative_openai(base64_image):
    """Gửi ảnh đến OpenAI API và nhận về lời dẫn chuyện."""
    from openai import OpenAI
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return "Lỗi: OpenAI API Key không hợp lệ."

    prompt_messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Bạn là người kể chuyện tóm tắt truyện tranh. Hãy nhìn vào ảnh và mô tả hành động, biểu cảm một cách ngắn gọn, kịch tính (tối đa 2 câu). Đừng mô tả chữ trong ô thoại."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4-vision-preview", messages=prompt_messages, max_tokens=150)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi khi gọi OpenAI API: {e}"

def generate_narrative_local(base64_image):
    """Gửi ảnh đến Ollama API (local) và nhận về lời dẫn chuyện."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",
        "prompt": "Mô tả hình ảnh này cho một video tóm tắt truyện tranh. Hãy viết một cách ngắn gọn và kịch tính, tập trung vào hành động và cảm xúc. Bỏ qua mọi văn bản trong ảnh.",
        "images": [base64_image],
        "stream": False # Nhận toàn bộ phản hồi trong một lần
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Báo lỗi nếu request thất bại
        response_data = response.json()
        return response_data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Lỗi: Không thể kết nối đến Ollama. Vui lòng đảm bảo Ollama đang chạy."
    except Exception as e:
        return f"Lỗi khi gọi Ollama API: {e}"

def generate_narrative_for_image(image_path):
    """Hàm trung gian để chọn nhà cung cấp AI phù hợp."""
    base64_image = encode_image_to_base64(image_path)
    
    if AI_PROVIDER == 'openai' and OPENAI_API_KEY:
        print("-> Sử dụng OpenAI API (Trả phí)...")
        return generate_narrative_openai(base64_image)
    else:
        print("-> Sử dụng Local AI (Miễn phí)...")
        return generate_narrative_local(base64_image)

def create_full_script(image_folder, ai_provider, api_key, log_callback):
    # THAY ĐỔI Ở ĐÂY: Sử dụng hàm sort_key mới
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
                         key=sort_key)

    full_script = []
    
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        print(f"Đang tạo kịch bản cho ảnh: {filename}...")
        narrative = generate_narrative_for_image(image_path)
        full_script.append(narrative)
        print(f"   Kịch bản: {narrative}")

    final_script_content = "\n---\n".join(full_script)
    
    with open(output_script_file, 'w', encoding='utf-8') as f:
        f.write(final_script_content)
    
    print(f"\nKịch bản tự động đã được lưu tại: {output_script_file}")