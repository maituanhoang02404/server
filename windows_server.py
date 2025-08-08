# server.py - MTH Recap Server for Windows VPS
from flask import Flask, request, jsonify, send_file
import os
import uuid
import shutil
from PIL import Image
import cv2
import numpy as np
import torch
import re
import tempfile
import gc
from threading import Lock
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Thread-safe model loading
model_lock = Lock()
model_loaded = False
processor = None
model = None

def load_model():
    """Load AI model với error handling tốt cho Windows"""
    global model_loaded, processor, model
    
    if model_loaded:
        return True
        
    try:
        with model_lock:
            if not model_loaded:
                logger.info("Đang tải model AI...")
                
                # Import ở đây để tránh lỗi nếu transformers chưa cài
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Sử dụng device: {device}")
                
                # Load model với error handling
                processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
                model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                    device_map="auto" if device == "cuda" else None
                )
                
                if device == "cpu":
                    model = model.to(device)
                
                model_loaded = True
                logger.info("Model đã được tải thành công!")
                return True
                
    except Exception as e:
        logger.error(f"Lỗi khi tải model: {e}")
        return False

# Thư mục lưu trữ tạm (Windows compatible)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'temp_sessions')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def sort_key(filename):
    """Trích xuất số ở đầu tên file để sắp xếp."""
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def safe_remove_folder(folder_path):
    """Xóa folder an toàn trên Windows"""
    if os.path.exists(folder_path):
        try:
            # Windows thường lock file, thử vài lần
            for _ in range(3):
                try:
                    shutil.rmtree(folder_path)
                    break
                except PermissionError:
                    import time
                    time.sleep(1)
        except Exception as e:
            logger.warning(f"Không thể xóa folder {folder_path}: {e}")

def remove_text_bubbles(image):
    """Xóa ô thoại từ ảnh truyện tranh - tối ưu cho Windows"""
    try:
        # Chuyển PIL Image sang OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = img_array
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện vùng trắng (ô thoại)
        _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Tìm contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tạo mask cho các ô thoại lớn
        bubble_mask = np.zeros_like(gray)
        for contour in contours:
            if cv2.contourArea(contour) > 300:  # Giảm threshold cho ảnh nhỏ hơn
                cv2.drawContours(bubble_mask, [contour], -1, (255), thickness=cv2.FILLED)
        
        # Inpainting
        if np.any(bubble_mask):
            result = cv2.inpaint(img, bubble_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            return image
            
    except Exception as e:
        logger.warning(f"Lỗi xử lý ảnh: {e}")
        return image

def generate_narrative(image):
    """Tạo kịch bản từ ảnh sử dụng Blip-2"""
    if not model_loaded:
        if not load_model():
            return "Không thể tải model AI để tạo kịch bản."
    
    try:
        # Resize ảnh để tiết kiệm memory
        max_size = (512, 512)
        image.thumbnail(max_size, Image.LANCZOS)
        
        # Tạo prompt tiếng Việt
        prompt = "Mô tả ngắn gọn hành động và cảm xúc trong ảnh này:"
        
        inputs = processor(image, prompt, return_tensors="pt")
        
        # Di chuyển inputs đến device phù hợp
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=80,
                num_beams=3,
                temperature=0.8,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Làm sạch text
        narrative = generated_text.replace(prompt, "").strip()
        
        if not narrative or len(narrative) < 5:
            return "Một cảnh hành động trong truyện tranh."
        
        # Đảm bảo có nội dung tiếng Việt
        if not any(ord(char) > 127 for char in narrative):
            narrative = f"Cảnh này cho thấy: {narrative}"
        
        return narrative[:200]  # Giới hạn độ dài
        
    except Exception as e:
        logger.error(f"Lỗi tạo narrative: {e}")
        return "Một cảnh trong truyện tranh đang diễn ra."
    finally:
        # Dọn dẹp memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def create_simple_video(image_paths, narratives, output_path):
    """Tạo video đơn giản mà không cần Ken Burns effect để tiết kiệm RAM"""
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips
        
        clips = []
        srt_content = []
        current_time = 0
        
        for i, (image_path, narrative) in enumerate(zip(image_paths, narratives)):
            try:
                # Tạo clip đơn giản
                clip = ImageClip(image_path, duration=4).resize((1280, 720)).set_fps(24)
                clips.append(clip)
                
                # Tạo subtitle
                start_time = f"{current_time//3600:02d}:{(current_time%3600)//60:02d}:{current_time%60:02d},000"
                end_time = f"{(current_time+4)//3600:02d}:{((current_time+4)%3600)//60:02d}:{(current_time+4)%60:02d},000"
                srt_content.append(f"{i+1}\n{start_time} --> {end_time}\n{narrative}\n")
                current_time += 4
                
            except Exception as e:
                logger.warning(f"Lỗi tạo clip cho {image_path}: {e}")
                continue
        
        if not clips:
            raise ValueError("Không thể tạo clip nào")
        
        # Ghép video
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Export với settings tối ưu cho Windows
        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
            threads=2  # Giới hạn threads để tránh quá tải RAM
        )
        
        # Cleanup
        final_video.close()
        for clip in clips:
            clip.close()
        
        return srt_content
        
    except Exception as e:
        logger.error(f"Lỗi tạo video: {e}")
        raise

@app.route('/api/process', methods=['POST'])
def process_images():
    try:
        # Tạo session ID
        session_id = str(uuid.uuid4())[:8]  # Rút ngắn để Windows dễ xử lý
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        logger.info(f"Tạo session: {session_id}")
        
        # Lưu ảnh upload
        uploaded_files = request.files.getlist('images')
        if not uploaded_files:
            return jsonify({'error': 'Không có ảnh nào được upload'}), 400
        
        image_paths = []
        narratives = []
        
        # Xử lý từng ảnh
        for file in uploaded_files:
            if file.filename == '':
                continue
                
            try:
                # Lưu ảnh gốc
                filename = file.filename
                filepath = os.path.join(session_folder, filename)
                file.save(filepath)
                
                # Mở và xử lý ảnh
                image = Image.open(filepath).convert('RGB')
                
                # Xóa ô thoại
                cleaned_image = remove_text_bubbles(image)
                cleaned_filename = f"cleaned_{filename}"
                cleaned_path = os.path.join(session_folder, cleaned_filename)
                cleaned_image.save(cleaned_path, quality=85)
                
                # Tạo kịch bản
                logger.info(f"Tạo kịch bản cho: {filename}")
                narrative = generate_narrative(cleaned_image)
                narratives.append(narrative)
                image_paths.append(cleaned_path)
                
            except Exception as e:
                logger.error(f"Lỗi xử lý file {file.filename}: {e}")
                continue
        
        if not image_paths:
            return jsonify({'error': 'Không thể xử lý ảnh nào'}), 400
        
        # Sắp xếp theo số thứ tự
        paired_data = list(zip(image_paths, narratives))
        paired_data.sort(key=lambda x: sort_key(os.path.basename(x[0])))
        image_paths, narratives = zip(*paired_data) if paired_data else ([], [])
        
        # Tạo video
        video_path = os.path.join(session_folder, "output.mp4")
        logger.info("Bắt đầu tạo video...")
        
        srt_content = create_simple_video(list(image_paths), list(narratives), video_path)
        
        # Lưu subtitle
        srt_path = os.path.join(session_folder, "output.srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_content))
        
        logger.info(f"Hoàn thành session: {session_id}")
        
        return jsonify({
            'message': 'Xử lý thành công',
            'session_id': session_id,
            'video_url': f'/download/{session_id}/output.mp4',
            'srt_url': f'/download/{session_id}/output.srt'
        })
        
    except Exception as e:
        logger.error(f"Lỗi xử lý: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, session_id, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File không tồn tại'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    try:
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        safe_remove_folder(session_folder)
        return jsonify({'message': 'Dọn dẹp thành công'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'OK',
        'model_loaded': model_loaded,
        'platform': 'Windows',
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        'upload_folder': UPLOAD_FOLDER
    })

@app.route('/')
def index():
    return """
    <h1>MTH Recap Server</h1>
    <p>Server đang chạy thành công!</p>
    <p><a href="/health">Kiểm tra trạng thái</a></p>
    """

if __name__ == '__main__':
    print("=== MTH RECAP SERVER ===")
    print(f"Thư mục làm việc: {os.getcwd()}")
    print(f"Thư mục upload: {UPLOAD_FOLDER}")
    
    # Load model khi khởi động (tùy chọn)
    print("Đang tải model AI...")
    if load_model():
        print("Model đã sẵn sàng!")
    else:
        print("Cảnh báo: Model chưa được tải. Sẽ tải khi có request đầu tiên.")
    
    print("Server đang khởi động tại http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
