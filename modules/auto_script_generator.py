# modules/auto_script_generator.py
import os
import re
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Cấu hình YOLOv5-tiny
MODEL_PATH = "models/yolov5s.onnx"
CONFIDENCE_THRESHOLD = 0.5

def sort_key(filename):
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_yolo_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session

def detect_objects(image_path, session):
    img = cv2.imread(image_path)
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # YOLOv5-tiny yêu cầu kích thước 640x640
    img = img.transpose((2, 0, 1))  # CHW format
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img})

    # Giả định đầu ra là bounding boxes, scores, classes (cần điều chỉnh dựa trên YOLO output)
    detections = outputs[0]
    boxes = detections[0][:, :4]  # x1, y1, x2, y2
    scores = detections[0][:, 4]
    classes = detections[0][:, 5].astype(np.int32)

    detected_objects = []
    for box, score, cls in zip(boxes, scores, classes):
        if score > CONFIDENCE_THRESHOLD:
            detected_objects.append({"class": cls, "score": score})
    return detected_objects

def generate_narrative_rule_based(image_path, image_index, detected_objects):
    if not detected_objects:
        return f"Cảnh {image_index + 1}: Một khoảnh khắc bí ẩn!"
    obj_names = {0: "người", 1: "vật thể", 2: "vũ khí"}  # Tùy chỉnh danh sách lớp theo YOLO
    objects = [obj_names.get(obj["class"], "đối tượng") for obj in detected_objects]
    return f"Cảnh {image_index + 1}: {', '.join(objects)} xuất hiện trong trận chiến kịch tính!"

def create_full_script(image_folder, ai_provider, api_key, log_callback):
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
                         key=sort_key)
    session = load_yolo_model()
    full_script = []
    for index, filename in enumerate(image_files):
        image_path = os.path.join(image_folder, filename)
        log_callback(f"Đang tạo kịch bản cho ảnh: {filename}...")
        detected_objects = detect_objects(image_path, session)
        narrative = generate_narrative_rule_based(image_path, index, detected_objects)
        full_script.append(narrative)
        log_callback(f"   Kịch bản: {narrative}")
    return "\n---\n".join(full_script)
