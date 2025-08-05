 

# text_remover.py
import cv2
import numpy as np
import os
import re

def sort_key(filename):
    """Trích xuất số ở đầu tên file để sắp xếp."""
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def remove_text_bubbles(image_path):
    """
    Phát hiện và xóa các ô thoại màu trắng khỏi ảnh truyện tranh.
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None

    # Chuyển sang ảnh xám để dễ xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sử dụng Threshold để phân tách các vùng màu trắng (ô thoại)
    # Các giá trị này có thể cần điều chỉnh tùy thuộc vào ảnh của bạn
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    # Đảo ngược mask: ô thoại sẽ màu đen, còn lại màu trắng
    mask_inv = cv2.bitwise_not(mask)
    
    # Tìm các đường viền của ô thoại
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo một mask trống để vẽ các đường viền đã tìm thấy
    bubble_mask = np.zeros_like(gray)
    
    # Vẽ các đường viền đủ lớn lên mask (loại bỏ các chấm nhiễu nhỏ)
    for contour in contours:
        if cv2.contourArea(contour) > 500: # Lọc theo diện tích
            cv2.drawContours(bubble_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Làm dày mask một chút để đảm bảo xóa hết viền đen của ô thoại
    kernel = np.ones((10, 10), np.uint8)
    bubble_mask = cv2.dilate(bubble_mask, kernel, iterations=1)

    # Sử dụng thuật toán inpainting để lấp đầy vùng bị xóa
    result = cv2.inpaint(img, bubble_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return result

def process_folder(input_folder, output_folder, log_callback):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # THAY ĐỔI Ở ĐÂY: Sử dụng hàm sort_key mới
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
                         key=sort_key)
    
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        log_callback(f"  - Đang xóa ô thoại từ: {filename}") # <-- SỬ DỤNG NÓ
        cleaned_image = remove_text_bubbles(input_path)
        if cleaned_image is not None:
            cv2.imwrite(output_path, cleaned_image)

    

if __name__ == '__main__':
    # Ví dụ sử dụng
    INPUT_IMAGE_DIR = "My_Video_Project/images_original"
    OUTPUT_IMAGE_DIR = "My_Video_Project/images_cleaned"
    process_folder(INPUT_IMAGE_DIR, OUTPUT_IMAGE_DIR,) # <-- THÊM THAM SỐ VÀO ĐÂY
    