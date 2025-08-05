# modules/video_creator.py
import os
from moviepy.editor import *
from PIL import Image
import numpy as np
import re


def sort_key(filename):
    """Trích xuất số ở đầu tên file để sắp xếp."""
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def create_ken_burns_clip(image_path, duration, size):
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    
    img_w, img_h = pil_img.size
    scale_ratio = size[1] / img_h
    new_w, new_h = int(img_w * scale_ratio), size[1]
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    if new_w < size[0]:
        new_img = Image.new('RGB', size, (0, 0, 0))
        new_img.paste(pil_img, ((size[0] - new_w) // 2, 0))
        pil_img = new_img

    img_array = np.array(pil_img)
    zoom_ratio = 1.15

    def make_frame(t):
        current_zoom = 1 + (zoom_ratio - 1) * (t / duration)
        zoomed_w = int(size[0] * current_zoom)
        zoomed_h = int(size[1] * current_zoom)
        zoomed_img = Image.fromarray(img_array).resize((zoomed_w, zoomed_h), Image.LANCZOS)
        x_pan = (zoomed_w - size[0]) * (t / duration)
        y_center = (zoomed_h - size[1]) // 2
        box = (int(x_pan), int(y_center), int(x_pan + size[0]), int(y_center + size[1]))
        frame = zoomed_img.crop(box)
        return np.array(frame)

    return VideoClip(make_frame, duration=duration).set_fps(24)

def parse_script(script_content):
    return [part.strip() for part in script_content.split('---')]

def format_time(seconds):
    millisec = int((seconds - int(seconds)) * 1000)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{millisec:03d}"

def build_video(cleaned_images_folder, script_content, music_file, output_path, log_callback):
    VIDEO_SIZE = (1280, 720)
    IMAGE_DURATION = 5
    FPS = 24

    log_callback(">>> Bắt đầu dựng các cảnh video...")
    image_files = sorted([f for f in os.listdir(cleaned_images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
                         key=sort_key)
    
    scripts = parse_script(script_content)

    clips = []
    srt_content = []
    current_time = 0.0

    if len(image_files) != len(scripts):
        log_callback(f"Cảnh báo: Số lượng ảnh ({len(image_files)}) và kịch bản ({len(scripts)}) không khớp. Phụ đề có thể bị lệch.")

    for i, filename in enumerate(image_files):
        log_callback(f"  - Đang tạo cảnh cho ảnh: {filename}")
        image_path = os.path.join(cleaned_images_folder, filename)
        image_clip = create_ken_burns_clip(image_path, IMAGE_DURATION, VIDEO_SIZE)
        if image_clip:
            clips.append(image_clip)
            start_time = format_time(current_time)
            end_time = format_time(current_time + IMAGE_DURATION)
            script_text = scripts[i].replace('\n', ' ') if i < len(scripts) else ""
            srt_content.append(f"{i+1}\n{start_time} --> {end_time}\n{script_text}\n")
            current_time += IMAGE_DURATION

    if not clips:
        raise ValueError("Không có cảnh nào được tạo. Vui lòng kiểm tra lại thư mục ảnh đã xử lý.")

    log_callback(">>> Đang ghép nối các cảnh...")
    final_video = concatenate_videoclips(clips, method="compose")

    if music_file and os.path.exists(music_file):
        log_callback(">>> Đang thêm nhạc nền...")
        audioclip = AudioFileClip(music_file)
        if audioclip.duration > final_video.duration:
            audioclip = audioclip.subclip(0, final_video.duration)
        final_video = final_video.set_audio(audioclip)

    srt_path = os.path.splitext(output_path)[0] + ".srt"
    log_callback(f">>> Đang ghi file phụ đề: {os.path.basename(srt_path)}")
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_content))
    
    log_callback(">>> Đang render video cuối cùng (bước này có thể mất nhiều thời gian)...")
    final_video.write_videofile(output_path, fps=FPS, codec="libx264", audio_codec="aac", logger=None, threads=os.cpu_count())