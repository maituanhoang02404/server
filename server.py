# server.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import tempfile
import time
from modules import text_remover, auto_script_generator, video_creator

app = FastAPI()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/api/process")
async def process_images(
    images: list[UploadFile] = File(...),
    ai_provider: str = Form("local"),
    api_key: str = Form("")
):
    try:
        session_id = str(int(time.time()))
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        original_folder = os.path.join(session_dir, "images_original")
        cleaned_folder = os.path.join(session_dir, "images_cleaned")
        output_video = os.path.join(session_dir, "final_video.mp4")
        output_srt = os.path.join(session_dir, "final_video.srt")
        os.makedirs(original_folder)
        os.makedirs(cleaned_folder)

        # Lưu ảnh
        for image in images:
            with open(os.path.join(original_folder, image.filename), "wb") as f:
                f.write(await image.read())

        # Xử lý ảnh
        text_remover.process_folder(original_folder, cleaned_folder, lambda msg: print(msg))

        # Tạo kịch bản
        script_content = auto_script_generator.create_full_script(
            original_folder, ai_provider, api_key, lambda msg: print(msg)
        )

        # Tạo video
        video_creator.build_video(
            cleaned_folder, script_content, None, output_video, lambda msg: print(msg)
        )

        return {
            "status": "success",
            "video_url": f"/api/download/{session_id}/final_video.mp4",
            "srt_url": f"/api/download/{session_id}/final_video.srt"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    file_path = os.path.join(UPLOAD_DIR, session_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/cleanup/{session_id}")
async def cleanup(session_id: str):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    return {"status": "cleaned"}
