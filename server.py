from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import shutil
import tempfile

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
    # Logic xử lý (tạm thời bỏ qua để kiểm tra)
    return {"message": "Processing started"}

@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    file_path = os.path.join(UPLOAD_DIR, session_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return {"error": "File not found"}

@app.post("/api/cleanup/{session_id}")
async def cleanup(session_id: str):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    return {"status": "cleaned"}
