# install_mth_server.ps1
Write-Host "=== CÀI ĐẶT MTH RECAP SERVER TRÊN WINDOWS ===" -ForegroundColor Green

# Kiểm tra quyền Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "Vui lòng chạy script này với quyền Administrator!" -ForegroundColor Red
    exit 1
}

# Tạo thư mục dự án
$ProjectPath = "C:\MTH_Server"
if (!(Test-Path $ProjectPath)) {
    New-Item -ItemType Directory -Path $ProjectPath
}
Set-Location $ProjectPath

Write-Host "Thư mục dự án: $ProjectPath" -ForegroundColor Yellow

# Kiểm tra Python
try {
    $pythonVersion = python --version
    Write-Host "Python đã cài đặt: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python chưa được cài đặt. Đang tải xuống..." -ForegroundColor Yellow
    
    # Tải Python
    $pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller
    
    Write-Host "Đang cài đặt Python..." -ForegroundColor Yellow
    Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait
    
    # Làm mới PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Remove-Item $pythonInstaller
}

# Tạo virtual environment
Write-Host "Tạo virtual environment..." -ForegroundColor Yellow
python -m venv mth_env
& ".\mth_env\Scripts\Activate.ps1"

# Cập nhật pip
Write-Host "Cập nhật pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Cài đặt PyTorch CPU (tiết kiệm RAM)
Write-Host "Cài đặt PyTorch..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Gỡ protobuf cũ và cài lại bản ổn định
Write-Host "Gỡ protobuf cũ (nếu có) và cài protobuf 3.20.3..." -ForegroundColor Yellow
pip uninstall protobuf -y
pip install protobuf==3.20.3

# Cài đặt các thư viện cần thiết
Write-Host "Cài đặt các thư viện AI và xử lý ảnh..." -ForegroundColor Yellow
pip install transformers accelerate bitsandbytes
pip install opencv-python pillow moviepy
pip install flask waitress requests

# Xóa cache model nếu cần
$cachePath = "$env:USERPROFILE\.cache\huggingface"
if (Test-Path $cachePath) {
    Write-Host "Đã phát hiện cache model cũ. Đang xóa..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $cachePath
}

# Tải sẵn model (tùy chọn)
Write-Host "Bạn có muốn tải sẵn model AI không? (y/n): " -NoNewline -ForegroundColor Cyan
$downloadModel = Read-Host

if ($downloadModel -eq "y" -or $downloadModel -eq "Y") {
    Write-Host "Đang tải model AI (có thể mất vài phút)..." -ForegroundColor Yellow
    python -c @"
from transformers import BlipProcessor, BlipForConditionalGeneration
print('Downloading model...')
BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
BlipForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
print('Model downloaded successfully!')
"@
}

# Tạo file batch để chạy server
$startServerBat = @"
@echo off
cd /d "$ProjectPath"
call mth_env\Scripts\activate.bat
python server.py
pause
"@

Set-Content -Path "start_server.bat" -Value $startServerBat

# Tạo file cấu hình firewall
Write-Host "Cấu hình Windows Firewall..." -ForegroundColor Yellow
try {
    New-NetFirewallRule -DisplayName "MTH Server" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
    Write-Host "Đã mở port 5000 trên firewall" -ForegroundColor Green
} catch {
    Write-Host "Không thể cấu hình firewall tự động. Vui lòng mở port 5000 thủ công." -ForegroundColor Red
}

Write-Host ""
Write-Host "=== CÀI ĐẶT HOÀN TẤT ===" -ForegroundColor Green
Write-Host "File server.py cần được đặt trong thư mục: $ProjectPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "Cách chạy server:" -ForegroundColor Cyan
Write-Host "1. Chạy file: start_server.bat" -ForegroundColor White
Write-Host "2. Hoặc chạy trực tiếp: python server.py" -ForegroundColor White
Write-Host ""
Write-Host "Server sẽ chạy tại: http://your-server-ip:5000" -ForegroundColor Green
Write-Host "Kiểm tra trạng thái: http://your-server-ip:5000/health" -ForegroundColor Green
