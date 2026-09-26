import os, sys, glob, shutil

print("Current Python:", sys.executable)
print("Sys path:", sys.path)

# Check for virtual environments
candidates = [
    r"F:\projects\adtech-video-pipeline\.venv\Scripts\python.exe",
    r"F:\projects\adtech-video-pipeline\venv\Scripts\python.exe",
    r"F:\projects\adtech-video-pipeline\env\Scripts\python.exe",
    r"F:\projects\hoichoi\.venv\Scripts\python.exe",
    r"F:\projects\hoichoi\venv\Scripts\python.exe",
    r"C:\Users\Lenovo\miniconda3\python.exe",
    r"C:\Users\Lenovo\anaconda3\python.exe",
]
for c in candidates:
    if os.path.exists(c):
        print("Found environment:", c)

# Check installed packages with pip list in current python
try:
    import subprocess
    res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if any(k in line.lower() for k in ["scene", "whisper", "torch", "clip", "cv2", "opencv", "pytest", "yaml"]):
            print("Pip package:", line)
except Exception as e:
    print("Pip check error:", e)
