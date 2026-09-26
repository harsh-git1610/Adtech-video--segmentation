import os
from pathlib import Path

cache_dir = Path(r"C:\Users\Lenovo\.cache\huggingface\hub\models--Systran--faster-whisper-medium")
if cache_dir.exists():
    total_size = 0
    for f in cache_dir.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            print(f"{f.name}: {size / (1024*1024):.2f} MB")
    print(f"Total downloaded: {total_size / (1024*1024):.2f} MB")
else:
    print("Cache dir does not exist yet.")
