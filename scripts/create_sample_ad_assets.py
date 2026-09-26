"""
Utility to generate synthetic demo ad video clips for the 8 catalogue brands.
Creates lightweight MP4 files with animated brand graphics, color schemes, and timer bars.
"""

import os
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ADS_DIR = REPO_ROOT / "assets" / "ads"
DEMO_ADS_DIR = REPO_ROOT / "demo" / "assets" / "ads"

BRANDS = [
    {
        "filename": "zestcola_15s.mp4",
        "name": "ZestCola",
        "tagline": "Taste the Electric Rush!",
        "duration": 5,  # 5 seconds for fast snappy demo playback
        "bg_color": (20, 20, 180),     # Crimson Red (BGR)
        "accent": (50, 220, 255)       # Gold
    },
    {
        "filename": "crunchbite_20s.mp4",
        "name": "CrunchBite",
        "tagline": "Crispy, Crunchy, Craveable.",
        "duration": 5,
        "bg_color": (0, 120, 220),     # Orange (BGR)
        "accent": (255, 255, 255)
    },
    {
        "filename": "streamflix_15s.mp4",
        "name": "StreamFlix",
        "tagline": "Unlimited Stories Await.",
        "duration": 5,
        "bg_color": (140, 10, 20),     # Dark Blue / Violet (BGR)
        "accent": (0, 215, 255)
    },
    {
        "filename": "shelterwise_30s.mp4",
        "name": "ShelterWise Insurance",
        "tagline": "Protecting What Matters Most.",
        "duration": 5,
        "bg_color": (80, 140, 20),     # Forest Green (BGR)
        "accent": (255, 255, 255)
    },
    {
        "filename": "novatel_20s.mp4",
        "name": "Novatel Mobile",
        "tagline": "Ultra-Fast 5G Everywhere.",
        "duration": 5,
        "bg_color": (160, 50, 50),     # Deep Cyan / Teal
        "accent": (0, 255, 200)
    },
    {
        "filename": "voltdrive_30s.mp4",
        "name": "VoltDrive Auto",
        "tagline": "Charge into the Future.",
        "duration": 5,
        "bg_color": (30, 30, 30),      # Sleek Charcoal
        "accent": (255, 180, 0)
    },
    {
        "filename": "munchpop_15s.mp4",
        "name": "MunchPop Snacks",
        "tagline": "Pop the Fun Anytime!",
        "duration": 5,
        "bg_color": (10, 180, 120),    # Vibrant Yellow-Green
        "accent": (255, 255, 255)
    },
    {
        "filename": "quickkart_20s.mp4",
        "name": "QuickKart",
        "tagline": "Delivered to Your Door in 10 Mins.",
        "duration": 5,
        "bg_color": (180, 80, 10),     # Royal Purple (BGR)
        "accent": (0, 240, 255)
    }
]


def create_ad_clip(brand: dict, out_paths: list):
    width, height = 960, 540
    fps = 24
    total_frames = brand["duration"] * fps

    # Try H264 / mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writers = []
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        w = cv2.VideoWriter(str(p), fourcc, fps, (width, height))
        writers.append(w)

    bg_color = np.array(brand["bg_color"], dtype=np.float32)
    accent_color = brand["accent"]

    for frame_idx in range(total_frames):
        # Create subtle pulsating gradient background
        pulse = 0.85 + 0.15 * np.sin(2 * np.pi * frame_idx / fps)
        frame_color = np.clip(bg_color * pulse, 0, 255).astype(np.uint8)
        img = np.full((height, width, 3), frame_color, dtype=np.uint8)

        # Header: "SPONSORED ADVERTISEMENT"
        cv2.putText(img, "SPONSORED ADVERTISEMENT", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        # Brand Title
        cv2.putText(img, brand["name"], (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, accent_color, 4, cv2.LINE_AA)

        # Tagline
        cv2.putText(img, brand["tagline"], (40, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)

        # Progress bar at bottom
        remaining_sec = max(0.0, brand["duration"] - (frame_idx / fps))
        cv2.putText(img, f"Ad ends in {remaining_sec:.1f}s", (40, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

        bar_width = int((frame_idx / total_frames) * (width - 80))
        cv2.rectangle(img, (40, 490), (width - 40, 502), (60, 60, 60), -1)
        if bar_width > 0:
            cv2.rectangle(img, (40, 490), (40 + bar_width, 502), accent_color, -1)

        for w in writers:
            w.write(img)

    for w in writers:
        w.release()

    # Convert to browser-compatible H.264 (avc1/yuv420p) via ffmpeg
    import subprocess
    ffmpeg_exe = r'F:\INSTALLS\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe'
    for p in out_paths:
        tmp = str(p) + '.tmp.mp4'
        cmd = [ffmpeg_exe, '-y', '-i', str(p), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0:
            os.replace(tmp, str(p))


def main():
    print(f"Generating {len(BRANDS)} demo ad video clips...")
    for b in BRANDS:
        p1 = ADS_DIR / b["filename"]
        p2 = DEMO_ADS_DIR / b["filename"]
        create_ad_clip(b, [p1, p2])
        print(f"  Created: {b['filename']}")
    print("Done! All sample ad video clips generated.")


if __name__ == "__main__":
    main()
