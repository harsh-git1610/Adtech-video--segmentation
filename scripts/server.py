#!/usr/bin/env python3
"""
Range-enabled HTTP Server for HTML5 Video Streaming & Demo Evaluation.

Python's default 'python -m http.server' does NOT implement RFC 7233 HTTP 206 Partial Content
(byte ranges). When browsers seek an HTML5 MP4 video on a server without 206 support, the browser
cannot seek and reloads the video from 0:00 every time.

This server provides standard HTTP 206 Partial Content support with zero external dependencies,
enabling instant seeking and smooth scrubbing on video files of any size.

Usage:
    python scripts/server.py [port] [directory]
    Example: python scripts/server.py 8000 demo
"""

import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with RFC 7233 byte-range support."""

    def end_headers(self):
        # Enable CORS and byte range advertising
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def send_head(self):
        if 'Range' not in self.headers:
            return super().send_head()

        path = self.translate_path(self.path)
        if not os.path.exists(path) or os.path.isdir(path):
            return super().send_head()

        file_size = os.path.getsize(path)
        range_header = self.headers['Range']
        match = re.match(r'bytes=(\d+)-(\d*)', range_header)

        if not match:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1

        if start >= file_size or start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        end = min(end, file_size - 1)
        content_length = end - start + 1

        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
        self.send_header('Content-Length', str(content_length))
        self.end_headers()

        f = open(path, 'rb')
        f.seek(start)
        return RangeFileWrapper(f, content_length)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.end_headers()

    def do_POST(self):
        if self.path.startswith('/api/upload'):
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self.send_error(400, "Bad Request: Empty Body")
                return

            upload_dir = Path("uploads").resolve()
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract filename from header or default
            raw_filename = self.headers.get('X-File-Name', 'user_uploaded_video.mp4')
            import urllib.parse
            filename = urllib.parse.unquote(raw_filename).replace(' ', '_')
            target_file = upload_dir / filename

            # Read and write chunks
            with open(target_file, 'wb') as f:
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(remaining, 65536)
                    chunk = self.rfile.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            import json
            resp = {
                "status": "success",
                "filename": filename,
                "video_url": f"./uploads/{filename}",
                "size_bytes": os.path.getsize(target_file)
            }
            self.wfile.write(json.dumps(resp).encode('utf-8'))
            return

        self.send_error(404, "Endpoint Not Found")

    def log_message(self, format, *args):
        # Suppress harmless socket disconnects during rapid seeks
        msg = format % args
        if "Broken pipe" not in msg and "forcibly closed" not in msg:
            super().log_message(format, *args)


class RangeFileWrapper:
    """Wraps a file object to read only up to content_length bytes."""
    def __init__(self, f, length):
        self.f = f
        self.remaining = length

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        read_size = self.remaining if size < 0 else min(size, self.remaining)
        data = self.f.read(read_size)
        self.remaining -= len(data)
        return data

    def close(self):
        self.f.close()


def run_server(port=8000, directory="demo"):
    target_dir = Path(directory).resolve()
    if not target_dir.exists():
        target_dir = Path(__file__).resolve().parent.parent / "demo"

    os.chdir(str(target_dir))
    server_address = ('', port)
    httpd = HTTPServer(server_address, RangeHTTPRequestHandler)
    print(f"\nAdTech Demo Server running at: http://localhost:{port}/")
    print(f"Serving directory: {target_dir}")
    print("HTTP 206 Partial Content enabled (instant HTML5 video seeking)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    directory = sys.argv[2] if len(sys.argv) > 2 else "demo"
    run_server(port, directory)
