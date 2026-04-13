#!/usr/bin/env python3
"""
Local HTTP server for the IC 3D Campus Explorer website.

Run from the 3D_city_model/ directory:
    python serve.py

Then open: http://localhost:8000/website/
"""

import http.server
import socketserver
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# Serve from the project root directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Only log non-image requests to keep output clean
        if not any(args[0].endswith(ext) for ext in ('.jpg', '.png', '.jpeg')):
            super().log_message(fmt, *args)


print("=" * 50)
print(f"  IC Campus 3D Explorer")
print(f"  Open: http://localhost:{PORT}/website/")
print(f"  Press Ctrl+C to stop")
print("=" * 50)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
