import io
import os
import sys
import time
import base64
import threading
import subprocess
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch

import engine

BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)


app = FastAPI(title="KuroLine AI Anime Studio Engine", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/api/status")
async def get_status():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
    vram_free = 0
    vram_total = 0
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        vram_free = round(free_bytes / (1024**3), 2)
        vram_total = round(total_bytes / (1024**3), 2)

    return {
        "status": "online",
        "engine": "KuroLine Studio Neural Backend",
        "device": str(engine.DEVICE),
        "gpu": gpu_name,
        "vram": f"{vram_free} GB free / {vram_total} GB total",
        "upscaler_loaded": engine.upscaler.model_loaded,
        "matting_loaded": engine.matting.session is not None,
    }


@app.post("/api/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Form(4)
):
    try:
        scale = 2 if scale == 2 else 4
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))

        t0 = time.time()
        result_img = engine.upscaler.upscale(pil_img, scale=scale)
        elapsed = round(time.time() - t0, 3)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG", optimize=True)
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Process-Time": f"{elapsed}s",
                "X-Resolution": f"{result_img.width}x{result_img.height}",
                "Access-Control-Expose-Headers": "X-Process-Time, X-Resolution"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))

        t0 = time.time()
        result_img = engine.matting.remove_background(pil_img)
        elapsed = round(time.time() - t0, 3)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Process-Time": f"{elapsed}s",
                "X-Resolution": f"{result_img.width}x{result_img.height}",
                "Access-Control-Expose-Headers": "X-Process-Time, X-Resolution"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def get_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "index.html not found"}


def poll_and_launch_browser():
    """
    Waits for the FastAPI HTTP server to start listening,
    then launches the browser directly via Windows Explorer shell verb.
    """
    url = "http://127.0.0.1:8000"
    for _ in range(60):
        time.sleep(0.3)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=0.8) as resp:
                if resp.status == 200:
                    print(f"[*] Server is ready! Launching browser at {url} ...")
                    break
        except Exception:
            pass

    time.sleep(0.5)
    try:
        # Use Windows Shell (explorer.exe) to open default browser
        subprocess.Popen(["explorer.exe", url])
    except Exception:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    # Start auto-open thread in background
    t = threading.Thread(target=poll_and_launch_browser, daemon=True)
    t.start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
