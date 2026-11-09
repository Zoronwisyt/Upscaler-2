# KuroLine AI Studio — Neural Anime Upscaler & Line Matting

[![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Windows%20%7C%20Linux-blue.svg)](https://github.com)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-100%25%20In--Browser%20Ready-00f0ff.svg)](https://github.com)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%204070%20Ti%20Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)

An anime-focused web studio featuring **dual-mode architecture**:
1. **GitHub Pages (Web Engine)**: Runs 100% client-side inside any desktop or mobile browser via HTML5 Canvas & WebGL without needing a backend server.
2. **Local PC (GPU Mode)**: Optional one-click local server utilizing NVIDIA PyTorch CUDA (Real-ESRGAN Anime 6B + IS-Net).

---

## Features

- **Anime Background Removal**: High-precision boundary isolation optimized for line art, hair strands, and cel-shading.
- **2x / 4x Super-Resolution**: Eliminates JPEG ringing, compression halos, and restores sharp vector-like ink lines.
- **Interactive Split Slider**: Real-time before/after comparison directly in the browser.
- **Instant Client-Side Fallback**: Deploys statically on GitHub Pages with zero server configuration.

---

## Deployment & Usage

### Option A: GitHub Pages (Zero-Setup Static Web Hosting)

1. Fork or push this repository to GitHub.
2. Go to **Settings** -> **Pages**.
3. Under **Branch**, select `main` / `root` and click **Save**.
4. Your website is live at `https://<your-username>.github.io/<repo-name>/`.
   - The in-browser Web Engine handles all upscaling and background removal directly on the client's GPU/CPU.
   - No Python installation or server required.

### Option B: Local PC with GPU Acceleration (NVIDIA CUDA)

To unlock maximum neural fidelity using your local graphics card (e.g. RTX 4070 Ti):

```cmd
start.bat
```
This launches the FastAPI PyTorch server, loads Real-ESRGAN and IS-Net ONNX weights onto your GPU, and opens `http://127.0.0.1:8000`.

---

## Project Structure

```
├── index.html             # Single-page studio with dual-mode (WebGPU/Canvas + REST API)
├── engine.py              # PyTorch CUDA & ONNX inference core (RRDBNet & IS-Net)
├── server.py              # FastAPI backend with REST endpoints & static file server
├── start.bat              # One-click Windows runner & browser watchdog
├── requirements.txt       # Python package dependencies
├── samples/               # Built-in anime test assets
│   └── sample_anime.png
└── weights/               # Auto-cached model checkpoints (git-ignored)
```
