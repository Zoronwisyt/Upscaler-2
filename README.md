# KuroLine AI Studio — Neural Anime Upscaler & Line Matting

[![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Windows%20%7C%20Linux-blue.svg)](https://github.com)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-100%25%20In--Browser%20Ready-00f0ff.svg)](https://github.com)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%204070%20Ti%20Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Models](https://img.shields.io/badge/Engines-Waifu2x%20CUnet%20%7C%20RMBG--1.4-ff3b82.svg)](https://github.com)

An anime-focused web studio featuring **dual-mode architecture**:
1. **GitHub Pages (Web Engine)**: Runs 100% client-side inside any desktop or mobile browser via HTML5 Canvas & WebGL without needing a backend server.
2. **Local PC (GPU Mode)**: One-click local server utilizing NVIDIA RTX 4070 Ti hardware acceleration powered by **Waifu2x CUnet** and **BRIA AI RMBG-1.4** (Pixelcut-class).

---

## Features

- **Pixelcut-Tier Background Removal**: High-precision boundary isolation via **BRIA AI RMBG-1.4** for studio-grade hair, ribbon, and cel art matting.
- **2x / 4x Waifu2x CUnet Super-Resolution**: Eliminates compression halos, sharpens linework with vector ink definition, and cleans textures.
- **Interactive Split Slider & 200% Loupe**: Real-time side-by-side comparison with a 200% magnification inspector for 1:1 pixel verification.
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

### Option B: Local PC with GPU Acceleration (NVIDIA CUDA / Vulkan)

To unlock maximum neural fidelity using your local graphics card (e.g. RTX 4070 Ti):

```cmd
start.bat
```
This launches the studio server, loads Waifu2x CUnet and RMBG-1.4 weights onto your GPU, and opens `http://127.0.0.1:8000`.

---

## Project Structure

```
├── index.html             # Single-page studio with dual-mode (WebGPU/Canvas + REST API)
├── engine.py              # Waifu2x CUnet, Real-ESRGAN & RMBG-1.4 inference core
├── server.py              # FastAPI backend with REST endpoints & static file server
├── start.bat              # One-click Windows runner & browser watchdog
├── requirements.txt       # Python package dependencies
└── samples/               # Built-in anime test assets
    └── sample_anime.png
```
