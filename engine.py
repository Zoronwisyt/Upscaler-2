import os
import io
import math
import time
import tempfile
import subprocess
import urllib.request
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path.home() / ".cache" / "kuroline" / "weights"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WAIFU2X_DIR = Path.home() / ".cache" / "kuroline" / "waifu2x"
WAIFU2X_EXE = WAIFU2X_DIR / "waifu2x-ncnn-vulkan_waifu2xEX.exe"
CUNET_MODELS = WAIFU2X_DIR / "models-cunet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Ritmada Engine] Initialized on device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Ritmada Engine] GPU: {torch.cuda.get_device_name(0)}")


# -------------------------------------------------------------
# 1. Waifu2x CUnet & Real-ESRGAN Anime Super-Resolution
# -------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class AnimeUpscaler:
    def __init__(self):
        self.waifu2x_available = WAIFU2X_EXE.exists() and CUNET_MODELS.exists()
        self.realesrgan_model = None
        self.model_loaded = False
        self.engine_name = "Waifu2x-CUnet" if self.waifu2x_available else "Real-ESRGAN"

        if self.waifu2x_available:
            print("[AnimeUpscaler] Native Waifu2x CUnet models active (Vulkan/CUDA).")
            self.model_loaded = True
        else:
            self._init_realesrgan()

    def _init_realesrgan(self):
        try:
            self.weights_path = MODELS_DIR / "RealESRGAN_x4plus_anime_6B.pth"
            self.realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32)
            if not self.weights_path.exists():
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
                print(f"[AnimeUpscaler] Downloading RealESRGAN_x4plus_anime_6B weights...")
                urllib.request.urlretrieve(url, str(self.weights_path))

            loadnet = torch.load(str(self.weights_path), map_location="cpu", weights_only=True)
            keyname = "params_ema" if "params_ema" in loadnet else ("params" if "params" in loadnet else None)
            state_dict = loadnet[keyname] if keyname else loadnet
            self.realesrgan_model.load_state_dict(state_dict, strict=True)
            self.realesrgan_model.eval().to(DEVICE)
            self.model_loaded = True
            print("[AnimeUpscaler] Real-ESRGAN anime-6B weights loaded successfully.")
        except Exception as e:
            print(f"[AnimeUpscaler] Fallback to high-order inking upscaler: {e}")
            self.realesrgan_model = None
            self.model_loaded = False

    def upscale(self, img: Image.Image, scale: int = 4) -> Image.Image:
        """
        Upscales an anime PIL Image by 2x or 4x using Waifu2x CUnet or Real-ESRGAN.
        """
        orig_mode = img.mode
        alpha = None
        if orig_mode == "RGBA":
            r, g, b, alpha = img.split()
            img_rgb = Image.merge("RGB", (r, g, b))
        else:
            img_rgb = img.convert("RGB")

        # Path A: Waifu2x CUnet (Highest anime fidelity)
        if self.waifu2x_available:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    in_path = os.path.join(tmpdir, "in.png")
                    out_path = os.path.join(tmpdir, "out.png")
                    img_rgb.save(in_path, "PNG")

                    # First pass (2x)
                    cmd1 = [
                        str(WAIFU2X_EXE),
                        "-i", in_path,
                        "-o", out_path,
                        "-s", "2",
                        "-n", "2",
                        "-m", str(CUNET_MODELS)
                    ]
                    subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                    if scale == 4:
                        # Second pass for 4x
                        in_path2 = out_path
                        out_path2 = os.path.join(tmpdir, "out4x.png")
                        cmd2 = [
                            str(WAIFU2X_EXE),
                            "-i", in_path2,
                            "-o", out_path2,
                            "-s", "2",
                            "-n", "1",
                            "-m", str(CUNET_MODELS)
                        ]
                        subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        out_img = Image.open(out_path2).copy()
                    else:
                        out_img = Image.open(out_path).copy()

                    # Apply vector inking pass
                    out_img = self._apply_inking(out_img)
                    if alpha is not None:
                        alpha_up = alpha.resize(out_img.size, Image.Resampling.LANCZOS)
                        out_img.putalpha(alpha_up)
                    return out_img
            except Exception as e:
                print(f"[AnimeUpscaler] Waifu2x subprocess notice: {e}. Using Real-ESRGAN.")

        # Path B: PyTorch Real-ESRGAN
        if self.realesrgan_model is not None:
            with torch.no_grad():
                img_np = np.array(img_rgb).astype(np.float32) / 255.0
                tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0).to(DEVICE)
                b, c, h, w = tensor.shape
                tile_size = 512
                tile_pad = 16

                if h <= tile_size and w <= tile_size:
                    output = self.realesrgan_model(tensor)
                else:
                    output = torch.zeros((b, c, h * 4, w * 4), device=DEVICE)
                    for y in range(0, h, tile_size):
                        for x in range(0, w, tile_size):
                            y_start = max(0, y - tile_pad)
                            y_end = min(h, y + tile_size + tile_pad)
                            x_start = max(0, x - tile_pad)
                            x_end = min(w, x + tile_size + tile_pad)

                            in_patch = tensor[:, :, y_start:y_end, x_start:x_end]
                            out_patch = self.realesrgan_model(in_patch)

                            out_y_start = (y - y_start) * 4
                            out_y_end = out_y_start + (min(h, y + tile_size) - y) * 4
                            out_x_start = (x - x_start) * 4
                            out_x_end = out_x_start + (min(w, x + tile_size) - x) * 4

                            patch_crop = out_patch[:, :, out_y_start:out_y_end, out_x_start:out_x_end]
                            target_y = y * 4
                            target_x = x * 4
                            output[:, :, target_y:target_y + patch_crop.shape[2], target_x:target_x + patch_crop.shape[3]] = patch_crop

                output = output.squeeze(0).clamp(0, 1).cpu().numpy().transpose((1, 2, 0))
                out_img = Image.fromarray((output * 255.0).round().astype(np.uint8))
                if scale == 2:
                    out_img = out_img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        else:
            out_img = img_rgb.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)

        out_img = self._apply_inking(out_img)
        if alpha is not None:
            alpha_up = alpha.resize(out_img.size, Image.Resampling.LANCZOS)
            out_img.putalpha(alpha_up)

        return out_img

    def _apply_inking(self, img_rgb: Image.Image) -> Image.Image:
        """
        Enhances line contrast and unsharp edge definition for dramatic vector clarity.
        """
        sharp = img_rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
        enh_contrast = ImageEnhance.Contrast(sharp).enhance(1.10)
        enh_color = ImageEnhance.Color(enh_contrast).enhance(1.06)

        arr = np.array(enh_color).astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        line_mask = (gray < 155)
        arr[line_mask] = np.clip(arr[line_mask] * 0.88, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


# -------------------------------------------------------------
# 2. Pixelcut-Class RMBG-1.4 / SOTA Background Removal (Matting)
# -------------------------------------------------------------
class AnimeMattingEngine:
    def __init__(self):
        self.session = None
        self.model_type = None
        self._init_session()

    def _init_session(self):
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE.type == 'cuda' else ['CPUExecutionProvider']
            
            # Primary: BRIA AI RMBG-1.4 (SOTA Pixelcut-tier segmentation)
            self.rmbg_path = MODELS_DIR / "rmbg-1.4.onnx"
            if not self.rmbg_path.exists():
                print(f"[AnimeMatting] Downloading RMBG-1.4 Pixelcut-tier ONNX weights...")
                try:
                    from huggingface_hub import hf_hub_download
                    import shutil
                    dl_path = hf_hub_download(repo_id='briaai/RMBG-1.4', filename='onnx/model.onnx')
                    shutil.copy(dl_path, str(self.rmbg_path))
                    print("[AnimeMatting] RMBG-1.4 download complete.")
                except Exception as dl_err:
                    print(f"[AnimeMatting] RMBG download notice: {dl_err}")

            if self.rmbg_path.exists():
                self.session = ort.InferenceSession(str(self.rmbg_path), providers=providers)
                self.model_type = "rmbg"
                print(f"[AnimeMatting] SOTA RMBG-1.4 model loaded on {self.session.get_providers()[0]}")
                return

            # Secondary fallback: isnet-anime
            self.isnet_path = MODELS_DIR / "isnet-anime.onnx"
            if self.isnet_path.exists():
                self.session = ort.InferenceSession(str(self.isnet_path), providers=providers)
                self.model_type = "isnet"
                print(f"[AnimeMatting] IS-Net Anime model loaded on {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"[AnimeMatting] ONNX session init notice: {e}. Fast bilateral edge matting active.")
            self.session = None

    def remove_background(self, img: Image.Image) -> Image.Image:
        """
        Removes background with studio-grade boundary precision (Pixelcut-class).
        Returns RGBA Image with alpha transparency.
        """
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size

        if self.session is not None:
            try:
                # Preprocess for 1024x1024 RMBG / IS-Net
                target_size = 1024
                resized = img_rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)
                arr = np.array(resized, dtype=np.float32) / 255.0
                mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                arr = (arr - mean) / std
                tensor = arr.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)

                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: tensor})
                mask = outputs[0][0][0]

                # Normalize mask to 0..255
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                mask = (mask * 255.0).astype(np.uint8)
                mask_img = Image.fromarray(mask).resize((w, h), Image.Resampling.BILINEAR)

                result = img_rgb.copy()
                result.putalpha(mask_img)
                return result
            except Exception as e:
                print(f"[AnimeMatting] Neural inference error: {e}. Using bilateral fallback.")

        # High-Fidelity Color Distance & Edge Saliency Matting Fallback
        img_np = np.array(img_rgb).astype(np.float32)
        top_border = img_np[0:5, :, :]
        bottom_border = img_np[-5:, :, :]
        left_border = img_np[:, 0:5, :]
        right_border = img_np[:, -5:, :]
        border_pixels = np.concatenate([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ], axis=0)

        bg_color = np.median(border_pixels, axis=0)
        diff = np.linalg.norm(img_np - bg_color, axis=2)

        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        grad_y, grad_x = np.gradient(gray)
        edge_mag = np.hypot(grad_x, grad_y)

        mask_raw = (diff > 26.0) | (edge_mag > 16.0)
        mask_pil = Image.fromarray((mask_raw * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(1.2))

        result = img_rgb.copy()
        result.putalpha(mask_pil)
        return result


# Global singletons
upscaler = AnimeUpscaler()
matting = AnimeMattingEngine()
