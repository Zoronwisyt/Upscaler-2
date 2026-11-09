import math
from pathlib import Path
from PIL import Image, ImageDraw

SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

def generate_sample_anime():
    w, h = 400, 400
    img = Image.new("RGB", (w, h), (30, 35, 60))
    draw = ImageDraw.Draw(img)

    # Background anime sky / gradient
    for y in range(h):
        r = int(25 + 50 * (y / h))
        g = int(30 + 35 * (y / h))
        b = int(70 + 80 * (y / h))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Anime sun / moon glow behind character
    draw.ellipse([w//2 - 90, 40, w//2 + 90, 220], fill=(255, 230, 180, 200), outline=(255, 180, 120), width=3)

    # Hair Back
    draw.polygon([(110, 160), (70, 360), (140, 340), (200, 370), (270, 340), (330, 360), (290, 160)], fill=(255, 80, 140), outline=(140, 20, 60))

    # Face Skin
    draw.polygon([(150, 150), (250, 150), (240, 240), (200, 280), (160, 240)], fill=(255, 235, 220), outline=(180, 130, 110), width=2)

    # Big Anime Eyes
    # Left eye
    draw.ellipse([165, 180, 192, 215], fill=(20, 20, 40), outline=(10, 10, 20), width=2)
    draw.ellipse([170, 186, 188, 210], fill=(0, 210, 240))
    draw.ellipse([174, 183, 182, 192], fill=(255, 255, 255))
    draw.line([(160, 175), (195, 175)], fill=(20, 20, 30), width=3)

    # Right eye
    draw.ellipse([208, 180, 235, 215], fill=(20, 20, 40), outline=(10, 10, 20), width=2)
    draw.ellipse([212, 186, 230, 210], fill=(0, 210, 240))
    draw.ellipse([216, 183, 224, 192], fill=(255, 255, 255))
    draw.line([(205, 175), (240, 175)], fill=(20, 20, 30), width=3)

    # Anime Mouth & Blush
    draw.ellipse([155, 225, 175, 235], fill=(255, 160, 180))
    draw.ellipse([225, 225, 245, 235], fill=(255, 160, 180))
    draw.arc([193, 240, 207, 252], 0, 180, fill=(180, 60, 60), width=2)

    # Front Bangs / Hair Strands
    draw.polygon([(130, 120), (160, 190), (175, 135), (200, 205), (225, 135), (245, 190), (270, 120), (200, 95)], fill=(255, 95, 155), outline=(160, 25, 75), width=2)
    draw.polygon([(110, 150), (135, 270), (150, 170)], fill=(255, 80, 140), outline=(160, 25, 75), width=2)
    draw.polygon([(290, 150), (265, 270), (250, 170)], fill=(255, 80, 140), outline=(160, 25, 75), width=2)

    # Shoulders / Collar
    draw.polygon([(120, 310), (200, 290), (280, 310), (320, 400), (80, 400)], fill=(30, 25, 45), outline=(0, 240, 255), width=2)

    img.save(SAMPLES_DIR / "sample_anime.png", format="PNG")
    print(f"[Samples] Created sample anime asset at {SAMPLES_DIR / 'sample_anime.png'}")

if __name__ == "__main__":
    generate_sample_anime()
