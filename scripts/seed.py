"""
Seed script — uploads a handful of synthetically generated sample images
against a running instance of the API, so a reviewer can hit /images and
/stats and immediately see populated data instead of an empty database.

Usage:
    python scripts/seed.py [--base-url http://localhost:8000]
"""
import argparse
import io
import time

import httpx
import numpy as np
from PIL import Image


def make_sharp_image() -> bytes:
    """A well-lit, sharp, random-noise 'clean' sample image."""
    arr = (np.random.rand(480, 640, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_blurry_image() -> bytes:
    """A heavily blurred image to trigger BlurAnalyzer."""
    arr = (np.random.rand(480, 640, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    # Simple box-blur via repeated resize down/up — no extra deps needed.
    small = img.resize((32, 24))
    blurred = small.resize((640, 480))
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_dark_image() -> bytes:
    """A low-light image to trigger BrightnessAnalyzer."""
    arr = (np.random.rand(480, 640, 3) * 30).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_screenshot_like_image() -> bytes:
    """A common phone-screen aspect ratio with a flat top/bottom bar,
    to trigger ScreenshotAnalyzer's heuristics."""
    # 960/1920 = 0.5, which matches KNOWN_SCREEN_RATIOS in screenshot_analyzer.py.
    arr = (np.random.rand(1920, 960, 3) * 255).astype("uint8")
    arr[:60, :, :] = 250  # flat status-bar-like strip
    arr[-60:, :, :] = 250  # flat nav-bar-like strip
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


SAMPLES = [
    ("clean_sample.jpg", make_sharp_image),
    ("blurry_sample.jpg", make_blurry_image),
    ("dark_sample.jpg", make_dark_image),
    ("screenshot_like_sample.jpg", make_screenshot_like_image),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the API with sample images")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--duplicate", action="store_true", help="Also upload one duplicate")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(f"Connected to {args.base_url} — {health.json()}")

        first_image_bytes = None
        for filename, generator in SAMPLES:
            image_bytes = generator()
            if first_image_bytes is None:
                first_image_bytes = image_bytes

            response = client.post(
                "/upload", files={"file": (filename, image_bytes, "image/jpeg")}
            )
            response.raise_for_status()
            print(f"Uploaded {filename}: {response.json()}")

        if args.duplicate and first_image_bytes:
            response = client.post(
                "/upload", files={"file": ("duplicate_of_clean.jpg", first_image_bytes, "image/jpeg")}
            )
            response.raise_for_status()
            print(f"Uploaded duplicate_of_clean.jpg: {response.json()}")

        print("\nWaiting for background processing to finish...")
        time.sleep(3)

        stats = client.get("/stats")
        print(f"\nDashboard stats: {stats.json()}")


if __name__ == "__main__":
    main()
