import io

import numpy as np
from PIL import Image


def sharp_jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    arr = (np.random.rand(height, width, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def blurry_jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    arr = (np.random.rand(height, width, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    small = img.resize((20, 15))
    blurred = small.resize((width, height))
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def dark_jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    """Produces a low-light image (mean intensity well below 60)."""
    arr = (np.random.rand(height, width, 3) * 25).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def corrupted_file_bytes() -> bytes:
    """Returns bytes that look like a JPEG header but are truncated garbage."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 50

