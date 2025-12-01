import imghdr
from PIL import Image
from io import BytesIO

def is_valid_image(content, min_size=(512,512)):
    try:
        img = Image.open(BytesIO(content))
        w, h = img.size
        return w >= min_size[0] and h >= min_size[1]
    except Exception:
        return False

def get_extension(content):
    fmt = imghdr.what(None, content)
    return fmt if fmt != "webp" else "png"

def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)