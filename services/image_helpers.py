import io
import base64
import numpy as np
from PIL import Image


def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return "data:image/png;base64," + encoded


def normalize_for_display(array):
    array = np.asarray(array, dtype=np.float32)

    arr_min = float(np.min(array))
    arr_max = float(np.max(array))

    if arr_max - arr_min < 0.000001:
        return np.zeros_like(array, dtype=np.float32)

    return (array - arr_min) / (arr_max - arr_min)


def array_to_base64_image(array):
    display = normalize_for_display(array) * 255
    img = Image.fromarray(display.astype(np.uint8)).convert("L")
    return image_to_base64(img)


def make_display_image_from_pixels(pixel_array, photometric=""):
    display_array = normalize_for_display(pixel_array) * 255

    if str(photometric).upper() == "MONOCHROME1":
        display_array = 255 - display_array

    return Image.fromarray(display_array.astype(np.uint8)).convert("L")