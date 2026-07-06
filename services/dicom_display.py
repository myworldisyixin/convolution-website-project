import numpy as np
from PIL import Image, ImageDraw
from skimage import filters

from services.dicom_config import MAX_STACK_IMAGES_RETURNED, ROBUST_VERSION
from services.dicom_cache import store_dicom_stack, get_cached_dicom_stack
from services.dicom_reader import read_dicom_stack
from services.image_helpers import (
    image_to_base64,
    normalize_for_display,
    array_to_base64_image,
    make_display_image_from_pixels,
)


def slice_to_display_image(slice_data):
    try:
        return make_display_image_from_pixels(
            slice_data["pixels"],
            slice_data.get("photometric", "MONOCHROME2")
        )
    except Exception:
        arr = normalize_for_display(slice_data["pixels"])
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")


def window_pixels_to_uint8_array(pixels, window_width, window_level, photometric=""):
    window_width = float(window_width)
    window_level = float(window_level)

    if window_width <= 0:
        window_width = 1.0

    lower = window_level - (window_width / 2.0)
    upper = window_level + (window_width / 2.0)

    if upper == lower:
        upper = lower + 1.0

    display = (pixels - lower) / (upper - lower)
    display = np.clip(display, 0, 1)
    display = (display * 255).astype(np.uint8)

    if str(photometric).upper() == "MONOCHROME1":
        display = 255 - display

    return display


def window_pixels_to_image(pixels, window_width, window_level, photometric=""):
    display = window_pixels_to_uint8_array(
        pixels=pixels,
        window_width=window_width,
        window_level=window_level,
        photometric=photometric,
    )
    return Image.fromarray(display).convert("RGB")


def create_dicom_stack_preview(uploaded_file):
    slices = read_dicom_stack(uploaded_file)

    filename = getattr(uploaded_file, "filename", "")
    stack_id = store_dicom_stack(slices, filename=filename)

    returned_slices = slices[:MAX_STACK_IMAGES_RETURNED]
    images = []

    for index, slice_data in enumerate(returned_slices):
        img = slice_to_display_image(slice_data).convert("RGB")

        images.append({
            "index": index,
            "image": image_to_base64(img),
            "label": slice_data["label"],
            "rows": int(slice_data["pixels"].shape[0]),
            "columns": int(slice_data["pixels"].shape[1]),
            "instanceNumber": slice_data["info"].get("instanceNumber", "Unknown"),
            "sliceLocation": slice_data["info"].get("sliceLocation", "Unknown"),
            "sourceName": slice_data.get("sourceName", ""),
        })

    first_info = returned_slices[0]["info"]
    warning = ""

    if len(slices) > MAX_STACK_IMAGES_RETURNED:
        warning = (
            f"Loaded first {MAX_STACK_IMAGES_RETURNED} preview images only. "
            f"The uploaded file contains {len(slices)} slices total."
        )

    color_count = sum(1 for s in slices if s.get("info", {}).get("isColorDicom"))

    if color_count:
        warning = (
            (warning + " " if warning else "")
            + f"{color_count} color DICOM image(s) were converted to grayscale for display. "
            "ACR HU analysis needs original grayscale CT DICOM."
        )

    return {
        "stackId": stack_id,
        "sliceCount": len(slices),
        "returnedSliceCount": len(returned_slices),
        "images": images,
        "info": first_info,
        "warning": warning,
        "readerVersion": ROBUST_VERSION,
    }


def _get_slices_from_stack_or_upload(stack_id=None, uploaded_file=None):
    if stack_id:
        return get_cached_dicom_stack(stack_id)

    if uploaded_file is not None:
        return read_dicom_stack(uploaded_file)

    raise ValueError("No DICOM stack or uploaded DICOM file provided.")


def create_dicom_window_preview(
    slice_index,
    window_width,
    window_level,
    stack_id=None,
    uploaded_file=None
):
    slices = _get_slices_from_stack_or_upload(
        stack_id=stack_id,
        uploaded_file=uploaded_file
    )

    slice_index = int(slice_index)

    if slice_index < 0 or slice_index >= len(slices):
        raise ValueError(f"Invalid slice index. This file contains {len(slices)} slice(s).")

    selected_slice = slices[slice_index]

    img = window_pixels_to_image(
        selected_slice["pixels"],
        window_width,
        window_level,
        selected_slice.get("photometric", "MONOCHROME2")
    )

    return {
        "image": image_to_base64(img),
        "sliceIndex": slice_index,
        "sliceLabel": selected_slice["label"],
        "windowWidth": float(window_width),
        "windowLevel": float(window_level),
        "info": selected_slice["info"],
        "readerVersion": ROBUST_VERSION,
    }


def dicom_to_image(uploaded_file):
    slices = read_dicom_stack(uploaded_file)
    first_slice = slices[0]
    img = slice_to_display_image(first_slice)
    return img, first_slice["info"]


def analyze_dicom_roi(
    ymin,
    ymax,
    xmin,
    xmax,
    slice_index=0,
    stack_id=None,
    uploaded_file=None
):
    slices = _get_slices_from_stack_or_upload(
        stack_id=stack_id,
        uploaded_file=uploaded_file
    )

    slice_index = int(slice_index)

    if slice_index < 0 or slice_index >= len(slices):
        raise ValueError(f"Invalid slice index. This file contains {len(slices)} slice(s).")

    selected_slice = slices[slice_index]
    raw_pixels = selected_slice["pixels"]
    info = selected_slice["info"]

    height, width = raw_pixels.shape

    ymin = int(ymin)
    ymax = int(ymax)
    xmin = int(xmin)
    xmax = int(xmax)

    if ymin < 0 or ymax > height or xmin < 0 or xmax > width:
        raise ValueError(f"ROI is outside the image. Image size is {height} rows x {width} columns.")

    if ymin >= ymax or xmin >= xmax:
        raise ValueError("Invalid ROI coordinates.")

    norm_pixels = normalize_for_display(raw_pixels)

    smooth_img = filters.gaussian(norm_pixels, sigma=2.0)
    sharp_img = filters.unsharp_mask(norm_pixels, radius=2.0, amount=1.5)
    edge_img = filters.sobel(norm_pixels)

    roi_pixels = raw_pixels[ymin:ymax, xmin:xmax]

    original_img = slice_to_display_image(selected_slice).convert("RGB")
    draw = ImageDraw.Draw(original_img)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

    return {
        "images": {
            "original": image_to_base64(original_img),
            "smooth": array_to_base64_image(smooth_img),
            "sharp": array_to_base64_image(sharp_img),
            "edge": array_to_base64_image(edge_img),
        },
        "roi": {
            "sliceIndex": slice_index,
            "sliceLabel": selected_slice["label"],
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
            "mean": round(float(np.mean(roi_pixels)), 2),
            "std": round(float(np.std(roi_pixels)), 2),
            "min": round(float(np.min(roi_pixels)), 2),
            "max": round(float(np.max(roi_pixels)), 2),
        },
        "imageInfo": info,
        "readerVersion": ROBUST_VERSION,
    }
