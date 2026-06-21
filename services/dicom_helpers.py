import io
import time
import uuid
import zipfile
import math

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from skimage import filters, measure

from services.image_helpers import (
    image_to_base64,
    normalize_for_display,
    array_to_base64_image,
    make_display_image_from_pixels
)


MAX_STACK_IMAGES_RETURNED = 500

# Local in-memory cache.
# This is for your desktop Flask app, not a production server.
DICOM_STACK_CACHE = {}
CACHE_MAX_ITEMS = 8
CACHE_MAX_AGE_SECONDS = 60 * 60


def _cleanup_stack_cache():
    now = time.time()

    expired_keys = []

    for key, value in DICOM_STACK_CACHE.items():
        age = now - value.get("created_at", now)

        if age > CACHE_MAX_AGE_SECONDS:
            expired_keys.append(key)

    for key in expired_keys:
        DICOM_STACK_CACHE.pop(key, None)

    while len(DICOM_STACK_CACHE) > CACHE_MAX_ITEMS:
        oldest_key = min(
            DICOM_STACK_CACHE.keys(),
            key=lambda k: DICOM_STACK_CACHE[k].get("created_at", now)
        )
        DICOM_STACK_CACHE.pop(oldest_key, None)


def store_dicom_stack(slices, filename=""):
    _cleanup_stack_cache()

    stack_id = str(uuid.uuid4())

    DICOM_STACK_CACHE[stack_id] = {
        "created_at": time.time(),
        "filename": filename,
        "slices": slices
    }

    return stack_id


def get_cached_dicom_stack(stack_id):
    _cleanup_stack_cache()

    if not stack_id:
        raise ValueError("Missing DICOM stack ID.")

    cached = DICOM_STACK_CACHE.get(stack_id)

    if not cached:
        raise ValueError(
            "DICOM stack cache expired or was not found. Please upload the DICOM file again."
        )

    cached["created_at"] = time.time()

    return cached["slices"]


def _read_uploaded_bytes(uploaded_file):
    uploaded_file.stream.seek(0)
    return uploaded_file.read()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _get_sort_value(ds, filename="", frame_index=0):
    instance_number = _safe_float(getattr(ds, "InstanceNumber", 0), 0)
    slice_location = _safe_float(getattr(ds, "SliceLocation", 0), 0)

    image_position_z = 0

    try:
        image_position = getattr(ds, "ImagePositionPatient", None)

        if image_position is not None and len(image_position) >= 3:
            image_position_z = float(image_position[2])
    except Exception:
        image_position_z = 0

    return (
        image_position_z,
        slice_location,
        instance_number,
        filename,
        frame_index
    )


def _get_pixel_spacing(ds):
    try:
        spacing = getattr(ds, "PixelSpacing", None)

        if spacing is not None and len(spacing) >= 2:
            row_spacing = float(spacing[0])
            col_spacing = float(spacing[1])

            if row_spacing > 0 and col_spacing > 0:
                return row_spacing, col_spacing
    except Exception:
        pass

    return None, None


def _dicom_info(ds, pixels):
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    row_spacing, col_spacing = _get_pixel_spacing(ds)

    return {
        "isDicom": True,
        "patientID": str(getattr(ds, "PatientID", "Unknown")),
        "modality": str(getattr(ds, "Modality", "Unknown")),
        "studyDescription": str(getattr(ds, "StudyDescription", "Unknown")),
        "seriesDescription": str(getattr(ds, "SeriesDescription", "Unknown")),
        "rows": int(pixels.shape[-2]),
        "columns": int(pixels.shape[-1]),
        "rescaleSlope": slope,
        "rescaleIntercept": intercept,
        "photometricInterpretation": photometric or "Unknown",
        "originalPixelMin": round(float(np.min(pixels)), 2),
        "originalPixelMax": round(float(np.max(pixels)), 2),
        "instanceNumber": str(getattr(ds, "InstanceNumber", "Unknown")),
        "sliceLocation": str(getattr(ds, "SliceLocation", "Unknown")),
        "pixelSpacingRow": row_spacing,
        "pixelSpacingCol": col_spacing,
    }


def _extract_slices_from_dicom_bytes(file_bytes, filename="uploaded.dcm"):
    ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)

    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"{filename} does not contain pixel data.")

    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    pixels = pixels * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()

    slices = []

    if pixels.ndim == 2:
        info = _dicom_info(ds, pixels)

        slices.append({
            "pixels": pixels,
            "info": info,
            "photometric": photometric,
            "filename": filename,
            "frameIndex": 0,
            "label": f"{filename}",
            "sort": _get_sort_value(ds, filename, 0)
        })

    elif pixels.ndim == 3:
        samples_per_pixel = _safe_int(getattr(ds, "SamplesPerPixel", 1), 1)

        if samples_per_pixel > 1 or photometric in ["RGB", "YBR_FULL", "YBR_FULL_422"]:
            raise ValueError(
                f"{filename} looks like a color image. "
                "This app currently supports grayscale DICOM stacks only."
            )

        for frame_index in range(pixels.shape[0]):
            frame_pixels = pixels[frame_index, :, :]
            info = _dicom_info(ds, frame_pixels)

            slices.append({
                "pixels": frame_pixels,
                "info": info,
                "photometric": photometric,
                "filename": filename,
                "frameIndex": frame_index,
                "label": f"{filename} - frame {frame_index + 1}",
                "sort": _get_sort_value(ds, filename, frame_index)
            })

    else:
        raise ValueError(
            f"{filename} has unsupported DICOM pixel shape: {pixels.shape}. "
            "Only 2D slices or 3D multi-frame grayscale DICOM files are supported."
        )

    return slices


def read_dicom_stack(uploaded_file):
    file_bytes = _read_uploaded_bytes(uploaded_file)

    all_slices = []

    if zipfile.is_zipfile(io.BytesIO(file_bytes)):
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
            names = z.namelist()

            for name in names:
                if name.endswith("/") or name.startswith("__MACOSX"):
                    continue

                lower_name = name.lower()

                if lower_name.endswith(".txt") or lower_name.endswith(".json"):
                    continue

                try:
                    inner_bytes = z.read(name)
                    slices = _extract_slices_from_dicom_bytes(inner_bytes, filename=name)
                    all_slices.extend(slices)
                except Exception:
                    continue

        if not all_slices:
            raise ValueError(
                "The ZIP file did not contain readable DICOM images. "
                "Make sure the ZIP contains .dcm files, not another folder zip inside it."
            )

    else:
        all_slices = _extract_slices_from_dicom_bytes(
            file_bytes,
            filename=getattr(uploaded_file, "filename", "uploaded.dcm")
        )

    all_slices.sort(key=lambda s: s["sort"])

    return all_slices


def slice_to_display_image(slice_data):
    return make_display_image_from_pixels(
        slice_data["pixels"],
        slice_data["photometric"]
    )


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

    photometric = str(photometric).upper()

    if photometric == "MONOCHROME1":
        display = 255 - display

    return display


def window_pixels_to_image(pixels, window_width, window_level, photometric=""):
    display = window_pixels_to_uint8_array(
        pixels=pixels,
        window_width=window_width,
        window_level=window_level,
        photometric=photometric
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
            "sliceLocation": slice_data["info"].get("sliceLocation", "Unknown")
        })

    first_info = returned_slices[0]["info"]

    warning = ""

    if len(slices) > MAX_STACK_IMAGES_RETURNED:
        warning = (
            f"Loaded first {MAX_STACK_IMAGES_RETURNED} slices only. "
            f"The uploaded file contains {len(slices)} slices total."
        )

    return {
        "stackId": stack_id,
        "sliceCount": len(slices),
        "returnedSliceCount": len(returned_slices),
        "images": images,
        "info": first_info,
        "warning": warning
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

    if slice_index < 0 or slice_index >= len(slices):
        raise ValueError(
            f"Invalid slice index. This file contains {len(slices)} slice(s)."
        )

    selected_slice = slices[slice_index]

    img = window_pixels_to_image(
        pixels=selected_slice["pixels"],
        window_width=window_width,
        window_level=window_level,
        photometric=selected_slice["photometric"]
    )

    return {
        "image": image_to_base64(img),
        "sliceIndex": slice_index,
        "sliceLabel": selected_slice["label"],
        "windowWidth": float(window_width),
        "windowLevel": float(window_level),
        "info": selected_slice["info"]
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

    if slice_index < 0 or slice_index >= len(slices):
        raise ValueError(
            f"Invalid slice index. This file contains {len(slices)} slice(s)."
        )

    selected_slice = slices[slice_index]
    raw_pixels = selected_slice["pixels"]
    info = selected_slice["info"]

    height, width = raw_pixels.shape

    if ymin < 0 or ymax > height or xmin < 0 or xmax > width:
        raise ValueError(
            f"ROI is outside the image. Image size is {height} rows x {width} columns."
        )

    if ymin >= ymax or xmin >= xmax:
        raise ValueError("Invalid ROI coordinates.")

    norm_pixels = normalize_for_display(raw_pixels)

    smooth_img = filters.gaussian(norm_pixels, sigma=2.0)
    sharp_img = filters.unsharp_mask(norm_pixels, radius=2.0, amount=1.5)
    edge_img = filters.sobel(norm_pixels)

    roi_pixels = raw_pixels[ymin:ymax, xmin:xmax]

    roi_mean = float(np.mean(roi_pixels))
    roi_std = float(np.std(roi_pixels))
    roi_min = float(np.min(roi_pixels))
    roi_max = float(np.max(roi_pixels))

    original_img = make_display_image_from_pixels(
        raw_pixels,
        info["photometricInterpretation"]
    ).convert("RGB")

    draw = ImageDraw.Draw(original_img)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

    return {
        "images": {
            "original": image_to_base64(original_img),
            "smooth": array_to_base64_image(smooth_img),
            "sharp": array_to_base64_image(sharp_img),
            "edge": array_to_base64_image(edge_img)
        },
        "roi": {
            "sliceIndex": slice_index,
            "sliceLabel": selected_slice["label"],
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
            "mean": round(roi_mean, 2),
            "std": round(roi_std, 2),
            "min": round(roi_min, 2),
            "max": round(roi_max, 2)
        },
        "imageInfo": info
    }


def _detect_phantom_circle(raw_pixels):
    norm = normalize_for_display(raw_pixels)

    try:
        threshold = filters.threshold_otsu(norm)
        mask = norm > threshold
    except Exception:
        finite_pixels = raw_pixels[np.isfinite(raw_pixels)]
        threshold = np.percentile(finite_pixels, 25)
        mask = raw_pixels > threshold

    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    if not regions:
        raise ValueError("Could not detect phantom boundary.")

    height, width = raw_pixels.shape
    image_area = height * width

    usable_regions = []

    for region in regions:
        if region.area < image_area * 0.05:
            continue

        usable_regions.append(region)

    if not usable_regions:
        usable_regions = regions

    main = sorted(usable_regions, key=lambda r: r.area, reverse=True)[0]

    cy, cx = main.centroid
    area = float(main.area)
    radius = math.sqrt(area / math.pi)

    return float(cx), float(cy), float(radius)


def _detect_two_bright_markers(
    raw_pixels,
    phantom_cx,
    phantom_cy,
    phantom_radius,
    detection_pixels=None
):
    """
    Detect the two small bright BB dots using the current windowed display image.

    The BB detection uses detection_pixels, which matches the current WW/WL display.
    Uniformity ROI values still use raw HU pixels later.
    """

    height, width = raw_pixels.shape

    if detection_pixels is None:
        detection_pixels = window_pixels_to_uint8_array(
            pixels=raw_pixels,
            window_width=100,
            window_level=0
        )

    detection_pixels = detection_pixels.astype(np.float32)

    finite_mask = np.isfinite(raw_pixels)

    yy, xx = np.ogrid[:height, :width]

    distance_from_center = np.sqrt(
        (xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2
    )

    # Search inside the phantom, away from the outer edge.
    search_mask = (
        finite_mask &
        (distance_from_center < phantom_radius * 0.86)
    )

    search_values = detection_pixels[search_mask]

    if search_values.size < 10:
        raise ValueError(
            "Could not build BB search region. Select the correct uniformity slice."
        )

    best_candidates = []

    for percentile in [99.98, 99.95, 99.9, 99.8, 99.6, 99.3, 99.0, 98.5, 98.0]:
        threshold = np.percentile(search_values, percentile)

        bright_mask = (detection_pixels >= threshold) & search_mask

        labels = measure.label(bright_mask)
        regions = measure.regionprops(labels, intensity_image=detection_pixels)

        candidates = []

        max_area = max(25, int(height * width * 0.0015))

        for region in regions:
            if region.area < 1 or region.area > max_area:
                continue

            minr, minc, maxr, maxc = region.bbox
            box_h = maxr - minr
            box_w = maxc - minc

            if box_h <= 0 or box_w <= 0:
                continue

            # BBs are small bright blobs, not long structures.
            if box_h > phantom_radius * 0.10 or box_w > phantom_radius * 0.10:
                continue

            aspect = box_w / box_h

            if aspect < 0.20 or aspect > 5.0:
                continue

            try:
                cy, cx = region.weighted_centroid
            except Exception:
                cy, cx = region.centroid

            dist = math.sqrt((cx - phantom_cx) ** 2 + (cy - phantom_cy) ** 2)

            if dist > phantom_radius * 0.86:
                continue

            candidates.append({
                "x": float(cx),
                "y": float(cy),
                "area": float(region.area),
                "meanIntensity": float(region.mean_intensity),
                "maxIntensity": float(region.max_intensity),
                "distanceFromCenter": float(dist)
            })

        if len(candidates) >= 2:
            best_candidates = candidates
            break

    if len(best_candidates) < 2:
        raise ValueError(
            "Could not find the two white BB dots. Adjust WW/WL until both dots are visible, then run again."
        )

    best_pair = None
    best_score = None

    for i in range(len(best_candidates)):
        for j in range(i + 1, len(best_candidates)):
            a = best_candidates[i]
            b = best_candidates[j]

            dx = a["x"] - b["x"]
            dy = a["y"] - b["y"]
            pair_distance = math.sqrt(dx * dx + dy * dy)

            # Avoid selecting two pieces of the same BB.
            if pair_distance < phantom_radius * 0.12:
                continue

            sorted_center_distances = sorted([
                a["distanceFromCenter"],
                b["distanceFromCenter"]
            ])

            # In the user's Module 3 display, one BB is closer to the center,
            # and one BB is farther away inside the phantom.
            one_near_center = sorted_center_distances[0] < phantom_radius * 0.40
            one_farther_out = sorted_center_distances[1] > phantom_radius * 0.45

            geometry_bonus = 0

            if one_near_center:
                geometry_bonus += 100

            if one_farther_out:
                geometry_bonus += 100

            if pair_distance > phantom_radius * 0.30:
                geometry_bonus += 40

            brightness_score = (
                a["meanIntensity"] +
                b["meanIntensity"] +
                a["maxIntensity"] +
                b["maxIntensity"]
            )

            area_score = min(a["area"], b["area"]) * 3.0
            area_mismatch_penalty = abs(a["area"] - b["area"]) * 2.0

            score = brightness_score + geometry_bonus + area_score - area_mismatch_penalty

            if best_score is None or score > best_score:
                best_score = score
                best_pair = (a, b, pair_distance)

    if best_pair is None:
        raise ValueError(
            "Bright dots were found, but the app could not identify the two BB dots."
        )

    marker_a, marker_b, marker_distance = best_pair

    return marker_a, marker_b, float(marker_distance)


def _make_circular_roi_mask(shape, cx, cy, radius):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]

    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2


def _roi_stats(raw_pixels, name, cx, cy, radius, center_mean=None):
    mask = _make_circular_roi_mask(raw_pixels.shape, cx, cy, radius)
    values = raw_pixels[mask]

    if values.size == 0:
        raise ValueError(f"{name} ROI did not contain pixels.")

    mean = float(np.mean(values))
    std = float(np.std(values))

    if center_mean is None:
        diff = 0.0
        result = "REFERENCE"
    else:
        diff = mean - center_mean
        abs_diff = abs(diff)

        if abs_diff <= 5:
            result = "PASS"
        elif abs_diff <= 7:
            result = "MINOR"
        else:
            result = "MAJOR"

    return {
        "name": name,
        "cx": int(round(cx)),
        "cy": int(round(cy)),
        "radius": int(round(radius)),
        "mean": round(mean, 2),
        "std": round(std, 2),
        "diffFromCenter": round(diff, 2),
        "result": result
    }


def create_acr_module3_analysis(
    slice_index=0,
    stack_id=None,
    uploaded_file=None,
    window_width=100,
    window_level=0
):
    slices = _get_slices_from_stack_or_upload(
        stack_id=stack_id,
        uploaded_file=uploaded_file
    )

    if slice_index < 0 or slice_index >= len(slices):
        raise ValueError(
            f"Invalid slice index. This file contains {len(slices)} slice(s)."
        )

    selected_slice = slices[slice_index]
    raw_pixels = selected_slice["pixels"]
    info = selected_slice["info"]

    phantom_cx, phantom_cy, phantom_radius = _detect_phantom_circle(raw_pixels)

    detection_pixels = window_pixels_to_uint8_array(
        pixels=raw_pixels,
        window_width=window_width,
        window_level=window_level,
        photometric=selected_slice["photometric"]
    )

    marker_a, marker_b, marker_distance_px = _detect_two_bright_markers(
        raw_pixels=raw_pixels,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        detection_pixels=detection_pixels
    )

    row_spacing = info.get("pixelSpacingRow", None)
    col_spacing = info.get("pixelSpacingCol", None)

    marker_distance_mm = None
    roi_area_mm2 = None

    if row_spacing and col_spacing:
        dx_mm = (marker_a["x"] - marker_b["x"]) * float(col_spacing)
        dy_mm = (marker_a["y"] - marker_b["y"]) * float(row_spacing)
        marker_distance_mm = math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)

        roi_area_mm2 = 400.0
        roi_radius_mm = math.sqrt(roi_area_mm2 / math.pi)
        avg_spacing = (float(row_spacing) + float(col_spacing)) / 2.0
        roi_radius_px = roi_radius_mm / avg_spacing
    else:
        roi_radius_px = phantom_radius * 0.113

    roi_radius_px = max(4.0, min(float(roi_radius_px), phantom_radius * 0.18))

    # ACR-style edge ROI placement:
    # edge of ROI approximately one ROI diameter away from phantom edge.
    offset = phantom_radius - (3.0 * roi_radius_px)
    offset = max(phantom_radius * 0.45, min(offset, phantom_radius * 0.72))

    roi_defs = [
        ("Center", phantom_cx, phantom_cy),
        ("Top", phantom_cx, phantom_cy - offset),
        ("Right", phantom_cx + offset, phantom_cy),
        ("Bottom", phantom_cx, phantom_cy + offset),
        ("Left", phantom_cx - offset, phantom_cy),
    ]

    center_roi = _roi_stats(
        raw_pixels=raw_pixels,
        name="Center",
        cx=roi_defs[0][1],
        cy=roi_defs[0][2],
        radius=roi_radius_px,
        center_mean=None
    )

    rois = [center_roi]
    center_mean = center_roi["mean"]

    for name, cx, cy in roi_defs[1:]:
        rois.append(
            _roi_stats(
                raw_pixels=raw_pixels,
                name=name,
                cx=cx,
                cy=cy,
                radius=roi_radius_px,
                center_mean=center_mean
            )
        )

    worst_abs_diff = max(abs(r["diffFromCenter"]) for r in rois[1:])

    if worst_abs_diff <= 5:
        final_result = "PASS"
    elif worst_abs_diff <= 7:
        final_result = "MINOR DEFICIENCY"
    else:
        final_result = "MAJOR DEFICIENCY"

    overlay = window_pixels_to_image(
        pixels=raw_pixels,
        window_width=window_width,
        window_level=window_level,
        photometric=selected_slice["photometric"]
    )

    draw = ImageDraw.Draw(overlay)

    draw.ellipse(
        [
            phantom_cx - phantom_radius,
            phantom_cy - phantom_radius,
            phantom_cx + phantom_radius,
            phantom_cy + phantom_radius
        ],
        outline="yellow",
        width=2
    )

    draw.line(
        [
            marker_a["x"],
            marker_a["y"],
            marker_b["x"],
            marker_b["y"]
        ],
        fill="cyan",
        width=3
    )

    for marker in [marker_a, marker_b]:
        x = marker["x"]
        y = marker["y"]
        r = 6

        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            outline="cyan",
            width=3
        )

    for roi in rois:
        cx = roi["cx"]
        cy = roi["cy"]
        r = roi["radius"]

        color = "lime"

        if roi["result"] == "MINOR":
            color = "orange"
        elif roi["result"] == "MAJOR":
            color = "red"

        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=color,
            width=3
        )

        draw.text(
            (cx + r + 4, cy - r),
            roi["name"],
            fill=color
        )

    return {
        "analysisType": "ACR CT Module 3 Uniformity",
        "sliceIndex": slice_index,
        "sliceLabel": selected_slice["label"],
        "windowWidth": float(window_width),
        "windowLevel": float(window_level),
        "finalResult": final_result,
        "phantomCenterX": int(round(phantom_cx)),
        "phantomCenterY": int(round(phantom_cy)),
        "phantomRadiusPixels": round(float(phantom_radius), 2),
        "markerA": {
            "x": round(marker_a["x"], 2),
            "y": round(marker_a["y"], 2)
        },
        "markerB": {
            "x": round(marker_b["x"], 2),
            "y": round(marker_b["y"], 2)
        },
        "markerDistancePixels": round(float(marker_distance_px), 2),
        "markerDistanceMm": round(float(marker_distance_mm), 2) if marker_distance_mm is not None else None,
        "roiAreaMm2": round(float(roi_area_mm2), 2) if roi_area_mm2 is not None else None,
        "roiRadiusPixels": round(float(roi_radius_px), 2),
        "rois": rois,
        "overlayImage": image_to_base64(overlay),
        "imageInfo": info
    }