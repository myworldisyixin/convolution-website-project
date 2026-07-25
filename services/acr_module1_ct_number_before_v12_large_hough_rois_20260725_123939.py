
"""
Quick ACR CT Phantom Module 1 CT-number/material insert analysis.

First working Sprint #2 version:
- uses OpenCV Hough circles to find visible circular inserts,
- uses reduced inner ROIs,
- calculates mean HU and standard deviation,
- generates one overlay image and a results table.

This is intentionally simple so we can refine it later.
"""

from __future__ import annotations

import base64
from io import BytesIO
import math
from typing import Any

import numpy as np
from PIL import ImageDraw

try:
    import cv2
except Exception:  # handled at runtime
    cv2 = None

from services.acr_module_classifier import (
    CLASSIFIER_VERSION,
    _estimate_phantom_geometry,
    create_acr_module_classification,
)
from services.dicom_display import (
    _get_slices_from_stack_or_upload,
    window_pixels_to_image,
)


MODULE1_ANALYSIS_VERSION = "ACR_MODULE1_CT_NUMBER_QUICK_V3_SMART_WATER_2026_07_15"
MODULE_1 = "MODULE_1_CT_NUMBER"


def _image_to_data_url(image) -> str:
    stream = BytesIO()
    image.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def _require_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{name} is missing or invalid.") from exc

    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number.")

    return number


def _normalize_to_uint8(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    finite = arr[np.isfinite(arr)]

    if finite.size < 20:
        raise ValueError("Not enough finite pixels for Module 1 detection.")

    low = float(np.percentile(finite, 0.5))
    high = float(np.percentile(finite, 99.5))

    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))

    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    normalized = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _merge_circles(circles: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    merged: list[tuple[float, float, float]] = []

    for x_value, y_value, radius in sorted(circles, key=lambda item: item[2], reverse=True):
        duplicate = False

        for kept_x, kept_y, kept_radius in merged:
            distance = math.hypot(x_value - kept_x, y_value - kept_y)

            if distance < max(radius, kept_radius) * 0.65:
                duplicate = True
                break

        if not duplicate:
            merged.append((float(x_value), float(y_value), float(radius)))

    return merged


def _detect_insert_circles(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    if cv2 is None:
        raise ValueError(
            "OpenCV is not installed. Run: py -m pip install opencv-python"
        )

    img_8bit = _normalize_to_uint8(raw)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(img_8bit)
    blurred = cv2.GaussianBlur(
        enhanced,
        (9, 9),
        2,
    )

    height, width = raw.shape
    min_dim = min(height, width)

    min_radius = max(
        8,
        int(round(min_dim * 0.035)),
    )
    max_radius = max(
        min_radius + 4,
        int(round(min_dim * 0.16)),
    )
    min_dist = max(
        35,
        int(round(min_dim * 0.10)),
    )

    raw_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=30,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    detected: list[tuple[float, float, float]] = []

    if raw_circles is not None:
        for x_value, y_value, radius in raw_circles[0, :]:
            x_float = float(x_value)
            y_float = float(y_value)
            r_float = float(radius)

            distance_from_center = math.hypot(
                x_float - phantom_cx,
                y_float - phantom_cy,
            )

            if distance_from_center + r_float > phantom_radius * 0.96:
                continue

            if r_float < min_radius or r_float > max_radius:
                continue

            detected.append((x_float, y_float, r_float))

    detected = _merge_circles(detected)

    detected = sorted(
        detected,
        key=lambda circle: (
            abs(math.hypot(circle[0] - phantom_cx, circle[1] - phantom_cy) - phantom_radius * 0.45),
            -circle[2],
        ),
    )[:8]

    detected = sorted(
        detected,
        key=lambda circle: math.atan2(circle[1] - phantom_cy, circle[0] - phantom_cx),
    )

    diagnostics = {
        "method": "CLAHE + Gaussian blur + OpenCV HoughCircles",
        "minRadiusPixels": int(min_radius),
        "maxRadiusPixels": int(max_radius),
        "minDistancePixels": int(min_dist),
        "detectedCircleCount": int(len(detected)),
    }

    return detected, diagnostics


def _ellipse_mask_pixels(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]

    return (
        ((xx - float(cx)) / max(float(radius_x), 1e-6)) ** 2
        + ((yy - float(cy)) / max(float(radius_y), 1e-6)) ** 2
    ) <= 1.0


def _guess_material_label(mean_hu: float, used_labels: set[str]) -> str:
    if mean_hu <= -600:
        label = "Air"
    elif -160 <= mean_hu <= -20:
        label = "Polyethylene / low-density insert"
    elif 40 <= mean_hu <= 180:
        label = "Acrylic / PMMA-like insert"
    elif mean_hu >= 250:
        label = "Bone / high-density insert"
    else:
        label = "Detected insert"

    if label not in used_labels:
        used_labels.add(label)
        return label

    suffix = 2
    while f"{label} {suffix}" in used_labels:
        suffix += 1

    final_label = f"{label} {suffix}"
    used_labels.add(final_label)
    return final_label


def _measure_roi(
    raw: np.ndarray,
    label: str,
    cx: float,
    cy: float,
    radius_x_pixels: float,
    radius_y_pixels: float,
    row_spacing: float,
    col_spacing: float,
    display_radius_pixels: float | None = None,
    full_detected_radius_pixels: float | None = None,
) -> dict[str, Any]:
    mask = _ellipse_mask_pixels(
        shape=raw.shape,
        cx=cx,
        cy=cy,
        radius_x=radius_x_pixels,
        radius_y=radius_y_pixels,
    )

    values = raw[mask]
    values = values[np.isfinite(values)]

    if values.size < 10:
        raise ValueError(f"{label} ROI contains too few valid pixels.")

    actual_area_mm2 = float(values.size * row_spacing * col_spacing)

    return {
        "label": label,
        "cx": round(float(cx), 3),
        "cy": round(float(cy), 3),
        "radiusX": round(float(radius_x_pixels), 3),
        "radiusY": round(float(radius_y_pixels), 3),
        "displayRadiusPixels": round(float(display_radius_pixels or radius_x_pixels), 3),
        "detectedRadiusPixels": (
            round(float(full_detected_radius_pixels), 3)
            if full_detected_radius_pixels is not None
            else None
        ),
        "actualAreaMm2": round(float(actual_area_mm2), 2),
        "pixelCount": int(values.size),
        "meanHU": round(float(np.mean(values)), 2),
        "stdHU": round(float(np.std(values)), 2),
        "result": "MEASURED",
    }


def _choose_module1_slice(
    classification: dict[str, Any],
    slice_count: int,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    slices = classification.get("slices") or []

    candidates = [
        item
        for item in slices
        if item.get("prediction") == MODULE_1
    ]

    if candidates:
        best = max(
            candidates,
            key=lambda item: float(
                (item.get("scores") or {}).get(MODULE_1, 0)
            ),
        )
        index = int(best.get("sliceIndex", int(best.get("sliceNumber", 1)) - 1))
        index = max(0, min(index, slice_count - 1))

        group = None
        for possible_group in classification.get("groups") or []:
            if (
                possible_group.get("prediction") == MODULE_1
                and int(possible_group.get("startSliceIndex", -1)) <= index <= int(possible_group.get("endSliceIndex", -1))
            ):
                group = possible_group
                break

        return index, best, group, warnings

    groups = [
        group
        for group in classification.get("groups") or []
        if group.get("prediction") == MODULE_1
    ]

    if groups:
        group = max(
            groups,
            key=lambda item: (
                int(item.get("sliceCount", 0)),
                float((item.get("averageScores") or {}).get(MODULE_1, 0)),
            ),
        )
        index = (
            int(group["startSliceIndex"])
            + int(group["endSliceIndex"])
        ) // 2
        index = max(0, min(index, slice_count - 1))
        warnings.append(
            "No single high-scoring Module 1 slice was found; using the middle of the detected Module 1 range."
        )
        return index, None, group, warnings

    warnings.append(
        "The classifier did not find Module 1 confidently. Using the first slice as a quick fallback."
    )
    return 0, None, None, warnings


def create_module1_ct_number_analysis(
    stack_id: str | None = None,
    uploaded_file=None,
    window_width: float = 400,
    window_level: float = 40,
    classification_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slices = _get_slices_from_stack_or_upload(
        stack_id=stack_id,
        uploaded_file=uploaded_file,
    )

    if not slices:
        raise ValueError("No DICOM slices were loaded.")

    if classification_result is None:
        classification_result = create_acr_module_classification(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            max_size=160,
        )

    slice_index, slice_record, module1_group, warnings = _choose_module1_slice(
        classification=classification_result,
        slice_count=len(slices),
    )

    selected = slices[slice_index]
    info = selected.get("info", {})

    if info.get("isColorDicom"):
        raise ValueError(
            "The selected DICOM is color/secondary-capture data. Module 1 HU measurements require original grayscale CT DICOM."
        )

    raw = np.asarray(selected["pixels"], dtype=np.float32)

    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D CT slice, got shape {raw.shape}.")

    row_spacing = _require_number(
        info.get("pixelSpacingRow"),
        "DICOM PixelSpacing row value",
    )
    col_spacing = _require_number(
        info.get("pixelSpacingCol"),
        "DICOM PixelSpacing column value",
    )

    phantom_cx, phantom_cy, phantom_radius = _estimate_phantom_geometry(raw)

    circles, detection_info = _detect_insert_circles(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    rois: list[dict[str, Any]] = []
    used_labels: set[str] = set()

    for circle_index, (x_value, y_value, radius) in enumerate(circles, start=1):
        inner_radius = max(
            4.0,
            float(radius) * 0.50,
        )

        preliminary = _measure_roi(
            raw=raw,
            label=f"Detected insert {circle_index}",
            cx=x_value,
            cy=y_value,
            radius_x_pixels=inner_radius,
            radius_y_pixels=inner_radius,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
            display_radius_pixels=inner_radius,
            full_detected_radius_pixels=radius,
        )

        label = _guess_material_label(
            mean_hu=float(preliminary["meanHU"]),
            used_labels=used_labels,
        )

        preliminary["label"] = label
        preliminary["circleNumber"] = int(circle_index)
        rois.append(preliminary)

    if not rois:
        warnings.append(
            "No insert circles were detected by HoughCircles. The overlay will only show the estimated water/background ROI."
        )

    if rois:
        water_radius_pixels = float(
            np.median([
                float(roi["radiusX"])
                for roi in rois
            ])
        )
    else:
        water_radius_pixels = max(
            8.0,
            min(22.0, phantom_radius * 0.09),
        )

    water_radius_pixels = max(
        5.0,
        min(water_radius_pixels, phantom_radius * 0.14),
    )

    water_location = _module1_v3_pick_water_roi(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        circles=circles,
        water_radius_pixels=water_radius_pixels,
    )

    water_roi = _measure_roi(
        raw=raw,
        label="Water / background",
        cx=float(water_location["cx"]),
        cy=float(water_location["cy"]),
        radius_x_pixels=water_radius_pixels,
        radius_y_pixels=water_radius_pixels,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        display_radius_pixels=water_radius_pixels,
        full_detected_radius_pixels=None,
    )
    water_roi["circleNumber"] = 0
    water_roi["waterSearchScore"] = round(float(water_location["score"]), 3)
    water_roi["waterSearchDiagnostics"] = water_location.get("searchDiagnostics", {})

    rois.append(water_roi)

    rois = sorted(
        rois,
        key=lambda roi: (
            1 if roi["label"].startswith("Water") else 0,
            float(roi["meanHU"]),
        ),
    )

    overlay = window_pixels_to_image(
        raw,
        float(window_width),
        float(window_level),
        selected.get("photometric", "MONOCHROME2"),
    ).convert("RGB")

    draw = ImageDraw.Draw(overlay)

    draw.ellipse(
        [
            phantom_cx - phantom_radius,
            phantom_cy - phantom_radius,
            phantom_cx + phantom_radius,
            phantom_cy + phantom_radius,
        ],
        outline="yellow",
        width=2,
    )

    for circle_index, (x_value, y_value, radius) in enumerate(circles, start=1):
        draw.ellipse(
            [
                x_value - radius,
                y_value - radius,
                x_value + radius,
                y_value + radius,
            ],
            outline="gray",
            width=1,
        )

    for roi in rois:
        cx = float(roi["cx"])
        cy = float(roi["cy"])
        radius_x = float(roi["radiusX"])
        radius_y = float(roi["radiusY"])

        if roi["label"].startswith("Water"):
            color = "cyan"
        elif "Air" in roi["label"]:
            color = "red"
        elif "Bone" in roi["label"]:
            color = "orange"
        else:
            color = "lime"

        draw.ellipse(
            [
                cx - radius_x,
                cy - radius_y,
                cx + radius_x,
                cy + radius_y,
            ],
            outline=color,
            width=3,
        )

        draw.text(
            (cx + radius_x + 5, cy - radius_y),
            roi["label"],
            fill=color,
        )

    overlay_data = _image_to_data_url(overlay)

    module1_range = None

    if module1_group:
        module1_range = {
            "startSliceIndex": int(module1_group["startSliceIndex"]),
            "endSliceIndex": int(module1_group["endSliceIndex"]),
            "startSliceNumber": int(module1_group["startSliceNumber"]),
            "endSliceNumber": int(module1_group["endSliceNumber"]),
            "sliceCount": int(module1_group["sliceCount"]),
        }

    return {
        "success": True,
        "analysisType": "Quick ACR Module 1 CT Number / Material Insert Analysis",
        "analysisVersion": MODULE1_ANALYSIS_VERSION,
        "classifierVersion": classification_result.get(
            "classifierVersion",
            CLASSIFIER_VERSION,
        ),
        "sliceCount": len(slices),
        "selectedSliceIndex": int(slice_index),
        "selectedSliceNumber": int(slice_index) + 1,
        "selectedSliceLabel": selected.get("label", ""),
        "module1Range": module1_range,
        "selectedSliceClassifierRecord": slice_record,
        "phantom": {
            "centerX": round(float(phantom_cx), 3),
            "centerY": round(float(phantom_cy), 3),
            "radiusPixels": round(float(phantom_radius), 3),
            "radiusMmApprox": round(
                float(
                    phantom_radius
                    * ((row_spacing + col_spacing) / 2.0)
                ),
                3,
            ),
        },
        "detection": detection_info,
        "rois": rois,
        "roiCount": int(len(rois)),
        "displayWindow": {
            "windowWidth": float(window_width),
            "windowLevel": float(window_level),
            "note": (
                "WW/WL controls only the displayed overlay. ROI statistics use raw CT HU values."
            ),
        },
        "overlayImage": overlay_data,
        "image": overlay_data,
        "warnings": warnings,
        "criteriaNote": (
            "Quick first-pass Module 1 analysis. Visible insert ROIs use 50% of each detected circle radius. "
            "The water/background ROI is searched automatically inside the phantom while avoiding detected inserts. Material labels are guessed from measured HU and will be refined later."
        ),
    }

# BEGIN MODULE1 V3 SMART WATER HELPERS

def _module1_v3_unique_label(label: str, used_labels: set[str]) -> str:
    if label not in used_labels:
        used_labels.add(label)
        return label

    suffix = 2
    while f"{label} {suffix}" in used_labels:
        suffix += 1

    final_label = f"{label} {suffix}"
    used_labels.add(final_label)
    return final_label


def _module1_v3_guess_material_label(mean_hu: float, used_labels: set[str]) -> str:
    if mean_hu <= -700:
        label = "Air"
    elif -220 <= mean_hu <= -35:
        label = "Low-density insert"
    elif -35 < mean_hu < 35:
        label = "Water-like insert"
    elif 35 <= mean_hu <= 180:
        label = "Acrylic / PMMA-like insert"
    elif 180 < mean_hu <= 600:
        label = "Dense insert"
    elif mean_hu > 600:
        label = "Bone / high-density insert"
    else:
        label = "Detected insert"

    return _module1_v3_unique_label(label, used_labels)


# Override the earlier quick HU label guess without touching the route or UI.
_guess_material_label = _module1_v3_guess_material_label


def _module1_v3_roi_stats(raw: np.ndarray, cx: float, cy: float, radius_pixels: float) -> dict[str, float] | None:
    mask = _ellipse_mask_pixels(
        shape=raw.shape,
        cx=cx,
        cy=cy,
        radius_x=radius_pixels,
        radius_y=radius_pixels,
    )
    values = raw[mask]
    values = values[np.isfinite(values)]

    if values.size < 12:
        return None

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(values.size),
    }


def _module1_v3_pick_water_roi(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
    circles: list[tuple[float, float, float]],
    water_radius_pixels: float,
) -> dict[str, Any]:
    """
    Search for a clean water/background ROI inside the phantom.
    It avoids the detected material inserts and prefers HU near 0 with low SD.
    """
    candidate_offsets: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]

    for radial_fraction in [0.12, 0.20, 0.28, 0.36, 0.44, 0.52]:
        radius = phantom_radius * radial_fraction
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            candidate_offsets.append((
                radius * math.cos(float(angle)),
                radius * math.sin(float(angle)),
                radial_fraction,
            ))

    best_candidate: dict[str, Any] | None = None
    checked_count = 0
    rejected_by_insert = 0
    rejected_by_edge = 0

    for dx, dy, radial_fraction in candidate_offsets:
        cx = float(phantom_cx + dx)
        cy = float(phantom_cy + dy)

        if math.hypot(cx - phantom_cx, cy - phantom_cy) + water_radius_pixels > phantom_radius * 0.74:
            rejected_by_edge += 1
            continue

        overlaps_insert = False
        for circle_x, circle_y, circle_radius in circles:
            if math.hypot(cx - circle_x, cy - circle_y) < circle_radius + water_radius_pixels * 1.65:
                overlaps_insert = True
                break

        if overlaps_insert:
            rejected_by_insert += 1
            continue

        stats = _module1_v3_roi_stats(raw, cx, cy, water_radius_pixels)
        if not stats:
            continue

        checked_count += 1
        score = abs(stats["mean"]) * 1.8 + stats["std"] * 3.8 + radial_fraction * 6.0
        candidate = {
            "cx": cx,
            "cy": cy,
            "radius": float(water_radius_pixels),
            "mean": stats["mean"],
            "std": stats["std"],
            "score": float(score),
            "radialFraction": float(radial_fraction),
        }

        if best_candidate is None or candidate["score"] < best_candidate["score"]:
            best_candidate = candidate

    if best_candidate is None:
        best_candidate = {
            "cx": float(phantom_cx),
            "cy": float(phantom_cy),
            "radius": float(water_radius_pixels),
            "score": 9999.0,
            "radialFraction": 0.0,
            "fallbackUsed": True,
        }

    best_candidate["searchDiagnostics"] = {
        "method": "grid search inside phantom, avoid insert circles, prefer HU near 0 and low SD",
        "checkedCandidateCount": int(checked_count),
        "rejectedByInsertCount": int(rejected_by_insert),
        "rejectedByEdgeCount": int(rejected_by_edge),
    }
    return best_candidate

# END MODULE1 V3 SMART WATER HELPERS

# BEGIN MODULE1 V10 CARDINAL EDGE-BB ANALYSIS SELECTOR

MODULE1_ANALYSIS_VERSION = "ACR_MODULE1_CT_NUMBER_QUICK_V10_CARDINAL_EDGE_BB_SELECTOR_2026_07_15"


def _module1_v10_candidate_indices(
    classification: dict[str, Any],
    slice_count: int,
) -> list[int]:
    indices: list[int] = []

    for group in classification.get("groups") or []:
        if group.get("prediction") != MODULE_1:
            continue

        start = int(group.get("startSliceIndex", 0))
        end = int(group.get("endSliceIndex", start))

        for index in range(start, end + 1):
            if 0 <= index < slice_count:
                indices.append(index)

    for item in classification.get("slices") or []:
        if item.get("prediction") != MODULE_1:
            continue

        index = int(
            item.get(
                "sliceIndex",
                int(item.get("sliceNumber", 1)) - 1,
            )
        )

        if 0 <= index < slice_count:
            indices.append(index)

    if not indices:
        indices = list(range(0, min(slice_count, 8)))

    return sorted(set(indices))


def _module1_v10_classifier_score_for_index(
    classification: dict[str, Any],
    index: int,
) -> float:
    for item in classification.get("slices") or []:
        item_index = int(
            item.get(
                "sliceIndex",
                int(item.get("sliceNumber", 1)) - 1,
            )
        )

        if item_index == index:
            return float(
                (item.get("scores") or {}).get(MODULE_1, 0.0)
            )

    return 0.0


def _module1_v10_group_for_index(
    classification: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    for group in classification.get("groups") or []:
        if group.get("prediction") != MODULE_1:
            continue

        if (
            int(group.get("startSliceIndex", -1))
            <= index
            <= int(group.get("endSliceIndex", -1))
        ):
            return group

    return None


def _module1_v10_slice_record_for_index(
    classification: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    for item in classification.get("slices") or []:
        item_index = int(
            item.get(
                "sliceIndex",
                int(item.get("sliceNumber", 1)) - 1,
            )
        )

        if item_index == index:
            return item

    return None


def _module1_v10_basic_roi_stats(
    raw: np.ndarray,
    cx: float,
    cy: float,
    radius_pixels: float,
) -> dict[str, float] | None:
    try:
        mask = _ellipse_mask_pixels(
            shape=raw.shape,
            cx=cx,
            cy=cy,
            radius_x=radius_pixels,
            radius_y=radius_pixels,
        )
        values = raw[mask]
        values = values[np.isfinite(values)]

        if values.size < 8:
            return None

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": int(values.size),
        }
    except Exception:
        return None


def _module1_v10_detect_small_edge_bbs(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
) -> list[dict[str, Any]]:
    """
    Detect the small cardinal BBs near the outside edge of the Module 1 slice.

    Wanted pattern:
        top BB
        bottom BB
        left BB
        right BB
    """
    if cv2 is None:
        return []

    image_8bit = _normalize_to_uint8(raw)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(image_8bit)

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        1.1,
    )

    height, width = raw.shape
    min_dim = min(height, width)

    min_radius = max(2, int(round(min_dim * 0.004)))
    max_radius = max(min_radius + 2, int(round(min_dim * 0.032)))
    min_dist = max(10, int(round(min_dim * 0.030)))

    raw_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=min_dist,
        param1=38,
        param2=8,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if raw_circles is None:
        return []

    candidates: list[dict[str, Any]] = []

    for x_value, y_value, radius in raw_circles[0, :]:
        x_float = float(x_value)
        y_float = float(y_value)
        r_float = float(radius)

        dx = x_float - float(phantom_cx)
        dy = y_float - float(phantom_cy)
        distance = math.hypot(dx, dy)
        ratio = distance / max(float(phantom_radius), 1e-6)

        # These are edge BBs, not the large material inserts.
        if ratio < 0.55 or ratio > 1.10:
            continue

        if r_float > float(phantom_radius) * 0.075:
            continue

        # Ignore pure noise: the BB should have measurable local contrast/noise.
        stats = _module1_v10_basic_roi_stats(
            raw=raw,
            cx=x_float,
            cy=y_float,
            radius_pixels=max(3.0, r_float),
        )

        if stats is None:
            continue

        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        candidates.append({
            "x": x_float,
            "y": y_float,
            "radius": r_float,
            "distanceRatio": ratio,
            "angleDegrees": angle,
            "meanHU": round(float(stats["mean"]), 2),
            "stdHU": round(float(stats["std"]), 2),
        })

    # Merge Hough duplicates.
    merged: list[dict[str, Any]] = []

    for candidate in sorted(candidates, key=lambda item: float(item["radius"]), reverse=True):
        duplicate = False

        for kept in merged:
            if math.hypot(
                float(candidate["x"]) - float(kept["x"]),
                float(candidate["y"]) - float(kept["y"]),
            ) < max(float(candidate["radius"]), float(kept["radius"])) * 2.2:
                duplicate = True
                break

        if not duplicate:
            merged.append(candidate)

    return sorted(
        merged,
        key=lambda item: abs(float(item["distanceRatio"]) - 0.84),
    )[:10]


def _module1_v10_cardinal_slot_for_bb(
    bb: dict[str, Any],
    phantom_cx: float,
    phantom_cy: float,
) -> tuple[str, float]:
    dx = float(bb["x"]) - float(phantom_cx)
    dy = float(bb["y"]) - float(phantom_cy)

    # Image coordinates: y smaller = top, y larger = bottom.
    if abs(dx) >= abs(dy):
        if dx < 0:
            slot = "left"
            ideal_angle = 180.0
        else:
            slot = "right"
            ideal_angle = 0.0
    else:
        if dy < 0:
            slot = "top"
            ideal_angle = 270.0
        else:
            slot = "bottom"
            ideal_angle = 90.0

    angle = float(bb["angleDegrees"])
    angular_error = abs(((angle - ideal_angle + 180.0) % 360.0) - 180.0)

    return slot, float(angular_error)


def _module1_v10_cardinal_edge_bb_score(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
) -> tuple[float, dict[str, Any]]:
    bbs = _module1_v10_detect_small_edge_bbs(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    slots: dict[str, dict[str, Any]] = {}

    for bb in bbs:
        slot, angular_error = _module1_v10_cardinal_slot_for_bb(
            bb=bb,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
        )

        radial_error = abs(float(bb["distanceRatio"]) - 0.84)
        candidate_score = angular_error * 1.8 + radial_error * 85.0

        chosen = {
            **bb,
            "slot": slot,
            "angularError": round(float(angular_error), 3),
            "radialError": round(float(radial_error), 4),
            "slotFitScore": round(float(candidate_score), 3),
        }

        if slot not in slots or candidate_score < float(slots[slot]["slotFitScore"]):
            slots[slot] = chosen

    wanted_slots = ["top", "bottom", "left", "right"]
    slot_count = sum(1 for slot in wanted_slots if slot in slots)

    if slot_count == 4:
        slot_score = 220.0
    elif slot_count == 3:
        slot_score = 110.0
    elif slot_count == 2:
        slot_score = 40.0
    elif slot_count == 1:
        slot_score = 10.0
    else:
        slot_score = 0.0

    selected_bbs = [slots[slot] for slot in wanted_slots if slot in slots]

    if selected_bbs:
        mean_angular_error = float(np.mean([float(bb["angularError"]) for bb in selected_bbs]))
        mean_radial_error = float(np.mean([float(bb["radialError"]) for bb in selected_bbs]))

        alignment_score = max(0.0, 80.0 - mean_angular_error * 2.2)
        radial_score = max(0.0, 45.0 - mean_radial_error * 160.0)
    else:
        mean_angular_error = 999.0
        mean_radial_error = 999.0
        alignment_score = 0.0
        radial_score = 0.0

    total_score = float(slot_score + alignment_score + radial_score)

    details = {
        "method": "cardinal edge BB pattern: top, bottom, left, right",
        "edgeBbRawCandidateCount": int(len(bbs)),
        "cardinalSlotCount": int(slot_count),
        "cardinalSlotsPresent": [slot for slot in wanted_slots if slot in slots],
        "missingCardinalSlots": [slot for slot in wanted_slots if slot not in slots],
        "meanAngularError": round(float(mean_angular_error), 3),
        "meanRadialError": round(float(mean_radial_error), 4),
        "cardinalEdgeBbScore": round(float(total_score), 3),
        "selectedEdgeBbs": selected_bbs,
        "rawEdgeBbCandidates": bbs,
    }

    return total_score, details


def _module1_score_candidate_slice(
    index: int,
    raw: np.ndarray,
    classification: dict[str, Any],
) -> dict[str, Any]:
    """
    V10 Module 1 analysis slice score.

    This brings the same cardinal edge-BB idea used in Split Modules into the
    Analyze Module 1 button.
    """
    phantom_cx, phantom_cy, phantom_radius = _estimate_phantom_geometry(raw)

    circles, detection_info = _detect_insert_circles(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    insert_count = len(circles)
    classifier_score = _module1_v10_classifier_score_for_index(
        classification,
        index,
    )

    if circles:
        radii = np.asarray(
            [circle[2] for circle in circles],
            dtype=np.float32,
        )
        radius_cv = float(
            np.std(radii)
            / max(float(np.mean(radii)), 1e-6)
        )
    else:
        radius_cv = 9.0

    if insert_count == 4:
        four_insert_bonus = 115.0
    elif insert_count == 5:
        four_insert_bonus = 65.0
    elif insert_count == 3:
        four_insert_bonus = 45.0
    else:
        four_insert_bonus = max(
            0.0,
            25.0 - abs(insert_count - 4) * 14.0,
        )

    cardinal_score, cardinal_details = _module1_v10_cardinal_edge_bb_score(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    score = (
        cardinal_score * 1.60
        + four_insert_bonus
        + min(insert_count, 4) * 14.0
        + classifier_score * 0.25
        - radius_cv * 28.0
    )

    result = {
        "sliceIndex": int(index),
        "sliceNumber": int(index) + 1,
        "score": round(float(score), 3),
        "detectedCircleCount": int(insert_count),
        "classifierScore": round(float(classifier_score), 3),
        "radiusCoefficientOfVariation": round(float(radius_cv), 4),
        "fourInsertBonus": round(float(four_insert_bonus), 3),
        "phantomCenterX": round(float(phantom_cx), 3),
        "phantomCenterY": round(float(phantom_cy), 3),
        "phantomRadius": round(float(phantom_radius), 3),
        "detection": detection_info,
        "circles": circles,
    }

    result.update(cardinal_details)

    return result


def _choose_best_module1_slice_by_four_circles(
    slices: list[dict[str, Any]],
    classification: dict[str, Any],
) -> tuple[
    int,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[str],
]:
    warnings: list[str] = []

    candidate_indices = _module1_v10_candidate_indices(
        classification=classification,
        slice_count=len(slices),
    )

    scored_candidates: list[dict[str, Any]] = []

    for index in candidate_indices:
        selected = slices[index]
        info = selected.get("info", {})

        if info.get("isColorDicom"):
            continue

        raw = np.asarray(selected["pixels"], dtype=np.float32)

        if raw.ndim != 2:
            continue

        try:
            scored_candidates.append(
                _module1_score_candidate_slice(
                    index=index,
                    raw=raw,
                    classification=classification,
                )
            )
        except Exception as exc:
            scored_candidates.append({
                "sliceIndex": int(index),
                "sliceNumber": int(index) + 1,
                "score": -9999.0,
                "detectedCircleCount": 0,
                "cardinalSlotCount": 0,
                "error": str(exc),
            })

    scored_candidates = sorted(
        scored_candidates,
        key=lambda item: (
            float(item.get("cardinalSlotCount", 0)),
            float(item.get("cardinalEdgeBbScore", 0.0)),
            float(item.get("detectedCircleCount", 0)),
            float(item.get("score", -9999.0)),
        ),
        reverse=True,
    )

    if not scored_candidates:
        warnings.append(
            "No valid Module 1 candidate slice could be scored. Using the first slice as fallback."
        )
        return (
            0,
            _module1_v10_slice_record_for_index(classification, 0),
            _module1_v10_group_for_index(classification, 0),
            [],
            warnings,
        )

    best = scored_candidates[0]
    best_index = int(best["sliceIndex"])

    if int(best.get("cardinalSlotCount", 0)) < 4:
        warnings.append(
            "The selected Module 1 slice did not show all four cardinal edge BBs "
            "(top, bottom, left, right). Using the strongest available candidate for now."
        )

    return (
        best_index,
        _module1_v10_slice_record_for_index(classification, best_index),
        _module1_v10_group_for_index(classification, best_index),
        scored_candidates,
        warnings,
    )

# END MODULE1 V10 CARDINAL EDGE-BB ANALYSIS SELECTOR

# BEGIN MODULE1 V11 ROI REGION DETECTION OVERRIDE

MODULE1_ANALYSIS_VERSION = "ACR_MODULE1_CT_NUMBER_QUICK_V11_ROI_REGION_DETECTION_2026_07_15"


def _module1_v11_component_candidates(
    raw: np.ndarray,
    mask: np.ndarray,
    label: str,
    short_label: str,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
) -> list[dict[str, Any]]:
    """
    Find connected image regions for one material category.
    This detects the actual visible insert region, not a fixed location.
    """
    if cv2 is None:
        return []

    binary = np.asarray(mask, dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    min_area = math.pi * (phantom_radius * 0.035) ** 2
    max_area = math.pi * (phantom_radius * 0.18) ** 2

    candidates: list[dict[str, Any]] = []

    for component_id in range(1, component_count):
        area = float(stats[component_id, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue

        x_value = float(centroids[component_id][0])
        y_value = float(centroids[component_id][1])

        width = float(stats[component_id, cv2.CC_STAT_WIDTH])
        height = float(stats[component_id, cv2.CC_STAT_HEIGHT])

        if width <= 0 or height <= 0:
            continue

        aspect_ratio = max(width, height) / max(min(width, height), 1e-6)

        if aspect_ratio > 1.8:
            continue

        distance_from_center = math.hypot(
            x_value - phantom_cx,
            y_value - phantom_cy,
        )

        # Material inserts should be inside the phantom and not right on the edge.
        if distance_from_center + max(width, height) * 0.5 > phantom_radius * 0.88:
            continue

        radius = math.sqrt(area / math.pi)

        component_mask = labels == component_id
        component_values = raw[component_mask]
        component_values = component_values[np.isfinite(component_values)]

        if component_values.size < 20:
            continue

        mean_hu = float(np.mean(component_values))
        std_hu = float(np.std(component_values))

        # Module 1 material inserts should sit roughly around the insert ring.
        ring_error = abs(distance_from_center - phantom_radius * 0.43)
        circularity_penalty = abs(aspect_ratio - 1.0) * 35.0
        noise_penalty = min(std_hu, 120.0) * 0.12

        score = ring_error + circularity_penalty + noise_penalty - area * 0.001

        candidates.append({
            "label": label,
            "shortLabel": short_label,
            "x": round(float(x_value), 3),
            "y": round(float(y_value), 3),
            "radius": round(float(radius), 3),
            "areaPixels": int(area),
            "meanHU": round(float(mean_hu), 2),
            "stdHU": round(float(std_hu), 2),
            "aspectRatio": round(float(aspect_ratio), 3),
            "distanceFromCenterPixels": round(float(distance_from_center), 3),
            "score": round(float(score), 3),
            "method": "HU connected component",
        })

    return sorted(candidates, key=lambda item: float(item["score"]))


def _module1_v11_detect_material_inserts(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Detect the four material inserts from actual HU/intensity regions.
    """
    if cv2 is None:
        raise ValueError("OpenCV is not installed. Run: py -m pip install opencv-python")

    height, width = raw.shape
    yy, xx = np.ogrid[:height, :width]

    inside_phantom = (
        (xx - phantom_cx) ** 2
        + (yy - phantom_cy) ** 2
    ) <= (phantom_radius * 0.86) ** 2

    valid = inside_phantom & np.isfinite(raw)

    categories = [
        ("Air", "Air", valid & (raw <= -500)),
        ("Low-density insert", "Low", valid & (raw > -350) & (raw <= -35)),
        ("Acrylic / PMMA-like insert", "Acrylic", valid & (raw >= 40) & (raw <= 260)),
        ("Bone / high-density insert", "Bone", valid & (raw >= 300)),
    ]

    selected: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "method": "HU threshold + connected components",
        "categories": [],
    }

    for label, short_label, mask in categories:
        candidates = _module1_v11_component_candidates(
            raw=raw,
            mask=mask,
            label=label,
            short_label=short_label,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
        )

        diagnostics["categories"].append({
            "label": label,
            "candidateCount": int(len(candidates)),
            "best": candidates[0] if candidates else None,
        })

        if candidates:
            selected.append(candidates[0])

    # Deduplicate in case two HU masks pick the same physical object.
    output: list[dict[str, Any]] = []

    for candidate in sorted(selected, key=lambda item: float(item["score"])):
        duplicate = False

        for kept in output:
            if math.hypot(
                float(candidate["x"]) - float(kept["x"]),
                float(candidate["y"]) - float(kept["y"]),
            ) < max(float(candidate["radius"]), float(kept["radius"])) * 1.30:
                duplicate = True
                break

        if not duplicate:
            output.append(candidate)

    return output, diagnostics


def _module1_v11_add_hough_fallback_inserts(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    If HU connected components miss one insert, add Hough circles as fallback.
    """
    try:
        circles, _info = _detect_insert_circles(
            raw=raw,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
        )
    except Exception:
        return existing

    output = list(existing)
    used_labels = {str(item["label"]) for item in output}

    for x_value, y_value, radius in circles:
        overlaps = False

        for item in output:
            if math.hypot(
                float(x_value) - float(item["x"]),
                float(y_value) - float(item["y"]),
            ) < max(float(radius), float(item["radius"])) * 1.35:
                overlaps = True
                break

        if overlaps:
            continue

        inner_radius = max(4.0, float(radius) * 0.50)
        mask = _ellipse_mask_pixels(
            shape=raw.shape,
            cx=x_value,
            cy=y_value,
            radius_x=inner_radius,
            radius_y=inner_radius,
        )

        values = raw[mask]
        values = values[np.isfinite(values)]

        if values.size < 12:
            continue

        mean_hu = float(np.mean(values))

        if mean_hu <= -500:
            label = "Air"
            short_label = "Air"
        elif -350 < mean_hu <= -35:
            label = "Low-density insert"
            short_label = "Low"
        elif 40 <= mean_hu <= 260:
            label = "Acrylic / PMMA-like insert"
            short_label = "Acrylic"
        elif mean_hu >= 300:
            label = "Bone / high-density insert"
            short_label = "Bone"
        else:
            label = "Detected insert"
            short_label = "Insert"

        if label in used_labels and label != "Detected insert":
            continue

        used_labels.add(label)

        output.append({
            "label": label,
            "shortLabel": short_label,
            "x": round(float(x_value), 3),
            "y": round(float(y_value), 3),
            "radius": round(float(radius), 3),
            "meanHU": round(float(np.mean(values)), 2),
            "stdHU": round(float(np.std(values)), 2),
            "score": 999.0,
            "method": "Hough fallback circle",
        })

        if len(output) >= 4:
            break

    return output


def _module1_v11_find_water_roi(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius: float,
    inserts: list[dict[str, Any]],
    water_radius_pixels: float,
) -> dict[str, Any]:
    """
    Find a water/background ROI from the image itself.

    It searches a water-like HU mask, removes insert regions, then uses distance
    transform to find a clean area where the ROI can fit.
    """
    if cv2 is None:
        return {
            "cx": round(float(phantom_cx), 3),
            "cy": round(float(phantom_cy), 3),
            "radius": round(float(water_radius_pixels), 3),
            "method": "fallback center because OpenCV unavailable",
            "fallbackUsed": True,
        }

    height, width = raw.shape
    yy, xx = np.ogrid[:height, :width]

    central_phantom = (
        (xx - phantom_cx) ** 2
        + (yy - phantom_cy) ** 2
    ) <= (phantom_radius * 0.72) ** 2

    water_like = central_phantom & np.isfinite(raw) & (raw >= -45) & (raw <= 45)

    if int(np.sum(water_like)) < 300:
        water_like = central_phantom & np.isfinite(raw) & (raw >= -90) & (raw <= 90)

    exclusion = np.zeros(raw.shape, dtype=bool)

    for insert in inserts:
        x_value = float(insert["x"])
        y_value = float(insert["y"])
        radius = float(insert["radius"])

        exclusion |= (
            (xx - x_value) ** 2
            + (yy - y_value) ** 2
        ) <= (radius + water_radius_pixels * 2.2) ** 2

    candidate_mask = water_like & (~exclusion)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_mask_u8 = cv2.morphologyEx(
        candidate_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
    )

    if int(np.sum(candidate_mask_u8)) < 100:
        candidate_mask_u8 = (central_phantom & (~exclusion)).astype(np.uint8)

    distance = cv2.distanceTransform(
        candidate_mask_u8.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    _min_value, max_value, _min_loc, max_loc = cv2.minMaxLoc(distance)

    if max_value < max(4.0, water_radius_pixels * 0.65):
        cx = float(phantom_cx)
        cy = float(phantom_cy)
        fallback = True
    else:
        cx = float(max_loc[0])
        cy = float(max_loc[1])
        fallback = False

    mask = _ellipse_mask_pixels(
        shape=raw.shape,
        cx=cx,
        cy=cy,
        radius_x=water_radius_pixels,
        radius_y=water_radius_pixels,
    )

    values = raw[mask]
    values = values[np.isfinite(values)]

    return {
        "cx": round(float(cx), 3),
        "cy": round(float(cy), 3),
        "radius": round(float(water_radius_pixels), 3),
        "method": "water-like HU mask + insert exclusion + distance transform",
        "fallbackUsed": bool(fallback),
        "candidatePixelCount": int(np.sum(candidate_mask_u8)),
        "distanceTransformMax": round(float(max_value), 3),
        "meanHU": round(float(np.mean(values)), 2) if values.size else None,
        "stdHU": round(float(np.std(values)), 2) if values.size else None,
    }


def _module1_v11_expected_range(label: str) -> dict[str, Any]:
    """
    Temporary quick expected bands. These are for UI flagging only and can be
    replaced with your final criteria later.
    """
    if label.startswith("Water"):
        return {"low": -7, "high": 7, "name": "Water approx 0 HU"}
    if "Air" in label:
        return {"low": -1040, "high": -950, "name": "Air approx -1000 HU"}
    if "Low-density" in label:
        return {"low": -150, "high": -40, "name": "Low-density insert rough band"}
    if "Acrylic" in label or "PMMA" in label:
        return {"low": 90, "high": 150, "name": "Acrylic/PMMA rough band"}
    if "Bone" in label or "high-density" in label:
        return {"low": 700, "high": 1100, "name": "Bone/high-density rough band"}

    return {"low": None, "high": None, "name": "No temporary band"}


def _module1_v11_add_quick_result_flag(roi: dict[str, Any]) -> dict[str, Any]:
    band = _module1_v11_expected_range(str(roi["label"]))
    mean_hu = float(roi["meanHU"])

    roi["temporaryExpectedRange"] = band

    if band["low"] is None or band["high"] is None:
        roi["quickFlag"] = "MEASURED"
    elif band["low"] <= mean_hu <= band["high"]:
        roi["quickFlag"] = "OK"
    else:
        roi["quickFlag"] = "CHECK"

    return roi


def create_module1_ct_number_analysis(
    stack_id: str | None = None,
    uploaded_file=None,
    window_width: float = 400,
    window_level: float = 40,
    classification_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slices = _get_slices_from_stack_or_upload(
        stack_id=stack_id,
        uploaded_file=uploaded_file,
    )

    if not slices:
        raise ValueError("No DICOM slices were loaded.")

    if classification_result is None:
        classification_result = create_acr_module_classification(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            max_size=160,
        )

    (
        slice_index,
        slice_record,
        module1_group,
        slice_candidates,
        warnings,
    ) = _choose_best_module1_slice_by_four_circles(
        slices=slices,
        classification=classification_result,
    )

    selected = slices[slice_index]
    info = selected.get("info", {})

    if info.get("isColorDicom"):
        raise ValueError(
            "The selected DICOM is color/secondary-capture data. Module 1 HU measurements require original grayscale CT DICOM."
        )

    raw = np.asarray(selected["pixels"], dtype=np.float32)

    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D CT slice, got shape {raw.shape}.")

    row_spacing = _require_number(
        info.get("pixelSpacingRow"),
        "DICOM PixelSpacing row value",
    )
    col_spacing = _require_number(
        info.get("pixelSpacingCol"),
        "DICOM PixelSpacing column value",
    )

    phantom_cx, phantom_cy, phantom_radius = _estimate_phantom_geometry(raw)

    inserts, insert_detection = _module1_v11_detect_material_inserts(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    if len(inserts) < 4:
        inserts = _module1_v11_add_hough_fallback_inserts(
            raw=raw,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
            existing=inserts,
        )

    # Keep the four expected material inserts first if extra fallback objects appear.
    material_priority = {
        "Air": 0,
        "Low-density insert": 1,
        "Acrylic / PMMA-like insert": 2,
        "Bone / high-density insert": 3,
    }

    inserts = sorted(
        inserts,
        key=lambda item: (
            material_priority.get(str(item["label"]), 99),
            float(item.get("score", 999.0)),
        ),
    )[:4]

    rois: list[dict[str, Any]] = []

    for insert_index, insert in enumerate(inserts, start=1):
        x_value = float(insert["x"])
        y_value = float(insert["y"])
        detected_radius = float(insert["radius"])
        roi_radius = max(4.0, detected_radius * 0.50)

        roi = _measure_roi(
            raw=raw,
            label=str(insert["label"]),
            cx=x_value,
            cy=y_value,
            radius_x_pixels=roi_radius,
            radius_y_pixels=roi_radius,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
            display_radius_pixels=roi_radius,
            full_detected_radius_pixels=detected_radius,
        )

        angle_degrees = (
            math.degrees(math.atan2(y_value - phantom_cy, x_value - phantom_cx))
            + 360.0
        ) % 360.0

        roi["shortLabel"] = str(insert.get("shortLabel", insert["label"]))
        roi["circleNumber"] = int(insert_index)
        roi["detectionMethod"] = str(insert.get("method", "HU connected component"))
        roi["angleDegrees"] = round(float(angle_degrees), 2)
        roi["distanceFromCenterPixels"] = round(
            float(math.hypot(x_value - phantom_cx, y_value - phantom_cy)),
            3,
        )
        roi = _module1_v11_add_quick_result_flag(roi)
        rois.append(roi)

    if len(inserts) < 4:
        warnings.append(
            f"Only {len(inserts)} material insert ROI(s) were detected. Review the overlay."
        )

    if rois:
        water_radius_pixels = float(np.median([float(roi["radiusX"]) for roi in rois]))
    else:
        water_radius_pixels = max(8.0, min(22.0, phantom_radius * 0.09))

    water_radius_pixels = max(5.0, min(water_radius_pixels, phantom_radius * 0.14))

    water_location = _module1_v11_find_water_roi(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        inserts=inserts,
        water_radius_pixels=water_radius_pixels,
    )

    water_roi = _measure_roi(
        raw=raw,
        label="Water / background",
        cx=float(water_location["cx"]),
        cy=float(water_location["cy"]),
        radius_x_pixels=water_radius_pixels,
        radius_y_pixels=water_radius_pixels,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        display_radius_pixels=water_radius_pixels,
        full_detected_radius_pixels=None,
    )
    water_roi["shortLabel"] = "Water"
    water_roi["circleNumber"] = 0
    water_roi["detectionMethod"] = str(water_location["method"])
    water_roi["waterSearch"] = water_location
    water_roi = _module1_v11_add_quick_result_flag(water_roi)
    rois.append(water_roi)

    # Display order: air/low/acrylic/bone/water.
    priority = {
        "Air": 0,
        "Low-density insert": 1,
        "Acrylic / PMMA-like insert": 2,
        "Bone / high-density insert": 3,
        "Water / background": 4,
    }

    rois = sorted(
        rois,
        key=lambda item: priority.get(str(item["label"]), 99),
    )

    overlay = window_pixels_to_image(
        raw,
        float(window_width),
        float(window_level),
        selected.get("photometric", "MONOCHROME2"),
    ).convert("RGB")

    draw = ImageDraw.Draw(overlay)

    draw.ellipse(
        [
            phantom_cx - phantom_radius,
            phantom_cy - phantom_radius,
            phantom_cx + phantom_radius,
            phantom_cy + phantom_radius,
        ],
        outline="yellow",
        width=2,
    )

    # Draw full detected insert regions faintly, then inner measurement ROIs.
    for insert in inserts:
        x_value = float(insert["x"])
        y_value = float(insert["y"])
        radius = float(insert["radius"])

        draw.ellipse(
            [
                x_value - radius,
                y_value - radius,
                x_value + radius,
                y_value + radius,
            ],
            outline="gray",
            width=1,
        )

    for roi in rois:
        cx = float(roi["cx"])
        cy = float(roi["cy"])
        radius_x = float(roi["radiusX"])
        radius_y = float(roi["radiusY"])
        label = str(roi["label"])

        if label.startswith("Water"):
            color = "cyan"
        elif "Air" in label:
            color = "red"
        elif "Bone" in label or "high-density" in label:
            color = "orange"
        elif "Low-density" in label:
            color = "lime"
        else:
            color = "springgreen"

        draw.ellipse(
            [
                cx - radius_x,
                cy - radius_y,
                cx + radius_x,
                cy + radius_y,
            ],
            outline=color,
            width=3,
        )

        draw.text(
            (cx + radius_x + 5, cy - radius_y),
            str(roi.get("shortLabel", label)),
            fill=color,
        )

    overlay_data = _image_to_data_url(overlay)

    module1_range = None

    if module1_group:
        module1_range = {
            "startSliceIndex": int(module1_group["startSliceIndex"]),
            "endSliceIndex": int(module1_group["endSliceIndex"]),
            "startSliceNumber": int(module1_group["startSliceNumber"]),
            "endSliceNumber": int(module1_group["endSliceNumber"]),
            "sliceCount": int(module1_group["sliceCount"]),
        }

    final_status = "REVIEW"
    measured_labels = {str(roi["label"]) for roi in rois}

    if {
        "Air",
        "Low-density insert",
        "Acrylic / PMMA-like insert",
        "Bone / high-density insert",
        "Water / background",
    }.issubset(measured_labels):
        final_status = "MEASURED"

    return {
        "success": True,
        "analysisType": "Quick ACR Module 1 CT Number / Material Insert Analysis",
        "analysisVersion": MODULE1_ANALYSIS_VERSION,
        "classifierVersion": classification_result.get(
            "classifierVersion",
            CLASSIFIER_VERSION,
        ),
        "sliceCount": len(slices),
        "selectedSliceIndex": int(slice_index),
        "selectedSliceNumber": int(slice_index) + 1,
        "selectedSliceLabel": selected.get("label", ""),
        "module1Range": module1_range,
        "selectedSliceClassifierRecord": slice_record,
        "sliceSelection": {
            "method": "cardinal edge-BB selector from Module 1 V10",
            "candidateCount": int(len(slice_candidates)),
            "candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "circles"
                }
                for candidate in slice_candidates[:12]
            ],
        },
        "insertDetection": insert_detection,
        "waterSearch": water_location,
        "phantom": {
            "centerX": round(float(phantom_cx), 3),
            "centerY": round(float(phantom_cy), 3),
            "radiusPixels": round(float(phantom_radius), 3),
            "radiusMmApprox": round(
                float(
                    phantom_radius
                    * ((row_spacing + col_spacing) / 2.0)
                ),
                3,
            ),
        },
        "detection": {
            "method": "HU connected components for inserts + water-like mask distance transform",
            "detectedInsertCount": int(len(inserts)),
            "expectedInsertCount": 4,
        },
        "rois": rois,
        "roiCount": int(len(rois)),
        "finalResult": {
            "status": final_status,
            "note": (
                "MEASURED means all four material inserts plus water/background were found. "
                "Temporary OK/CHECK flags use rough bands only and are not final ACR criteria yet."
            ),
        },
        "displayWindow": {
            "windowWidth": float(window_width),
            "windowLevel": float(window_level),
            "note": (
                "WW/WL controls only the displayed overlay. ROI statistics use raw CT HU values."
            ),
        },
        "overlayImage": overlay_data,
        "image": overlay_data,
        "warnings": warnings,
        "criteriaNote": (
            "Module 1 V11 uses the cardinal edge-BB selected slice, then detects the four material inserts from HU connected components. "
            "Water/background is found from a water-like region inside the phantom while avoiding the material inserts. "
            "ROI locations are image-derived, not hard-coded fixed positions. Temporary expected ranges are included only as rough review flags."
        ),
    }

# END MODULE1 V11 ROI REGION DETECTION OVERRIDE
