
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

# BEGIN MODULE1 V12 LARGE HOUGH INSERT ROIS

MODULE1_ANALYSIS_VERSION = "ACR_MODULE1_CT_NUMBER_QUICK_V12_LARGE_HOUGH_INSERT_ROIS_2026_07_15"


def _m1v12_stats(raw, cx, cy, radius):
    mask = _ellipse_mask_pixels(raw.shape, cx, cy, radius, radius)
    values = raw[mask]
    values = values[np.isfinite(values)]
    if values.size < 12:
        return None
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(values.size),
    }


def _m1v12_label(mean_hu):
    if mean_hu <= -500:
        return "Air", "Air", 0
    if -350 < mean_hu <= -35:
        return "Low-density insert", "Low", 1
    if 40 <= mean_hu <= 260:
        return "Acrylic / PMMA-like insert", "Acrylic", 2
    if mean_hu >= 300:
        return "Bone / high-density insert", "Bone", 3
    return "Detected insert", "Insert", 99


def _m1v12_detect_insert_rois(raw, phantom_cx, phantom_cy, phantom_radius):
    """
    Detect Module 1 material inserts as large circles.
    This avoids V11 connected-component thresholding that could grab wrong patches.
    """
    circles, info = _detect_insert_circles(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    candidates = []

    for x, y, r in circles:
        x = float(x)
        y = float(y)
        r = float(r)
        d = math.hypot(x - phantom_cx, y - phantom_cy)
        ratio = d / max(float(phantom_radius), 1e-6)

        # Inserts should be mid-phantom circles, not cardinal edge BBs and not center water.
        if ratio < 0.16 or ratio > 0.70:
            continue

        roi_r = max(4.0, r * 0.50)
        stats = _m1v12_stats(raw, x, y, roi_r)
        if not stats:
            continue

        mean_hu = float(stats["mean"])
        sd_hu = float(stats["std"])
        label, short_label, priority = _m1v12_label(mean_hu)

        # Do not accept water-like circles as material insert ROIs.
        if priority == 99 and abs(mean_hu) < 35:
            continue

        # Prefer the true material insert ring and HU that is clearly not water.
        ring_error = abs(ratio - 0.43)
        hu_strength = min(1.0, abs(mean_hu) / 1000.0)
        score = ring_error * 120.0 + min(sd_hu, 120.0) * 0.10 - hu_strength * 22.0

        candidates.append({
            "label": label,
            "shortLabel": short_label,
            "priority": int(priority),
            "x": round(x, 3),
            "y": round(y, 3),
            "radius": round(r, 3),
            "roiRadius": round(roi_r, 3),
            "meanHU": round(mean_hu, 2),
            "stdHU": round(sd_hu, 2),
            "distanceRatio": round(ratio, 4),
            "score": round(float(score), 3),
            "method": "large Hough circle + HU label",
        })

    # Choose one per expected material if possible.
    chosen = []
    for priority in [0, 1, 2, 3]:
        matches = [c for c in candidates if int(c.get("priority", 99)) == priority]
        if matches:
            chosen.append(sorted(matches, key=lambda c: float(c["score"]))[0])

    # Fill missing categories with best remaining circles, but avoid duplicates.
    for cand in sorted(candidates, key=lambda c: float(c["score"])):
        if len(chosen) >= 4:
            break
        duplicate = False
        for kept in chosen:
            if math.hypot(float(cand["x"]) - float(kept["x"]), float(cand["y"]) - float(kept["y"])) < max(float(cand["radius"]), float(kept["radius"])) * 1.1:
                duplicate = True
                break
        if not duplicate:
            chosen.append(cand)

    chosen = chosen[:4]

    return chosen, {
        "method": "large Hough circles filtered to material insert ring, then labeled by measured HU",
        "rawCircleCount": int(len(circles)),
        "candidateCount": int(len(candidates)),
        "chosenCount": int(len(chosen)),
        "chosenLabels": [c["label"] for c in chosen],
        "sourceHoughDiagnostics": info,
        "candidates": candidates[:12],
    }


def _m1v12_find_water(raw, phantom_cx, phantom_cy, phantom_radius, inserts, water_radius):
    """
    Automatic water ROI: searches water-like HU inside phantom and avoids insert circles.
    """
    if cv2 is None:
        return {"cx": float(phantom_cx), "cy": float(phantom_cy), "method": "fallback center"}

    h, w = raw.shape
    yy, xx = np.ogrid[:h, :w]

    inside = ((xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2) <= (phantom_radius * 0.72) ** 2
    water_like = inside & np.isfinite(raw) & (raw >= -45) & (raw <= 45)
    if int(np.sum(water_like)) < 200:
        water_like = inside & np.isfinite(raw) & (raw >= -80) & (raw <= 80)

    exclude = np.zeros(raw.shape, dtype=bool)
    for ins in inserts:
        x = float(ins["x"])
        y = float(ins["y"])
        r = float(ins["radius"])
        exclude |= ((xx - x) ** 2 + (yy - y) ** 2) <= (r + water_radius * 2.2) ** 2

    candidate = water_like & (~exclude)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_u8 = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    if int(np.sum(candidate_u8)) < 80:
        candidate_u8 = (inside & (~exclude)).astype(np.uint8)

    dist = cv2.distanceTransform(candidate_u8.astype(np.uint8), cv2.DIST_L2, 5)
    _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(dist)

    if max_v < max(4.0, water_radius * 0.60):
        cx = float(phantom_cx)
        cy = float(phantom_cy)
        fallback = True
    else:
        cx = float(max_loc[0])
        cy = float(max_loc[1])
        fallback = False

    stats = _m1v12_stats(raw, cx, cy, water_radius) or {"mean": 0.0, "std": 0.0, "count": 0}

    return {
        "cx": round(cx, 3),
        "cy": round(cy, 3),
        "radius": round(float(water_radius), 3),
        "method": "water-like HU mask + insert exclusion + distance transform",
        "fallbackUsed": bool(fallback),
        "candidatePixelCount": int(np.sum(candidate_u8)),
        "distanceTransformMax": round(float(max_v), 3),
        "meanHU": round(float(stats["mean"]), 2),
        "stdHU": round(float(stats["std"]), 2),
        "pixelCount": int(stats["count"]),
    }


def _m1v12_band(label):
    if label.startswith("Water"):
        return {"low": -7, "high": 7, "name": "Water approx 0 HU"}
    if "Air" in label:
        return {"low": -1040, "high": -950, "name": "Air approx -1000 HU"}
    if "Low-density" in label:
        return {"low": -150, "high": -40, "name": "Low-density rough band"}
    if "Acrylic" in label or "PMMA" in label:
        return {"low": 90, "high": 150, "name": "Acrylic/PMMA rough band"}
    if "Bone" in label or "high-density" in label:
        return {"low": 700, "high": 1100, "name": "Bone/high-density rough band"}
    return {"low": None, "high": None, "name": "Measured only"}


def _m1v12_flag(roi):
    band = _m1v12_band(str(roi["label"]))
    roi["temporaryExpectedRange"] = band
    if band["low"] is None or band["high"] is None:
        roi["quickFlag"] = "MEASURED"
    elif float(band["low"]) <= float(roi["meanHU"]) <= float(band["high"]):
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
    slices = _get_slices_from_stack_or_upload(stack_id=stack_id, uploaded_file=uploaded_file)
    if not slices:
        raise ValueError("No DICOM slices were loaded.")

    if classification_result is None:
        classification_result = create_acr_module_classification(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            max_size=160,
        )

    slice_index, slice_record, module1_group, slice_candidates, warnings = _choose_best_module1_slice_by_four_circles(
        slices=slices,
        classification=classification_result,
    )

    selected = slices[slice_index]
    info = selected.get("info", {})
    if info.get("isColorDicom"):
        raise ValueError("The selected DICOM is color/secondary-capture data. Module 1 HU measurements require original grayscale CT DICOM.")

    raw = np.asarray(selected["pixels"], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D CT slice, got shape {raw.shape}.")

    row_spacing = _require_number(info.get("pixelSpacingRow"), "DICOM PixelSpacing row value")
    col_spacing = _require_number(info.get("pixelSpacingCol"), "DICOM PixelSpacing column value")

    phantom_cx, phantom_cy, phantom_radius = _estimate_phantom_geometry(raw)

    inserts, insert_detection = _m1v12_detect_insert_rois(raw, phantom_cx, phantom_cy, phantom_radius)

    rois = []
    for i, ins in enumerate(inserts, start=1):
        x = float(ins["x"])
        y = float(ins["y"])
        r = float(ins["radius"])
        roi_r = max(4.0, r * 0.50)

        roi = _measure_roi(
            raw=raw,
            label=str(ins["label"]),
            cx=x,
            cy=y,
            radius_x_pixels=roi_r,
            radius_y_pixels=roi_r,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
            display_radius_pixels=roi_r,
            full_detected_radius_pixels=r,
        )
        roi["shortLabel"] = str(ins.get("shortLabel", ins["label"]))
        roi["circleNumber"] = int(i)
        roi["detectionMethod"] = str(ins.get("method", "large Hough circle + HU label"))
        roi["angleDegrees"] = round(float((math.degrees(math.atan2(y - phantom_cy, x - phantom_cx)) + 360.0) % 360.0), 2)
        roi["distanceFromCenterPixels"] = round(float(math.hypot(x - phantom_cx, y - phantom_cy)), 3)
        rois.append(_m1v12_flag(roi))

    if len(inserts) < 4:
        warnings.append(f"Only {len(inserts)} material insert ROI(s) were detected. Review the overlay.")

    water_radius = float(np.median([float(r["radiusX"]) for r in rois])) if rois else max(8.0, min(22.0, phantom_radius * 0.09))
    water_radius = max(5.0, min(water_radius, phantom_radius * 0.14))

    water = _m1v12_find_water(raw, phantom_cx, phantom_cy, phantom_radius, inserts, water_radius)
    water_roi = _measure_roi(
        raw=raw,
        label="Water / background",
        cx=float(water["cx"]),
        cy=float(water["cy"]),
        radius_x_pixels=water_radius,
        radius_y_pixels=water_radius,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        display_radius_pixels=water_radius,
        full_detected_radius_pixels=None,
    )
    water_roi["shortLabel"] = "Water"
    water_roi["circleNumber"] = 0
    water_roi["detectionMethod"] = str(water["method"])
    water_roi["waterSearch"] = water
    rois.append(_m1v12_flag(water_roi))

    priority = {
        "Air": 0,
        "Low-density insert": 1,
        "Acrylic / PMMA-like insert": 2,
        "Bone / high-density insert": 3,
        "Water / background": 4,
    }
    rois = sorted(rois, key=lambda r: priority.get(str(r["label"]), 99))

    overlay = window_pixels_to_image(
        raw,
        float(window_width),
        float(window_level),
        selected.get("photometric", "MONOCHROME2"),
    ).convert("RGB")
    draw = ImageDraw.Draw(overlay)

    draw.ellipse(
        [phantom_cx - phantom_radius, phantom_cy - phantom_radius, phantom_cx + phantom_radius, phantom_cy + phantom_radius],
        outline="yellow",
        width=2,
    )

    for ins in inserts:
        x = float(ins["x"])
        y = float(ins["y"])
        r = float(ins["radius"])
        draw.ellipse([x-r, y-r, x+r, y+r], outline="gray", width=1)

    for roi in rois:
        cx = float(roi["cx"])
        cy = float(roi["cy"])
        rx = float(roi["radiusX"])
        ry = float(roi["radiusY"])
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

        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], outline=color, width=3)
        draw.text((cx + rx + 5, cy - ry), str(roi.get("shortLabel", label)), fill=color)

    module1_range = None
    if module1_group:
        module1_range = {
            "startSliceIndex": int(module1_group["startSliceIndex"]),
            "endSliceIndex": int(module1_group["endSliceIndex"]),
            "startSliceNumber": int(module1_group["startSliceNumber"]),
            "endSliceNumber": int(module1_group["endSliceNumber"]),
            "sliceCount": int(module1_group["sliceCount"]),
        }

    expected = {"Air", "Low-density insert", "Acrylic / PMMA-like insert", "Bone / high-density insert", "Water / background"}
    measured = {str(r["label"]) for r in rois}
    status = "MEASURED" if expected.issubset(measured) else "REVIEW"

    overlay_data = _image_to_data_url(overlay)

    return {
        "success": True,
        "analysisType": "Quick ACR Module 1 CT Number / Material Insert Analysis",
        "analysisVersion": MODULE1_ANALYSIS_VERSION,
        "classifierVersion": classification_result.get("classifierVersion", CLASSIFIER_VERSION),
        "sliceCount": len(slices),
        "selectedSliceIndex": int(slice_index),
        "selectedSliceNumber": int(slice_index) + 1,
        "selectedSliceLabel": selected.get("label", ""),
        "module1Range": module1_range,
        "selectedSliceClassifierRecord": slice_record,
        "sliceSelection": {
            "method": "cardinal edge-BB selected Module 1 slice",
            "candidateCount": int(len(slice_candidates)),
            "candidates": [{k: v for k, v in c.items() if k != "circles"} for c in slice_candidates[:12]],
        },
        "insertDetection": insert_detection,
        "waterSearch": water,
        "phantom": {
            "centerX": round(float(phantom_cx), 3),
            "centerY": round(float(phantom_cy), 3),
            "radiusPixels": round(float(phantom_radius), 3),
            "radiusMmApprox": round(float(phantom_radius * ((row_spacing + col_spacing) / 2.0)), 3),
        },
        "detection": {
            "method": "large Hough circle material inserts + automatic water-like background",
            "detectedInsertCount": int(len(inserts)),
            "expectedInsertCount": 4,
        },
        "rois": rois,
        "roiCount": int(len(rois)),
        "finalResult": {
            "status": status,
            "note": "MEASURED means all four material inserts plus water/background were found. Temporary OK/CHECK flags use rough review bands only.",
        },
        "displayWindow": {
            "windowWidth": float(window_width),
            "windowLevel": float(window_level),
            "note": "WW/WL controls only the displayed overlay. ROI statistics use raw CT HU values.",
        },
        "overlayImage": overlay_data,
        "image": overlay_data,
        "warnings": warnings,
        "criteriaNote": "Module 1 V12 detects material ROIs as large circular inserts and labels them by measured HU. It avoids fixed x/y positions and avoids the V11 random threshold-patch behavior.",
    }

# END MODULE1 V12 LARGE HOUGH INSERT ROIS

# BEGIN MODULE1 V13 PHYSICAL ROI SIZE OVERRIDE

MODULE1_ANALYSIS_VERSION = "ACR_MODULE1_CT_NUMBER_QUICK_V13_PHYSICAL_ROI_SIZE_2026_07_15"

M1_V13_MATERIAL_ROI_AREA_MM2 = 100.0
M1_V13_WATER_ROI_AREA_MM2 = 400.0


def _m1v13_radius_for_area_mm2(
    area_mm2: float,
    row_spacing_mm: float,
    col_spacing_mm: float,
    min_radius_pixels: float = 4.0,
    max_radius_pixels: float | None = None,
) -> float:
    """
    Convert target physical ROI area to pixel radius.

    area = pi * rx * ry * row_spacing * col_spacing
    Here rx = ry = radius_pixels.
    """
    pixel_area_mm2 = max(float(row_spacing_mm) * float(col_spacing_mm), 1e-6)
    radius_pixels = math.sqrt(float(area_mm2) / (math.pi * pixel_area_mm2))

    radius_pixels = max(float(min_radius_pixels), radius_pixels)

    if max_radius_pixels is not None:
        radius_pixels = min(radius_pixels, float(max_radius_pixels))

    return float(radius_pixels)


def _m1v13_add_area_policy(roi: dict[str, Any], target_area_mm2: float, capped: bool) -> dict[str, Any]:
    roi["targetAreaMm2"] = round(float(target_area_mm2), 2)
    roi["areaPolicy"] = (
        "physical-area ROI, capped to remain inside detected insert"
        if capped
        else "physical-area ROI"
    )
    return roi


def create_module1_ct_number_analysis(
    stack_id: str | None = None,
    uploaded_file=None,
    window_width: float = 400,
    window_level: float = 40,
    classification_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    V13 keeps the V12 large-circle ROI locations but makes ROI size controlled
    by physical area instead of arbitrary 50% circle radius.
    """
    slices = _get_slices_from_stack_or_upload(stack_id=stack_id, uploaded_file=uploaded_file)

    if not slices:
        raise ValueError("No DICOM slices were loaded.")

    if classification_result is None:
        classification_result = create_acr_module_classification(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            max_size=160,
        )

    slice_index, slice_record, module1_group, slice_candidates, warnings = _choose_best_module1_slice_by_four_circles(
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

    row_spacing = _require_number(info.get("pixelSpacingRow"), "DICOM PixelSpacing row value")
    col_spacing = _require_number(info.get("pixelSpacingCol"), "DICOM PixelSpacing column value")

    phantom_cx, phantom_cy, phantom_radius = _estimate_phantom_geometry(raw)

    # Reuse the V12 large-circle detector.
    inserts, insert_detection = _m1v12_detect_insert_rois(
        raw,
        phantom_cx,
        phantom_cy,
        phantom_radius,
    )

    rois: list[dict[str, Any]] = []

    for index, insert in enumerate(inserts, start=1):
        x_value = float(insert["x"])
        y_value = float(insert["y"])
        detected_radius = float(insert["radius"])

        uncapped_radius = _m1v13_radius_for_area_mm2(
            area_mm2=M1_V13_MATERIAL_ROI_AREA_MM2,
            row_spacing_mm=row_spacing,
            col_spacing_mm=col_spacing,
            min_radius_pixels=4.0,
            max_radius_pixels=None,
        )

        max_safe_radius = max(4.0, detected_radius * 0.55)
        roi_radius = min(uncapped_radius, max_safe_radius)
        was_capped = roi_radius < uncapped_radius - 0.01

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
        roi["circleNumber"] = int(index)
        roi["detectionMethod"] = str(insert.get("method", "large Hough circle + HU label"))
        roi["angleDegrees"] = round(float(angle_degrees), 2)
        roi["distanceFromCenterPixels"] = round(
            float(math.hypot(x_value - phantom_cx, y_value - phantom_cy)),
            3,
        )
        roi = _m1v12_flag(roi)
        roi = _m1v13_add_area_policy(
            roi,
            target_area_mm2=M1_V13_MATERIAL_ROI_AREA_MM2,
            capped=was_capped,
        )
        rois.append(roi)

    if len(inserts) < 4:
        warnings.append(
            f"Only {len(inserts)} material insert ROI(s) were detected. Review the overlay."
        )

    water_radius = _m1v13_radius_for_area_mm2(
        area_mm2=M1_V13_WATER_ROI_AREA_MM2,
        row_spacing_mm=row_spacing,
        col_spacing_mm=col_spacing,
        min_radius_pixels=5.0,
        max_radius_pixels=phantom_radius * 0.16,
    )

    water = _m1v12_find_water(
        raw,
        phantom_cx,
        phantom_cy,
        phantom_radius,
        inserts,
        water_radius,
    )

    water_roi = _measure_roi(
        raw=raw,
        label="Water / background",
        cx=float(water["cx"]),
        cy=float(water["cy"]),
        radius_x_pixels=water_radius,
        radius_y_pixels=water_radius,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        display_radius_pixels=water_radius,
        full_detected_radius_pixels=None,
    )
    water_roi["shortLabel"] = "Water"
    water_roi["circleNumber"] = 0
    water_roi["detectionMethod"] = str(water["method"])
    water_roi["waterSearch"] = water
    water_roi = _m1v12_flag(water_roi)
    water_roi = _m1v13_add_area_policy(
        water_roi,
        target_area_mm2=M1_V13_WATER_ROI_AREA_MM2,
        capped=False,
    )
    rois.append(water_roi)

    priority = {
        "Air": 0,
        "Low-density insert": 1,
        "Acrylic / PMMA-like insert": 2,
        "Bone / high-density insert": 3,
        "Water / background": 4,
    }
    rois = sorted(rois, key=lambda r: priority.get(str(r["label"]), 99))

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

    # Full detected insert circle.
    for insert in inserts:
        x_value = float(insert["x"])
        y_value = float(insert["y"])
        detected_radius = float(insert["radius"])
        draw.ellipse(
            [
                x_value - detected_radius,
                y_value - detected_radius,
                x_value + detected_radius,
                y_value + detected_radius,
            ],
            outline="gray",
            width=1,
        )

    # Measurement ROIs.
    for roi in rois:
        cx = float(roi["cx"])
        cy = float(roi["cy"])
        rx = float(roi["radiusX"])
        ry = float(roi["radiusY"])
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

        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=color, width=3)
        draw.text((cx + rx + 5, cy - ry), str(roi.get("shortLabel", label)), fill=color)

    module1_range = None

    if module1_group:
        module1_range = {
            "startSliceIndex": int(module1_group["startSliceIndex"]),
            "endSliceIndex": int(module1_group["endSliceIndex"]),
            "startSliceNumber": int(module1_group["startSliceNumber"]),
            "endSliceNumber": int(module1_group["endSliceNumber"]),
            "sliceCount": int(module1_group["sliceCount"]),
        }

    expected = {
        "Air",
        "Low-density insert",
        "Acrylic / PMMA-like insert",
        "Bone / high-density insert",
        "Water / background",
    }
    measured = {str(r["label"]) for r in rois}
    status = "MEASURED" if expected.issubset(measured) else "REVIEW"

    overlay_data = _image_to_data_url(overlay)

    return {
        "success": True,
        "analysisType": "Quick ACR Module 1 CT Number / Material Insert Analysis",
        "analysisVersion": MODULE1_ANALYSIS_VERSION,
        "classifierVersion": classification_result.get("classifierVersion", CLASSIFIER_VERSION),
        "sliceCount": len(slices),
        "selectedSliceIndex": int(slice_index),
        "selectedSliceNumber": int(slice_index) + 1,
        "selectedSliceLabel": selected.get("label", ""),
        "module1Range": module1_range,
        "selectedSliceClassifierRecord": slice_record,
        "sliceSelection": {
            "method": "cardinal edge-BB selected Module 1 slice",
            "candidateCount": int(len(slice_candidates)),
            "candidates": [
                {k: v for k, v in c.items() if k != "circles"}
                for c in slice_candidates[:12]
            ],
        },
        "roiSizePolicy": {
            "materialInsertTargetAreaMm2": M1_V13_MATERIAL_ROI_AREA_MM2,
            "waterTargetAreaMm2": M1_V13_WATER_ROI_AREA_MM2,
            "note": "Material insert ROIs are capped at 55% of the detected insert radius so they do not cross insert edges.",
        },
        "insertDetection": insert_detection,
        "waterSearch": water,
        "phantom": {
            "centerX": round(float(phantom_cx), 3),
            "centerY": round(float(phantom_cy), 3),
            "radiusPixels": round(float(phantom_radius), 3),
            "radiusMmApprox": round(float(phantom_radius * ((row_spacing + col_spacing) / 2.0)), 3),
        },
        "detection": {
            "method": "large Hough circle material inserts + fixed physical-area ROIs",
            "detectedInsertCount": int(len(inserts)),
            "expectedInsertCount": 4,
        },
        "rois": rois,
        "roiCount": int(len(rois)),
        "finalResult": {
            "status": status,
            "note": "MEASURED means all four material inserts plus water/background were found. Temporary OK/CHECK flags use rough review bands only.",
        },
        "displayWindow": {
            "windowWidth": float(window_width),
            "windowLevel": float(window_level),
            "note": "WW/WL controls only the displayed overlay. ROI statistics use raw CT HU values.",
        },
        "overlayImage": overlay_data,
        "image": overlay_data,
        "warnings": warnings,
        "criteriaNote": (
            "Module 1 V13 keeps V12 insert locations, but uses physical-area ROIs: "
            "100 mm² for material inserts and 400 mm² for water/background. "
            "Material insert ROIs are capped to stay inside detected circles."
        ),
    }

# END MODULE1 V13 PHYSICAL ROI SIZE OVERRIDE

# BEGIN MODULE1 V14 RESULT SUMMARY WRAPPER

MODULE1_ANALYSIS_VERSION_V14 = "ACR_MODULE1_CT_NUMBER_QUICK_V14_RESULT_SUMMARY_2026_07_15"


def _m1v14_label_key(label):
    text = str(label or "").lower()

    if "water" in text:
        return "water"
    if "air" in text:
        return "air"
    if "low-density" in text or "polyethylene" in text or "low density" in text:
        return "low_density"
    if "acrylic" in text or "pmma" in text:
        return "acrylic"
    if "bone" in text or "high-density" in text or "high density" in text:
        return "bone"

    return "unknown"


def _m1v14_expected_range_for_key(key):
    # These are review bands, not final regulatory criteria.
    ranges = {
        "water": {
            "label": "Water / background",
            "low": -7.0,
            "high": 7.0,
            "target": 0.0,
            "units": "HU",
            "note": "Review band around 0 HU",
        },
        "air": {
            "label": "Air",
            "low": -1040.0,
            "high": -950.0,
            "target": -1000.0,
            "units": "HU",
            "note": "Review band around -1000 HU",
        },
        "low_density": {
            "label": "Low-density insert",
            "low": -150.0,
            "high": -40.0,
            "target": -95.0,
            "units": "HU",
            "note": "Temporary rough review band",
        },
        "acrylic": {
            "label": "Acrylic / PMMA-like insert",
            "low": 90.0,
            "high": 150.0,
            "target": 120.0,
            "units": "HU",
            "note": "Temporary rough review band",
        },
        "bone": {
            "label": "Bone / high-density insert",
            "low": 700.0,
            "high": 1100.0,
            "target": 900.0,
            "units": "HU",
            "note": "Temporary rough review band",
        },
    }

    return ranges.get(key)


def _m1v14_reflag_roi(roi):
    label = str(roi.get("label", ""))
    key = _m1v14_label_key(label)
    band = _m1v14_expected_range_for_key(key)

    roi["materialKey"] = key

    if band is None:
        roi["temporaryExpectedRange"] = {
            "low": None,
            "high": None,
            "name": "Measured only",
        }
        roi["quickFlag"] = "MEASURED"
        roi["rangeDeviationHU"] = None
        return roi

    roi["temporaryExpectedRange"] = {
        "low": band["low"],
        "high": band["high"],
        "name": band["label"] + " review band",
        "note": band["note"],
    }

    try:
        mean_hu = float(roi.get("meanHU"))
    except Exception:
        roi["quickFlag"] = "CHECK"
        roi["rangeDeviationHU"] = None
        return roi

    if band["low"] <= mean_hu <= band["high"]:
        roi["quickFlag"] = "OK"
        roi["rangeDeviationHU"] = 0.0
    elif mean_hu < band["low"]:
        roi["quickFlag"] = "CHECK"
        roi["rangeDeviationHU"] = round(float(mean_hu - band["low"]), 2)
    else:
        roi["quickFlag"] = "CHECK"
        roi["rangeDeviationHU"] = round(float(mean_hu - band["high"]), 2)

    return roi


def _m1v14_build_summary(result):
    rois = result.get("rois") or []
    warnings = list(result.get("warnings") or [])

    updated_rois = []
    roi_by_key = {}

    for roi in rois:
        roi = dict(roi)
        roi = _m1v14_reflag_roi(roi)
        updated_rois.append(roi)

        key = roi.get("materialKey", "unknown")

        # Keep first matching ROI per material. If duplicated, keep the one with lower SD.
        if key not in roi_by_key:
            roi_by_key[key] = roi
        else:
            try:
                old_sd = float(roi_by_key[key].get("stdHU", 999999))
                new_sd = float(roi.get("stdHU", 999999))
                if new_sd < old_sd:
                    roi_by_key[key] = roi
            except Exception:
                pass

    expected_keys = ["air", "low_density", "acrylic", "bone", "water"]
    missing_keys = [key for key in expected_keys if key not in roi_by_key]
    check_keys = [
        key
        for key in expected_keys
        if key in roi_by_key and roi_by_key[key].get("quickFlag") == "CHECK"
    ]

    high_noise = []

    for key in expected_keys:
        roi = roi_by_key.get(key)

        if not roi:
            continue

        try:
            sd = float(roi.get("stdHU", 0.0))
        except Exception:
            sd = 0.0

        if key == "water" and sd > 15.0:
            high_noise.append({
                "materialKey": key,
                "label": roi.get("label"),
                "stdHU": round(sd, 2),
                "note": "Water ROI SD is high; review ROI placement or image noise.",
            })
        elif key != "water" and sd > 35.0:
            high_noise.append({
                "materialKey": key,
                "label": roi.get("label"),
                "stdHU": round(sd, 2),
                "note": "Insert ROI SD is high; review ROI placement.",
            })

    if missing_keys:
        status = "REVIEW"
        summary_note = "One or more expected Module 1 ROIs were not detected."
    elif check_keys:
        status = "CHECK"
        summary_note = "All expected ROIs were measured, but one or more HU values are outside the temporary review bands."
    else:
        status = "MEASURED_OK"
        summary_note = "All expected ROIs were measured and are inside the temporary review bands."

    if high_noise and status == "MEASURED_OK":
        status = "MEASURED_REVIEW"
        summary_note = "All expected ROIs were measured, but at least one ROI has high SD and should be reviewed."

    expected_ranges = {
        key: _m1v14_expected_range_for_key(key)
        for key in expected_keys
    }

    measured_values = {}

    for key in expected_keys:
        roi = roi_by_key.get(key)

        if not roi:
            measured_values[key] = None
            continue

        measured_values[key] = {
            "label": roi.get("label"),
            "meanHU": roi.get("meanHU"),
            "stdHU": roi.get("stdHU"),
            "actualAreaMm2": roi.get("actualAreaMm2"),
            "targetAreaMm2": roi.get("targetAreaMm2"),
            "quickFlag": roi.get("quickFlag"),
            "rangeDeviationHU": roi.get("rangeDeviationHU"),
            "cx": roi.get("cx"),
            "cy": roi.get("cy"),
        }

    if missing_keys:
        warnings.append(
            "Module 1 summary is REVIEW because missing expected ROI(s): "
            + ", ".join(missing_keys)
        )

    if check_keys:
        warnings.append(
            "Module 1 summary is CHECK because these ROI(s) are outside temporary review bands: "
            + ", ".join(check_keys)
        )

    for item in high_noise:
        warnings.append(str(item.get("note", "High ROI SD")) + " " + str(item.get("label", "")))

    result["analysisVersion"] = MODULE1_ANALYSIS_VERSION_V14
    result["rois"] = updated_rois
    result["warnings"] = warnings

    result["module1Summary"] = {
        "status": status,
        "note": summary_note,
        "expectedRoiKeys": expected_keys,
        "missingRoiKeys": missing_keys,
        "checkRoiKeys": check_keys,
        "highNoiseRois": high_noise,
        "measuredValues": measured_values,
        "expectedRanges": expected_ranges,
        "detectedRoiCount": int(len(updated_rois)),
        "detectedExpectedRoiCount": int(len([key for key in expected_keys if key in roi_by_key])),
    }

    result["finalResult"] = {
        "status": status,
        "note": summary_note,
        "missingRoiKeys": missing_keys,
        "checkRoiKeys": check_keys,
        "highNoiseRois": high_noise,
        "important": (
            "This is a review/check result using temporary HU bands, not a final regulatory pass/fail."
        ),
    }

    result["criteriaNote"] = (
        "Module 1 V14 summarizes the V13 measurements into expected ROI keys "
        "(air, low-density, acrylic, bone, water). Temporary HU review bands are used "
        "for OK/CHECK flags only; replace these with your final criteria later."
    )

    return result


_ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V14 = create_module1_ct_number_analysis


def create_module1_ct_number_analysis(*args, **kwargs):
    result = _ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V14(*args, **kwargs)

    try:
        return _m1v14_build_summary(result)
    except Exception as exc:
        result["module1SummaryError"] = str(exc)
        return result

# END MODULE1 V14 RESULT SUMMARY WRAPPER
