
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

# BEGIN MODULE1 V15 SMART WATER AND POLYETHYLENE

MODULE1_ANALYSIS_VERSION_V15 = "ACR_MODULE1_CT_NUMBER_QUICK_V15_SMART_WATER_POLYETHYLENE_2026_07_25"


def _m1v12_label(mean_hu):
    """
    V15 label override.
    Rename the low-density material insert to Polyethylene.
    """
    if mean_hu <= -500:
        return "Air", "Air", 0
    if -350 < mean_hu <= -35:
        return "Polyethylene", "Polyethylene", 1
    if 40 <= mean_hu <= 260:
        return "Acrylic / PMMA-like insert", "Acrylic", 2
    if mean_hu >= 300:
        return "Bone / high-density insert", "Bone", 3
    return "Detected insert", "Insert", 99


def _m1v12_band(label):
    """
    V15 temporary review-band override.
    Low-density is now reported as Polyethylene.
    """
    text = str(label or "")
    lower = text.lower()

    if text.startswith("Water"):
        return {"low": -7, "high": 7, "name": "Water approx 0 HU"}
    if "air" in lower:
        return {"low": -1040, "high": -950, "name": "Air approx -1000 HU"}
    if "polyethylene" in lower or "low-density" in lower or "low density" in lower:
        return {"low": -150, "high": -40, "name": "Polyethylene rough band"}
    if "acrylic" in lower or "pmma" in lower:
        return {"low": 90, "high": 150, "name": "Acrylic/PMMA rough band"}
    if "bone" in lower or "high-density" in lower or "high density" in lower:
        return {"low": 700, "high": 1100, "name": "Bone/high-density rough band"}
    return {"low": None, "high": None, "name": "Measured only"}


def _m1v15_roi_values(raw, cx, cy, radius):
    mask = _ellipse_mask_pixels(
        raw.shape,
        float(cx),
        float(cy),
        float(radius),
        float(radius),
    )
    values = raw[mask]
    values = values[np.isfinite(values)]

    if values.size < 12:
        return None

    return values


def _m1v15_roi_texture_values(texture_map, cx, cy, radius):
    mask = _ellipse_mask_pixels(
        texture_map.shape,
        float(cx),
        float(cy),
        float(radius),
        float(radius),
    )
    values = texture_map[mask]
    values = values[np.isfinite(values)]

    if values.size < 12:
        return None

    return values


def _m1v15_detect_edge_bb_exclusion(raw, phantom_cx, phantom_cy, phantom_radius, water_radius):
    """
    Optional exclusion mask for tiny edge BBs. This is defensive; the main
    water search also stays inside the phantom away from the outer edge.
    """
    try:
        import cv2 as _cv2
    except Exception:
        return np.zeros(raw.shape, dtype=bool), []

    try:
        bbs = _module1_v10_detect_small_edge_bbs(
            raw=raw,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
        )
    except Exception:
        bbs = []

    height, width = raw.shape
    yy, xx = np.ogrid[:height, :width]
    mask = np.zeros(raw.shape, dtype=bool)

    for bb in bbs:
        x_value = float(bb.get("x", 0.0))
        y_value = float(bb.get("y", 0.0))
        radius = float(bb.get("radius", 3.0))
        exclude_radius = max(float(water_radius) * 1.6, radius + float(water_radius))
        mask |= ((xx - x_value) ** 2 + (yy - y_value) ** 2) <= exclude_radius ** 2

    return mask, bbs


def _m1v15_build_texture_masks(raw, phantom_mask, roi_radius):
    """
    Build texture/edge maps so water does not land on line-pair bars or streaks.

    Water/background should be smooth gray phantom material. Line-pair regions
    may have mean HU near water, but they have high local texture/gradient.
    """
    try:
        import cv2 as _cv2
    except Exception:
        zeros = np.zeros(raw.shape, dtype=np.float32)
        return zeros, np.zeros(raw.shape, dtype=bool), {
            "textureMethod": "OpenCV unavailable; texture mask disabled",
        }

    arr = np.asarray(raw, dtype=np.float32)
    finite = arr[np.isfinite(arr)]

    if finite.size < 100:
        zeros = np.zeros(raw.shape, dtype=np.float32)
        return zeros, np.zeros(raw.shape, dtype=bool), {
            "textureMethod": "not enough finite pixels; texture mask disabled",
        }

    # Clip extreme insert values so gradient/texture thresholds are not only
    # driven by air/bone insert edges.
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    clipped = np.clip(arr, low, high).astype(np.float32)

    grad_x = _cv2.Sobel(clipped, _cv2.CV_32F, 1, 0, ksize=3)
    grad_y = _cv2.Sobel(clipped, _cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    kernel_size = int(max(5, round(float(roi_radius) * 0.75)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    mean = _cv2.blur(clipped, (kernel_size, kernel_size))
    mean_sq = _cv2.blur(clipped * clipped, (kernel_size, kernel_size))
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    local_sd = np.sqrt(local_var)

    inside_values = local_sd[phantom_mask & np.isfinite(local_sd)]
    grad_values = grad[phantom_mask & np.isfinite(grad)]

    if inside_values.size < 100 or grad_values.size < 100:
        texture_mask = np.zeros(raw.shape, dtype=bool)
        texture_map = local_sd
        diagnostics = {
            "textureMethod": "local SD and Sobel gradient; insufficient percentile data",
            "kernelSize": int(kernel_size),
        }
        return texture_map, texture_mask, diagnostics

    # Adaptive threshold. This catches line-pair bars without hard-coding their location.
    sd_threshold = max(
        float(np.percentile(inside_values, 82.0)),
        float(np.median(inside_values) + np.std(inside_values) * 0.75),
    )
    grad_threshold = max(
        float(np.percentile(grad_values, 82.0)),
        float(np.median(grad_values) + np.std(grad_values) * 0.75),
    )

    texture_mask = phantom_mask & (
        (local_sd >= sd_threshold)
        | (grad >= grad_threshold)
    )

    try:
        dilate_radius = int(max(3, round(float(roi_radius) * 0.80)))
        kernel = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE,
            (dilate_radius * 2 + 1, dilate_radius * 2 + 1),
        )
        texture_mask = _cv2.dilate(texture_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        pass

    # Use local SD as texture map for candidate scoring.
    texture_map = local_sd.astype(np.float32)

    diagnostics = {
        "textureMethod": "local SD + Sobel gradient, adaptive threshold, dilated",
        "kernelSize": int(kernel_size),
        "localSdThreshold": round(float(sd_threshold), 3),
        "gradientThreshold": round(float(grad_threshold), 3),
        "texturePixelCount": int(np.sum(texture_mask)),
    }

    return texture_map, texture_mask, diagnostics


def _m1v15_min_distance_to_inserts(cx, cy, inserts):
    if not inserts:
        return 999999.0

    min_distance = 999999.0

    for insert in inserts:
        x_value = float(insert.get("x", 0.0))
        y_value = float(insert.get("y", 0.0))
        radius = float(insert.get("radius", 0.0))
        edge_distance = math.hypot(float(cx) - x_value, float(cy) - y_value) - radius

        if edge_distance < min_distance:
            min_distance = edge_distance

    return float(min_distance)


def _m1v15_score_water_candidate(
    raw,
    texture_map,
    forbidden_mask,
    cx,
    cy,
    radius,
    phantom_cx,
    phantom_cy,
    phantom_radius,
    inserts,
):
    values = _m1v15_roi_values(raw, cx, cy, radius)

    if values is None:
        return None

    roi_mask = _ellipse_mask_pixels(raw.shape, float(cx), float(cy), float(radius), float(radius))

    if bool(np.any(forbidden_mask[roi_mask])):
        return None

    texture_values = _m1v15_roi_texture_values(texture_map, cx, cy, radius)
    texture_mean = float(np.mean(texture_values)) if texture_values is not None else 0.0

    mean_hu = float(np.mean(values))
    std_hu = float(np.std(values))

    distance_from_center = math.hypot(float(cx) - float(phantom_cx), float(cy) - float(phantom_cy))
    distance_ratio = distance_from_center / max(float(phantom_radius), 1e-6)

    edge_clearance = float(phantom_radius) - distance_from_center - float(radius)
    insert_clearance = _m1v15_min_distance_to_inserts(cx, cy, inserts) - float(radius)

    # Water can be center-ish, but exact center often intersects line-pair/detail
    # structure in this module. This is only a soft penalty; clean center can still win.
    exact_center_penalty = max(0.0, 0.12 - distance_ratio) * 35.0

    # Stay in open phantom material, not too close to edge.
    edge_penalty = max(0.0, float(radius) * 1.5 - edge_clearance) * 2.0
    insert_penalty = max(0.0, float(radius) * 1.4 - insert_clearance) * 3.0

    score = (
        abs(mean_hu) * 1.9
        + std_hu * 3.8
        + texture_mean * 1.8
        + edge_penalty
        + insert_penalty
        + exact_center_penalty
    )

    return {
        "cx": round(float(cx), 3),
        "cy": round(float(cy), 3),
        "radius": round(float(radius), 3),
        "meanHU": round(float(mean_hu), 2),
        "stdHU": round(float(std_hu), 2),
        "textureMean": round(float(texture_mean), 3),
        "distanceRatio": round(float(distance_ratio), 4),
        "edgeClearancePixels": round(float(edge_clearance), 3),
        "insertClearancePixels": round(float(insert_clearance), 3),
        "score": round(float(score), 3),
    }


def _m1v12_find_water(raw, phantom_cx, phantom_cy, phantom_radius, inserts, water_radius):
    """
    V15 smart water/background ROI finder.

    It does not place water at a fixed center. It scans for a clean water-like
    gray region where a 400 mm² ROI fits while avoiding inserts, edge BBs,
    line-pair bars, high-texture areas, and the phantom edge.
    """
    try:
        import cv2 as _cv2
    except Exception:
        stats = _m1v15_roi_values(raw, phantom_cx, phantom_cy, water_radius)
        return {
            "cx": round(float(phantom_cx), 3),
            "cy": round(float(phantom_cy), 3),
            "radius": round(float(water_radius), 3),
            "method": "fallback center because OpenCV unavailable",
            "fallbackUsed": True,
            "meanHU": round(float(np.mean(stats)), 2) if stats is not None else None,
            "stdHU": round(float(np.std(stats)), 2) if stats is not None else None,
        }

    arr = np.asarray(raw, dtype=np.float32)
    height, width = arr.shape
    yy, xx = np.ogrid[:height, :width]

    # Stay inside the phantom, but do not force center.
    inside_phantom = (
        (xx - float(phantom_cx)) ** 2
        + (yy - float(phantom_cy)) ** 2
    ) <= (float(phantom_radius) * 0.76) ** 2

    # Water-like gray background. Use a strict band first, widen only if needed.
    water_like = inside_phantom & np.isfinite(arr) & (arr >= -45.0) & (arr <= 45.0)

    if int(np.sum(water_like)) < 250:
        water_like = inside_phantom & np.isfinite(arr) & (arr >= -80.0) & (arr <= 80.0)

    forbidden = np.zeros(arr.shape, dtype=bool)

    # Exclude the four material inserts and their partial-volume halos.
    for insert in inserts:
        x_value = float(insert.get("x", 0.0))
        y_value = float(insert.get("y", 0.0))
        radius = float(insert.get("radius", water_radius))
        exclusion_radius = radius + float(water_radius) * 2.25
        forbidden |= ((xx - x_value) ** 2 + (yy - y_value) ** 2) <= exclusion_radius ** 2

    # Exclude edge BBs if detected.
    edge_bb_mask, edge_bbs = _m1v15_detect_edge_bb_exclusion(
        raw=arr,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        water_radius=water_radius,
    )
    forbidden |= edge_bb_mask

    texture_map, texture_mask, texture_diagnostics = _m1v15_build_texture_masks(
        raw=arr,
        phantom_mask=inside_phantom,
        roi_radius=water_radius,
    )

    # Exclude line-pair bars/high-texture areas.
    forbidden |= texture_mask

    # Candidate center must be water-like and not forbidden.
    allowed = water_like & (~forbidden)

    # Distance transform enforces enough open room for the ROI.
    allowed_u8 = allowed.astype(np.uint8)
    dist = _cv2.distanceTransform(allowed_u8, _cv2.DIST_L2, 5)
    enough_room = dist >= max(3.0, float(water_radius) * 0.92)

    if int(np.sum(enough_room)) < 20:
        # Relax texture exclusion if it was too aggressive, but still avoid inserts.
        allowed = water_like & (~(forbidden & (~texture_mask)))
        allowed_u8 = allowed.astype(np.uint8)
        dist = _cv2.distanceTransform(allowed_u8, _cv2.DIST_L2, 5)
        enough_room = dist >= max(3.0, float(water_radius) * 0.85)

    if int(np.sum(enough_room)) < 20:
        # Last resort: any open spot inside phantom away from inserts/edge.
        allowed = inside_phantom & np.isfinite(arr) & (~forbidden)
        allowed_u8 = allowed.astype(np.uint8)
        dist = _cv2.distanceTransform(allowed_u8, _cv2.DIST_L2, 5)
        enough_room = dist >= max(3.0, float(water_radius) * 0.75)

    candidate_points = np.column_stack(np.nonzero(enough_room))

    candidates = []
    fallback_used = False

    if candidate_points.size == 0:
        # Final fallback is still not hard-coded as primary logic; it is only for failure.
        fallback_used = True
        candidate = _m1v15_score_water_candidate(
            raw=arr,
            texture_map=texture_map,
            forbidden_mask=np.zeros(arr.shape, dtype=bool),
            cx=phantom_cx,
            cy=phantom_cy,
            radius=water_radius,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
            inserts=inserts,
        )
        if candidate:
            candidates.append(candidate)
    else:
        # Sample the mask so this stays fast and avoids checking every pixel.
        step = int(max(3, round(float(water_radius) * 0.45)))

        for y_value, x_value in candidate_points:
            if int(x_value) % step != 0 or int(y_value) % step != 0:
                continue

            candidate = _m1v15_score_water_candidate(
                raw=arr,
                texture_map=texture_map,
                forbidden_mask=forbidden,
                cx=float(x_value),
                cy=float(y_value),
                radius=water_radius,
                phantom_cx=phantom_cx,
                phantom_cy=phantom_cy,
                phantom_radius=phantom_radius,
                inserts=inserts,
            )

            if candidate:
                candidates.append(candidate)

        # If grid sampling missed too much, score up to 800 strongest room pixels.
        if not candidates:
            room_values = dist[enough_room]
            sorted_indices = np.argsort(room_values)[-800:]
            selected_points = candidate_points[sorted_indices]

            for y_value, x_value in selected_points:
                candidate = _m1v15_score_water_candidate(
                    raw=arr,
                    texture_map=texture_map,
                    forbidden_mask=forbidden,
                    cx=float(x_value),
                    cy=float(y_value),
                    radius=water_radius,
                    phantom_cx=phantom_cx,
                    phantom_cy=phantom_cy,
                    phantom_radius=phantom_radius,
                    inserts=inserts,
                )
                if candidate:
                    candidates.append(candidate)

    if not candidates:
        fallback_used = True
        cx = float(phantom_cx)
        cy = float(phantom_cy)
        values = _m1v15_roi_values(arr, cx, cy, water_radius)
        return {
            "cx": round(cx, 3),
            "cy": round(cy, 3),
            "radius": round(float(water_radius), 3),
            "method": "fallback center after smart water search found no candidate",
            "fallbackUsed": True,
            "meanHU": round(float(np.mean(values)), 2) if values is not None else None,
            "stdHU": round(float(np.std(values)), 2) if values is not None else None,
            "candidateCount": 0,
            "textureDiagnostics": texture_diagnostics,
        }

    candidates = sorted(candidates, key=lambda item: float(item["score"]))
    best = candidates[0]

    return {
        "cx": float(best["cx"]),
        "cy": float(best["cy"]),
        "radius": round(float(water_radius), 3),
        "method": "V15 smart water search: water-like HU + insert/BB/texture exclusion + ROI scoring",
        "fallbackUsed": bool(fallback_used),
        "meanHU": best["meanHU"],
        "stdHU": best["stdHU"],
        "textureMean": best["textureMean"],
        "score": best["score"],
        "distanceRatio": best["distanceRatio"],
        "edgeClearancePixels": best["edgeClearancePixels"],
        "insertClearancePixels": best["insertClearancePixels"],
        "candidateCount": int(len(candidates)),
        "allowedPixelCount": int(np.sum(allowed)),
        "enoughRoomPixelCount": int(np.sum(enough_room)),
        "edgeBbCountExcluded": int(len(edge_bbs)),
        "textureDiagnostics": texture_diagnostics,
        "topCandidates": candidates[:8],
    }


def _m1v14_expected_range_for_key(key):
    """
    V15 summary-range override.
    """
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
            "label": "Polyethylene",
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


_ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V15 = create_module1_ct_number_analysis


def create_module1_ct_number_analysis(*args, **kwargs):
    result = _ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V15(*args, **kwargs)

    # Make sure returned labels and final summary use Polyethylene wording.
    try:
        for roi in result.get("rois") or []:
            label = str(roi.get("label", ""))
            if "Low-density" in label or "low-density" in label or "Low density" in label:
                roi["label"] = "Polyethylene"
                roi["shortLabel"] = "Polyethylene"
            if "temporaryExpectedRange" in roi:
                rng = roi["temporaryExpectedRange"] or {}
                name = str(rng.get("name", ""))
                if "Low-density" in name or "low-density" in name or "Low density" in name:
                    rng["name"] = name.replace("Low-density", "Polyethylene").replace("low-density", "Polyethylene").replace("Low density", "Polyethylene")
                    roi["temporaryExpectedRange"] = rng

        summary = result.get("module1Summary")
        if isinstance(summary, dict):
            expected_ranges = summary.get("expectedRanges")
            if isinstance(expected_ranges, dict) and "low_density" in expected_ranges:
                expected_ranges["low_density"] = _m1v14_expected_range_for_key("low_density")
            measured = summary.get("measuredValues")
            if isinstance(measured, dict) and "low_density" in measured and measured["low_density"]:
                measured["low_density"]["label"] = "Polyethylene"

        result["analysisVersion"] = MODULE1_ANALYSIS_VERSION_V15

        if isinstance(result.get("finalResult"), dict):
            result["finalResult"]["waterRoiMethod"] = "smart search, not fixed center"
            result["finalResult"]["labelUpdate"] = "Low-density insert renamed to Polyethylene"

        result["criteriaNote"] = (
            "Module 1 V15 keeps existing insert ROI logic but replaces water/background placement with a smart search. "
            "The water ROI is chosen from plain gray water-like phantom material by avoiding inserts, edge BBs, line-pair/high-texture regions, and the phantom edge. "
            "Low-density insert is labeled as Polyethylene."
        )

    except Exception as exc:
        result["module1V15PostprocessError"] = str(exc)

    return result

# END MODULE1 V15 SMART WATER AND POLYETHYLENE

# BEGIN MODULE1 V16 WATER SIDE PATCH

MODULE1_ANALYSIS_VERSION_V16 = "ACR_MODULE1_CT_NUMBER_QUICK_V16_WATER_SIDE_SEARCH_2026_07_25"


def _m1v16_circle_mask(shape, cx, cy, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= float(radius) ** 2


def _m1v16_roi_stats(raw, cx, cy, radius):
    values = _m1v15_roi_values(raw, cx, cy, radius)
    if values is None:
        return None
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(values.size),
    }


def _m1v16_texture_map(raw, phantom_mask, roi_radius):
    """
    Lightweight texture map used only to avoid line-pair / bar regions.
    """
    try:
        import cv2 as _cv2
    except Exception:
        return np.zeros(raw.shape, dtype=np.float32), np.zeros(raw.shape, dtype=bool), {
            "textureMethod": "OpenCV unavailable",
        }

    arr = np.asarray(raw, dtype=np.float32)
    finite = arr[phantom_mask & np.isfinite(arr)]

    if finite.size < 200:
        zeros = np.zeros(raw.shape, dtype=np.float32)
        return zeros, np.zeros(raw.shape, dtype=bool), {
            "textureMethod": "not enough pixels",
        }

    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    clipped = np.clip(arr, low, high).astype(np.float32)

    k = int(max(5, round(float(roi_radius) * 0.75)))
    if k % 2 == 0:
        k += 1

    mean = _cv2.blur(clipped, (k, k))
    mean_sq = _cv2.blur(clipped * clipped, (k, k))
    local_sd = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)

    gx = _cv2.Sobel(clipped, _cv2.CV_32F, 1, 0, ksize=3)
    gy = _cv2.Sobel(clipped, _cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    sd_values = local_sd[phantom_mask & np.isfinite(local_sd)]
    grad_values = grad[phantom_mask & np.isfinite(grad)]

    sd_threshold = float(np.percentile(sd_values, 78.0))
    grad_threshold = float(np.percentile(grad_values, 78.0))

    texture_mask = phantom_mask & ((local_sd >= sd_threshold) | (grad >= grad_threshold))

    try:
        dilate_radius = int(max(3, round(float(roi_radius) * 0.65)))
        kernel = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE,
            (dilate_radius * 2 + 1, dilate_radius * 2 + 1),
        )
        texture_mask = _cv2.dilate(texture_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        pass

    return local_sd, texture_mask, {
        "textureMethod": "local SD + gradient percentile mask",
        "kernelSize": int(k),
        "localSdThreshold": round(sd_threshold, 3),
        "gradientThreshold": round(grad_threshold, 3),
        "texturePixelCount": int(np.sum(texture_mask)),
    }


def _m1v16_score_water(raw, texture_map, forbidden, cx, cy, radius, phantom_cx, phantom_cy, phantom_radius, inserts):
    roi_mask = _ellipse_mask_pixels(raw.shape, float(cx), float(cy), float(radius), float(radius))

    if bool(np.any(forbidden[roi_mask])):
        return None

    stats = _m1v16_roi_stats(raw, cx, cy, radius)
    if stats is None:
        return None

    texture_values = texture_map[roi_mask]
    texture_values = texture_values[np.isfinite(texture_values)]
    texture_mean = float(np.mean(texture_values)) if texture_values.size else 0.0

    dx = float(cx) - float(phantom_cx)
    dy = float(cy) - float(phantom_cy)
    r = float(phantom_radius)

    distance = math.hypot(dx, dy)
    distance_ratio = distance / max(r, 1e-6)
    side_ratio = abs(dx) / max(r, 1e-6)
    vertical_ratio = abs(dy) / max(r, 1e-6)

    insert_clearance = _m1v15_min_distance_to_inserts(cx, cy, inserts) - float(radius)
    edge_clearance = r - distance - float(radius)

    # Key behavior change:
    # Do not pick the center/central-detail corridor. Prefer a clean side
    # background patch between inserts, like the gray open phantom material.
    central_penalty = max(0.0, 0.28 - side_ratio) * 95.0

    # Prefer side-middle water background, not top/bottom line-pair zones.
    side_target_penalty = abs(side_ratio - 0.36) * 26.0
    vertical_penalty = vertical_ratio * 18.0

    insert_penalty = max(0.0, float(radius) * 1.25 - insert_clearance) * 4.0
    edge_penalty = max(0.0, float(radius) * 1.4 - edge_clearance) * 2.0

    score = (
        abs(float(stats["mean"])) * 1.8
        + float(stats["std"]) * 4.1
        + texture_mean * 2.2
        + central_penalty
        + side_target_penalty
        + vertical_penalty
        + insert_penalty
        + edge_penalty
    )

    return {
        "cx": round(float(cx), 3),
        "cy": round(float(cy), 3),
        "radius": round(float(radius), 3),
        "meanHU": round(float(stats["mean"]), 2),
        "stdHU": round(float(stats["std"]), 2),
        "textureMean": round(float(texture_mean), 3),
        "sideRatio": round(float(side_ratio), 4),
        "verticalRatio": round(float(vertical_ratio), 4),
        "distanceRatio": round(float(distance_ratio), 4),
        "insertClearancePixels": round(float(insert_clearance), 3),
        "edgeClearancePixels": round(float(edge_clearance), 3),
        "score": round(float(score), 3),
    }


def _m1v12_find_water(raw, phantom_cx, phantom_cy, phantom_radius, inserts, water_radius):
    """
    V16 water/background finder.

    The previous smart search still allowed a central ROI. This version makes
    the intended behavior stricter: water is the plain gray phantom background,
    preferably in a side-middle open area, not in the central line-pair/detail
    corridor.
    """
    try:
        import cv2 as _cv2
    except Exception:
        stats = _m1v16_roi_stats(raw, phantom_cx, phantom_cy, water_radius)
        return {
            "cx": round(float(phantom_cx), 3),
            "cy": round(float(phantom_cy), 3),
            "radius": round(float(water_radius), 3),
            "method": "fallback center because OpenCV unavailable",
            "fallbackUsed": True,
            "meanHU": round(float(stats["mean"]), 2) if stats else None,
            "stdHU": round(float(stats["std"]), 2) if stats else None,
        }

    arr = np.asarray(raw, dtype=np.float32)
    height, width = arr.shape
    yy, xx = np.ogrid[:height, :width]

    r = float(phantom_radius)
    cx0 = float(phantom_cx)
    cy0 = float(phantom_cy)
    roi_r = float(water_radius)

    distance_from_center = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)

    # Keep ROI well inside phantom.
    inside = distance_from_center <= (r * 0.74)

    # Central/detail corridor is where the line-pair structures live in this
    # Module 1 slice. Exclude it from water ROI center selection.
    central_detail_corridor = (
        (np.abs(xx - cx0) <= r * 0.25)
        & (np.abs(yy - cy0) <= r * 0.72)
    )

    # Prefer side-middle areas. This still includes both left and right sides.
    side_middle_zone = (
        (np.abs(xx - cx0) >= r * 0.24)
        & (np.abs(xx - cx0) <= r * 0.56)
        & (np.abs(yy - cy0) <= r * 0.42)
    )

    water_like = inside & np.isfinite(arr) & (arr >= -45.0) & (arr <= 45.0)

    if int(np.sum(water_like & side_middle_zone)) < 120:
        water_like = inside & np.isfinite(arr) & (arr >= -75.0) & (arr <= 75.0)

    forbidden = np.zeros(arr.shape, dtype=bool)

    # Exclude inserts with safety halo.
    for insert in inserts:
        x_value = float(insert.get("x", 0.0))
        y_value = float(insert.get("y", 0.0))
        radius = float(insert.get("radius", roi_r))
        exclusion_radius = radius + roi_r * 2.10
        forbidden |= ((xx - x_value) ** 2 + (yy - y_value) ** 2) <= exclusion_radius ** 2

    # Exclude edge BBs.
    edge_bb_mask, edge_bbs = _m1v15_detect_edge_bb_exclusion(
        raw=arr,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        water_radius=water_radius,
    )
    forbidden |= edge_bb_mask

    texture_map, texture_mask, texture_diag = _m1v16_texture_map(
        raw=arr,
        phantom_mask=inside,
        roi_radius=roi_r,
    )

    # Exclude texture/detail bars.
    forbidden |= texture_mask

    # Main allowed zone: water-like, side-middle, not central corridor, not forbidden.
    allowed = water_like & side_middle_zone & (~central_detail_corridor) & (~forbidden)

    dist = _cv2.distanceTransform(allowed.astype(np.uint8), _cv2.DIST_L2, 5)
    enough_room = dist >= max(3.0, roi_r * 0.88)

    # Relax in stages, but never go back to exact center unless no choice.
    relax_stage = "strict side-middle"

    if int(np.sum(enough_room)) < 10:
        relax_stage = "relaxed texture, side-middle"
        forbidden_without_texture = forbidden & (~texture_mask)
        allowed = water_like & side_middle_zone & (~central_detail_corridor) & (~forbidden_without_texture)
        dist = _cv2.distanceTransform(allowed.astype(np.uint8), _cv2.DIST_L2, 5)
        enough_room = dist >= max(3.0, roi_r * 0.82)

    if int(np.sum(enough_room)) < 10:
        relax_stage = "side zone, wider water band"
        wide_water_like = inside & np.isfinite(arr) & (arr >= -100.0) & (arr <= 100.0)
        allowed = wide_water_like & side_middle_zone & (~central_detail_corridor) & (~(forbidden & (~texture_mask)))
        dist = _cv2.distanceTransform(allowed.astype(np.uint8), _cv2.DIST_L2, 5)
        enough_room = dist >= max(3.0, roi_r * 0.75)

    candidate_points = np.column_stack(np.nonzero(enough_room))
    candidates = []

    if candidate_points.size:
        step = int(max(3, round(roi_r * 0.42)))

        for y_value, x_value in candidate_points:
            if int(x_value) % step != 0 or int(y_value) % step != 0:
                continue

            candidate = _m1v16_score_water(
                raw=arr,
                texture_map=texture_map,
                forbidden=forbidden,
                cx=float(x_value),
                cy=float(y_value),
                radius=roi_r,
                phantom_cx=phantom_cx,
                phantom_cy=phantom_cy,
                phantom_radius=phantom_radius,
                inserts=inserts,
            )

            if candidate:
                candidates.append(candidate)

        if not candidates:
            room_values = dist[enough_room]
            best_indices = np.argsort(room_values)[-700:]
            for y_value, x_value in candidate_points[best_indices]:
                candidate = _m1v16_score_water(
                    raw=arr,
                    texture_map=texture_map,
                    forbidden=forbidden,
                    cx=float(x_value),
                    cy=float(y_value),
                    radius=roi_r,
                    phantom_cx=phantom_cx,
                    phantom_cy=phantom_cy,
                    phantom_radius=phantom_radius,
                    inserts=inserts,
                )
                if candidate:
                    candidates.append(candidate)

    if not candidates:
        # Last-resort side location, still not center.
        relax_stage = "fallback side sample"
        manual_points = [
            (cx0 - r * 0.36, cy0),
            (cx0 + r * 0.36, cy0),
            (cx0 - r * 0.32, cy0 + r * 0.18),
            (cx0 + r * 0.32, cy0 - r * 0.18),
        ]

        zero_forbidden = np.zeros(arr.shape, dtype=bool)
        for px, py in manual_points:
            stats = _m1v16_score_water(
                raw=arr,
                texture_map=texture_map,
                forbidden=zero_forbidden,
                cx=float(px),
                cy=float(py),
                radius=roi_r,
                phantom_cx=phantom_cx,
                phantom_cy=phantom_cy,
                phantom_radius=phantom_radius,
                inserts=inserts,
            )
            if stats:
                candidates.append(stats)

    if not candidates:
        stats = _m1v16_roi_stats(arr, cx0, cy0, roi_r)
        return {
            "cx": round(cx0, 3),
            "cy": round(cy0, 3),
            "radius": round(roi_r, 3),
            "method": "last fallback center after V16 side-water search failed",
            "fallbackUsed": True,
            "meanHU": round(float(stats["mean"]), 2) if stats else None,
            "stdHU": round(float(stats["std"]), 2) if stats else None,
            "candidateCount": 0,
        }

    candidates = sorted(candidates, key=lambda item: float(item["score"]))
    best = candidates[0]

    return {
        "cx": float(best["cx"]),
        "cy": float(best["cy"]),
        "radius": round(roi_r, 3),
        "method": "V16 side-water search: plain gray side-middle background, excludes center/detail corridor",
        "fallbackUsed": False,
        "relaxStage": relax_stage,
        "meanHU": best["meanHU"],
        "stdHU": best["stdHU"],
        "textureMean": best["textureMean"],
        "score": best["score"],
        "sideRatio": best["sideRatio"],
        "verticalRatio": best["verticalRatio"],
        "distanceRatio": best["distanceRatio"],
        "edgeClearancePixels": best["edgeClearancePixels"],
        "insertClearancePixels": best["insertClearancePixels"],
        "candidateCount": int(len(candidates)),
        "allowedPixelCount": int(np.sum(allowed)),
        "enoughRoomPixelCount": int(np.sum(enough_room)),
        "edgeBbCountExcluded": int(len(edge_bbs)),
        "textureDiagnostics": texture_diag,
        "topCandidates": candidates[:8],
    }


_ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V16 = create_module1_ct_number_analysis


def create_module1_ct_number_analysis(*args, **kwargs):
    result = _ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V16(*args, **kwargs)

    try:
        result["analysisVersion"] = MODULE1_ANALYSIS_VERSION_V16

        if isinstance(result.get("finalResult"), dict):
            result["finalResult"]["waterRoiMethod"] = "V16 side-middle water search"
            result["finalResult"]["waterLocationFix"] = "Water ROI should be in plain gray background, not the central detail corridor."

        result["criteriaNote"] = (
            "Module 1 V16 uses the existing insert detection but replaces water placement with a stricter side-water search. "
            "The water ROI is selected from plain gray phantom background in a side-middle open area, avoiding inserts, edge BBs, high-texture line-pair regions, the phantom edge, and the central detail corridor. "
            "Polyethylene label is preserved."
        )
    except Exception as exc:
        result["module1V16PostprocessError"] = str(exc)

    return result

# END MODULE1 V16 WATER SIDE PATCH

# BEGIN MODULE1 V17 FAINT WATER CIRCLE DETECTOR

MODULE1_ANALYSIS_VERSION_V17 = "ACR_MODULE1_CT_NUMBER_QUICK_V17_FAINT_WATER_CIRCLE_2026_07_25"


def _m1v17_safe_stats(raw, cx, cy, radius):
    values = _m1v15_roi_values(raw, cx, cy, radius)
    if values is None:
        return None
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(values.size),
    }


def _m1v17_ring_mask(shape, cx, cy, inner_radius, outer_radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    dist2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
    return (dist2 >= float(inner_radius) ** 2) & (dist2 <= float(outer_radius) ** 2)


def _m1v17_circle_inside_mask(shape, cx, cy, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= float(radius) ** 2


def _m1v17_overlaps_inserts(cx, cy, radius, inserts, safety=1.20):
    for insert in inserts or []:
        ix = float(insert.get("x", 0.0))
        iy = float(insert.get("y", 0.0))
        ir = float(insert.get("radius", radius))
        if math.hypot(float(cx) - ix, float(cy) - iy) < (float(radius) + ir) * float(safety):
            return True
    return False


def _m1v17_min_insert_clearance(cx, cy, radius, inserts):
    if not inserts:
        return 999999.0

    best = 999999.0

    for insert in inserts:
        ix = float(insert.get("x", 0.0))
        iy = float(insert.get("y", 0.0))
        ir = float(insert.get("radius", radius))
        clear = math.hypot(float(cx) - ix, float(cy) - iy) - ir - float(radius)
        best = min(best, clear)

    return float(best)


def _m1v17_make_edge_maps(raw):
    try:
        import cv2 as _cv2
    except Exception:
        return None, None, {"edgeMethod": "OpenCV unavailable"}

    arr = np.asarray(raw, dtype=np.float32)
    finite = arr[np.isfinite(arr)]

    if finite.size < 100:
        return None, None, {"edgeMethod": "not enough finite pixels"}

    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))

    if hi <= lo:
        return None, None, {"edgeMethod": "flat image"}

    clipped = np.clip(arr, lo, hi)
    norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    clahe = _cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # Difference-of-Gaussians brings out faint circular boundaries.
    small = _cv2.GaussianBlur(enhanced, (5, 5), 0)
    large = _cv2.GaussianBlur(enhanced, (31, 31), 0)
    dog = _cv2.absdiff(small, large)
    dog = _cv2.normalize(dog, None, 0, 255, _cv2.NORM_MINMAX).astype(np.uint8)

    gx = _cv2.Sobel(enhanced, _cv2.CV_32F, 1, 0, ksize=3)
    gy = _cv2.Sobel(enhanced, _cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    diagnostics = {
        "edgeMethod": "CLAHE + Difference-of-Gaussians + Sobel gradient",
        "clipLow": round(lo, 3),
        "clipHigh": round(hi, 3),
    }

    return dog, grad, diagnostics


def _m1v17_hough_faint_circle_candidates(raw, dog, phantom_cx, phantom_cy, phantom_radius, inserts):
    try:
        import cv2 as _cv2
    except Exception:
        return [], {"houghMethod": "OpenCV unavailable"}

    if dog is None:
        return [], {"houghMethod": "no DoG map"}

    if inserts:
        median_insert_radius = float(np.median([float(item.get("radius", 0.0)) for item in inserts if float(item.get("radius", 0.0)) > 0.0]))
    else:
        median_insert_radius = float(phantom_radius) * 0.105

    min_radius = max(8, int(round(median_insert_radius * 0.72)))
    max_radius = max(min_radius + 3, int(round(median_insert_radius * 1.32)))
    min_dist = max(24, int(round(median_insert_radius * 1.45)))

    circles = _cv2.HoughCircles(
        dog,
        _cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=min_dist,
        param1=28,
        param2=8,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return [], {
            "houghMethod": "DoG HoughCircles",
            "rawCircleCount": 0,
            "minRadius": int(min_radius),
            "maxRadius": int(max_radius),
        }

    candidates = []

    for x, y, radius in circles[0, :]:
        x = float(x)
        y = float(y)
        radius = float(radius)

        center_dist = math.hypot(x - float(phantom_cx), y - float(phantom_cy))
        center_ratio = center_dist / max(float(phantom_radius), 1e-6)

        # Water circle is inside phantom and is not the outer edge BB.
        if center_ratio < 0.10 or center_ratio > 0.66:
            continue

        if center_dist + radius > float(phantom_radius) * 0.82:
            continue

        if _m1v17_overlaps_inserts(x, y, radius, inserts, safety=1.08):
            continue

        candidates.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "radius": round(radius, 3),
            "source": "DoG Hough faint-circle candidate",
        })

    return candidates, {
        "houghMethod": "DoG HoughCircles",
        "rawCircleCount": int(circles.shape[1]),
        "keptCircleCount": int(len(candidates)),
        "minRadius": int(min_radius),
        "maxRadius": int(max_radius),
    }


def _m1v17_grid_faint_circle_candidates(raw, phantom_cx, phantom_cy, phantom_radius, inserts):
    """
    Fallback search that scans possible circle centers and uses ring evidence.
    This prevents us from falling back to a random clean background spot.
    """
    if inserts:
        radius = float(np.median([float(item.get("radius", 0.0)) for item in inserts if float(item.get("radius", 0.0)) > 0.0]))
    else:
        radius = float(phantom_radius) * 0.105

    radius = max(8.0, min(radius, float(phantom_radius) * 0.16))
    step = max(5, int(round(radius * 0.30)))

    candidates = []
    h, w = raw.shape

    # Search the inside of phantom. Include the whole usable area, but avoid
    # very center and insert overlaps. This is still an image search, not a fixed location.
    for y in range(int(max(radius, phantom_cy - phantom_radius * 0.62)), int(min(h - radius, phantom_cy + phantom_radius * 0.62)), step):
        for x in range(int(max(radius, phantom_cx - phantom_radius * 0.62)), int(min(w - radius, phantom_cx + phantom_radius * 0.62)), step):
            center_dist = math.hypot(float(x) - float(phantom_cx), float(y) - float(phantom_cy))
            center_ratio = center_dist / max(float(phantom_radius), 1e-6)

            if center_ratio < 0.12 or center_ratio > 0.66:
                continue

            if center_dist + radius > float(phantom_radius) * 0.82:
                continue

            if _m1v17_overlaps_inserts(x, y, radius, inserts, safety=1.10):
                continue

            candidates.append({
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "radius": round(float(radius), 3),
                "source": "grid ring-evidence candidate",
            })

    return candidates


def _m1v17_score_faint_water_circle(raw, dog, grad, candidate, phantom_cx, phantom_cy, phantom_radius, inserts):
    cx = float(candidate["x"])
    cy = float(candidate["y"])
    radius = float(candidate["radius"])

    inner_roi_radius = max(5.0, radius * 0.50)
    stats = _m1v17_safe_stats(raw, cx, cy, inner_roi_radius)

    if stats is None:
        return None

    inside_mask = _m1v17_circle_inside_mask(raw.shape, cx, cy, inner_roi_radius)
    ring_mask = _m1v17_ring_mask(raw.shape, cx, cy, radius * 0.82, radius * 1.18)

    if dog is not None:
        dog_ring = dog[ring_mask]
        dog_inside = dog[inside_mask]
        dog_ring_mean = float(np.mean(dog_ring)) if dog_ring.size else 0.0
        dog_inside_mean = float(np.mean(dog_inside)) if dog_inside.size else 0.0
    else:
        dog_ring_mean = 0.0
        dog_inside_mean = 0.0

    if grad is not None:
        grad_ring = grad[ring_mask]
        grad_inside = grad[inside_mask]
        grad_ring_mean = float(np.mean(grad_ring)) if grad_ring.size else 0.0
        grad_inside_mean = float(np.mean(grad_inside)) if grad_inside.size else 0.0
    else:
        grad_ring_mean = 0.0
        grad_inside_mean = 0.0

    # The faint water circle should have a ring/boundary, but the inner ROI
    # should still look like water/background.
    ring_evidence = max(0.0, dog_ring_mean - dog_inside_mean * 0.65) + max(0.0, grad_ring_mean - grad_inside_mean * 0.65)

    center_dist = math.hypot(cx - float(phantom_cx), cy - float(phantom_cy))
    center_ratio = center_dist / max(float(phantom_radius), 1e-6)
    side_ratio = abs(cx - float(phantom_cx)) / max(float(phantom_radius), 1e-6)
    vertical_ratio = abs(cy - float(phantom_cy)) / max(float(phantom_radius), 1e-6)

    insert_clearance = _m1v17_min_insert_clearance(cx, cy, radius, inserts)

    # This is a faint-circle detector, so ring evidence matters most.
    # HU/SD still matter so it does not choose air/bone/acrylic.
    quality = (
        ring_evidence * 3.2
        - abs(float(stats["mean"])) * 1.10
        - float(stats["std"]) * 2.10
        + max(0.0, insert_clearance) * 0.08
        - max(0.0, 0.10 - center_ratio) * 50.0
        - max(0.0, center_ratio - 0.66) * 50.0
        - vertical_ratio * 3.0
    )

    # Softly prefer off-center side water circle over random background.
    if side_ratio >= 0.18:
        quality += 8.0

    return {
        **candidate,
        "roiRadius": round(float(inner_roi_radius), 3),
        "meanHU": round(float(stats["mean"]), 2),
        "stdHU": round(float(stats["std"]), 2),
        "ringEvidence": round(float(ring_evidence), 3),
        "dogRingMean": round(float(dog_ring_mean), 3),
        "gradRingMean": round(float(grad_ring_mean), 3),
        "centerRatio": round(float(center_ratio), 4),
        "sideRatio": round(float(side_ratio), 4),
        "verticalRatio": round(float(vertical_ratio), 4),
        "insertClearancePixels": round(float(insert_clearance), 3),
        "quality": round(float(quality), 3),
    }


def _m1v12_find_water(raw, phantom_cx, phantom_cy, phantom_radius, inserts, water_radius):
    """
    V17 water finder.

    Finds the faint circular water structure by low-contrast circle evidence,
    not by picking a random clean background patch.
    """
    dog, grad, edge_diag = _m1v17_make_edge_maps(raw)

    hough_candidates, hough_diag = _m1v17_hough_faint_circle_candidates(
        raw=raw,
        dog=dog,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        inserts=inserts,
    )

    grid_candidates = _m1v17_grid_faint_circle_candidates(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        inserts=inserts,
    )

    # De-duplicate hough + grid candidates.
    all_candidates = []
    for candidate in hough_candidates + grid_candidates:
        duplicate = False
        for kept in all_candidates:
            if math.hypot(float(candidate["x"]) - float(kept["x"]), float(candidate["y"]) - float(kept["y"])) < max(float(candidate["radius"]), float(kept["radius"])) * 0.55:
                duplicate = True
                break
        if not duplicate:
            all_candidates.append(candidate)

    scored = []
    for candidate in all_candidates:
        item = _m1v17_score_faint_water_circle(
            raw=raw,
            dog=dog,
            grad=grad,
            candidate=candidate,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
            inserts=inserts,
        )
        if item is not None:
            scored.append(item)

    if not scored:
        # If faint-circle search fails, fall back to the last smart search if it exists.
        try:
            return _m1v16_find_water_fallback(raw, phantom_cx, phantom_cy, phantom_radius, inserts, water_radius)
        except Exception:
            stats = _m1v17_safe_stats(raw, phantom_cx, phantom_cy, water_radius)
            return {
                "cx": round(float(phantom_cx), 3),
                "cy": round(float(phantom_cy), 3),
                "radius": round(float(water_radius), 3),
                "method": "fallback center; V17 faint circle search found no candidates",
                "fallbackUsed": True,
                "meanHU": round(float(stats["mean"]), 2) if stats else None,
                "stdHU": round(float(stats["std"]), 2) if stats else None,
                "candidateCount": 0,
                "edgeDiagnostics": edge_diag,
                "houghDiagnostics": hough_diag,
            }

    scored = sorted(scored, key=lambda item: float(item["quality"]), reverse=True)
    best = scored[0]

    # Use the physical ROI radius requested by V13 for measurement, but the
    # detected faint circle center for location.
    final_stats = _m1v17_safe_stats(raw, float(best["x"]), float(best["y"]), water_radius) or {
        "mean": best["meanHU"],
        "std": best["stdHU"],
        "count": 0,
    }

    return {
        "cx": float(best["x"]),
        "cy": float(best["y"]),
        "radius": round(float(water_radius), 3),
        "detectedWaterCircleRadius": best["radius"],
        "method": "V17 faint water-circle detector: DoG/Hough + ring-evidence scoring",
        "fallbackUsed": False,
        "meanHU": round(float(final_stats["mean"]), 2),
        "stdHU": round(float(final_stats["std"]), 2),
        "pixelCount": int(final_stats["count"]),
        "bestCandidate": best,
        "candidateCount": int(len(scored)),
        "topCandidates": scored[:10],
        "edgeDiagnostics": edge_diag,
        "houghDiagnostics": hough_diag,
    }


_ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V17 = create_module1_ct_number_analysis


def create_module1_ct_number_analysis(*args, **kwargs):
    result = _ORIGINAL_CREATE_MODULE1_CT_NUMBER_ANALYSIS_V17(*args, **kwargs)
    try:
        result["analysisVersion"] = MODULE1_ANALYSIS_VERSION_V17
        if isinstance(result.get("finalResult"), dict):
            result["finalResult"]["waterRoiMethod"] = "V17 faint water-circle detector"
        result["criteriaNote"] = (
            "Module 1 V17 finds the water ROI as a faint circular structure using low-contrast edge/ring evidence, "
            "instead of choosing any clean gray background location. Polyethylene label is preserved."
        )
    except Exception as exc:
        result["module1V17PostprocessError"] = str(exc)
    return result

# END MODULE1 V17 FAINT WATER CIRCLE DETECTOR
