"""
ACR CT phantom four-module classifier — V4.

Designed for fast Module 3 range discovery:
- Module 1: opposing high-HU and low-HU material inserts
- Module 2: subtle structures that persist at the same positions across adjacent slices
- Module 3: uniform interior with low cross-slice structured persistence
- Module 4: repeated bright high-contrast resolution objects

After classifying the four module ranges, V4 performs a second full-resolution
ranking only inside the probable Module 3 range to select the exact slice where
the two small bright BB dots are strongest. It still does not run the final five
ROI uniformity measurement. NumPy only; SciPy is intentionally not imported.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

import numpy as np

from services.dicom_display import _get_slices_from_stack_or_upload


CLASSIFIER_VERSION = "ACR_MODULE_CLASSIFIER_V4_EXACT_M3_BB_SLICE_2026_07_05"

MODULE_1 = "MODULE_1_CT_NUMBER"
MODULE_2 = "MODULE_2_LOW_CONTRAST"
MODULE_3 = "MODULE_3_UNIFORMITY"
MODULE_4 = "MODULE_4_HIGH_CONTRAST"
UNKNOWN = "TRANSITION_OR_UNKNOWN"

MODULE_KEYS = (MODULE_1, MODULE_2, MODULE_3, MODULE_4)

MODULE_LABELS = {
    MODULE_1: "Module 1 — CT Number / Materials",
    MODULE_2: "Module 2 — Low Contrast",
    MODULE_3: "Module 3 — Uniformity",
    MODULE_4: "Module 4 — High-Contrast Resolution",
    UNKNOWN: "Transition / Unknown",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


def _downsample_mean(array: np.ndarray, max_size: int = 160) -> tuple[np.ndarray, float]:
    arr = np.asarray(array, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D DICOM slice, got {arr.shape}.")

    height, width = arr.shape
    block = max(1, int(math.ceil(max(height, width) / max_size)))

    if block == 1:
        return arr.copy(), 1.0

    padded_height = int(math.ceil(height / block) * block)
    padded_width = int(math.ceil(width / block) * block)

    padded = np.pad(
        arr,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )

    small = padded.reshape(
        padded_height // block,
        block,
        padded_width // block,
        block,
    ).mean(axis=(1, 3))

    return small.astype(np.float32), float(block)


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)

    if radius <= 0:
        return arr.copy()

    size = radius * 2 + 1
    padded = np.pad(arr, radius, mode="edge")
    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    ).cumsum(axis=0).cumsum(axis=1)

    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )

    return (total / float(size * size)).astype(np.float32)


def _estimate_phantom_geometry(raw: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(raw, dtype=np.float32)
    height, width = arr.shape
    finite = np.isfinite(arr)

    if np.sum(finite) < 100:
        raise ValueError("Not enough finite pixels to detect the phantom.")

    border_width = max(2, int(round(min(height, width) * 0.06)))
    border_values = np.concatenate([
        arr[:border_width, :].ravel(),
        arr[-border_width:, :].ravel(),
        arr[:, :border_width].ravel(),
        arr[:, -border_width:].ravel(),
    ])
    border_values = border_values[np.isfinite(border_values)]

    center_values = arr[
        int(height * 0.30):int(height * 0.70),
        int(width * 0.30):int(width * 0.70),
    ]
    center_values = center_values[np.isfinite(center_values)]

    if border_values.size < 20 or center_values.size < 20:
        raise ValueError("Could not estimate phantom/background intensity.")

    border_median = float(np.median(border_values))
    center_median = float(np.median(center_values))
    difference = center_median - border_median

    if abs(difference) < 5.0:
        finite_values = arr[finite]
        low = float(np.percentile(finite_values, 10))
        high = float(np.percentile(finite_values, 70))
        threshold = (low + high) / 2.0
        body = arr >= threshold
    else:
        threshold = border_median + difference * 0.35
        body = arr >= threshold if difference > 0 else arr <= threshold

    yy, xx = np.indices(arr.shape)
    broad_center = (
        (xx - width / 2.0) ** 2
        + (yy - height / 2.0) ** 2
    ) <= (min(height, width) * 0.49) ** 2

    body &= finite & broad_center
    neighborhood_vote = _box_blur(body.astype(np.float32), 1)
    body &= neighborhood_vote >= 0.35

    area = float(np.sum(body))

    if area < height * width * 0.08:
        raise ValueError("Detected phantom area is too small.")

    cx = float(np.mean(xx[body]))
    cy = float(np.mean(yy[body]))
    radius = math.sqrt(area / math.pi)

    if radius < min(height, width) * 0.14:
        raise ValueError("Detected phantom radius is too small.")

    return cx, cy, radius


def _radial_detrend(
    normalized: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    interior_mask: np.ndarray,
    bin_count: int = 32,
) -> np.ndarray:
    yy, xx = np.indices(normalized.shape)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    radial_bin = np.floor(
        np.clip(distance / max(radius * 0.75, 1.0), 0.0, 0.9999)
        * bin_count
    ).astype(np.int16)

    profile = np.zeros(bin_count, dtype=np.float32)

    for index in range(bin_count):
        values = normalized[interior_mask & (radial_bin == index)]
        if values.size:
            profile[index] = float(np.median(values))

    return normalized - profile[radial_bin]


def _pearson_correlation(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> float:
    x = np.asarray(first[mask], dtype=np.float64)
    y = np.asarray(second[mask], dtype=np.float64)

    if x.size < 100 or y.size != x.size:
        return 0.0

    x -= float(np.mean(x))
    y -= float(np.mean(y))

    denominator = math.sqrt(
        float(np.sum(x * x))
        * float(np.sum(y * y))
    )

    if denominator <= 1e-12:
        return 0.0

    return float(np.sum(x * y) / denominator)


def _connected_component_count(mask: np.ndarray, minimum_area: int) -> int:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    count = 0

    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue

        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        area = 0

        while queue:
            y, x = queue.pop()
            area += 1

            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= height:
                    continue

                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx = x + dx
                    if nx < 0 or nx >= width:
                        continue

                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        if area >= minimum_area:
            count += 1

    return count


def _prepare_slice(
    slice_data: dict[str, Any],
    slice_index: int,
    max_size: int,
) -> dict[str, Any]:
    info = slice_data.get("info", {})

    if info.get("isColorDicom"):
        return {
            "sliceIndex": int(slice_index),
            "sliceNumber": int(slice_index) + 1,
            "status": "skipped_color",
            "sourceName": slice_data.get("sourceName", ""),
            "sliceLabel": slice_data.get("label", ""),
        }

    raw = np.asarray(slice_data["pixels"], dtype=np.float32)
    small, downsample_factor = _downsample_mean(raw, max_size=max_size)

    cx, cy, radius = _estimate_phantom_geometry(small)
    yy, xx = np.indices(small.shape)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    interior = np.isfinite(small) & (distance <= radius * 0.65)

    if np.sum(interior) < 200:
        raise ValueError("Not enough phantom interior pixels.")

    median_hu = float(np.median(small[interior]))
    normalized = small - median_hu

    structure_radius = max(1, int(round(radius * 0.018)))
    quantile_radius = max(1, int(round(radius * 0.015)))

    smooth = _box_blur(normalized, quantile_radius)
    radial_detrended = _radial_detrend(
        normalized=normalized,
        cx=cx,
        cy=cy,
        radius=radius,
        interior_mask=interior,
    )
    structure = _box_blur(radial_detrended, structure_radius)

    values = smooth[interior]
    p01, p05, p95, p99 = np.percentile(values, [1, 5, 95, 99])

    robust_range = float(p95 - p05)
    wide_range = float(p99 - p01)

    positive_extreme = interior & (smooth >= 100.0)
    negative_extreme = interior & (smooth <= -100.0)
    interior_area = max(float(np.sum(interior)), 1.0)

    positive_fraction = float(np.sum(positive_extreme) / interior_area)
    negative_fraction = float(np.sum(negative_extreme) / interior_area)

    minimum_component_area = max(5, int(round(interior_area * 0.0006)))
    bright_component_count = _connected_component_count(
        positive_extreme,
        minimum_area=minimum_component_area,
    )
    dark_component_count = _connected_component_count(
        negative_extreme,
        minimum_area=minimum_component_area,
    )

    gradient_x = np.zeros_like(smooth)
    gradient_y = np.zeros_like(smooth)
    gradient_x[:, 1:] = np.abs(smooth[:, 1:] - smooth[:, :-1])
    gradient_y[1:, :] = np.abs(smooth[1:, :] - smooth[:-1, :])
    gradient = np.hypot(gradient_x, gradient_y)
    gradient_values = gradient[interior]

    edge_threshold = max(
        float(np.percentile(gradient_values, 92)),
        float(np.median(gradient_values) + 2.0 * np.std(gradient_values)),
    )
    edge_density = float(np.mean(gradient_values >= edge_threshold))

    return {
        "sliceIndex": int(slice_index),
        "sliceNumber": int(slice_index) + 1,
        "status": "ok",
        "sourceName": slice_data.get("sourceName", ""),
        "sliceLabel": slice_data.get("label", ""),
        "_structure": structure,
        "_interior": interior,
        "features": {
            "interiorMedianHu": round(median_hu, 3),
            "robustRangeHu": round(robust_range, 3),
            "wideRangeHu": round(wide_range, 3),
            "positiveExtremeFraction": round(positive_fraction, 6),
            "negativeExtremeFraction": round(negative_fraction, 6),
            "brightHighContrastComponentCount": int(bright_component_count),
            "darkHighContrastComponentCount": int(dark_component_count),
            "edgeDensity": round(edge_density, 6),
            "structureBlurRadius": int(structure_radius),
            "downsampleFactor": round(float(downsample_factor), 3),
            "phantomCenterXSmall": round(float(cx), 3),
            "phantomCenterYSmall": round(float(cy), 3),
            "phantomRadiusSmall": round(float(radius), 3),
        },
    }


def _add_cross_slice_persistence(prepared: list[dict[str, Any]]) -> None:
    for index, item in enumerate(prepared):
        if item.get("status") != "ok":
            continue

        correlations: list[float] = []
        immediate_correlations: list[float] = []

        for offset in (-2, -1, 1, 2):
            neighbor_index = index + offset

            if neighbor_index < 0 or neighbor_index >= len(prepared):
                continue

            neighbor = prepared[neighbor_index]
            if neighbor.get("status") != "ok":
                continue

            first_structure = item["_structure"]
            second_structure = neighbor["_structure"]

            if first_structure.shape != second_structure.shape:
                continue

            overlap = item["_interior"] & neighbor["_interior"]
            correlation = _pearson_correlation(
                first_structure,
                second_structure,
                overlap,
            )
            correlations.append(correlation)

            if abs(offset) == 1:
                immediate_correlations.append(correlation)

        sorted_correlations = sorted(correlations, reverse=True)
        strongest = sorted_correlations[:2]

        top_two_mean = float(np.mean(strongest)) if strongest else 0.0
        immediate_mean = (
            float(np.mean(immediate_correlations))
            if immediate_correlations
            else top_two_mean
        )

        item["features"]["neighborPersistence"] = round(top_two_mean, 6)
        item["features"]["immediateNeighborPersistence"] = round(
            immediate_mean,
            6,
        )
        item["features"]["neighborCorrelations"] = [
            round(float(value), 6)
            for value in correlations
        ]


def _raw_evidence(features: dict[str, Any]) -> dict[str, float]:
    wide_range = float(features["wideRangeHu"])
    robust_range = float(features["robustRangeHu"])
    positive_fraction = float(features["positiveExtremeFraction"])
    negative_fraction = float(features["negativeExtremeFraction"])
    bright_components = int(features["brightHighContrastComponentCount"])
    dark_components = int(features["darkHighContrastComponentCount"])
    persistence = float(features.get("neighborPersistence", 0.0))

    positive_strength = _scale(positive_fraction, 0.002, 0.010)
    negative_strength = _scale(negative_fraction, 0.001, 0.006)
    both_extremes = min(positive_strength, negative_strength)
    high_contrast_range = _scale(wide_range, 120.0, 900.0)

    module_1 = _clamp01(
        0.62 * both_extremes
        + 0.20 * high_contrast_range * both_extremes
        + 0.10 * _scale(bright_components, 1, 3)
        + 0.08 * _scale(dark_components, 1, 3)
    )

    module_4 = _clamp01(
        0.45 * positive_strength * (1.0 - _scale(negative_fraction, 0.0005, 0.003))
        + 0.25 * high_contrast_range
        + 0.20 * _scale(bright_components, 4, 8)
        + 0.10 * (1.0 - _scale(dark_components, 0, 2))
    )

    obvious_high_contrast = max(module_1, module_4)
    low_contrast_range = _scale(wide_range, 6.7, 10.5)
    persistent_structure = _scale(persistence, 0.46, 0.68)

    module_2 = _clamp01(
        0.64 * persistent_structure
        + 0.24 * low_contrast_range
        + 0.12 * (1.0 - _scale(robust_range, 8.0, 35.0))
    )

    # A transition tail from Module 1 may still be spatially persistent, but its
    # HU range is far too large to be a true low-contrast module.
    module_2 *= 1.0 - _scale(wide_range, 25.0, 90.0)

    uniform_persistence = 1.0 - _scale(persistence, 0.32, 0.64)
    uniform_wide_range = 1.0 - _scale(wide_range, 6.0, 12.0)
    uniform_robust_range = 1.0 - _scale(robust_range, 4.0, 10.0)

    module_3 = _clamp01(
        0.58 * uniform_persistence
        + 0.27 * uniform_wide_range
        + 0.15 * uniform_robust_range
    )

    high_contrast_penalty = 1.0 - 0.96 * obvious_high_contrast
    module_2 *= max(0.0, high_contrast_penalty)
    module_3 *= max(0.0, high_contrast_penalty)

    return {
        MODULE_1: float(module_1),
        MODULE_2: float(module_2),
        MODULE_3: float(module_3),
        MODULE_4: float(module_4),
    }


def _smooth_evidence(prepared: list[dict[str, Any]]) -> None:
    raw = [item.get("rawEvidence", {}) for item in prepared]
    weights = {-2: 1.0, -1: 2.0, 0: 3.0, 1: 2.0, 2: 1.0}

    for index, item in enumerate(prepared):
        if item.get("status") != "ok":
            continue

        smoothed: dict[str, float] = {}

        for module in MODULE_KEYS:
            weighted_total = 0.0
            weight_total = 0.0

            for offset in (-2, -1, 0, 1, 2):
                neighbor_index = index + offset
                if neighbor_index < 0 or neighbor_index >= len(prepared):
                    continue

                neighbor = prepared[neighbor_index]
                if neighbor.get("status") != "ok":
                    continue

                weight = weights[offset]
                weighted_total += float(raw[neighbor_index][module]) * weight
                weight_total += weight

            neighbor_average = (
                weighted_total / weight_total
                if weight_total > 0
                else float(raw[index][module])
            )

            own = float(raw[index][module])
            combined = 0.72 * own + 0.28 * neighbor_average

            # Never let smoothing erase an obvious material or resolution slice.
            if module in (MODULE_1, MODULE_4) and own >= 0.65:
                combined = max(combined, own)

            smoothed[module] = _clamp01(combined)

        item["evidence"] = smoothed


def _competitive_scores(evidence: dict[str, float], temperature: float = 0.20) -> dict[str, int]:
    values = np.array(
        [float(evidence[module]) for module in MODULE_KEYS],
        dtype=np.float64,
    )
    values = (values - float(np.max(values))) / max(temperature, 1e-6)
    exponential = np.exp(values)
    probabilities = exponential / max(float(np.sum(exponential)), 1e-12)

    raw_scores = probabilities * 100.0
    rounded = np.floor(raw_scores).astype(int)
    remainder = int(100 - int(np.sum(rounded)))

    if remainder > 0:
        order = np.argsort(-(raw_scores - rounded))
        for index in order[:remainder]:
            rounded[index] += 1

    return {
        module: int(rounded[index])
        for index, module in enumerate(MODULE_KEYS)
    }


def _reasons(prediction: str, features: dict[str, Any]) -> dict[str, list[str]]:
    wide_range = float(features.get("wideRangeHu", 0.0))
    persistence = float(features.get("neighborPersistence", 0.0))
    positive_fraction = float(features.get("positiveExtremeFraction", 0.0))
    negative_fraction = float(features.get("negativeExtremeFraction", 0.0))
    bright_components = int(features.get("brightHighContrastComponentCount", 0))
    dark_components = int(features.get("darkHighContrastComponentCount", 0))

    reasons = {
        MODULE_1: [
            f"bright and dark high-contrast material evidence",
            f"wide interior HU range ({wide_range:.1f} HU)",
            f"{bright_components} bright and {dark_components} dark high-contrast components",
        ],
        MODULE_2: [
            f"persistent subtle structure across adjacent slices ({persistence:.2f})",
            f"low-contrast structured HU range ({wide_range:.1f} HU)",
            "no opposing high-HU/air material pattern",
        ],
        MODULE_3: [
            f"low cross-slice structured persistence ({persistence:.2f})",
            f"narrow interior HU range ({wide_range:.1f} HU)",
            "no large material inserts or resolution pattern",
        ],
        MODULE_4: [
            f"{bright_components} bright high-contrast components",
            f"wide positive HU structure ({wide_range:.1f} HU)",
            "little matching dark-material evidence",
        ],
        UNKNOWN: [
            "competing or transitional module evidence",
            f"structured persistence {persistence:.2f}; HU range {wide_range:.1f} HU",
        ],
    }

    if prediction == MODULE_1 and negative_fraction <= 0.001:
        reasons[MODULE_1].append("dark-material evidence is weak")
    if prediction == MODULE_4 and positive_fraction <= 0.002:
        reasons[MODULE_4].append("bright-object evidence is weak")

    return reasons


def _finalize_predictions(prepared: list[dict[str, Any]]) -> None:
    for item in prepared:
        if item.get("status") != "ok":
            item.update({
                "prediction": UNKNOWN,
                "predictionLabel": MODULE_LABELS[UNKNOWN],
                "rawScores": {module: 0 for module in MODULE_KEYS},
                "scores": {module: 0 for module in MODULE_KEYS},
                "topScore": 0,
                "secondScore": 0,
                "scoreMargin": 0,
                "neighborAgreement": 0.0,
                "confidence": 0.0,
                "confidenceLabel": "Low",
                "reasons": {
                    UNKNOWN: [
                        "color/secondary-capture or unreadable slice was not classified"
                    ]
                },
            })
            continue

        scores = _competitive_scores(item["evidence"])
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        winner, top_score = ranked[0]
        second_score = ranked[1][1]
        margin = int(top_score - second_score)
        top_evidence = float(item["evidence"][winner])

        features = item["features"]
        wide_range = float(features["wideRangeHu"])

        # A broad intermediate HU range without a clear material or resolution
        # signature is usually a boundary slice rather than Module 2 or 3.
        transition_range = (
            25.0 < wide_range < 250.0
            and winner in (MODULE_2, MODULE_3)
        )

        if top_evidence < 0.40 or top_score < 52 or margin < 16 or transition_range:
            prediction = UNKNOWN
        else:
            prediction = winner

        item["rawScores"] = {
            module: int(round(item["rawEvidence"][module] * 100.0))
            for module in MODULE_KEYS
        }
        item["scores"] = scores
        item["prediction"] = prediction
        item["predictionLabel"] = MODULE_LABELS[prediction]
        item["topScore"] = int(top_score)
        item["secondScore"] = int(second_score)
        item["scoreMargin"] = int(margin)
        item["confidence"] = round(float(top_score / 100.0), 3)

        if prediction == UNKNOWN:
            item["confidenceLabel"] = "Low"
        elif top_score >= 82 and margin >= 55:
            item["confidenceLabel"] = "High"
        elif top_score >= 65 and margin >= 30:
            item["confidenceLabel"] = "Medium"
        else:
            item["confidenceLabel"] = "Low"

        item["reasons"] = _reasons(prediction, features)


def _add_neighbor_agreement(prepared: list[dict[str, Any]], radius: int = 2) -> None:
    predictions = [item.get("prediction", UNKNOWN) for item in prepared]

    for index, item in enumerate(prepared):
        prediction = item.get("prediction", UNKNOWN)
        neighbors = predictions[
            max(0, index - radius):min(len(prepared), index + radius + 1)
        ]

        agreement = (
            sum(1 for value in neighbors if value == prediction)
            / max(len(neighbors), 1)
        )
        item["neighborAgreement"] = round(float(agreement), 3)


def _make_groups(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []

    groups: list[dict[str, Any]] = []
    start = 0
    current = results[0]["prediction"]

    def append_group(first: int, last: int, prediction: str) -> None:
        members = results[first:last + 1]
        average_scores = {
            module: round(
                float(np.mean([
                    member.get("scores", {}).get(module, 0)
                    for member in members
                ])),
                2,
            )
            for module in MODULE_KEYS
        }

        groups.append({
            "prediction": prediction,
            "predictionLabel": MODULE_LABELS[prediction],
            "startSliceIndex": int(first),
            "endSliceIndex": int(last),
            "startSliceNumber": int(first) + 1,
            "endSliceNumber": int(last) + 1,
            "sliceCount": int(last - first + 1),
            "averageScores": average_scores,
            "averageConfidence": round(
                float(np.mean([
                    member.get("confidence", 0.0)
                    for member in members
                ])),
                3,
            ),
        })

    for index in range(1, len(results)):
        prediction = results[index]["prediction"]
        if prediction != current:
            append_group(start, index - 1, current)
            start = index
            current = prediction

    append_group(start, len(results) - 1, current)
    return groups


def _probable_module3_group(groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        group
        for group in groups
        if group["prediction"] == MODULE_3
    ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda group: (
            int(group["sliceCount"]),
            float(group["averageScores"][MODULE_3]),
            float(group["averageConfidence"]),
        ),
        reverse=True,
    )

    return dict(candidates[0])



def _local_maximum_mask(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return compact local maxima without SciPy."""
    maxima = np.asarray(valid_mask, dtype=bool).copy()

    for delta_y in (-2, -1, 0, 1, 2):
        for delta_x in (-2, -1, 0, 1, 2):
            if delta_x == 0 and delta_y == 0:
                continue

            shifted = np.roll(
                np.roll(values, delta_y, axis=0),
                delta_x,
                axis=1,
            )
            maxima &= values >= shifted

    maxima[:3, :] = False
    maxima[-3:, :] = False
    maxima[:, :3] = False
    maxima[:, -3:] = False
    return maxima


def _analyze_module3_target_slice(
    slice_data: dict[str, Any],
    slice_index: int,
) -> dict[str, Any]:
    """
    Rank one already-classified Module 3 slice by two-bright-BB evidence.

    This score is deliberately separate from the Module 3 classification score:
    - classification score says "this belongs to the uniformity module"
    - target score says "this is the best exact slice for the two BB dots"
    """
    info = slice_data.get("info", {})

    if info.get("isColorDicom"):
        raise ValueError("Color DICOM cannot be ranked for raw-HU BB dots.")

    raw = np.asarray(slice_data["pixels"], dtype=np.float32)
    cx, cy, radius = _estimate_phantom_geometry(raw)

    yy, xx = np.indices(raw.shape)
    interior = (
        np.isfinite(raw)
        & (
            (xx - cx) ** 2
            + (yy - cy) ** 2
            <= (radius * 0.82) ** 2
        )
    )

    if np.sum(interior) < 1000:
        raise ValueError("Not enough Module 3 interior pixels for BB ranking.")

    interior_median = float(np.median(raw[interior]))
    background_radius = max(3, int(round(radius * 0.022)))
    local_background = _box_blur(raw, background_radius)
    positive_residual = raw - local_background

    residual_values = positive_residual[interior]
    residual_median = float(np.median(residual_values))
    noise_sigma = max(
        1.0,
        float(
            np.median(
                np.abs(residual_values - residual_median)
            )
            * 1.4826
        ),
    )

    maxima = _local_maximum_mask(positive_residual, interior)
    coordinates = np.argwhere(maxima)

    ranked_points = sorted(
        [
            {
                "residual": float(positive_residual[y, x]),
                "rawAboveMedian": float(raw[y, x] - interior_median),
                "rawHu": float(raw[y, x]),
                "x": int(x),
                "y": int(y),
            }
            for y, x in coordinates
        ],
        key=lambda point: point["residual"],
        reverse=True,
    )

    suppression_distance = max(7, int(round(radius * 0.035)))
    candidates: list[dict[str, Any]] = []

    for point in ranked_points:
        if point["residual"] < noise_sigma * 3.0:
            break

        is_separate = all(
            (
                (point["x"] - existing["x"]) ** 2
                + (point["y"] - existing["y"]) ** 2
            ) > suppression_distance ** 2
            for existing in candidates
        )

        if not is_separate:
            continue

        point["zScore"] = round(
            float(point["residual"] / noise_sigma),
            4,
        )
        candidates.append(point)

        if len(candidates) >= 24:
            break

    pairs: list[dict[str, Any]] = []

    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            distance_pixels = math.hypot(
                first["x"] - second["x"],
                first["y"] - second["y"],
            )
            distance_ratio = distance_pixels / max(radius, 1.0)

            # Broad physical sanity only. No fixed orientation or location.
            if distance_ratio < 0.35 or distance_ratio > 1.45:
                continue

            minimum_z = min(first["zScore"], second["zScore"])
            sum_z = first["zScore"] + second["zScore"]
            separation_quality = max(
                0.0,
                1.0 - abs(distance_ratio - 0.95) / 0.65,
            )

            pair_evidence = (
                minimum_z * 0.70
                + sum_z * 0.15
                + separation_quality * 3.0
            )

            pairs.append({
                "markerA": first,
                "markerB": second,
                "distancePixels": float(distance_pixels),
                "distanceRatio": float(distance_ratio),
                "minimumZ": float(minimum_z),
                "pairEvidence": float(pair_evidence),
            })

    pairs.sort(
        key=lambda pair: pair["pairEvidence"],
        reverse=True,
    )

    if not pairs:
        return {
            "sliceIndex": int(slice_index),
            "sliceNumber": int(slice_index) + 1,
            "targetScore": 0.0,
            "pairAccepted": False,
            "reason": "No separated two-bright-dot pair found.",
            "candidateCount": len(candidates),
            "noiseSigma": round(float(noise_sigma), 3),
        }

    best_pair = pairs[0]
    selected_positions = {
        (
            int(best_pair["markerA"]["x"]),
            int(best_pair["markerA"]["y"]),
        ),
        (
            int(best_pair["markerB"]["x"]),
            int(best_pair["markerB"]["y"]),
        ),
    }

    strongest_competitor_z = 0.0

    for candidate in candidates:
        position = (int(candidate["x"]), int(candidate["y"]))
        if position not in selected_positions:
            strongest_competitor_z = max(
                strongest_competitor_z,
                float(candidate["zScore"]),
            )

    minimum_z = float(best_pair["minimumZ"])
    minimum_residual = min(
        float(best_pair["markerA"]["residual"]),
        float(best_pair["markerB"]["residual"]),
    )
    dominance_ratio = minimum_z / max(strongest_competitor_z, 1.0)

    target_score = 100.0 * _clamp01(
        0.55 * _scale(minimum_z, 5.0, 40.0)
        + 0.30 * _scale(minimum_residual, 25.0, 200.0)
        + 0.15 * _scale(dominance_ratio, 1.25, 5.0)
    )

    row_spacing = info.get("pixelSpacingRow")
    col_spacing = info.get("pixelSpacingCol")
    distance_mm = None

    if row_spacing and col_spacing:
        delta_x_mm = (
            best_pair["markerA"]["x"]
            - best_pair["markerB"]["x"]
        ) * float(col_spacing)
        delta_y_mm = (
            best_pair["markerA"]["y"]
            - best_pair["markerB"]["y"]
        ) * float(row_spacing)
        distance_mm = math.hypot(delta_x_mm, delta_y_mm)

    pair_accepted = (
        target_score >= 35.0
        and minimum_z >= 9.0
        and dominance_ratio >= 1.8
    )

    return {
        "sliceIndex": int(slice_index),
        "sliceNumber": int(slice_index) + 1,
        "targetScore": round(float(target_score), 2),
        "pairAccepted": bool(pair_accepted),
        "markerA": {
            "x": int(best_pair["markerA"]["x"]),
            "y": int(best_pair["markerA"]["y"]),
            "rawHu": round(float(best_pair["markerA"]["rawHu"]), 2),
            "localPositiveContrast": round(
                float(best_pair["markerA"]["residual"]),
                2,
            ),
            "zScore": round(float(best_pair["markerA"]["zScore"]), 2),
        },
        "markerB": {
            "x": int(best_pair["markerB"]["x"]),
            "y": int(best_pair["markerB"]["y"]),
            "rawHu": round(float(best_pair["markerB"]["rawHu"]), 2),
            "localPositiveContrast": round(
                float(best_pair["markerB"]["residual"]),
                2,
            ),
            "zScore": round(float(best_pair["markerB"]["zScore"]), 2),
        },
        "distancePixels": round(
            float(best_pair["distancePixels"]),
            3,
        ),
        "distanceMm": (
            round(float(distance_mm), 3)
            if distance_mm is not None
            else None
        ),
        "minimumDotZScore": round(float(minimum_z), 3),
        "minimumLocalPositiveContrast": round(
            float(minimum_residual),
            3,
        ),
        "strongestCompetitorZScore": round(
            float(strongest_competitor_z),
            3,
        ),
        "dominanceRatio": round(float(dominance_ratio), 3),
        "candidateCount": len(candidates),
        "noiseSigma": round(float(noise_sigma), 3),
        "reason": (
            "Two dominant small positive BB peaks found."
            if pair_accepted
            else "A possible pair was found, but it did not dominate image noise strongly enough."
        ),
    }


def _same_bb_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    tolerance_pixels: float = 14.0,
) -> bool:
    if not first.get("markerA") or not second.get("markerA"):
        return False

    first_points = [first["markerA"], first["markerB"]]
    second_points = [second["markerA"], second["markerB"]]

    direct = (
        math.hypot(
            first_points[0]["x"] - second_points[0]["x"],
            first_points[0]["y"] - second_points[0]["y"],
        ) <= tolerance_pixels
        and math.hypot(
            first_points[1]["x"] - second_points[1]["x"],
            first_points[1]["y"] - second_points[1]["y"],
        ) <= tolerance_pixels
    )

    swapped = (
        math.hypot(
            first_points[0]["x"] - second_points[1]["x"],
            first_points[0]["y"] - second_points[1]["y"],
        ) <= tolerance_pixels
        and math.hypot(
            first_points[1]["x"] - second_points[0]["x"],
            first_points[1]["y"] - second_points[0]["y"],
        ) <= tolerance_pixels
    )

    return bool(direct or swapped)


def _rank_exact_module3_slice(
    slices: list[dict[str, Any]],
    public_results: list[dict[str, Any]],
    probable_module3: dict[str, Any] | None,
) -> dict[str, Any] | None:
    for result in public_results:
        result["module3TargetScore"] = None
        result["module3TargetRank"] = None
        result["module3TargetAccepted"] = False

    if not probable_module3:
        return None

    first_index = int(probable_module3["startSliceIndex"])
    last_index = int(probable_module3["endSliceIndex"])
    rankings: list[dict[str, Any]] = []

    for slice_index in range(first_index, last_index + 1):
        if public_results[slice_index].get("prediction") != MODULE_3:
            continue

        try:
            ranking = _analyze_module3_target_slice(
                slice_data=slices[slice_index],
                slice_index=slice_index,
            )
        except Exception as exc:
            ranking = {
                "sliceIndex": int(slice_index),
                "sliceNumber": int(slice_index) + 1,
                "targetScore": 0.0,
                "pairAccepted": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

        rankings.append(ranking)

    rankings.sort(
        key=lambda ranking: (
            float(ranking.get("targetScore", 0.0)),
            float(ranking.get("minimumDotZScore", 0.0)),
        ),
        reverse=True,
    )

    for rank, ranking in enumerate(rankings, start=1):
        slice_index = int(ranking["sliceIndex"])
        public_results[slice_index]["module3TargetScore"] = round(
            float(ranking.get("targetScore", 0.0)),
            2,
        )
        public_results[slice_index]["module3TargetRank"] = int(rank)
        public_results[slice_index]["module3TargetAccepted"] = bool(
            ranking.get("pairAccepted", False)
        )
        public_results[slice_index]["module3TargetReason"] = ranking.get(
            "reason",
            "",
        )

    if not rankings:
        return None

    best = dict(rankings[0])
    second = dict(rankings[1]) if len(rankings) > 1 else None
    score_gap = (
        float(best.get("targetScore", 0.0))
        - float(second.get("targetScore", 0.0))
        if second
        else float(best.get("targetScore", 0.0))
    )

    adjacent_support_slices: list[int] = []

    for candidate in rankings[1:]:
        if abs(
            int(candidate["sliceIndex"])
            - int(best["sliceIndex"])
        ) > 2:
            continue

        if _same_bb_pair(best, candidate):
            adjacent_support_slices.append(
                int(candidate["sliceNumber"])
            )

    best_verified = (
        bool(best.get("pairAccepted", False))
        and float(best.get("targetScore", 0.0)) >= 70.0
        and score_gap >= 15.0
    )

    best.update({
        "verifiedBestSlice": bool(best_verified),
        "scoreGapToSecond": round(float(score_gap), 2),
        "secondBestSliceNumber": (
            int(second["sliceNumber"])
            if second
            else None
        ),
        "secondBestTargetScore": (
            round(float(second.get("targetScore", 0.0)), 2)
            if second
            else None
        ),
        "adjacentSupportSlices": adjacent_support_slices,
        "rankedSlices": [
            {
                "sliceIndex": int(ranking["sliceIndex"]),
                "sliceNumber": int(ranking["sliceNumber"]),
                "targetScore": round(
                    float(ranking.get("targetScore", 0.0)),
                    2,
                ),
                "pairAccepted": bool(
                    ranking.get("pairAccepted", False)
                ),
            }
            for ranking in rankings
        ],
        "scoreMeaning": (
            "This is the exact-slice BB target score, separate from the Module 3 category score."
        ),
    })

    return best

def _public_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if not key.startswith("_") and key not in {"rawEvidence", "evidence"}
    }


def create_acr_module_classification(
    stack_id: str | None = None,
    uploaded_file=None,
    max_size: int = 160,
) -> dict[str, Any]:
    started = time.perf_counter()

    slices = _get_slices_from_stack_or_upload(
        stack_id=stack_id,
        uploaded_file=uploaded_file,
    )

    prepared: list[dict[str, Any]] = []
    failure_count = 0

    for index, slice_data in enumerate(slices):
        try:
            item = _prepare_slice(
                slice_data=slice_data,
                slice_index=index,
                max_size=max_size,
            )
        except Exception as exc:
            failure_count += 1
            item = {
                "sliceIndex": int(index),
                "sliceNumber": int(index) + 1,
                "status": "error",
                "sourceName": slice_data.get("sourceName", ""),
                "sliceLabel": slice_data.get("label", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "features": {},
            }

        prepared.append(item)

    _add_cross_slice_persistence(prepared)

    for item in prepared:
        if item.get("status") == "ok":
            item["rawEvidence"] = _raw_evidence(item["features"])

    _smooth_evidence(prepared)
    _finalize_predictions(prepared)
    _add_neighbor_agreement(prepared)

    public_results = [_public_result(item) for item in prepared]
    groups = _make_groups(public_results)
    probable_module3 = _probable_module3_group(groups)
    best_module3_slice = _rank_exact_module3_slice(
        slices=slices,
        public_results=public_results,
        probable_module3=probable_module3,
    )

    counts = {
        module: sum(
            1
            for result in public_results
            if result["prediction"] == module
        )
        for module in (*MODULE_KEYS, UNKNOWN)
    }

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "success": True,
        "analysisType": "ACR Four-Module Competitive Classification",
        "classifierVersion": CLASSIFIER_VERSION,
        "sliceCount": len(slices),
        "classifiedSliceCount": len(slices) - failure_count,
        "failedSliceCount": int(failure_count),
        "processingTimeMs": round(float(elapsed_ms), 2),
        "moduleLabels": MODULE_LABELS,
        "counts": counts,
        "groups": groups,
        "probableModule3Group": probable_module3,
        "bestModule3Slice": best_module3_slice,
        "slices": public_results,
        "scoreNote": (
            "M1–M4 scores classify which module each slice belongs to and sum to 100. "
            "Module 3 slices can therefore have similar category scores. The separate M3 target score "
            "ranks the exact slice where the two bright BB dots are strongest."
        ),
        "nextStepNote": (
            "V4 identifies the Module 3 range and separately ranks the exact two-BB target slice. "
            "The final distance overlay and five 400 mm² ROI measurement are still not run here."
        ),
    }
