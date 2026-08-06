"""Low-level detector for the four cardinal Module 4 perimeter BB markers.

Candidate gating and final Module 4 scoring belong to the shared classifier.
This helper only analyzes one raw CT slice and contains no slice selector.
"""

from __future__ import annotations

import html as html_lib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

from services.dicom_display import window_pixels_to_image
from services.image_helpers import image_to_base64


_CARDINAL_ANGLES = {
    "right": 0.0,
    "bottom": 90.0,
    "left": 180.0,
    "top": 270.0,
}

MODULE4_FALLBACK_PIN_ANGLE_DEGREES = 90.0
MODULE4_INSERT_RADIUS_RATIO = 0.46
MODULE4_PHI_OFFSET_DEGREES = 0.0
MODULE4_ROI_SIDE_RATIO = 0.14
MODULE4_ROI_ANGLE_OFFSET_DEGREES = 45.0
MODULE4_TARGETED_POLISH_IDS = frozenset({"B6", "B7", "B8"})
MODULE4_STD_VISIBLE_RATIO_THRESHOLD = 3.0
MODULE4_STD_REVIEW_RATIO_THRESHOLD = 2.5
MODULE4_STD_THRESHOLD_STATUS = "provisional"
MODULE4_STD_THRESHOLD_SOURCE = (
    "Software-development threshold pending validation against physicist "
    "review across multiple scans."
)
MODULE4_NOMINAL_LP_CM_BY_ID = {
    "B8": 4,
    "B7": 5,
    "B6": 6,
    "B5": 7,
    "B4": 8,
    "B3": 9,
    "B2": 10,
    "B1": 12,
}
# This is a development/review mapping only. Confirm physical ROI-to-lp/cm
# mapping before using preliminary resolved lp/cm. Null values are intentional:
# suggestions must never become active configuration automatically.
MODULE4_MANUAL_DEVELOPMENT_LP_CM_MAPPING = {
    "B1": None, "B2": None, "B3": None, "B4": None,
    "B5": None, "B6": None, "B7": None, "B8": None,
}
MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING = False
MODULE4_REQUIRED_LP_CM = None
MODULE4_SUGGESTED_MAPPING_REVIEW = {
    "suggested_mapping_available": False,
    "suggested_mapping": {},
    "suggested_mapping_confidence": None,
    "can_apply_suggested_mapping": False,
    "reason": "Suggested mapping requires physicist confirmation before use.",
}
_MODULE4_ACTIVE_EXPECTED_LP_CM_MAPPING = (
    dict(MODULE4_MANUAL_DEVELOPMENT_LP_CM_MAPPING)
    if MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING
    else {target_id: None for target_id in MODULE4_MANUAL_DEVELOPMENT_LP_CM_MAPPING}
)
_MODULE4_MAPPING_COMPLETE = all(
    value in MODULE4_NOMINAL_LP_CM_BY_ID.values()
    for value in _MODULE4_ACTIVE_EXPECTED_LP_CM_MAPPING.values()
)
MODULE4_ANALYSIS_CONFIG = {
    "development_mapping_enabled": MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING,
    "mapping_source": (
        "manual_development_config"
        if MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING else "not_configured"
    ),
    "expected_lp_cm_mapping": _MODULE4_ACTIVE_EXPECTED_LP_CM_MAPPING,
    "allowed_lp_cm_values": [4, 5, 6, 7, 8, 9, 10, 12],
    "required_lp_cm": MODULE4_REQUIRED_LP_CM,
    "physicist_review_required": True,
}
MODULE4_EXPECTED_LP_CM_CONFIG = {
    "status": (
        "development_configured"
        if MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING and _MODULE4_MAPPING_COMPLETE
        else "not_configured"
    ),
    "mapping_source": MODULE4_ANALYSIS_CONFIG["mapping_source"],
    "targets": MODULE4_ANALYSIS_CONFIG["expected_lp_cm_mapping"],
    "allowed_lp_cm_values": MODULE4_ANALYSIS_CONFIG["allowed_lp_cm_values"],
    "required_lp_cm": MODULE4_REQUIRED_LP_CM,
    "physicist_review_required": True,
}
# Compatibility alias for graph/vote consumers; its values are controlled only
# by MODULE4_EXPECTED_LP_CM_CONFIG and the explicit development enable flag.
MODULE4_EXPECTED_LP_CM = MODULE4_EXPECTED_LP_CM_CONFIG["targets"]

MODULE4_ENABLE_AUTO_PRELIMINARY_SCORING = True
MODULE4_AUTO_PRELIMINARY_SCORING_CONFIG = {
    "scoring_enabled": True,
    "scoring_type": "automatic_evidence_based_resolution",
    "physicist_review_required": True,
    "official_acr_result": False,
    "target_lp_cm_order": [
        {"target": "B8", "lp_cm": 4},
        {"target": "B7", "lp_cm": 5},
        {"target": "B6", "lp_cm": 6},
        {"target": "B5", "lp_cm": 7},
        {"target": "B4", "lp_cm": 8},
        {"target": "B3", "lp_cm": 9},
        {"target": "B2", "lp_cm": 10},
        {"target": "B1", "lp_cm": 12},
    ],
    "continuity_required": True,
    "stop_at_first_unresolved_target": True,
    "use_target_specific_thresholds": True,
    "use_signal_curve_break_detection": True,
    "use_internal_contrast_ladder": True,
    "use_profile_ladder": True,
    "use_peak_valley_ladder": True,
    "use_fft_support": True,
}

MODULE4_ADAPTIVE_THRESHOLD_CONFIG = {
    "status": "development_adaptive",
    "physicist_review_required": True,
    "adaptive_thresholds_enabled": True,
    "fft": {
        "use_adaptive_fft_snr": True,
        "fft_snr_min_guardrail": 2.5,
        "fft_snr_max_guardrail": 6.0,
        "fft_score_min_guardrail": 0.45,
        "fft_score_max_guardrail": 0.75,
        "max_frequency_error_percent": 20.0,
    },
    "profile": {
        "use_adaptive_profile_snr": True,
        "profile_snr_min_guardrail": 1.8,
        "profile_snr_max_guardrail": 5.0,
        "profile_score_min_guardrail": 0.45,
        "profile_score_max_guardrail": 0.75,
        "min_peak_count": 3,
        "min_valley_count": 2,
        "max_spacing_error_percent": 25.0,
    },
    "std_support": {
        "use_adaptive_std_ratio": True,
        "std_ratio_min_guardrail": 1.8,
        "std_ratio_max_guardrail": 5.0,
        "peak_to_valley_noise_min_guardrail": 2.5,
        "peak_to_valley_noise_max_guardrail": 8.0,
        "std_score_min_guardrail": 0.45,
        "std_score_max_guardrail": 0.75,
    },
    "combined": {
        "fft_weight": 0.45,
        "profile_weight": 0.35,
        "std_weight": 0.20,
        "combined_score_min_guardrail": 0.50,
        "combined_score_max_guardrail": 0.75,
    },
}

MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG = {
    "status": "development_target_specific",
    "target_specific_thresholds_enabled": True,
    "physicist_review_required": True,
    "strategy": {
        "use_run_context": True,
        "use_target_local_context": True,
        "use_nominal_class_for_display_only": True,
        "mapping_required_for_frequency_pass": True,
    },
    "guardrails": {
        "fft_snr_min": 2.5, "fft_snr_max": 100.0,
        "fft_score_min": 0.45, "fft_score_max": 0.75,
        "profile_snr_min": 0.05, "profile_snr_max": 2.5,
        "profile_score_min": 0.25, "profile_score_max": 0.80,
        "combined_score_min": 0.45, "combined_score_max": 0.75,
    },
    "target_families": {
        "primary": ["B8", "B7", "B6", "B5"],
        "secondary": ["B4"],
        "reference_high_frequency": ["B3", "B2", "B1"],
    },
}


def _module4_safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _module4_clamp(value, low, high) -> float | None:
    number = _module4_safe_float(value)
    low_value = _module4_safe_float(low)
    high_value = _module4_safe_float(high)
    if number is None or low_value is None or high_value is None:
        return None
    return max(low_value, min(high_value, number))


def _module4_safe_median(values) -> float | None:
    clean = [_module4_safe_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    return float(np.median(clean)) if clean else None


def _module4_safe_mad(values) -> float | None:
    clean = [_module4_safe_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    median = _module4_safe_median(clean)
    return (
        float(np.median(np.abs(np.asarray(clean) - median)))
        if clean and median is not None else None
    )


def _module4_safe_margin(actual, threshold) -> float | None:
    actual_value = _module4_safe_float(actual)
    threshold_value = _module4_safe_float(threshold)
    if actual_value is None or threshold_value is None:
        return None
    return round(actual_value - threshold_value, 4)


def _module4_safe_error_margin(max_allowed, actual_error) -> float | None:
    maximum = _module4_safe_float(max_allowed)
    error = _module4_safe_float(actual_error)
    if maximum is None or error is None:
        return None
    return round(maximum - error, 4)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_boolean_values(
    array_name: str,
    array: np.ndarray,
    mask_name: str,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply a boolean mask only when its source array has the same shape."""
    if array.shape != mask.shape:
        raise ValueError(
            f"Module 4 shape mismatch: {array_name}{array.shape} cannot use "
            f"{mask_name}{mask.shape}."
        )
    return array[mask]


def _square_points(
    center_x: float,
    center_y: float,
    side: float,
    angle_degrees: float,
) -> list[dict]:
    half_side = side / 2.0
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        {
            "x": round(center_x + dx * cosine - dy * sine, 2),
            "y": round(center_y + dx * sine + dy * cosine, 2),
        }
        for dx, dy in (
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        )
    ]


def _fit_square_target(
    working: np.ndarray,
    residual: np.ndarray,
    component_bounds: tuple[slice, slice],
    center_x: float,
    center_y: float,
    noise: float,
    radius: float,
) -> dict:
    """Fit a robust min-area square to the local target envelope."""
    margin = max(4, int(round(radius * 0.035)))
    y0 = max(0, component_bounds[0].start - margin)
    y1 = min(working.shape[0], component_bounds[0].stop + margin)
    x0 = max(0, component_bounds[1].start - margin)
    x1 = min(working.shape[1], component_bounds[1].stop + margin)
    local = working[y0:y1, x0:x1]
    local_residual = residual[y0:y1, x0:x1]

    # Reconstruct the target at a less aggressive scale than the discovery
    # mask. This closes gaps between bars without inheriting its large dilation.
    local_threshold = max(
        noise * 2.2,
        float(np.percentile(local_residual, 72.0)),
    )
    target_mask = local_residual >= local_threshold
    close_size = max(3, min(7, int(round(radius * 0.022)) | 1))
    target_mask = ndimage.binary_closing(
        target_mask,
        structure=np.ones((close_size, close_size), dtype=bool),
        iterations=1,
    )
    target_mask = ndimage.binary_fill_holes(target_mask)
    target_labels, target_count = ndimage.label(target_mask)
    local_center_x = int(round(center_x - x0))
    local_center_y = int(round(center_y - y0))
    local_center_x = max(0, min(local_center_x, target_mask.shape[1] - 1))
    local_center_y = max(0, min(local_center_y, target_mask.shape[0] - 1))
    target_label = int(target_labels[local_center_y, local_center_x])
    if target_label == 0 and target_count:
        label_centers = ndimage.center_of_mass(
            target_mask,
            target_labels,
            range(1, target_count + 1),
        )
        target_label = min(
            range(1, target_count + 1),
            key=lambda label: math.hypot(
                label_centers[label - 1][1] - local_center_x,
                label_centers[label - 1][0] - local_center_y,
            ),
        )
    fitted_mask = target_labels == target_label if target_label else target_mask
    fit_y, fit_x = np.nonzero(fitted_mask)
    if fit_x.size < 12:
        fit_y, fit_x = np.nonzero(target_mask)
    if fit_x.size < 12:
        fit_y, fit_x = np.indices(local.shape)
        fit_y, fit_x = fit_y.ravel(), fit_x.ravel()

    absolute_x = fit_x.astype(float) + x0
    absolute_y = fit_y.astype(float) + y0
    fits = []
    for angle_degrees in np.arange(-45.0, 45.01, 0.5):
        angle = math.radians(float(angle_degrees))
        rotated_x = absolute_x * math.cos(angle) + absolute_y * math.sin(angle)
        rotated_y = -absolute_x * math.sin(angle) + absolute_y * math.cos(angle)
        # Robust extents keep isolated threshold pixels from inflating the box.
        x_low, x_high = np.percentile(rotated_x, (2.0, 98.0))
        y_low, y_high = np.percentile(rotated_y, (2.0, 98.0))
        span_x = float(x_high - x_low + 1.0)
        span_y = float(y_high - y_low + 1.0)
        area = span_x * span_y
        fits.append({
            "angle": float(angle_degrees),
            "area": area,
            "span_x": span_x,
            "span_y": span_y,
            "x_mid": float((x_low + x_high) / 2.0),
            "y_mid": float((y_low + y_high) / 2.0),
        })
    minimum_area = min(item["area"] for item in fits)
    near_minimum = [item for item in fits if item["area"] <= minimum_area * 1.025]
    expected_angles = (-45.0, 0.0, 45.0)
    best = min(
        near_minimum,
        key=lambda item: min(abs(item["angle"] - expected) for expected in expected_angles),
    )
    angle_degrees = min(
        expected_angles,
        key=lambda expected: abs(best["angle"] - expected),
    )
    angle = math.radians(angle_degrees)
    rotated_x = absolute_x * math.cos(angle) + absolute_y * math.sin(angle)
    rotated_y = -absolute_x * math.sin(angle) + absolute_y * math.cos(angle)
    x_low, x_high = np.percentile(rotated_x, (2.0, 98.0))
    y_low, y_high = np.percentile(rotated_y, (2.0, 98.0))
    span_x = float(x_high - x_low + 1.0)
    span_y = float(y_high - y_low + 1.0)
    x_mid = float((x_low + x_high) / 2.0)
    y_mid = float((y_low + y_high) / 2.0)
    center_fit_x = (
        x_mid * math.cos(angle)
        - y_mid * math.sin(angle)
    )
    center_fit_y = (
        x_mid * math.sin(angle)
        + y_mid * math.cos(angle)
    )
    boundary_side = max(span_x, span_y)
    return {
        "center_x": float(center_fit_x),
        "center_y": float(center_fit_y),
        "angle_degrees": float(angle_degrees),
        "boundary_side": float(boundary_side),
        "fit_area": float(span_x * span_y),
        "fit_pixel_count": int(fit_x.size),
        "local_threshold": float(local_threshold),
    }


def _profile_periodicity(profile: np.ndarray) -> dict:
    profile = ndimage.gaussian_filter1d(np.asarray(profile, dtype=float), sigma=0.7)
    centered = profile - float(np.mean(profile))
    scale = max(float(np.std(centered)), 1e-6)
    normalized = centered / scale
    peaks = np.where(
        (normalized[1:-1] > normalized[:-2])
        & (normalized[1:-1] >= normalized[2:])
        & (normalized[1:-1] >= 0.45)
    )[0] + 1
    valleys = np.where(
        (normalized[1:-1] < normalized[:-2])
        & (normalized[1:-1] <= normalized[2:])
        & (normalized[1:-1] <= -0.45)
    )[0] + 1
    autocorrelation = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    periodicity = 0.0
    average_period = None
    if autocorrelation.size > 4 and autocorrelation[0] > 0:
        autocorrelation /= autocorrelation[0]
        search = autocorrelation[2:max(4, len(autocorrelation) // 2)]
        if search.size:
            average_period = int(np.argmax(search) + 2)
            periodicity = _clamp01(float(np.max(search)))
    peak_score = _clamp01(len(peaks) / 5.0)
    peak_spacings = np.diff(peaks).astype(float)
    period_consistency = (
        _clamp01(
            1.0
            - float(np.std(peak_spacings))
            / max(float(np.mean(peak_spacings)), 1.0)
        )
        if peak_spacings.size >= 2
        else 0.0
    )
    return {
        "periodicity_score": periodicity,
        "peaks_found": int(len(peaks)),
        "peak_score": peak_score,
        "average_period_px": average_period,
        "period_consistency": period_consistency,
        "peak_positions": peaks.astype(int).tolist(),
        "valley_positions": valleys.astype(int).tolist(),
        "valleys_found": int(len(valleys)),
        "profile": profile,
    }


def estimate_module4_stripe_orientation(
    working: np.ndarray,
    residual: np.ndarray,
    center_x: float,
    center_y: float,
    estimated_side: float,
    noise: float,
) -> dict:
    """Find repeated stripe and perpendicular boundary profile directions."""
    crop_side = max(16, int(round(estimated_side * 1.45)))
    half = crop_side // 2
    x0 = max(0, int(round(center_x)) - half)
    y0 = max(0, int(round(center_y)) - half)
    x1 = min(working.shape[1], x0 + crop_side)
    y1 = min(working.shape[0], y0 + crop_side)
    local = working[y0:y1, x0:x1]
    local_residual = residual[y0:y1, x0:x1]
    if min(local.shape) < 12:
        return {"success": False, "reason": "Local crop was too small for profile fitting."}

    hypotheses = []
    for box_angle in np.arange(-60.0, 60.01, 2.0):
        rotated = ndimage.rotate(
            local,
            -float(box_angle),
            reshape=False,
            order=1,
            mode="nearest",
        )
        rotated_residual = ndimage.rotate(
            local_residual,
            -float(box_angle),
            reshape=False,
            order=1,
            mode="nearest",
        )
        margin_y = max(2, int(round(rotated.shape[0] * 0.18)))
        margin_x = max(2, int(round(rotated.shape[1] * 0.18)))
        core = rotated[margin_y:-margin_y, margin_x:-margin_x]
        if min(core.shape) < 6:
            continue
        profile_x = np.mean(core, axis=0)
        profile_y = np.mean(core, axis=1)
        evidence_x = _profile_periodicity(profile_x)
        evidence_y = _profile_periodicity(profile_y)
        if (
            evidence_x["periodicity_score"] + evidence_x["peak_score"]
            >= evidence_y["periodicity_score"] + evidence_y["peak_score"]
        ):
            stripe_evidence = evidence_x
            stripe_direction = float(box_angle + 90.0)
            boundary_direction = float(box_angle)
        else:
            stripe_evidence = evidence_y
            stripe_direction = float(box_angle)
            boundary_direction = float(box_angle + 90.0)

        support = rotated_residual >= max(
            noise * 1.8,
            float(np.percentile(rotated_residual, 72.0)),
        )
        support_x = np.mean(support, axis=0)
        support_y = np.mean(support, axis=1)
        x_indices = np.where(support_x >= max(0.08, float(np.max(support_x)) * 0.28))[0]
        y_indices = np.where(support_y >= max(0.08, float(np.max(support_y)) * 0.28))[0]
        if not x_indices.size or not y_indices.size:
            continue
        extent_x = float(x_indices[-1] - x_indices[0] + 1)
        extent_y = float(y_indices[-1] - y_indices[0] + 1)
        side_ratio = min(extent_x, extent_y) / max(extent_x, extent_y, 1.0)
        extent_score = _clamp01((side_ratio - 0.45) / 0.45)
        contrast_score = _clamp01(float(np.std(core)) / max(noise * 8.0, 1.0))
        quality = _clamp01(
            0.44 * stripe_evidence["periodicity_score"]
            + 0.22 * stripe_evidence["peak_score"]
            + 0.22 * extent_score
            + 0.12 * contrast_score
        )
        hypotheses.append({
            "box_angle_degrees": float(box_angle),
            "stripe_direction_degrees": stripe_direction,
            "boundary_direction_degrees": boundary_direction,
            "stripe_periodicity_score": stripe_evidence["periodicity_score"],
            "stripe_peaks_found": stripe_evidence["peaks_found"],
            "average_period_px": stripe_evidence["average_period_px"],
            "boundary_extent_score": extent_score,
            "extent_x": extent_x,
            "extent_y": extent_y,
            "quality": quality,
        })
    if not hypotheses:
        return {"success": False, "reason": "No stable stripe/boundary profile hypothesis was found."}
    best = max(hypotheses, key=lambda item: item["quality"])
    best["success"] = bool(
        best["stripe_peaks_found"] >= 3
        and best["stripe_periodicity_score"] >= 0.22
        and best["boundary_extent_score"] >= 0.25
    )
    best["orientation_confidence"] = round(best["quality"], 4)
    best["reason"] = (
        "Repeated stripe family and perpendicular block extent were detected."
        if best["success"]
        else "Profile evidence was incomplete; envelope geometry fallback is required."
    )
    return best


def fit_module4_block_from_profiles(
    working: np.ndarray,
    residual: np.ndarray,
    envelope_fit: dict,
    noise: float,
    radius: float,
) -> dict:
    """Construct outer/inner ROI geometry from stripe and boundary profiles."""
    profile_fit = estimate_module4_stripe_orientation(
        working,
        residual,
        envelope_fit["center_x"],
        envelope_fit["center_y"],
        envelope_fit["boundary_side"],
        noise,
    )
    if not profile_fit.get("success"):
        return {
            **envelope_fit,
            "success": False,
            "geometry_source": "local_target_envelope_fallback",
            "orientation_method": "envelope_fallback_after_weak_profiles",
            "orientation_confidence": round(float(profile_fit.get("orientation_confidence", 0.0)), 4),
            "stripe_direction_degrees": None,
            "boundary_direction_degrees": None,
            "stripe_periodicity_score": round(float(profile_fit.get("stripe_periodicity_score", 0.0)), 4),
            "stripe_peaks_found": int(profile_fit.get("stripe_peaks_found", 0)),
            "boundary_extent_score": round(float(profile_fit.get("boundary_extent_score", 0.0)), 4),
            "profile_reason": profile_fit.get("reason", "Profile fitting failed."),
        }
    profile_side = max(profile_fit["extent_x"], profile_fit["extent_y"])
    minimum_side = envelope_fit["boundary_side"] * 0.78
    maximum_side = envelope_fit["boundary_side"] * 1.18
    fitted_side = max(minimum_side, min(profile_side, maximum_side))
    fitted_side = max(radius * 0.065, min(fitted_side * 1.035, radius * 0.20))
    return {
        **envelope_fit,
        "success": True,
        "boundary_side": float(fitted_side),
        "angle_degrees": float(profile_fit["box_angle_degrees"]),
        "geometry_source": "stripe_boundary_profile_fit",
        "orientation_method": "repeated_stripe_and_perpendicular_boundary_profiles",
        "orientation_confidence": profile_fit["orientation_confidence"],
        "stripe_direction_degrees": round(profile_fit["stripe_direction_degrees"], 3),
        "boundary_direction_degrees": round(profile_fit["boundary_direction_degrees"], 3),
        "stripe_periodicity_score": round(profile_fit["stripe_periodicity_score"], 4),
        "stripe_peaks_found": profile_fit["stripe_peaks_found"],
        "boundary_extent_score": round(profile_fit["boundary_extent_score"], 4),
        "average_period_px": profile_fit["average_period_px"],
        "profile_reason": profile_fit["reason"],
    }


def _angle_error(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _scaled_pixels(raw: np.ndarray, max_dimension: int = 384) -> tuple[np.ndarray, float, float]:
    height, width = raw.shape
    scale = min(1.0, float(max_dimension) / max(height, width))
    if scale >= 0.999:
        return raw.copy(), 1.0, 1.0
    scaled = ndimage.zoom(raw, zoom=scale, order=1, prefilter=False)
    return scaled.astype(np.float32), width / scaled.shape[1], height / scaled.shape[0]


def _estimate_phantom_geometry(raw: np.ndarray) -> tuple[float, float, float]:
    height, width = raw.shape
    finite = np.isfinite(raw)
    if np.sum(finite) < 100:
        raise ValueError("Not enough finite pixels to detect the phantom.")
    border_width = max(2, int(round(min(height, width) * 0.06)))
    border = np.concatenate((
        raw[:border_width].ravel(),
        raw[-border_width:].ravel(),
        raw[:, :border_width].ravel(),
        raw[:, -border_width:].ravel(),
    ))
    border = border[np.isfinite(border)]
    center = raw[int(height * .30):int(height * .70), int(width * .30):int(width * .70)]
    center = center[np.isfinite(center)]
    background = float(np.median(border))
    interior = float(np.median(center))
    threshold = background + (interior - background) * .35
    body = finite & (raw >= threshold if interior >= background else raw <= threshold)
    labels, count = ndimage.label(body)
    if count < 1:
        raise ValueError("Could not identify the phantom body.")
    center_label = int(labels[height // 2, width // 2])
    if center_label == 0:
        sizes = ndimage.sum(body, labels, range(1, count + 1))
        center_label = int(np.argmax(sizes)) + 1
    yy, xx = np.nonzero(labels == center_label)
    if xx.size < 100:
        raise ValueError("Detected phantom body is too small.")
    return float(np.mean(xx)), float(np.mean(yy)), float(math.sqrt(xx.size / math.pi))


def _compact_perimeter_components(
    raw: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> tuple[list[dict], dict]:
    yy, xx = np.indices(raw.shape)
    finite = np.isfinite(raw)
    radial = np.hypot(xx - cx, yy - cy) / max(radius, 1.0)
    # The reference dots sit at the phantom circumference, not in the line-pair core.
    perimeter = finite & (radial >= 0.82) & (radial <= 1.08)
    if np.sum(perimeter) < 100:
        return [], {"threshold_hu": 0.0, "noise_sigma": 0.0}

    fill = float(np.median(raw[finite]))
    working = np.where(finite, raw, fill)
    filter_size = max(5, int(round(radius * .035)) | 1)
    local_background = ndimage.median_filter(working, size=filter_size)
    residual = working - local_background
    perimeter_residual = residual[perimeter]
    residual_median = float(np.median(perimeter_residual))
    noise_sigma = max(
        1.0,
        float(np.median(np.abs(perimeter_residual - residual_median)) * 1.4826),
    )
    threshold_hu = float(np.percentile(working[perimeter], 97.0))
    residual_threshold = max(
        residual_median + 4.5 * noise_sigma,
        float(np.percentile(perimeter_residual, 98.0)),
    )
    mask = perimeter & (working >= threshold_hu) & (residual >= residual_threshold)
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    minimum_area = max(2, int(round(radius * radius * .00005)))
    maximum_area = max(minimum_area + 1, int(round(radius * radius * .0028)))
    components: list[dict] = []

    for label_id in range(1, count + 1):
        bounds = objects[label_id - 1]
        if bounds is None:
            continue
        region = labels[bounds] == label_id
        area = int(np.sum(region))
        height, width = region.shape
        if area < minimum_area or area > maximum_area or min(height, width) < 2:
            continue
        aspect = min(width, height) / max(width, height)
        fill_ratio = area / float(width * height)
        if aspect < .62 or fill_ratio < .35:
            continue
        # Large, completely filled rectangles are typical bar/block fragments.
        if area >= 8 and fill_ratio > .94:
            continue

        local_y, local_x = np.nonzero(region)
        x = float(bounds[1].start + np.mean(local_x))
        y = float(bounds[0].start + np.mean(local_y))
        radial_ratio = float(math.hypot(x - cx, y - cy) / max(radius, 1.0))
        angle = float((math.degrees(math.atan2(y - cy, x - cx)) + 360.0) % 360.0)
        contrast_z = float(np.median(_safe_boolean_values(
            "perimeter_component_residual",
            residual[bounds],
            "perimeter_component_region",
            region,
        )) / noise_sigma)
        compactness = _clamp01(
            .55 * aspect
            + .45 * (1.0 - min(1.0, abs(fill_ratio - .68) / .45))
        )
        radial_quality = _clamp01(1.0 - abs(radial_ratio - .96) / .14)
        contrast_quality = _clamp01((contrast_z - 4.0) / 12.0)
        components.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "angle_degrees": round(angle, 3),
            "radial_ratio": round(radial_ratio, 4),
            "area": area,
            "aspect_ratio": round(aspect, 4),
            "fill_ratio": round(fill_ratio, 4),
            "contrast_z": round(contrast_z, 3),
            "compactness_score": round(compactness, 4),
            "radial_score": round(radial_quality, 4),
            "contrast_score": round(contrast_quality, 4),
        })

    # Bound later slot assignment and avoid combinatorial work.
    components.sort(
        key=lambda item: (
            item["radial_score"] + item["compactness_score"] + item["contrast_score"]
        ),
        reverse=True,
    )
    return components[:20], {
        "threshold_hu": round(threshold_hu, 3),
        "residual_threshold": round(residual_threshold, 3),
        "noise_sigma": round(noise_sigma, 3),
        "raw_component_count": int(count),
    }


def score_module4_outer_bbs(pixel_array: np.ndarray, max_dimension: int = 384) -> dict:
    """Score tiny top/right/bottom/left BB dots near the phantom circumference."""
    original = np.asarray(pixel_array, dtype=np.float32)
    if original.ndim != 2:
        raise ValueError(f"Module 4 outer-BB scoring requires 2D CT pixels, got {original.shape}.")
    raw, scale_x, scale_y = _scaled_pixels(original, max_dimension=max_dimension)
    cx, cy, radius = _estimate_phantom_geometry(raw)
    components, threshold_debug = _compact_perimeter_components(raw, cx, cy, radius)
    selected: dict[str, dict] = {}

    for name, target_angle in _CARDINAL_ANGLES.items():
        ranked = []
        for component in components:
            error = _angle_error(component["angle_degrees"], target_angle)
            if error > 22.0:
                continue
            angular_score = _clamp01(1.0 - error / 22.0)
            score = (
                .38 * angular_score
                + .27 * component["radial_score"]
                + .20 * component["compactness_score"]
                + .15 * component["contrast_score"]
            )
            ranked.append((score, error, component))
        if ranked:
            score, error, component = max(ranked, key=lambda item: item[0])
            if score >= .48:
                selected[name] = {
                    **component,
                    "cardinal_position": name,
                    "angular_error_degrees": round(error, 3),
                    "score": round(float(score), 4),
                }

    markers = list(selected.values())
    marker_count = len(markers)
    if markers:
        angular_score = float(np.mean([
            _clamp01(1.0 - marker["angular_error_degrees"] / 22.0)
            for marker in markers
        ]))
        radial_values = [marker["radial_ratio"] for marker in markers]
        radial_consistency = _clamp01(1.0 - float(np.std(radial_values)) / .08)
        contrast_values = [marker["contrast_z"] for marker in markers]
        contrast_consistency = _clamp01(
            1.0 - float(np.std(contrast_values)) / max(float(np.mean(contrast_values)), 1.0)
        )
        geometry_score = _clamp01(
            .48 * angular_score
            + .34 * radial_consistency
            + .18 * contrast_consistency
        )
        marker_quality = float(np.mean([marker["score"] for marker in markers]))
    else:
        geometry_score = 0.0
        marker_quality = 0.0

    count_score = marker_count / 4.0
    outer_score = _clamp01(
        .52 * count_score
        + .33 * geometry_score
        + .15 * marker_quality
    )
    if marker_count < 4:
        outer_score *= count_score

    found_names = [name for name in ("top", "right", "bottom", "left") if name in selected]
    cardinal_markers_found = {
        name: name in selected
        for name in ("top", "right", "bottom", "left")
    }
    if marker_count == 4:
        reason = "Detected 4/4 perimeter BB markers near top, right, bottom, and left phantom edge."
    else:
        reason = (
            f"Detected {marker_count}/4 perimeter BB markers"
            + (f" near {', '.join(found_names)}" if found_names else "")
            + "; needs review."
        )

    # Convert centers and phantom geometry back to original pixel coordinates.
    output_markers = []
    for marker in markers:
        output_markers.append({
            **marker,
            "x": round(marker["x"] * scale_x, 2),
            "y": round(marker["y"] * scale_y, 2),
        })

    return {
        "outer_4bb_score": round(float(outer_score), 4),
        "outer_bbs_detected": int(marker_count),
        "detected_outer_bbs": output_markers,
        "cardinal_markers_found": cardinal_markers_found,
        "geometry_score": round(float(geometry_score), 4),
        "reason": reason,
        "phantom_center": {
            "x": round(cx * scale_x, 2),
            "y": round(cy * scale_y, 2),
        },
        "phantom_radius": round(radius * ((scale_x + scale_y) / 2.0), 2),
        "component_count": len(components),
        "analysis_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "coordinate_scale": {"x": round(scale_x, 5), "y": round(scale_y, 5)},
        **threshold_debug,
    }


def _detect_module4_square_blocks_pass(
    slice_pixels: np.ndarray,
    phantom_geometry: tuple[float, float, float],
    pass_config: dict,
) -> dict:
    """Run one phantom-normalized block-detection parameter set."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    cx, cy, radius = phantom_geometry
    yy, xx = np.indices(raw.shape)
    finite = np.isfinite(raw)
    interior = finite & (np.hypot(xx - cx, yy - cy) <= radius * 0.76)
    values = raw[interior]
    if values.size < 500:
        raise ValueError("Not enough phantom interior pixels for Module 4 block detection.")

    fill = float(np.median(raw[finite]))
    working = np.where(finite, raw, fill)
    background = ndimage.median_filter(
        working,
        size=max(7, int(round(radius * 0.055)) | 1),
    )
    residual = working - background
    residual_values = residual[interior]
    noise = max(1.0, float(np.median(np.abs(residual_values - np.median(residual_values))) * 1.4826))
    hu_threshold = float(np.percentile(values, pass_config["hu_percentile"]))
    residual_threshold = max(
        noise * pass_config["noise_multiplier"],
        float(np.percentile(residual_values, pass_config["residual_percentile"])),
    )
    mask = interior & (working >= hu_threshold) & (residual >= residual_threshold)
    if pass_config.get("edge_supported"):
        gradient = np.hypot(
            ndimage.sobel(working, axis=0),
            ndimage.sobel(working, axis=1),
        )
        gradient_threshold = float(np.percentile(gradient[interior], 88.0))
        edge_mask = interior & (gradient >= gradient_threshold)
        edge_mask = ndimage.binary_dilation(edge_mask, iterations=1)
        mask |= edge_mask & (working >= np.percentile(values, 78.0))
    merge_size = max(
        3,
        min(13, int(round(radius * pass_config["merge_radius_fraction"])) | 1),
    )
    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((merge_size, merge_size), dtype=bool),
        iterations=pass_config["closing_iterations"],
    )
    if pass_config["dilation_iterations"]:
        mask = ndimage.binary_dilation(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=pass_config["dilation_iterations"],
        )
    labels, component_count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    minimum_area = max(12, int(radius * radius * pass_config["minimum_area_fraction"]))
    maximum_area = max(
        minimum_area + 1,
        int(radius * radius * pass_config["maximum_area_fraction"]),
    )
    candidates = []
    rejected = 0
    rejection_summary = {
        "outside_perimeter_region": 0,
        "too_small_bb_like": 0,
        "circular_hole_like": 0,
        "too_close_to_phantom_edge": 0,
        "duplicate_same_block": 0,
        "weak_square_geometry": 0,
        "low_contrast": 0,
        "oversized_region": 0,
    }

    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        region = labels[bounds] == label_id
        area = int(np.sum(region))
        height, width = region.shape
        if area < minimum_area or min(height, width) < 5:
            rejected += 1
            rejection_summary["too_small_bb_like"] += 1
            continue
        if area > maximum_area:
            rejected += 1
            rejection_summary["oversized_region"] += 1
            continue
        aspect = min(width, height) / max(width, height)
        rectangularity = area / float(width * height)
        eroded = ndimage.binary_erosion(region)
        perimeter_px = max(1, int(np.sum(region & ~eroded)))
        circularity = float(4.0 * math.pi * area / (perimeter_px * perimeter_px))
        if aspect < 0.38 or rectangularity < 0.22:
            rejected += 1
            rejection_summary["weak_square_geometry"] += 1
            continue
        rough_center_x = float(bounds[1].start + np.mean(np.nonzero(region)[1]))
        rough_center_y = float(bounds[0].start + np.mean(np.nonzero(region)[0]))
        rough_radial_ratio = math.hypot(rough_center_x - cx, rough_center_y - cy) / max(radius, 1.0)
        if circularity >= 0.76 and rough_radial_ratio >= 0.52:
            rejected += 1
            rejection_summary["circular_hole_like"] += 1
            continue
        if rough_radial_ratio > 0.76:
            rejected += 1
            rejection_summary["outside_perimeter_region"] += 1
            continue
        envelope_fit = _fit_square_target(
            working,
            residual,
            bounds,
            rough_center_x,
            rough_center_y,
            noise,
            radius,
        )
        target_fit = fit_module4_block_from_profiles(
            working,
            residual,
            envelope_fit,
            noise,
            radius,
        )
        center_x = target_fit["center_x"]
        center_y = target_fit["center_y"]
        angle_degrees = target_fit["angle_degrees"]
        boundary_side = target_fit["boundary_side"]
        radial_ratio = math.hypot(center_x - cx, center_y - cy) / max(radius, 1.0)
        if radial_ratio > 0.76:
            rejected += 1
            rejection_summary["too_close_to_phantom_edge"] += 1
            continue
        # Contrast belongs to the original labeled discovery component. Keep
        # that mask paired with the crop created from the same `bounds`.
        discovery_crop_residual = residual[bounds]
        component_residual_values = _safe_boolean_values(
            "discovery_crop_residual",
            discovery_crop_residual,
            "component_region",
            region,
        )
        contrast_score = _clamp01(
            float(np.median(component_residual_values)) / (noise * 12.0)
        )

        # Profile/measurement sampling uses a separate center-aligned crop and
        # must not reuse the discovery component's boolean mask.
        analysis_side = max(12, int(round(boundary_side * 1.16)))
        analysis_half = analysis_side // 2
        analysis_x0 = max(0, int(round(center_x)) - analysis_half)
        analysis_y0 = max(0, int(round(center_y)) - analysis_half)
        analysis_x1 = min(working.shape[1], analysis_x0 + analysis_side)
        analysis_y1 = min(working.shape[0], analysis_y0 + analysis_side)
        crop = working[analysis_y0:analysis_y1, analysis_x0:analysis_x1]
        gx = np.abs(np.diff(crop, axis=1))
        gy = np.abs(np.diff(crop, axis=0))
        edge_density = float((np.mean(gx > noise * 3.0) + np.mean(gy > noise * 3.0)) / 2.0)
        stripe_score = _clamp01(edge_density / 0.30)
        oriented_crop = ndimage.rotate(
            crop,
            -angle_degrees,
            reshape=False,
            order=1,
            mode="nearest",
        )
        inner_margin = max(2, int(round(min(oriented_crop.shape) * 0.20)))
        inner = oriented_crop[
            inner_margin:max(inner_margin + 1, oriented_crop.shape[0] - inner_margin),
            inner_margin:max(inner_margin + 1, oriented_crop.shape[1] - inner_margin),
        ]
        std_hu = float(np.std(inner))
        peak_to_valley = float(np.percentile(inner, 90) - np.percentile(inner, 10))
        inner_gx = np.abs(np.diff(inner, axis=1))
        inner_gy = np.abs(np.diff(inner, axis=0))
        edge_score = _clamp01((float(np.mean(inner_gx)) + float(np.mean(inner_gy))) / max(noise * 12.0, 1.0))
        profile_x = np.mean(inner, axis=0)
        profile_y = np.mean(inner, axis=1)
        profile = profile_x if np.std(profile_x) >= np.std(profile_y) else profile_y
        profile = ndimage.gaussian_filter1d(profile.astype(float), sigma=0.7)
        centered_profile = profile - np.mean(profile)
        autocorrelation = np.correlate(centered_profile, centered_profile, mode="full")[len(profile) - 1:]
        if autocorrelation.size > 2 and autocorrelation[0] > 0:
            autocorrelation = autocorrelation / autocorrelation[0]
            search = autocorrelation[2:max(3, len(autocorrelation) // 2)]
            periodicity_score = _clamp01(float(np.max(search)) if search.size else 0.0)
            average_period = int(np.argmax(search) + 2) if search.size else None
        else:
            periodicity_score, average_period = 0.0, None
        profile_periodicity = (
            target_fit["stripe_periodicity_score"]
            if target_fit.get("success")
            else periodicity_score
        )
        profile_quality = _clamp01(0.45 * profile_periodicity + 0.35 * edge_score + 0.20 * stripe_score)
        preliminary_visibility = (
            "visible" if profile_quality >= 0.68
            else "partial" if profile_quality >= 0.46
            else "weak" if profile_quality >= 0.28
            else "needs_review"
        )
        shape_score = _clamp01(0.55 * aspect + 0.45 * rectangularity)
        confidence = _clamp01(
            0.32 * shape_score
            + 0.30 * contrast_score
            + 0.28 * stripe_score
            + 0.10 * _clamp01(1.0 - radial_ratio / 0.72)
        )
        if confidence < 0.34:
            rejected += 1
            rejection_summary["low_contrast"] += 1
            continue
        side = boundary_side * (1.0 if target_fit.get("success") else 1.045)
        side = max(radius * 0.065, min(side, radius * 0.20))
        rotated_box = _square_points(center_x, center_y, side, angle_degrees)
        side = int(round(side))
        roi_x = int(round(center_x - side / 2))
        roi_y = int(round(center_y - side / 2))
        roi_x = max(0, min(roi_x, raw.shape[1] - side))
        roi_y = max(0, min(roi_y, raw.shape[0] - side))
        candidates.append({
            "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
            "bbox": {
                "x": roi_x,
                "y": roi_y,
                "width": int(side),
                "height": int(side),
            },
            "area_px": area,
            "boundary_side_px": round(boundary_side, 2),
            "geometry_source": target_fit["geometry_source"],
            "orientation_method": target_fit["orientation_method"],
            "orientation_confidence": target_fit["orientation_confidence"],
            "box_fit_quality": round(_clamp01(
                0.30 * shape_score
                + 0.30 * confidence
                + 0.22 * target_fit.get("boundary_extent_score", 0.0)
                + 0.18 * target_fit.get("orientation_confidence", 0.0)
            ), 4),
            "size_normalization_applied": False,
            "needs_review": bool(
                not target_fit.get("success")
                or confidence < 0.50
                or shape_score < 0.48
            ),
            "fit_area_px": round(target_fit["fit_area"], 2),
            "fit_pixel_count": target_fit["fit_pixel_count"],
            "fit_residual_threshold": round(target_fit["local_threshold"], 3),
            "rotated_box": rotated_box,
            "angle_degrees": round(angle_degrees, 3),
            "inner_roi": {
                "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
                "width": round(side * 0.66, 2),
                "height": round(side * 0.66, 2),
                "angle_degrees": round(angle_degrees, 3),
            },
            "aspect_ratio": round(float(width / max(height, 1)), 4),
            "rectangularity": round(rectangularity, 4),
            "radial_ratio": round(radial_ratio, 4),
            "contrast_score": round(contrast_score, 4),
            "stripe_score": round(stripe_score, 4),
            "std_hu": round(std_hu, 3),
            "peak_to_valley": round(peak_to_valley, 3),
            "periodicity_score": round(profile_periodicity, 4),
            "edge_score": round(edge_score, 4),
            "stripe_direction_degrees": target_fit.get("stripe_direction_degrees"),
            "boundary_direction_degrees": target_fit.get("boundary_direction_degrees"),
            "stripe_periodicity_score": round(
                float(target_fit.get("stripe_periodicity_score", profile_periodicity)),
                4,
            ),
            "stripe_peaks_found": int(target_fit.get("stripe_peaks_found", 0)),
            "boundary_extent_score": round(
                float(target_fit.get("boundary_extent_score", 0.0)),
                4,
            ),
            "profile_length": int(len(profile)),
            "average_period_px": (
                target_fit.get("average_period_px")
                if target_fit.get("success")
                else average_period
            ),
            "profile_quality": round(profile_quality, 4),
            "preliminary_visibility": preliminary_visibility,
            "review_note": "Preliminary stripe evidence only; formal measurement is not implemented.",
            "confidence": round(confidence, 4),
            "reason": (
                "ROI fitted from repeated stripe family and perpendicular block boundary."
                if target_fit.get("success")
                else target_fit.get(
                    "profile_reason",
                    "Envelope fallback used because stripe/boundary profile fitting was uncertain.",
                )
            ),
        })

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    if candidates:
        median_side = float(np.median([item["bbox"]["width"] for item in candidates]))
        minimum_side = median_side * 0.78
        maximum_side = median_side * 1.22
        size_filtered = []
        for candidate in candidates:
            current_side = float(candidate["bbox"]["width"])
            if current_side < minimum_side:
                rejected += 1
                rejection_summary["too_small_bb_like"] += 1
                continue
            final_side = int(round(min(current_side, maximum_side)))
            candidate["size_normalization_applied"] = final_side != int(round(current_side))
            center_x = candidate["center"]["x"]
            center_y = candidate["center"]["y"]
            angle_degrees = candidate["angle_degrees"]
            half_side = final_side / 2.0
            candidate["rotated_box"] = _square_points(
                center_x,
                center_y,
                final_side,
                angle_degrees,
            )
            candidate["bbox"] = {
                "x": max(0, int(round(center_x - half_side))),
                "y": max(0, int(round(center_y - half_side))),
                "width": final_side,
                "height": final_side,
            }
            candidate["inner_roi"]["width"] = round(final_side * 0.60, 2)
            candidate["inner_roi"]["height"] = round(final_side * 0.60, 2)
            size_filtered.append(candidate)
        candidates = size_filtered
    else:
        median_side = None
    merged = []
    merge_groups = []
    for candidate in candidates:
        box = candidate["bbox"]
        cx1, cy1 = candidate["center"]["x"], candidate["center"]["y"]
        duplicate_index = None
        for index, existing in enumerate(merged):
            other = existing["bbox"]
            ix1, iy1 = max(box["x"], other["x"]), max(box["y"], other["y"])
            ix2 = min(box["x"] + box["width"], other["x"] + other["width"])
            iy2 = min(box["y"] + box["height"], other["y"] + other["height"])
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = box["width"] * box["height"] + other["width"] * other["height"] - intersection
            iou = intersection / max(union, 1)
            distance = math.hypot(cx1 - existing["center"]["x"], cy1 - existing["center"]["y"])
            if iou >= 0.18 or distance <= max(box["width"], other["width"]) * 0.42:
                duplicate_index = index
                break
        if duplicate_index is None:
            merged.append(candidate)
            merge_groups.append([candidate["center"]])
        else:
            merge_groups[duplicate_index].append(candidate["center"])
            rejection_summary["duplicate_same_block"] += 1
            if candidate["confidence"] > merged[duplicate_index]["confidence"]:
                merged[duplicate_index] = candidate

    global_layout = fit_module4_global_block_layout(
        merged[:24],
        cx,
        cy,
        radius,
        working,
        residual,
        noise,
    )
    candidates = global_layout["accepted_targets"]
    return {
        "candidates": candidates,
        "phantom_center": {"x": round(cx, 2), "y": round(cy, 2)},
        "phantom_radius": round(radius, 2),
        "thresholds": {
            "hu": round(hu_threshold, 3),
            "positive_residual": round(residual_threshold, 3),
            "noise_sigma": round(noise, 3),
        },
        "components_considered": int(component_count),
        "components_rejected": int(rejected),
        "merge_groups": merge_groups,
        "merge_kernel_size": int(merge_size),
        "median_block_side_px": round(median_side, 2) if median_side is not None else None,
        "rejection_summary": rejection_summary,
        "detection_pass": pass_config["name"],
        "expected_ring_radius": global_layout["expected_ring_radius"],
        "internal_ring_radius_px": global_layout["internal_ring_radius_px"],
        "common_side_px": global_layout["common_side_px"],
        "side_votes": global_layout["side_votes"],
        "global_target_angle_degrees": global_layout["global_target_angle_degrees"],
        "angle_votes": global_layout["angle_votes"],
        "anchor_targets_used": global_layout["anchor_targets_used"],
        "strong_anchor_count": global_layout["strong_anchor_count"],
        "orientation_confidence": global_layout["orientation_confidence"],
        "global_layout_model_used": global_layout["global_layout_model_used"],
        "layout_needs_review": global_layout["layout_needs_review"],
        "target_slots": global_layout["target_slots"],
        "missing_slots": global_layout["missing_slots"],
        "weak_slots": global_layout["weak_slots"],
        "rejected_candidates": global_layout["rejected_candidates"],
        "assignment_scores": global_layout["assignment_scores"],
        "global_layout_quality": global_layout["global_layout_quality"],
    }


def _polar_angle_error(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _stable_square_angle(local_angle: float, confidence: float) -> tuple[float, bool]:
    normalized = (float(local_angle) + 90.0) % 180.0 - 90.0
    stable_angles = (-45.0, 0.0, 45.0, 90.0)
    nearest = min(
        stable_angles,
        key=lambda expected: abs(normalized - expected),
    )
    if confidence >= 0.72 and abs(normalized - nearest) <= 10.0:
        return normalized, False
    return nearest, abs(normalized - nearest) > 0.01


def _module4_center_alignment_score(
    working: np.ndarray,
    residual: np.ndarray,
    gradient: np.ndarray,
    center_x: float,
    center_y: float,
    expected_x: float,
    expected_y: float,
    side: float,
    angle_degrees: float,
    search_radius: float,
    noise: float,
) -> dict:
    padding = side * 0.78
    x0 = max(0, int(math.floor(center_x - padding)))
    x1 = min(working.shape[1], int(math.ceil(center_x + padding + 1)))
    y0 = max(0, int(math.floor(center_y - padding)))
    y1 = min(working.shape[0], int(math.ceil(center_y + padding + 1)))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return {"score": 0.0}
    yy, xx = np.indices((y1 - y0, x1 - x0))
    dx = xx + x0 - center_x
    dy = yy + y0 - center_y
    angle = math.radians(angle_degrees)
    rotated_x = dx * math.cos(angle) + dy * math.sin(angle)
    rotated_y = -dx * math.sin(angle) + dy * math.cos(angle)
    square_distance = np.maximum(np.abs(rotated_x), np.abs(rotated_y))
    half_side = side / 2.0
    outer_mask = square_distance <= half_side
    inner_mask = square_distance <= side * 0.30
    edge_width = max(1.5, side * 0.055)
    edge_mask = np.abs(square_distance - half_side) <= edge_width
    surround_mask = (square_distance > half_side) & (square_distance <= side * 0.68)
    local_residual = residual[y0:y1, x0:x1]
    local_gradient = gradient[y0:y1, x0:x1]
    local_working = working[y0:y1, x0:x1]
    if not np.any(inner_mask) or not np.any(edge_mask):
        return {"score": 0.0}
    bright_threshold = max(
        noise * 1.6,
        float(np.percentile(local_residual[outer_mask], 62.0)),
    )
    target_fill_score = _clamp01(float(np.mean(
        local_residual[outer_mask] >= bright_threshold
    )) / 0.42)
    edge_reference = max(
        noise * 6.0,
        float(np.percentile(local_gradient, 82.0)),
        1.0,
    )
    edge_alignment_score = _clamp01(
        float(np.mean(local_gradient[edge_mask])) / edge_reference
    )
    inner_values = local_working[inner_mask]
    inner_gx = np.abs(np.diff(inner_values)) if inner_values.size > 1 else np.array([0.0])
    stripe_alignment_score = _clamp01(
        float(np.std(inner_values)) / max(noise * 9.0, 1.0)
        + float(np.mean(inner_gx)) / max(noise * 18.0, 1.0)
    )
    background_leakage = (
        float(np.mean(local_residual[surround_mask] >= bright_threshold))
        if np.any(surround_mask)
        else 0.0
    )
    leakage_score = _clamp01(1.0 - background_leakage / 0.35)
    expected_distance = math.hypot(center_x - expected_x, center_y - expected_y)
    proximity_score = _clamp01(1.0 - expected_distance / max(search_radius, 1.0))
    score = _clamp01(
        0.30 * target_fill_score
        + 0.28 * edge_alignment_score
        + 0.20 * stripe_alignment_score
        + 0.12 * leakage_score
        + 0.10 * proximity_score
    )
    return {
        "score": score,
        "local_alignment_score": score,
        "edge_alignment_score": edge_alignment_score,
        "target_fill_score": target_fill_score,
        "stripe_alignment_score": stripe_alignment_score,
        "background_leakage_score": leakage_score,
    }


def refine_module4_target_center(
    working: np.ndarray,
    residual: np.ndarray,
    expected_center_x: float,
    expected_center_y: float,
    pre_refine_center_x: float,
    pre_refine_center_y: float,
    side: float,
    angle_degrees: float,
    phantom_radius: float,
    noise: float,
) -> dict:
    """Align a globally sized/oriented ROI to local raw-image evidence."""
    search_radius = min(
        side * 0.60,
        max(side * 0.35, phantom_radius * 0.055),
    )
    gradient = np.hypot(
        ndimage.sobel(working, axis=0),
        ndimage.sobel(working, axis=1),
    )
    candidates = []
    coarse_step = 2
    integer_radius = int(math.ceil(search_radius))
    for offset_y in range(-integer_radius, integer_radius + 1, coarse_step):
        for offset_x in range(-integer_radius, integer_radius + 1, coarse_step):
            if math.hypot(offset_x, offset_y) > search_radius:
                continue
            center_x = expected_center_x + offset_x
            center_y = expected_center_y + offset_y
            metrics = _module4_center_alignment_score(
                working, residual, gradient,
                center_x, center_y,
                expected_center_x, expected_center_y,
                side, angle_degrees, search_radius, noise,
            )
            candidates.append((metrics["score"], center_x, center_y, metrics))
    if not candidates:
        return {
            "final_center_x": pre_refine_center_x,
            "final_center_y": pre_refine_center_y,
            "center_refinement_method": "pre_refine_center_fallback",
            "center_refinement_shift_px": 0.0,
            "center_refinement_score": 0.0,
            "local_alignment_score": 0.0,
            "edge_alignment_score": 0.0,
            "target_fill_score": 0.0,
            "search_radius_px": round(search_radius, 3),
            "needs_review": True,
        }
    _, coarse_x, coarse_y, _ = max(candidates, key=lambda item: item[0])
    fine_candidates = []
    for offset_y in range(-2, 3):
        for offset_x in range(-2, 3):
            center_x = coarse_x + offset_x
            center_y = coarse_y + offset_y
            if math.hypot(
                center_x - expected_center_x,
                center_y - expected_center_y,
            ) > search_radius:
                continue
            metrics = _module4_center_alignment_score(
                working, residual, gradient,
                center_x, center_y,
                expected_center_x, expected_center_y,
                side, angle_degrees, search_radius, noise,
            )
            fine_candidates.append((metrics["score"], center_x, center_y, metrics))
    score, final_x, final_y, metrics = max(
        fine_candidates or candidates,
        key=lambda item: item[0],
    )
    return {
        "final_center_x": float(final_x),
        "final_center_y": float(final_y),
        "center_refinement_method": "global_template_raw_hu_coarse_to_fine_alignment",
        "center_refinement_shift_px": round(
            math.hypot(final_x - pre_refine_center_x, final_y - pre_refine_center_y),
            3,
        ),
        "center_refinement_score": round(score, 4),
        "local_alignment_score": round(metrics["local_alignment_score"], 4),
        "edge_alignment_score": round(metrics["edge_alignment_score"], 4),
        "target_fill_score": round(metrics["target_fill_score"], 4),
        "stripe_alignment_score": round(metrics["stripe_alignment_score"], 4),
        "background_leakage_score": round(metrics["background_leakage_score"], 4),
        "search_radius_px": round(search_radius, 3),
        "needs_review": score < 0.42,
    }


def _jump_cluster_envelope(
    gradient_profile: np.ndarray,
    expected_side: float,
) -> dict | None:
    profile = ndimage.gaussian_filter1d(
        np.asarray(gradient_profile, dtype=float),
        sigma=0.8,
    )
    if profile.size < 7 or float(np.max(profile)) <= 0:
        return None
    threshold = float(np.percentile(profile, 68.0))
    peak_indices = np.where(
        (profile[1:-1] > profile[:-2])
        & (profile[1:-1] >= profile[2:])
        & (profile[1:-1] >= threshold)
    )[0] + 1
    if peak_indices.size < 2:
        return None
    crop_center = (len(profile) - 1) / 2.0
    scale = max(float(np.percentile(profile, 90.0)), 1e-6)
    maximum_gap = max(3, int(round(expected_side * 0.26)))
    clusters = []
    current_cluster = [int(peak_indices[0])]
    for peak in peak_indices[1:]:
        peak = int(peak)
        if peak - current_cluster[-1] <= maximum_gap:
            current_cluster.append(peak)
        else:
            clusters.append(current_cluster)
            current_cluster = [peak]
    clusters.append(current_cluster)
    candidates = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        start = cluster[0]
        end = cluster[-1]
        width = float(end - start)
        if not expected_side * 0.35 <= width <= expected_side * 1.32:
            continue
        midpoint = (float(start) + float(end)) / 2.0
        center_error = abs(midpoint - crop_center)
        if center_error > expected_side * 0.60:
            continue
        strengths = [float(profile[index]) for index in cluster]
        strength = _clamp01(float(np.mean(strengths)) / scale)
        repeated_peak_score = _clamp01((len(cluster) - 1) / 5.0)
        width_score = _clamp01(
            1.0 - abs(width - expected_side) / max(expected_side * 0.52, 1.0)
        )
        center_score = _clamp01(
            1.0 - center_error / max(expected_side * 0.60, 1.0)
        )
        score = _clamp01(
            0.34 * strength
            + 0.30 * repeated_peak_score
            + 0.22 * width_score
            + 0.14 * center_score
        )
        candidates.append({
            "first": start,
            "second": end,
            "midpoint": midpoint,
            "separation": width,
            "strength": strength,
            "score": score,
            "peaks": cluster,
            "peak_count": len(cluster),
            "cluster_start": start,
            "cluster_end": end,
            "cluster_width": width,
        })
    return max(candidates, key=lambda item: item["score"]) if candidates else None


def fit_module4_jump_line_roi(
    working: np.ndarray,
    residual: np.ndarray,
    global_expected_center: tuple[float, float],
    initial_center: tuple[float, float],
    global_side: float,
    global_angle: float,
    phantom_radius: float,
    noise: float,
) -> dict:
    """Fit final target geometry from two parallel raw-HU jump lines."""
    crop_side = max(18, int(round(global_side * 1.90)))
    half_crop = crop_side // 2
    initial_x, initial_y = initial_center
    x0 = max(0, int(round(initial_x)) - half_crop)
    y0 = max(0, int(round(initial_y)) - half_crop)
    x1 = min(working.shape[1], x0 + crop_side)
    y1 = min(working.shape[0], y0 + crop_side)
    local = working[y0:y1, x0:x1]
    local_residual = residual[y0:y1, x0:x1]
    if min(local.shape) < 12:
        return {
            "success": False,
            "partial": False,
            "jump_cluster_found": False,
            "reason": "Jump-cluster boundary fit incomplete; global fallback used.",
        }
    hypotheses = []
    for angle_offset in np.arange(-8.0, 8.01, 2.0):
        box_angle = float(global_angle + angle_offset)
        rotated = ndimage.rotate(
            local, -box_angle, reshape=False, order=1, mode="nearest",
        )
        rotated_residual = ndimage.rotate(
            local_residual, -box_angle, reshape=False, order=1, mode="nearest",
        )
        gradient_x = np.mean(np.abs(np.gradient(rotated, axis=1)), axis=0)
        gradient_y = np.mean(np.abs(np.gradient(rotated, axis=0)), axis=1)
        for pair_axis, profile in (("x", gradient_x), ("y", gradient_y)):
            jump_cluster = _jump_cluster_envelope(profile, global_side)
            if jump_cluster is None:
                continue
            support_threshold = max(
                noise * 1.7,
                float(np.percentile(rotated_residual, 70.0)),
            )
            support = rotated_residual >= support_threshold
            perpendicular_profile = (
                np.mean(support, axis=1)
                if pair_axis == "x"
                else np.mean(support, axis=0)
            )
            support_indices = np.where(
                perpendicular_profile >= max(
                    0.08,
                    float(np.max(perpendicular_profile)) * 0.25,
                )
            )[0]
            if support_indices.size < 2:
                continue
            perpendicular_low = int(support_indices[0])
            perpendicular_high = int(support_indices[-1])
            perpendicular_extent = float(
                perpendicular_high - perpendicular_low + 1
            )
            perpendicular_midpoint = (
                perpendicular_low + perpendicular_high
            ) / 2.0
            extent_score = _clamp01(
                1.0
                - abs(perpendicular_extent - global_side)
                / max(global_side * 0.45, 1.0)
            )
            angle_score = _clamp01(1.0 - abs(angle_offset) / 12.0)
            line_fit_score = _clamp01(
                0.48 * jump_cluster["score"]
                + 0.34 * extent_score
                + 0.18 * angle_score
            )
            hypotheses.append({
                "pair_axis": pair_axis,
                "box_angle": box_angle,
                "pair": jump_cluster,
                "perpendicular_midpoint": perpendicular_midpoint,
                "perpendicular_start": perpendicular_low,
                "perpendicular_end": perpendicular_high,
                "perpendicular_extent": perpendicular_extent,
                "boundary_extent_score": extent_score,
                "line_fit_score": line_fit_score,
            })
    if not hypotheses:
        return {
            "success": False,
            "partial": False,
            "jump_cluster_found": False,
            "reason": "Jump-cluster boundary fit incomplete; global fallback used.",
        }
    best = max(hypotheses, key=lambda item: item["line_fit_score"])
    rotated_center_x = (local.shape[1] - 1) / 2.0
    rotated_center_y = (local.shape[0] - 1) / 2.0
    if best["pair_axis"] == "x":
        offset_rotated_x = best["pair"]["midpoint"] - rotated_center_x
        offset_rotated_y = best["perpendicular_midpoint"] - rotated_center_y
        jump_line_angle = best["box_angle"] + 90.0
    else:
        offset_rotated_x = best["perpendicular_midpoint"] - rotated_center_x
        offset_rotated_y = best["pair"]["midpoint"] - rotated_center_y
        jump_line_angle = best["box_angle"]
    angle = math.radians(best["box_angle"])
    center_x = (
        initial_x
        + offset_rotated_x * math.cos(angle)
        - offset_rotated_y * math.sin(angle)
    )
    center_y = (
        initial_y
        + offset_rotated_x * math.sin(angle)
        + offset_rotated_y * math.cos(angle)
    )
    global_x, global_y = global_expected_center
    shift_from_global = math.hypot(center_x - global_x, center_y - global_y)
    shift_limit = min(global_side * 0.65, phantom_radius * 0.075)
    shift_clamped = False
    if shift_from_global > shift_limit and shift_from_global > 0:
        scale = shift_limit / shift_from_global
        center_x = global_x + (center_x - global_x) * scale
        center_y = global_y + (center_y - global_y) * scale
        shift_clamped = True
    jump_side = max(
        best["pair"]["separation"],
        best["perpendicular_extent"],
    )
    strong_fit = (
        best["line_fit_score"] >= 0.58
        and best["pair"]["peak_count"] >= 3
    )
    partial_fit = best["line_fit_score"] >= 0.42
    allowed_variation = 0.20 if strong_fit else 0.12
    final_side = max(
        global_side * (1.0 - allowed_variation),
        min(jump_side, global_side * (1.0 + allowed_variation)),
    )
    side_clamped = abs(final_side - jump_side) > 0.5
    angle_difference = abs(
        (best["box_angle"] - global_angle + 45.0) % 90.0 - 45.0
    )
    if strong_fit and angle_difference <= 8.0:
        final_angle = best["box_angle"]
        angle_source = "jump_lines_stabilized_by_global_angle"
    else:
        final_angle = global_angle
        angle_source = "global_angle_fallback"
    return {
        "success": strong_fit,
        "partial": partial_fit,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "global_expected_center": {
            "x": round(global_x, 2), "y": round(global_y, 2),
        },
        "jump_line_center": {
            "x": round(center_x, 2), "y": round(center_y, 2),
        },
        "jump_cluster_found": True,
        "jump_peaks_found": best["pair"]["peak_count"],
        "jump_peak_positions": best["pair"]["peaks"],
        "jump_cluster_start_px": best["pair"]["cluster_start"],
        "jump_cluster_end_px": best["pair"]["cluster_end"],
        "jump_cluster_width_px": round(best["pair"]["cluster_width"], 3),
        "selected_outer_jump_lines": [
            best["pair"]["cluster_start"],
            best["pair"]["cluster_end"],
        ],
        "jump_lines_found": 2,
        "jump_line_angles": [
            round(jump_line_angle, 3),
            round(jump_line_angle, 3),
        ],
        "jump_line_angle_degrees": round(jump_line_angle, 3),
        "jump_line_separation_px": round(best["pair"]["separation"], 3),
        "perpendicular_extent_px": round(best["perpendicular_extent"], 3),
        "perpendicular_support_start_px": best["perpendicular_start"],
        "perpendicular_support_end_px": best["perpendicular_end"],
        "line_fit_score": round(best["line_fit_score"], 4),
        "envelope_fit_score": round(best["line_fit_score"], 4),
        "edge_jump_score": round(best["pair"]["strength"], 4),
        "global_angle_degrees": round(global_angle, 3),
        "final_angle_degrees": round(final_angle, 3),
        "angle_source": angle_source,
        "angle_confidence": round(best["line_fit_score"], 4),
        "global_side_px": round(global_side, 3),
        "jump_fit_side_px": round(jump_side, 3),
        "final_side_px": round(final_side, 3),
        "side_source": (
            "jump_line_and_perpendicular_extent"
            if strong_fit and not side_clamped
            else "jump_line_extent_clamped_by_global_side"
        ),
        "side_clamp_applied": side_clamped,
        "center_shift_from_global": round(
            math.hypot(center_x - global_x, center_y - global_y),
            3,
        ),
        "center_shift_from_jump_fit": round(
            math.hypot(center_x - initial_x, center_y - initial_y),
            3,
        ),
        "center_refinement_reason": (
            "Center derived from the outer jump-cluster envelope and perpendicular support extent."
            if partial_fit
            else "Jump-line evidence was weak; global geometry should be retained."
        ),
        "center_shift_clamped": shift_clamped,
        "needs_review": bool(not strong_fit or shift_clamped or side_clamped),
        "reason": (
            "ROI fitted from the outer repeated-jump cluster boundaries and perpendicular completion."
            if strong_fit
            else (
                "Partial jump-cluster envelope used with global constraints; needs review."
                if partial_fit
                else "Jump-cluster boundary fit incomplete; global fallback used."
            )
        ),
    }


def _module4_metrics_from_rotated_inner_roi(
    working: np.ndarray,
    center_x: float,
    center_y: float,
    inner_side: float,
    angle_degrees: float,
    noise: float,
) -> dict:
    sample_size = max(8, int(round(inner_side)))
    coordinates = np.linspace(
        -(inner_side - 1.0) / 2.0,
        (inner_side - 1.0) / 2.0,
        sample_size,
    )
    local_y, local_x = np.meshgrid(coordinates, coordinates, indexing="ij")
    angle = math.radians(angle_degrees)
    image_x = center_x + local_x * math.cos(angle) - local_y * math.sin(angle)
    image_y = center_y + local_x * math.sin(angle) + local_y * math.cos(angle)
    inner = ndimage.map_coordinates(
        working,
        [image_y, image_x],
        order=1,
        mode="nearest",
    )
    outer_side = inner_side * 1.35
    outer_size = max(sample_size + 4, int(round(outer_side)))
    outer_coordinates = np.linspace(
        -(outer_side - 1.0) / 2.0,
        (outer_side - 1.0) / 2.0,
        outer_size,
    )
    outer_y, outer_x = np.meshgrid(
        outer_coordinates, outer_coordinates, indexing="ij",
    )
    outer_image_x = (
        center_x + outer_x * math.cos(angle) - outer_y * math.sin(angle)
    )
    outer_image_y = (
        center_y + outer_x * math.sin(angle) + outer_y * math.cos(angle)
    )
    outer_sample = ndimage.map_coordinates(
        working,
        [outer_image_y, outer_image_x],
        order=1,
        mode="nearest",
    )
    background_ring = np.maximum(np.abs(outer_x), np.abs(outer_y)) > inner_side / 2.0
    mean_hu = float(np.mean(inner))
    background_mean_hu = float(np.mean(outer_sample[background_ring]))
    signal_background_difference = abs(mean_hu - background_mean_hu)
    std_hu = float(np.std(inner))
    peak_to_valley = float(np.percentile(inner, 90.0) - np.percentile(inner, 10.0))
    gradient_x = np.abs(np.diff(inner, axis=1))
    gradient_y = np.abs(np.diff(inner, axis=0))
    edge_score = _clamp01(
        (float(np.mean(gradient_x)) + float(np.mean(gradient_y)))
        / max(noise * 12.0, 1.0)
    )
    profile_x = np.mean(inner, axis=0)
    profile_y = np.mean(inner, axis=1)
    evidence_x = _profile_periodicity(profile_x)
    evidence_y = _profile_periodicity(profile_y)
    directional_evidence = (
        ("horizontal", evidence_x),
        ("vertical", evidence_y),
    )
    best_profile_direction, evidence = max(
        directional_evidence,
        key=lambda item: (
            item[1]["periodicity_score"]
            + item[1]["peak_score"]
            + item[1]["period_consistency"]
        ),
    )
    profile_quality = _clamp01(
        0.58 * evidence["periodicity_score"]
        + 0.22 * evidence["peak_score"]
        + 0.20 * edge_score
    )
    preliminary_visibility = (
        "visible" if profile_quality >= 0.68
        else "partial" if profile_quality >= 0.46
        else "weak" if profile_quality >= 0.28
        else "needs_review"
    )
    return {
        "std_hu": round(std_hu, 3),
        "mean_hu": round(mean_hu, 3),
        "background_mean_hu": round(background_mean_hu, 3),
        "peak_to_valley": round(peak_to_valley, 3),
        "contrast_to_noise": round(peak_to_valley / max(noise, 1.0), 4),
        "signal_background_separation": round(
            signal_background_difference / max(noise, 1.0), 4,
        ),
        "local_noise_hu": round(noise, 4),
        "periodicity_score": round(evidence["periodicity_score"], 4),
        "edge_score": round(edge_score, 4),
        "stripe_peaks_found": evidence["peaks_found"],
        "average_period_px": evidence["average_period_px"],
        "period_consistency": round(evidence["period_consistency"], 4),
        "best_profile_direction": best_profile_direction,
        "peaks_found": evidence["peaks_found"],
        "valleys_found": evidence["valleys_found"],
        "peak_positions": evidence["peak_positions"],
        "valley_positions": evidence["valley_positions"],
        "profile_samples": [
            round(float(value), 3) for value in evidence["profile"][:64]
        ],
        "profile_quality": round(profile_quality, 4),
        "profile_failure_reason": (
            None
            if profile_quality >= 0.46
            else "Periodic stripe evidence is weak in both sampled directions."
        ),
        "preliminary_visibility": preliminary_visibility,
    }


def _module4_draft_roi_scoring(candidate: dict) -> dict:
    """Create explicitly preliminary ROI scoring from raw-HU review metrics."""
    peak_to_valley_quality = _clamp01(
        float(candidate.get("contrast_to_noise", 0.0)) / 12.0
    )
    periodicity_quality = _clamp01(candidate.get("periodicity_score", 0.0))
    edge_quality = _clamp01(candidate.get("edge_score", 0.0))
    peaks_quality = _clamp01(candidate.get("stripe_peaks_found", 0) / 4.0)
    cnr_quality = _clamp01(
        float(candidate.get("signal_background_separation", 0.0)) / 8.0
    )
    geometry_quality = _clamp01(candidate.get("fit_quality", 0.0))
    breakdown = {
        "peak_to_valley": round(25.0 * peak_to_valley_quality, 2),
        "periodicity": round(20.0 * periodicity_quality, 2),
        "edge_strength": round(20.0 * edge_quality, 2),
        "stripe_peaks": round(15.0 * peaks_quality, 2),
        "contrast_to_noise": round(10.0 * cnr_quality, 2),
        "geometry_confidence": round(10.0 * geometry_quality, 2),
    }
    roi_score = round(sum(breakdown.values()), 1)
    passes_draft_rules = (
        roi_score >= 70.0
        and candidate.get("profile_quality", 0.0) >= 0.60
        and candidate.get("periodicity_score", 0.0) >= 0.45
        and candidate.get("contrast_to_noise", 0.0) >= 3.0
        and candidate.get("stripe_peaks_found", 0) >= 3
    )
    fails_draft_rules = (
        roi_score < 45.0
        and candidate.get("profile_quality", 0.0) < 0.35
        and candidate.get("periodicity_score", 0.0) < 0.25
    )
    draft_status = (
        "draft_pass"
        if passes_draft_rules
        else "draft_fail"
        if fails_draft_rules
        else "needs_review"
    )
    reason = (
        "Preliminary ROI score suggests a visible repeated pattern. Formal "
        "lp/cm measurement remains pending."
        if draft_status == "draft_pass"
        else "Draft ROI evidence is weak across score and profile metrics; "
        "formal measurement remains pending."
        if draft_status == "draft_fail"
        else "Preliminary ROI evidence or approximate geometry needs review; "
        "formal lp/cm measurement remains pending."
    )
    return {
        "roi_available": True,
        "roi_score": roi_score,
        "draft_status": draft_status,
        "preliminary_pass_fail": draft_status,
        "score_breakdown": breakdown,
        "roi_geometry_confidence": round(geometry_quality, 4),
        "draft_scoring_thresholds": {
            "draft_pass_min_score": 70,
            "draft_fail_max_score": 45,
            "draft_pass_min_profile_quality": 0.60,
            "draft_pass_min_periodicity": 0.45,
            "draft_pass_min_contrast_to_noise": 3.0,
            "draft_pass_min_peaks": 3,
        },
        "draft_score_reason": reason,
        "formal_measurement_status": "pending",
    }


def _debug_mask_image(mask: np.ndarray) -> str:
    pixels = np.where(mask, 255, 0).astype(np.uint8)
    return image_to_base64(Image.fromarray(pixels).convert("RGB"))


def localize_module4_target(
    slice_pixels: np.ndarray,
    phantom_center: tuple[float, float],
    phantom_radius: float,
    ideal_center: tuple[float, float],
    expected_side: float,
    target_id: str,
    target_slot: str,
    sector_angle_degrees: float,
    sector_half_width_degrees: float = 38.0,
    radial_minimum: float = 0.24,
    radial_maximum: float = 0.76,
    minimum_aspect_ratio: float = 0.34,
) -> dict:
    """Locate one internal Module 4 target inside its expected angular sector."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    finite = np.isfinite(raw)
    fill = float(np.median(raw[finite]))
    working = np.where(finite, raw, fill)
    cx, cy = phantom_center
    yy, xx = np.indices(raw.shape)
    radial_ratio = np.hypot(xx - cx, yy - cy) / max(phantom_radius, 1.0)
    angle = np.degrees(np.arctan2(yy - cy, xx - cx))
    angular_error = np.abs(
        (angle - sector_angle_degrees + 180.0) % 360.0 - 180.0
    )
    sector_internal = (
        finite
        & (radial_ratio >= radial_minimum)
        & (radial_ratio <= radial_maximum)
        & (angular_error <= sector_half_width_degrees)
    )
    background_size = max(7, int(round(phantom_radius * 0.045)) | 1)
    background = ndimage.median_filter(working, size=background_size)
    residual = working - background
    sector_values = working[sector_internal]
    sector_residual = residual[sector_internal]
    residual_median = float(np.median(sector_residual))
    noise = max(
        1.0,
        float(np.median(np.abs(sector_residual - residual_median)) * 1.4826),
    )
    hu_threshold = float(np.percentile(sector_values, 82.0))
    residual_threshold = max(
        residual_median + 2.6 * noise,
        float(np.percentile(sector_residual, 84.0)),
    )
    mask = (
        sector_internal
        & (working >= hu_threshold)
        & (residual >= residual_threshold)
    )
    merge_size = max(5, min(13, int(round(expected_side * 0.18)) | 1))
    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((merge_size, merge_size), dtype=bool),
        iterations=2,
    )
    mask = ndimage.binary_dilation(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    candidates = []
    minimum_area = max(10, int(round(expected_side ** 2 * 0.10)))
    maximum_area = max(minimum_area + 1, int(round(expected_side ** 2 * 2.80)))
    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        region = labels[bounds] == label_id
        area = int(np.sum(region))
        local_y, local_x = np.nonzero(region)
        center_x = float(bounds[1].start + np.mean(local_x))
        center_y = float(bounds[0].start + np.mean(local_y))
        candidate_radial = math.hypot(center_x - cx, center_y - cy) / max(phantom_radius, 1.0)
        candidate_angle = math.degrees(math.atan2(center_y - cy, center_x - cx))
        sector_error = _polar_angle_error(
            candidate_angle,
            sector_angle_degrees,
        )
        height, width = region.shape
        aspect = min(width, height) / max(width, height)
        perimeter = int(np.sum(region & ~ndimage.binary_erosion(region)))
        circularity = 4.0 * math.pi * area / max(float(perimeter * perimeter), 1.0)
        contrast = float(np.median(residual[bounds][region]))
        b6_bright_square_override = bool(
            target_id == "B6"
            and sector_error <= 8.0
            and minimum_area <= area <= maximum_area
            and aspect >= 0.75
            and contrast >= max(8.0, noise * 1.5)
        )
        rejection_reason = None
        if area < minimum_area:
            rejection_reason = "too_small_bb_like"
        elif area > maximum_area:
            rejection_reason = "oversized_region"
        elif candidate_radial > radial_maximum:
            rejection_reason = "outside_perimeter_region"
        elif sector_error > sector_half_width_degrees:
            rejection_reason = "outside_target_sector"
        elif (
            circularity >= 0.88
            and candidate_radial >= 0.50
            and not b6_bright_square_override
        ):
            rejection_reason = "circular_hole_like"
        elif aspect < minimum_aspect_ratio:
            rejection_reason = "weak_square_geometry"
        angle_quality = _clamp01(
            1.0 - sector_error / sector_half_width_degrees
        )
        radial_quality = _clamp01(1.0 - abs(candidate_radial - 0.55) / 0.28)
        area_quality = _clamp01(
            1.0 - abs(area - expected_side ** 2) / max(expected_side ** 2 * 1.3, 1.0)
        )
        contrast_quality = _clamp01(contrast / max(noise * 9.0, 1.0))
        score = _clamp01(
            0.34 * angle_quality
            + 0.25 * radial_quality
            + 0.18 * area_quality
            + 0.13 * aspect
            + 0.10 * contrast_quality
        )
        candidates.append({
            "label_id": label_id,
            "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
            "area": area,
            "radial_ratio": round(candidate_radial, 4),
            "angle_degrees": round(candidate_angle, 3),
            "sector_angular_error_degrees": round(sector_error, 3),
            "aspect_ratio": round(aspect, 4),
            "circularity": round(circularity, 4),
            "contrast_hu": round(contrast, 3),
            "localization_score": round(score, 4),
            "accepted": rejection_reason is None,
            "rejection_reason": rejection_reason,
            "b6_bright_square_override": b6_bright_square_override,
        })
    accepted = [item for item in candidates if item["accepted"]]
    selected = (
        max(accepted, key=lambda item: item["localization_score"])
        if accepted
        else None
    )
    return {
        "success": selected is not None,
        "target_id": target_id,
        "target_slot": target_slot,
        "sector_angle_degrees": sector_angle_degrees,
        "sector_half_width_degrees": sector_half_width_degrees,
        "radial_search_range": [radial_minimum, radial_maximum],
        "minimum_aspect_ratio": minimum_aspect_ratio,
        "original_ideal_rough_center": {
            "x": round(ideal_center[0], 2),
            "y": round(ideal_center[1], 2),
        },
        "detected_component_center": selected["center"] if selected else None,
        "final_crop_center": selected["center"] if selected else {
            "x": round(ideal_center[0], 2),
            "y": round(ideal_center[1], 2),
        },
        "sector_candidates_considered": candidates,
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted),
        "selected_localization_reason": (
            "Selected highest-scoring internal high-contrast component in "
            f"the {target_slot} sector."
            if selected
            else "No top-sector component passed localization filters; expanded ideal-sector crop used."
        ),
        "selected_candidate": selected,
        "localization_mask_image": _debug_mask_image(mask),
        "hu_threshold": round(hu_threshold, 3),
        "residual_threshold": round(residual_threshold, 3),
        "noise_sigma": round(noise, 3),
    }


def fit_single_block_square_template(
    crop: np.ndarray,
    residual: np.ndarray,
    selected_component_mask: np.ndarray,
    start_center: tuple[float, float],
    expected_side: float,
    expected_angle: float,
    noise: float,
    crop_origin: tuple[int, int],
    phantom_center: tuple[float, float],
    phantom_radius: float,
) -> dict:
    """Optimize a perfect square from image evidence near the B1 component."""
    height, width = crop.shape
    yy, xx = np.indices(crop.shape, dtype=np.float32)
    gradient = np.hypot(
        ndimage.sobel(crop, axis=0),
        ndimage.sobel(crop, axis=1),
    )
    gradient_scale = max(float(np.percentile(gradient, 92.0)), noise, 1.0)
    positive_residual = np.clip(residual / max(noise * 6.0, 1.0), 0.0, 1.0)
    component_area = max(int(np.sum(selected_component_mask)), 1)
    has_component = bool(np.any(selected_component_mask))
    origin_x, origin_y = crop_origin
    phantom_x, phantom_y = phantom_center

    center_range = max(15.0, min(25.0, expected_side * 0.60))
    side_min = expected_side * 0.75
    side_max = expected_side * 1.25
    angle_min = expected_angle - 10.0
    angle_max = expected_angle + 10.0

    def score_template(center_x: float, center_y: float, side: float, angle_degrees: float) -> dict | None:
        half = side / 2.0
        angle = math.radians(angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        dx = xx - center_x
        dy = yy - center_y
        local_x = dx * cosine + dy * sine
        local_y = -dx * sine + dy * cosine
        square_distance = np.maximum(np.abs(local_x), np.abs(local_y))
        edge_band_width = max(1.25, side * 0.045)
        outside_width = max(2.0, side * 0.12)
        inner_mask = square_distance <= max(half - edge_band_width, 1.0)
        boundary_mask = np.abs(square_distance - half) <= edge_band_width
        side_extent = half + edge_band_width
        side_masks = (
            (np.abs(local_x + half) <= edge_band_width) & (np.abs(local_y) <= side_extent),
            (np.abs(local_x - half) <= edge_band_width) & (np.abs(local_y) <= side_extent),
            (np.abs(local_y + half) <= edge_band_width) & (np.abs(local_x) <= side_extent),
            (np.abs(local_y - half) <= edge_band_width) & (np.abs(local_x) <= side_extent),
        )
        outside_mask = (
            (square_distance > half + edge_band_width)
            & (square_distance <= half + edge_band_width + outside_width)
        )
        if not np.any(inner_mask) or not np.any(boundary_mask) or not np.any(outside_mask):
            return None
        corners = _square_points(center_x, center_y, side, angle_degrees)
        if any(
            point["x"] < 1
            or point["y"] < 1
            or point["x"] >= width - 1
            or point["y"] >= height - 1
            for point in corners
        ):
            return None
        if any(
            math.hypot(
                point["x"] + origin_x - phantom_x,
                point["y"] + origin_y - phantom_y,
            ) > phantom_radius * 0.82
            for point in corners
        ):
            return None
        boundary_edge_score = _clamp01(
            float(np.mean(gradient[boundary_mask])) / gradient_scale
        )
        side_edge_scores = [
            _clamp01(float(np.mean(gradient[side_mask])) / gradient_scale)
            for side_mask in side_masks
            if np.any(side_mask)
        ]
        edge_symmetry_score = (
            _clamp01(
                min(side_edge_scores) / max(max(side_edge_scores), 0.01)
            )
            if len(side_edge_scores) == 4
            else 0.0
        )
        interior_fill_score = _clamp01(
            (
                0.65 * float(np.mean(positive_residual[inner_mask]))
                + 0.35 * float(np.percentile(positive_residual[inner_mask], 70.0))
            ) * 1.55
        )
        outside_leakage_penalty = _clamp01(
            (
                0.55 * float(np.mean(positive_residual[outside_mask]))
                + 0.45 * float(np.percentile(positive_residual[outside_mask], 75.0))
            ) * 1.75
        )
        mask_coverage_score = _clamp01(
            float(np.sum(selected_component_mask & (square_distance <= half)))
            / component_area
        )
        center_distance = math.hypot(
            center_x - start_center[0],
            center_y - start_center[1],
        )
        center_sigma = max(2.0, expected_side * (0.16 if has_component else 0.38))
        center_proximity_score = _clamp01(
            math.exp(-0.5 * (center_distance / center_sigma) ** 2)
        )
        if has_component:
            total_score = _clamp01(
                0.30 * boundary_edge_score
                + 0.15 * edge_symmetry_score
                + 0.20 * mask_coverage_score
                + 0.12 * interior_fill_score
                + 0.13 * center_proximity_score
                + 0.10 * (1.0 - outside_leakage_penalty)
            )
        else:
            total_score = _clamp01(
                0.38 * boundary_edge_score
                + 0.18 * edge_symmetry_score
                + 0.24 * interior_fill_score
                + 0.08 * center_proximity_score
                + 0.12 * (1.0 - outside_leakage_penalty)
            )
        return {
            "center_x": center_x,
            "center_y": center_y,
            "side_px": side,
            "angle_degrees": angle_degrees,
            "template_score": total_score,
            "boundary_edge_score": boundary_edge_score,
            "edge_symmetry_score": edge_symmetry_score,
            "interior_fill_score": interior_fill_score,
            "outside_leakage_penalty": outside_leakage_penalty,
            "mask_coverage_score": mask_coverage_score,
            "center_proximity_score": center_proximity_score,
        }

    trials = []
    center_step = max(4.0, expected_side * 0.24)
    side_step = max(3.0, expected_side * 0.125)
    center_offsets = np.arange(-center_range, center_range + 0.01, center_step)
    sides = np.arange(side_min, side_max + 0.01, side_step)
    angles = np.arange(angle_min, angle_max + 0.01, 5.0)
    for offset_y in center_offsets:
        for offset_x in center_offsets:
            for side in sides:
                for angle_degrees in angles:
                    trial = score_template(
                        start_center[0] + float(offset_x),
                        start_center[1] + float(offset_y),
                        float(side),
                        float(angle_degrees),
                    )
                    if trial is not None:
                        trials.append(trial)
    if not trials:
        return {
            "success": False,
            "failure_reason": "No valid square template stayed inside the crop and phantom.",
            "search_center_range": center_range,
            "search_side_range": [side_min, side_max],
            "search_angle_range": [angle_min, angle_max],
            "top_trials": [],
        }
    coarse_best = max(trials, key=lambda item: item["template_score"])
    fine_center_offsets = (
        -center_step,
        -center_step / 3.0,
        0.0,
        center_step / 3.0,
        center_step,
    )
    fine_side_offsets = (
        -side_step,
        -side_step / 3.0,
        0.0,
        side_step / 3.0,
        side_step,
    )
    for offset_y in fine_center_offsets:
        for offset_x in fine_center_offsets:
            for side_offset in fine_side_offsets:
                for angle_offset in np.arange(-2.5, 2.51, 1.25):
                    trial = score_template(
                        coarse_best["center_x"] + float(offset_x),
                        coarse_best["center_y"] + float(offset_y),
                        max(side_min, min(side_max, coarse_best["side_px"] + float(side_offset))),
                        max(angle_min, min(angle_max, coarse_best["angle_degrees"] + float(angle_offset))),
                    )
                    if trial is not None:
                        trials.append(trial)
    pre_micro_best = max(trials, key=lambda item: item["template_score"])
    micro_trials = []
    for offset_y in range(-3, 4):
        for offset_x in range(-3, 4):
            for side_offset in range(-3, 4):
                for angle_offset in (-1.0, 0.0, 1.0):
                    trial = score_template(
                        pre_micro_best["center_x"] + offset_x,
                        pre_micro_best["center_y"] + offset_y,
                        max(side_min, min(side_max, pre_micro_best["side_px"] + side_offset)),
                        max(angle_min, min(angle_max, pre_micro_best["angle_degrees"] + angle_offset)),
                    )
                    if trial is not None:
                        micro_trials.append(trial)
    trials.extend(micro_trials)
    best = max(
        micro_trials or [pre_micro_best],
        key=lambda item: item["template_score"],
    )
    top_trials = sorted(
        trials,
        key=lambda item: item["template_score"],
        reverse=True,
    )[:5]

    def rounded_trial(trial: dict) -> dict:
        return {
            "center": {
                "x": round(trial["center_x"] + origin_x, 2),
                "y": round(trial["center_y"] + origin_y, 2),
            },
            "side_px": round(trial["side_px"], 3),
            "angle_degrees": round(trial["angle_degrees"], 3),
            "template_score": round(trial["template_score"], 4),
            "boundary_edge_score": round(trial["boundary_edge_score"], 4),
            "edge_symmetry_score": round(trial["edge_symmetry_score"], 4),
            "interior_fill_score": round(trial["interior_fill_score"], 4),
            "outside_leakage_penalty": round(trial["outside_leakage_penalty"], 4),
            "mask_coverage_score": round(trial["mask_coverage_score"], 4),
            "center_proximity_score": round(trial["center_proximity_score"], 4),
        }

    return {
        "success": True,
        "best": best,
        "search_center_range": center_range,
        "search_side_range": [side_min, side_max],
        "search_angle_range": [angle_min, angle_max],
        "top_trials": [rounded_trial(trial) for trial in top_trials],
        "trial_count": len(trials),
        "pre_micro_template_center": {
            "x": round(pre_micro_best["center_x"] + origin_x, 2),
            "y": round(pre_micro_best["center_y"] + origin_y, 2),
        },
        "pre_micro_template_side_px": round(pre_micro_best["side_px"], 3),
        "pre_micro_template_angle_degrees": round(
            pre_micro_best["angle_degrees"], 3,
        ),
        "post_micro_template_center": {
            "x": round(best["center_x"] + origin_x, 2),
            "y": round(best["center_y"] + origin_y, 2),
        },
        "micro_shift_px": round(math.hypot(
            best["center_x"] - pre_micro_best["center_x"],
            best["center_y"] - pre_micro_best["center_y"],
        ), 3),
        "pre_micro_score": round(pre_micro_best["template_score"], 4),
        "post_micro_score": round(best["template_score"], 4),
        "micro_refinement_applied": True,
        "micro_refinement_reason": (
            "Applied ±3 px center/side and ±1 degree image-evidence refinement."
        ),
    }


def fit_single_module4_block_roi(
    slice_pixels: np.ndarray,
    phantom_center: tuple[float, float],
    phantom_radius: float,
    rough_center: tuple[float, float],
    rough_expected_side: float,
    rough_expected_angle: float = 45.0,
    target_id: str = "B1",
    target_slot: str = "top",
    crop_side_override: int | None = None,
    localization_debug: dict | None = None,
) -> dict:
    """Fit one local Module 4 block without invoking the eight-target pipeline."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    finite = np.isfinite(raw)
    fill = float(np.median(raw[finite]))
    working = np.where(finite, raw, fill)
    crop_side = (
        int(crop_side_override)
        if crop_side_override is not None
        else max(24, int(round(rough_expected_side * 2.35)))
    )
    half_crop = crop_side // 2
    rough_x, rough_y = rough_center
    x0 = max(0, int(round(rough_x)) - half_crop)
    y0 = max(0, int(round(rough_y)) - half_crop)
    x1 = min(raw.shape[1], x0 + crop_side)
    y1 = min(raw.shape[0], y0 + crop_side)
    crop = working[y0:y1, x0:x1]
    local_rough_x = rough_x - x0
    local_rough_y = rough_y - y0
    background_size = max(5, int(round(rough_expected_side * 0.35)) | 1)
    local_background = ndimage.median_filter(crop, size=background_size)
    residual = crop - local_background
    residual_median = float(np.median(residual))
    noise = max(
        1.0,
        float(np.median(np.abs(residual - residual_median)) * 1.4826),
    )
    local_threshold = max(
        residual_median + 2.25 * noise,
        float(np.percentile(residual, 74.0)),
    )
    threshold_mask = residual >= local_threshold
    raw_labels, components_before = ndimage.label(threshold_mask)
    merge_size = max(3, min(9, int(round(rough_expected_side * 0.14)) | 1))
    merged_mask = ndimage.binary_closing(
        threshold_mask,
        structure=np.ones((merge_size, merge_size), dtype=bool),
        iterations=2,
    )
    merged_mask = ndimage.binary_fill_holes(merged_mask)
    merged_labels, components_after = ndimage.label(merged_mask)
    objects = ndimage.find_objects(merged_labels)
    minimum_area = max(12, int(round(rough_expected_side ** 2 * 0.18)))
    maximum_area = max(minimum_area + 1, int(round(rough_expected_side ** 2 * 2.20)))
    component_options = []
    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        region = merged_labels[bounds] == label_id
        area = int(np.sum(region))
        if area < minimum_area or area > maximum_area:
            continue
        local_y, local_x = np.nonzero(region)
        center_x = float(bounds[1].start + np.mean(local_x))
        center_y = float(bounds[0].start + np.mean(local_y))
        distance = math.hypot(center_x - local_rough_x, center_y - local_rough_y)
        if distance > max(rough_expected_side * 1.35, crop_side * 0.42):
            continue
        height, width = region.shape
        aspect = min(width, height) / max(width, height)
        fill_ratio = area / max(float(width * height), 1.0)
        contrast = float(np.median(residual[bounds][region]))
        score = _clamp01(
            0.38 * _clamp01(1.0 - distance / max(rough_expected_side * 0.85, 1.0))
            + 0.24 * _clamp01(1.0 - abs(area - rough_expected_side ** 2) / max(rough_expected_side ** 2, 1.0))
            + 0.20 * aspect
            + 0.10 * _clamp01(fill_ratio / 0.60)
            + 0.08 * _clamp01(contrast / max(noise * 8.0, 1.0))
        )
        component_options.append({
            "label_id": label_id,
            "bounds": bounds,
            "region": region,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "distance": distance,
            "score": score,
        })
    debug_base = {
        "single_block_debug_enabled": True,
        "debug_target_id": target_id,
        "debug_target_slot": target_slot,
        "crop_bounds": {
            "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
        },
        "rough_center": {"x": round(rough_x, 2), "y": round(rough_y, 2)},
        "localization": localization_debug or {},
        "local_threshold": round(local_threshold, 3),
        "local_threshold_noise_multiplier": 2.25,
        "local_threshold_percentile": 74.0,
        "noise_sigma": round(noise, 3),
        "merge_kernel_size": merge_size,
        "merge_iterations": 2,
        "merge_dilation_iterations": 0,
        "components_before_merge": int(components_before),
        "components_after_merge": int(components_after),
        "threshold_mask_image": _debug_mask_image(threshold_mask),
        "merged_mask_image": _debug_mask_image(merged_mask),
        "local_crop_image": image_to_base64(window_pixels_to_image(
            crop,
            max(250.0, float(np.percentile(crop, 98.0) - np.percentile(crop, 5.0))),
            float((np.percentile(crop, 98.0) + np.percentile(crop, 5.0)) / 2.0),
        )),
    }
    if not component_options:
        return {
            **debug_base,
            "success": False,
            "candidate": None,
            "selected_component_area": None,
            "selected_component_bbox": None,
            "selected_component_center": None,
            "selected_component_reason": "No merged component matched the rough B1 location and size.",
            "fitted_angle_degrees": None,
            "component_fit_side_px": None,
            "side_margin_ratio": None,
            "side_before_clamp_px": None,
            "final_side_px": None,
            "fitted_side_px": None,
            "final_center": None,
            "final_rotated_box": [],
            "inner_roi": None,
            "inner_roi_size_ratio": None,
            "fit_quality": 0.0,
            "fallback_used": False,
            "failure_reason": "Single-block component detection needs review.",
            "fitted_crop_image": None,
        }
    selected = max(component_options, key=lambda item: item["score"])
    bounds = selected["bounds"]
    selected_component_bbox = {
        "x": int(x0 + bounds[1].start),
        "y": int(y0 + bounds[0].start),
        "width": int(bounds[1].stop - bounds[1].start),
        "height": int(bounds[0].stop - bounds[0].start),
    }
    local_y, local_x = np.nonzero(selected["region"])
    absolute_x = local_x.astype(float) + bounds[1].start
    absolute_y = local_y.astype(float) + bounds[0].start
    fits = []
    for angle_degrees in np.arange(-60.0, 60.01, 0.5):
        angle = math.radians(float(angle_degrees))
        rotated_x = absolute_x * math.cos(angle) + absolute_y * math.sin(angle)
        rotated_y = -absolute_x * math.sin(angle) + absolute_y * math.cos(angle)
        x_low, x_high = np.percentile(rotated_x, (0.75, 99.25))
        y_low, y_high = np.percentile(rotated_y, (0.75, 99.25))
        span_x = float(x_high - x_low + 1.0)
        span_y = float(y_high - y_low + 1.0)
        fits.append({
            "angle": float(angle_degrees),
            "area": span_x * span_y,
            "span_x": span_x,
            "span_y": span_y,
            "x_mid": float((x_low + x_high) / 2.0),
            "y_mid": float((y_low + y_high) / 2.0),
        })
    best_fit = min(fits, key=lambda item: item["area"])
    fitted_angle = best_fit["angle"]
    stable_angle = min(
        (-45.0, 0.0, 45.0),
        key=lambda angle: abs(angle - fitted_angle),
    )
    if abs(stable_angle - fitted_angle) <= 8.0:
        fitted_angle = stable_angle
    angle = math.radians(fitted_angle)
    rotated_x = absolute_x * math.cos(angle) + absolute_y * math.sin(angle)
    rotated_y = -absolute_x * math.sin(angle) + absolute_y * math.cos(angle)
    x_low, x_high = np.percentile(rotated_x, (0.75, 99.25))
    y_low, y_high = np.percentile(rotated_y, (0.75, 99.25))
    center_rotated_x = float((x_low + x_high) / 2.0)
    center_rotated_y = float((y_low + y_high) / 2.0)
    center_local_x = center_rotated_x * math.cos(angle) - center_rotated_y * math.sin(angle)
    center_local_y = center_rotated_x * math.sin(angle) + center_rotated_y * math.cos(angle)
    component_fit_side = max(
        float(x_high - x_low + 1.0),
        float(y_high - y_low + 1.0),
    )
    side_margin_ratio = 1.025
    side_before_clamp = component_fit_side * side_margin_ratio
    fitted_side = max(
        rough_expected_side * 0.62,
        min(side_before_clamp, rough_expected_side * 1.38),
    )
    pre_template_center_x = center_local_x + x0
    pre_template_center_y = center_local_y + y0
    pre_template_fit_box = _square_points(
        pre_template_center_x,
        pre_template_center_y,
        fitted_side,
        fitted_angle,
    )
    selected_mask = np.zeros(crop.shape, dtype=bool)
    selected_mask[bounds] = selected["region"]
    template_fit = fit_single_block_square_template(
        crop,
        residual,
        selected_mask,
        start_center=(selected["center_x"], selected["center_y"]),
        expected_side=rough_expected_side,
        expected_angle=rough_expected_angle,
        noise=noise,
        crop_origin=(x0, y0),
        phantom_center=phantom_center,
        phantom_radius=phantom_radius,
    )
    if template_fit["success"]:
        best_template = template_fit["best"]
        center_x = best_template["center_x"] + x0
        center_y = best_template["center_y"] + y0
        fitted_side = best_template["side_px"]
        fitted_angle = best_template["angle_degrees"]
        geometry_source = "module4_block_square_template_optimized"
        template_score = best_template["template_score"]
    else:
        best_template = None
        center_x = pre_template_center_x
        center_y = pre_template_center_y
        geometry_source = "module4_block_component_square_template_fallback"
        template_score = 0.0
    angle = math.radians(fitted_angle)
    rotated_box = _square_points(center_x, center_y, fitted_side, fitted_angle)
    fitted_crop = window_pixels_to_image(
        crop,
        max(250.0, float(np.percentile(crop, 98.0) - np.percentile(crop, 5.0))),
        float((np.percentile(crop, 98.0) + np.percentile(crop, 5.0)) / 2.0),
    )
    fitted_crop_pixels = np.asarray(fitted_crop, dtype=np.uint8).copy()
    fitted_crop_pixels[selected_mask] = (
        fitted_crop_pixels[selected_mask].astype(np.float32) * 0.58
        + np.array([34, 211, 238], dtype=np.float32) * 0.42
    ).astype(np.uint8)
    fitted_crop = Image.fromarray(fitted_crop_pixels, mode="RGB")
    fitted_crop_draw = ImageDraw.Draw(fitted_crop)
    pre_template_crop_points = [
        (point["x"] - x0, point["y"] - y0)
        for point in pre_template_fit_box
    ]
    fitted_crop_draw.line(
        pre_template_crop_points + [pre_template_crop_points[0]],
        fill="#facc15",
        width=1,
    )
    if template_fit["success"]:
        pre_micro_box = _square_points(
            template_fit["pre_micro_template_center"]["x"],
            template_fit["pre_micro_template_center"]["y"],
            template_fit["pre_micro_template_side_px"],
            template_fit["pre_micro_template_angle_degrees"],
        )
        pre_micro_crop_points = [
            (point["x"] - x0, point["y"] - y0)
            for point in pre_micro_box
        ]
        fitted_crop_draw.line(
            pre_micro_crop_points + [pre_micro_crop_points[0]],
            fill="#fb923c",
            width=1,
        )
    crop_box_points = [
        (point["x"] - x0, point["y"] - y0)
        for point in rotated_box
    ]
    fitted_crop_draw.line(
        crop_box_points + [crop_box_points[0]],
        fill="#22d3ee",
        width=2,
    )
    inner_roi_ratio = 0.58
    inner_roi = {
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "width": round(fitted_side * inner_roi_ratio, 2),
        "height": round(fitted_side * inner_roi_ratio, 2),
        "angle_degrees": round(fitted_angle, 3),
    }
    half_inner = fitted_side * inner_roi_ratio / 2.0
    crop_inner_points = []
    for dx, dy in (
        (-half_inner, -half_inner),
        (half_inner, -half_inner),
        (half_inner, half_inner),
        (-half_inner, half_inner),
    ):
        crop_inner_points.append((
            center_x - x0 + dx * math.cos(angle) - dy * math.sin(angle),
            center_y - y0 + dx * math.sin(angle) + dy * math.cos(angle),
        ))
    fitted_crop_draw.line(
        crop_inner_points + [crop_inner_points[0]],
        fill="#a5f3fc",
        width=1,
    )
    metrics = _module4_metrics_from_rotated_inner_roi(
        working,
        center_x,
        center_y,
        fitted_side * inner_roi_ratio,
        fitted_angle,
        noise,
    )
    fit_quality = _clamp01(
        0.80 * template_score
        + 0.20 * selected["score"]
    )
    weak_template_terms = []
    if best_template is not None:
        if best_template["boundary_edge_score"] < 0.35:
            weak_template_terms.append("boundary edge evidence")
        if best_template["interior_fill_score"] < 0.30:
            weak_template_terms.append("interior target evidence")
        if best_template["mask_coverage_score"] < 0.72:
            weak_template_terms.append("component-mask coverage")
        if best_template["outside_leakage_penalty"] > 0.45:
            weak_template_terms.append("outside leakage")
    needs_review = (
        not template_fit["success"]
        or template_score < 0.43
        or bool(weak_template_terms)
    )
    draw_on_overlay = bool(
        best_template
        and template_score >= 0.42
        and best_template["boundary_edge_score"] >= 0.28
        and best_template["edge_symmetry_score"] >= 0.30
        and best_template["mask_coverage_score"] >= 0.68
        and best_template["interior_fill_score"] >= 0.20
        and best_template["outside_leakage_penalty"] <= 0.55
    )
    selected_component_center = {
        "x": round(selected["center_x"] + x0, 2),
        "y": round(selected["center_y"] + y0, 2),
    }
    expected_slot_center = (
        (localization_debug or {}).get("original_ideal_rough_center")
    )
    center_delta_from_component = round(math.hypot(
        center_x - selected_component_center["x"],
        center_y - selected_component_center["y"],
    ), 3)
    center_delta_from_slot = (
        round(math.hypot(
            center_x - expected_slot_center["x"],
            center_y - expected_slot_center["y"],
        ), 3)
        if expected_slot_center else None
    )
    if not template_fit["success"]:
        candidate_reason = (
            "Square-template search had no valid trial; component square kept "
            "for review only."
        )
    elif weak_template_terms:
        candidate_reason = (
            "Best perfect-square template selected, but review is needed for "
            + ", ".join(weak_template_terms)
            + "."
        )
    else:
        candidate_reason = (
            "Best perfect-square template selected from raw-HU residual, "
            "gradient, leakage, coverage, and center evidence."
        )
    candidate = {
        "id": target_id,
        "target_slot": target_slot,
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "bbox": {
            "x": max(0, int(round(center_x - fitted_side / 2.0))),
            "y": max(0, int(round(center_y - fitted_side / 2.0))),
            "width": int(round(fitted_side)),
            "height": int(round(fitted_side)),
        },
        "rotated_box": rotated_box,
        "inner_roi": inner_roi,
        "angle_degrees": round(fitted_angle, 3),
        "side_px": round(fitted_side, 3),
        "box_fit_quality": round(fit_quality, 4),
        "assignment_confidence": round(selected["score"], 4),
        "geometry_source": geometry_source,
        "template_score": round(template_score, 4),
        "boundary_edge_score": (
            round(best_template["boundary_edge_score"], 4)
            if best_template else None
        ),
        "edge_symmetry_score": (
            round(best_template["edge_symmetry_score"], 4)
            if best_template else None
        ),
        "interior_fill_score": (
            round(best_template["interior_fill_score"], 4)
            if best_template else None
        ),
        "outside_leakage_penalty": (
            round(best_template["outside_leakage_penalty"], 4)
            if best_template else None
        ),
        "mask_coverage_score": (
            round(best_template["mask_coverage_score"], 4)
            if best_template else None
        ),
        "center_proximity_score": (
            round(best_template["center_proximity_score"], 4)
            if best_template else None
        ),
        "template_center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "template_side_px": round(fitted_side, 3),
        "template_angle_degrees": round(fitted_angle, 3),
        "selected_component_center": selected_component_center,
        "selected_component_bbox": selected_component_bbox,
        "selected_component_bbox_size": {
            "width": selected_component_bbox["width"],
            "height": selected_component_bbox["height"],
        },
        "center_delta_from_component": center_delta_from_component,
        "expected_slot_center": expected_slot_center,
        "center_delta_from_slot": center_delta_from_slot,
        "draw_on_overlay": draw_on_overlay,
        "fit_quality": round(fit_quality, 4),
        "needs_review": needs_review,
        "reason": candidate_reason,
        **metrics,
    }
    candidate["geometry_reason"] = candidate["reason"]
    candidate.update(_module4_draft_roi_scoring(candidate))
    candidate["reason"] = candidate["draft_score_reason"]
    return {
        **debug_base,
        "success": True,
        "candidate": candidate,
        "selected_component_area": selected["area"],
        "selected_component_bbox": selected_component_bbox,
        "selected_component_center": selected_component_center,
        "selected_component_reason": (
            "Selected highest-scoring plausible merged component near the "
            "image-localized B1 center."
        ),
        "fitted_angle_degrees": round(fitted_angle, 3),
        "pre_template_component_center": {
            "x": round(selected["center_x"] + x0, 2),
            "y": round(selected["center_y"] + y0, 2),
        },
        "pre_template_component_bbox": selected_component_bbox,
        "pre_template_fit_box": pre_template_fit_box,
        "template_search_center_range": round(
            template_fit["search_center_range"], 3,
        ),
        "template_search_side_range": [
            round(value, 3) for value in template_fit["search_side_range"]
        ],
        "template_search_angle_range": [
            round(value, 3) for value in template_fit["search_angle_range"]
        ],
        "best_template_score": round(template_score, 4),
        "best_template_score_breakdown": {
            "boundary_edge_score": candidate["boundary_edge_score"],
            "edge_symmetry_score": candidate["edge_symmetry_score"],
            "interior_fill_score": candidate["interior_fill_score"],
            "outside_leakage_penalty": candidate["outside_leakage_penalty"],
            "mask_coverage_score": candidate["mask_coverage_score"],
            "center_proximity_score": candidate["center_proximity_score"],
        },
        "top_template_trials": template_fit["top_trials"],
        "template_trial_count": template_fit.get("trial_count", 0),
        "pre_micro_template_center": template_fit.get(
            "pre_micro_template_center"
        ),
        "post_micro_template_center": template_fit.get(
            "post_micro_template_center"
        ),
        "micro_shift_px": template_fit.get("micro_shift_px"),
        "pre_micro_score": template_fit.get("pre_micro_score"),
        "post_micro_score": template_fit.get("post_micro_score"),
        "micro_refinement_applied": template_fit.get(
            "micro_refinement_applied", False,
        ),
        "micro_refinement_reason": template_fit.get(
            "micro_refinement_reason"
        ),
        "final_square_is_template_fit": template_fit["success"],
        "component_fit_side_px": round(component_fit_side, 3),
        "side_margin_ratio": side_margin_ratio,
        "side_before_clamp_px": round(side_before_clamp, 3),
        "final_side_px": round(fitted_side, 3),
        "fitted_side_px": round(fitted_side, 3),
        "final_center": candidate["center"],
        "final_rotated_box": rotated_box,
        "inner_roi": inner_roi,
        "inner_roi_size_ratio": inner_roi_ratio,
        "fit_quality": round(fit_quality, 4),
        "fallback_used": False,
        "failure_reason": None,
        "fitted_crop_image": image_to_base64(fitted_crop),
    }


def fit_module4_global_block_layout(
    rough_candidates: list[dict],
    phantom_center_x: float,
    phantom_center_y: float,
    phantom_radius: float,
    working: np.ndarray,
    residual: np.ndarray,
    noise: float,
) -> dict:
    """Assign rough block candidates to one of eight normalized ring slots."""
    slot_definitions = (
        ("top", -90.0),
        ("upper_right", -45.0),
        ("right", 0.0),
        ("lower_right", 45.0),
        ("bottom", 90.0),
        ("lower_left", 135.0),
        ("left", 180.0),
        ("upper_left", -135.0),
    )
    prepared = []
    for index, candidate in enumerate(rough_candidates):
        center = candidate["center"]
        dx = float(center["x"]) - phantom_center_x
        dy = float(center["y"]) - phantom_center_y
        radial_ratio = math.hypot(dx, dy) / max(phantom_radius, 1.0)
        polar_angle = math.degrees(math.atan2(dy, dx))
        prepared.append({
            "index": index,
            "candidate": candidate,
            "radial_ratio": radial_ratio,
            "polar_angle": polar_angle,
        })

    ring_values = [
        item["radial_ratio"]
        for item in prepared
        if 0.16 <= item["radial_ratio"] <= 0.72
    ]
    expected_ring_ratio = float(np.median(ring_values)) if ring_values else 0.46
    expected_ring_ratio = max(0.28, min(0.62, expected_ring_ratio))
    pairings = []
    assignment_scores = []
    for item in prepared:
        candidate = item["candidate"]
        for slot_index, (slot_name, slot_angle) in enumerate(slot_definitions):
            angular_error = _polar_angle_error(item["polar_angle"], slot_angle)
            radial_error = abs(item["radial_ratio"] - expected_ring_ratio)
            angular_quality = _clamp01(1.0 - angular_error / 32.0)
            radial_quality = _clamp01(1.0 - radial_error / 0.20)
            score = _clamp01(
                0.38 * angular_quality
                + 0.24 * radial_quality
                + 0.16 * candidate["confidence"]
                + 0.12 * candidate["box_fit_quality"]
                + 0.10 * candidate["contrast_score"]
            )
            record = {
                "slot_index": slot_index,
                "target_slot": slot_name,
                "candidate_index": item["index"],
                "score": score,
                "angular_error_degrees": angular_error,
                "radial_error": radial_error,
            }
            assignment_scores.append({
                **record,
                "score": round(score, 4),
                "angular_error_degrees": round(angular_error, 3),
                "radial_error": round(radial_error, 4),
            })
            if (
                angular_error <= 32.0
                and radial_error <= 0.20
                and 0.16 <= item["radial_ratio"] <= 0.72
                and score >= 0.42
            ):
                pairings.append(record)

    assigned_slots = {}
    used_candidates = set()
    for pairing in sorted(pairings, key=lambda item: item["score"], reverse=True):
        if pairing["target_slot"] in assigned_slots:
            continue
        if pairing["candidate_index"] in used_candidates:
            continue
        assigned_slots[pairing["target_slot"]] = pairing
        used_candidates.add(pairing["candidate_index"])

    assigned_items = [
        {
            "slot": slot_name,
            "pairing": pairing,
            "prepared": prepared[pairing["candidate_index"]],
            "candidate": prepared[pairing["candidate_index"]]["candidate"],
        }
        for slot_name, pairing in assigned_slots.items()
    ]
    anchor_items = [
        item for item in assigned_items
        if (
            item["pairing"]["score"] >= 0.56
            and item["candidate"]["confidence"] >= 0.50
            and item["candidate"]["box_fit_quality"] >= 0.48
            and item["candidate"]["contrast_score"] >= 0.25
            and not item["candidate"].get("needs_review", False)
        )
    ]
    strong_anchor_count = len(anchor_items)
    if len(anchor_items) < 3:
        anchor_items = sorted(
            assigned_items,
            key=lambda item: (
                item["pairing"]["score"]
                + item["candidate"]["box_fit_quality"]
                + item["candidate"]["contrast_score"]
            ),
            reverse=True,
        )[:min(3, len(assigned_items))]
    anchor_targets_used = [item["slot"] for item in anchor_items]
    side_votes = [
        {
            "target_slot": item["slot"],
            "side_px": float(item["candidate"]["bbox"]["width"]),
            "weight": round(float(item["pairing"]["score"]), 4),
        }
        for item in anchor_items
    ]
    common_side = (
        float(np.median([item["side_px"] for item in side_votes]))
        if side_votes
        else None
    )
    anchor_ring_ratios = [item["prepared"]["radial_ratio"] for item in anchor_items]
    final_ring_ratio = (
        float(np.median(anchor_ring_ratios))
        if anchor_ring_ratios
        else expected_ring_ratio
    )
    final_ring_ratio = max(
        expected_ring_ratio - 0.08,
        min(final_ring_ratio, expected_ring_ratio + 0.08),
    )
    internal_ring_radius = final_ring_ratio * phantom_radius

    angle_votes = []
    axis_vote = 0.0
    diamond_vote = 0.0
    for item in anchor_items:
        candidate = item["candidate"]
        local_angle = (float(candidate["angle_degrees"]) + 90.0) % 90.0
        distance_axis = min(local_angle, 90.0 - local_angle)
        distance_diamond = abs(local_angle - 45.0)
        weight = (
            float(item["pairing"]["score"])
            * max(0.15, float(candidate["orientation_confidence"]))
            * max(0.15, float(candidate["box_fit_quality"]))
        )
        axis_support = weight * _clamp01(1.0 - distance_axis / 30.0)
        diamond_support = weight * _clamp01(1.0 - distance_diamond / 30.0)
        axis_vote += axis_support
        diamond_vote += diamond_support
        angle_votes.append({
            "target_slot": item["slot"],
            "local_angle_degrees": round(float(candidate["angle_degrees"]), 3),
            "axis_support": round(axis_support, 4),
            "diamond_support": round(diamond_support, 4),
            "weight": round(weight, 4),
        })
    global_target_angle = 45.0 if diamond_vote > axis_vote else 0.0
    vote_total = axis_vote + diamond_vote
    global_orientation_confidence = (
        abs(diamond_vote - axis_vote) / vote_total
        if vote_total > 0
        else 0.0
    )
    layout_needs_review = (
        strong_anchor_count < 3
        or global_orientation_confidence < 0.18
    )
    accepted_targets = []
    for target_index, (slot_name, _) in enumerate(slot_definitions, start=1):
        pairing = assigned_slots.get(slot_name)
        if not pairing:
            continue
        candidate = dict(prepared[pairing["candidate_index"]]["candidate"])
        evidence_side = float(candidate["bbox"]["width"])
        strong_size_evidence = (
            candidate["box_fit_quality"] >= 0.80
            and pairing["score"] >= 0.68
        )
        final_side = (
            int(round(max(
                common_side * 0.90,
                min(evidence_side, common_side * 1.10),
            )))
            if strong_size_evidence
            else int(round(common_side))
        )
        local_angle = float(candidate["angle_degrees"])
        local_angle_modulo = (local_angle + 90.0) % 90.0
        global_angle_modulo = global_target_angle % 90.0
        angle_deviation = min(
            abs(local_angle_modulo - global_angle_modulo),
            90.0 - abs(local_angle_modulo - global_angle_modulo),
        )
        allow_local_angle = bool(
            candidate["orientation_confidence"] >= 0.92
            and candidate["box_fit_quality"] >= 0.85
            and angle_deviation <= 4.0
        )
        final_angle = (
            _stable_square_angle(local_angle, 1.0)[0]
            if allow_local_angle
            else global_target_angle
        )
        slot_angle = next(
            angle for name, angle in slot_definitions if name == slot_name
        )
        expected_center_x = (
            phantom_center_x
            + internal_ring_radius * math.cos(math.radians(slot_angle))
        )
        expected_center_y = (
            phantom_center_y
            + internal_ring_radius * math.sin(math.radians(slot_angle))
        )
        detected_center_x = float(candidate["center"]["x"])
        detected_center_y = float(candidate["center"]["y"])
        center_error_before = math.hypot(
            detected_center_x - expected_center_x,
            detected_center_y - expected_center_y,
        )
        center_refinement = refine_module4_target_center(
            working,
            residual,
            expected_center_x,
            expected_center_y,
            detected_center_x,
            detected_center_y,
            final_side,
            final_angle,
            phantom_radius,
            noise,
        )
        center_x = center_refinement["final_center_x"]
        center_y = center_refinement["final_center_y"]
        old_refined_center = {
            "x": round(center_x, 2),
            "y": round(center_y, 2),
        }
        jump_fit = fit_module4_jump_line_roi(
            working,
            residual,
            (expected_center_x, expected_center_y),
            (center_x, center_y),
            final_side,
            final_angle,
            phantom_radius,
            noise,
        )
        if jump_fit.get("partial"):
            center_x = jump_fit["center_x"]
            center_y = jump_fit["center_y"]
            final_side = int(round(jump_fit["final_side_px"]))
            final_angle = jump_fit["final_angle_degrees"]
        final_shift = math.hypot(
            center_x - expected_center_x,
            center_y - expected_center_y,
        )
        candidate["center"] = {
            "x": round(center_x, 2),
            "y": round(center_y, 2),
        }
        candidate["id"] = f"B{target_index}"
        candidate["target_slot"] = slot_name
        candidate["candidate_score"] = round(pairing["score"], 4)
        candidate["assignment_confidence"] = round(pairing["score"], 4)
        candidate["angular_error_degrees"] = round(pairing["angular_error_degrees"], 3)
        candidate["radial_error"] = round(pairing["radial_error"], 4)
        candidate["common_side_px"] = round(common_side, 2)
        candidate["side_px"] = final_side
        candidate["local_side_px"] = round(evidence_side, 2)
        candidate["final_side_px"] = final_side
        candidate["angle_degrees"] = round(final_angle, 3)
        candidate["per_block_local_angle"] = round(local_angle, 3)
        candidate["per_block_final_angle"] = round(final_angle, 3)
        candidate["per_block_angle_override_reason"] = (
            "Very strong local evidence agreed with the global orientation."
            if allow_local_angle
            else "Global slice orientation replaced independent local angle."
        )
        candidate["rotated_box"] = _square_points(
            center_x,
            center_y,
            final_side,
            final_angle,
        )
        half_side = final_side / 2.0
        candidate["bbox"] = {
            "x": max(0, int(round(center_x - half_side))),
            "y": max(0, int(round(center_y - half_side))),
            "width": final_side,
            "height": final_side,
        }
        candidate["inner_roi"] = {
            "center": {"x": center_x, "y": center_y},
            "width": round(final_side * 0.60, 2),
            "height": round(final_side * 0.60, 2),
            "angle_degrees": round(final_angle, 3),
        }
        candidate.update(_module4_metrics_from_rotated_inner_roi(
            working,
            center_x,
            center_y,
            final_side * 0.60,
            final_angle,
            noise,
        ))
        candidate["size_normalization_applied"] = (
            candidate.get("size_normalization_applied", False)
            or final_side != int(round(evidence_side))
        )
        candidate["side_clamp_applied"] = final_side != int(round(evidence_side))
        candidate["orientation_normalization_applied"] = not allow_local_angle
        candidate["expected_center"] = {
            "x": round(expected_center_x, 2),
            "y": round(expected_center_y, 2),
        }
        candidate["detected_center"] = {
            "x": round(detected_center_x, 2),
            "y": round(detected_center_y, 2),
        }
        candidate["pre_refine_center"] = candidate["detected_center"]
        candidate["final_center"] = candidate["center"]
        candidate["old_refined_center"] = old_refined_center
        candidate["global_expected_center"] = jump_fit.get(
            "global_expected_center",
            candidate["expected_center"],
        )
        candidate["jump_line_center"] = jump_fit.get("jump_line_center")
        candidate["jump_cluster_found"] = bool(jump_fit.get("jump_cluster_found", False))
        candidate["jump_peaks_found"] = int(jump_fit.get("jump_peaks_found", 0))
        candidate["jump_peak_positions"] = jump_fit.get("jump_peak_positions", [])
        candidate["jump_cluster_start_px"] = jump_fit.get("jump_cluster_start_px")
        candidate["jump_cluster_end_px"] = jump_fit.get("jump_cluster_end_px")
        candidate["jump_cluster_width_px"] = jump_fit.get("jump_cluster_width_px")
        candidate["selected_outer_jump_lines"] = jump_fit.get(
            "selected_outer_jump_lines", [],
        )
        candidate["jump_lines_found"] = int(jump_fit.get("jump_lines_found", 0))
        candidate["jump_line_angles"] = jump_fit.get("jump_line_angles", [])
        candidate["jump_line_separation_px"] = jump_fit.get("jump_line_separation_px")
        candidate["perpendicular_extent_px"] = jump_fit.get("perpendicular_extent_px")
        candidate["perpendicular_support_start_px"] = jump_fit.get(
            "perpendicular_support_start_px",
        )
        candidate["perpendicular_support_end_px"] = jump_fit.get(
            "perpendicular_support_end_px",
        )
        candidate["line_fit_score"] = jump_fit.get("line_fit_score", 0.0)
        candidate["envelope_fit_score"] = jump_fit.get("envelope_fit_score", 0.0)
        candidate["edge_jump_score"] = jump_fit.get("edge_jump_score", 0.0)
        candidate["global_angle_degrees"] = jump_fit.get(
            "global_angle_degrees", global_target_angle,
        )
        candidate["jump_line_angle_degrees"] = jump_fit.get("jump_line_angle_degrees")
        candidate["final_angle_degrees"] = round(final_angle, 3)
        candidate["angle_source"] = jump_fit.get("angle_source", "global_angle_fallback")
        candidate["angle_confidence"] = jump_fit.get("angle_confidence", 0.0)
        candidate["global_side_px"] = jump_fit.get("global_side_px", common_side)
        candidate["jump_fit_side_px"] = jump_fit.get("jump_fit_side_px")
        candidate["side_source"] = jump_fit.get("side_source", "global_side_fallback")
        candidate["center_shift_from_global"] = jump_fit.get(
            "center_shift_from_global", round(final_shift, 3),
        )
        candidate["center_shift_from_jump_fit"] = jump_fit.get(
            "center_shift_from_jump_fit", 0.0,
        )
        candidate["center_refinement_reason"] = jump_fit.get(
            "center_refinement_reason",
            "Jump-cluster boundary fit incomplete; global fallback used.",
        )
        candidate["center_error_before_refinement_px"] = round(center_error_before, 3)
        candidate["center_shift_px"] = round(final_shift, 3)
        candidate["center_shift_limit_px"] = center_refinement["search_radius_px"]
        candidate["center_refinement_method"] = center_refinement["center_refinement_method"]
        candidate["center_refinement_shift_px"] = center_refinement["center_refinement_shift_px"]
        candidate["center_refinement_score"] = center_refinement["center_refinement_score"]
        candidate["local_alignment_score"] = center_refinement["local_alignment_score"]
        candidate["edge_alignment_score"] = center_refinement["edge_alignment_score"]
        candidate["target_fill_score"] = center_refinement["target_fill_score"]
        candidate["stripe_alignment_score"] = center_refinement.get("stripe_alignment_score", 0.0)
        candidate["background_leakage_score"] = center_refinement.get("background_leakage_score", 0.0)
        candidate["slot_assignment_confidence"] = round(pairing["score"], 4)
        candidate["geometry_source"] = (
            "jump_cluster_envelope_perpendicular_roi_fit"
            if jump_fit.get("partial")
            else "global_slot_model_jump_cluster_fallback"
        )
        candidate["needs_review"] = bool(
            candidate.get("needs_review")
            or pairing["score"] < 0.56
            or layout_needs_review
            or center_refinement["needs_review"]
            or jump_fit.get("needs_review", True)
        )
        candidate["reason"] = (
            jump_fit["reason"]
            if not candidate["needs_review"]
            else (
                jump_fit.get(
                    "reason",
                    "Jump-cluster boundary fit incomplete; global fallback used.",
                )
            )
        )
        accepted_targets.append(candidate)

    missing_slots = [
        slot_name for slot_name, _ in slot_definitions
        if slot_name not in assigned_slots
    ]
    rejected_candidates = []
    for item in prepared:
        if item["index"] in used_candidates:
            continue
        nearest_slot, nearest_angle = min(
            slot_definitions,
            key=lambda slot: _polar_angle_error(item["polar_angle"], slot[1]),
        )
        radial_error = abs(item["radial_ratio"] - expected_ring_ratio)
        reason = (
            "far_from_expected_ring"
            if radial_error > 0.20
            else "duplicate_or_lower_score_in_target_sector"
        )
        rejected_candidates.append({
            "center": item["candidate"]["center"],
            "nearest_slot": nearest_slot,
            "angular_error_degrees": round(
                _polar_angle_error(item["polar_angle"], nearest_angle),
                3,
            ),
            "radial_error": round(radial_error, 4),
            "reason": reason,
        })
    mean_assignment = (
        float(np.mean([item["assignment_confidence"] for item in accepted_targets]))
        if accepted_targets
        else 0.0
    )
    count_quality = _clamp01(len(accepted_targets) / 8.0)
    global_layout_quality = _clamp01(0.58 * count_quality + 0.42 * mean_assignment)
    weak_slots = [
        item["target_slot"]
        for item in accepted_targets
        if item["needs_review"]
    ]
    return {
        "accepted_targets": accepted_targets,
        "expected_ring_radius": round(internal_ring_radius, 2),
        "expected_ring_radius_ratio": round(final_ring_ratio, 4),
        "internal_ring_radius_px": round(internal_ring_radius, 2),
        "common_side_px": round(common_side, 2) if common_side is not None else None,
        "side_votes": side_votes,
        "global_target_angle_degrees": round(global_target_angle, 3),
        "angle_votes": angle_votes,
        "anchor_targets_used": anchor_targets_used,
        "strong_anchor_count": strong_anchor_count,
        "orientation_confidence": round(global_orientation_confidence, 4),
        "global_layout_model_used": True,
        "layout_needs_review": layout_needs_review,
        "target_slots": [slot_name for slot_name, _ in slot_definitions],
        "missing_slots": missing_slots,
        "weak_slots": weak_slots,
        "rejected_candidates": rejected_candidates,
        "assignment_scores": assignment_scores,
        "global_layout_quality": round(global_layout_quality, 4),
    }


def _detection_set_quality(result: dict) -> float:
    candidates = result["candidates"]
    count = len(candidates)
    count_quality = _clamp01(1.0 - abs(count - 8) / 8.0)
    if not candidates:
        return 0.0
    fit_quality = float(np.mean([item["box_fit_quality"] for item in candidates]))
    contrast_quality = float(np.mean([item["contrast_score"] for item in candidates]))
    sides = [item["bbox"]["width"] for item in candidates]
    size_quality = _clamp01(1.0 - float(np.std(sides)) / max(float(np.mean(sides)), 1.0))
    radial_quality = float(np.mean([
        _clamp01(1.0 - max(0.0, item["radial_ratio"] - 0.68) / 0.08)
        for item in candidates
    ]))
    return _clamp01(
        0.25 * count_quality
        + 0.20 * fit_quality
        + 0.18 * contrast_quality
        + 0.15 * size_quality
        + 0.10 * radial_quality
        + 0.12 * result.get("global_layout_quality", 0.0)
    )


def detect_module4_square_blocks(
    slice_pixels: np.ndarray,
    pixel_spacing: tuple[float | None, float | None] | None = None,
) -> dict:
    """Select the safest result from multiple adaptive detection passes."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    phantom_detection_status = "detected"
    try:
        phantom_geometry = _estimate_phantom_geometry(raw)
    except Exception:
        height, width = raw.shape
        phantom_geometry = (
            (width - 1) / 2.0,
            (height - 1) / 2.0,
            min(height, width) * 0.42,
        )
        phantom_detection_status = "fallback_needs_review"

    pass_configs = (
        {
            "name": "standard",
            "hu_percentile": 88.0, "residual_percentile": 90.0,
            "noise_multiplier": 3.5, "merge_radius_fraction": 0.045,
            "closing_iterations": 2, "dilation_iterations": 1,
            "minimum_area_fraction": 0.002, "maximum_area_fraction": 0.085,
        },
        {
            "name": "strict",
            "hu_percentile": 92.0, "residual_percentile": 94.0,
            "noise_multiplier": 4.2, "merge_radius_fraction": 0.030,
            "closing_iterations": 1, "dilation_iterations": 0,
            "minimum_area_fraction": 0.0014, "maximum_area_fraction": 0.060,
        },
        {
            "name": "relaxed",
            "hu_percentile": 82.0, "residual_percentile": 84.0,
            "noise_multiplier": 2.6, "merge_radius_fraction": 0.055,
            "closing_iterations": 2, "dilation_iterations": 1,
            "minimum_area_fraction": 0.0015, "maximum_area_fraction": 0.095,
        },
        {
            "name": "edge_supported",
            "hu_percentile": 84.0, "residual_percentile": 87.0,
            "noise_multiplier": 2.9, "merge_radius_fraction": 0.040,
            "closing_iterations": 1, "dilation_iterations": 1,
            "minimum_area_fraction": 0.0015, "maximum_area_fraction": 0.080,
            "edge_supported": True,
        },
    )
    attempts = []
    for config in pass_configs:
        try:
            result = _detect_module4_square_blocks_pass(
                raw,
                phantom_geometry,
                config,
            )
            result["detection_set_quality"] = round(_detection_set_quality(result), 4)
            attempts.append(result)
        except Exception as exc:
            attempts.append({
                "candidates": [],
                "detection_pass": config["name"],
                "detection_set_quality": 0.0,
                "pass_error": str(exc),
                "rejection_summary": {},
            })
    selected = max(attempts, key=lambda item: item["detection_set_quality"])
    selected = dict(selected)
    selected["detection_passes_tried"] = [item["detection_pass"] for item in attempts]
    selected["selected_detection_pass"] = selected["detection_pass"]
    selected["candidates_per_pass"] = {
        item["detection_pass"]: len(item["candidates"])
        for item in attempts
    }
    selected["rejection_counts_per_pass"] = {
        item["detection_pass"]: item.get("rejection_summary", {})
        for item in attempts
    }
    selected["detection_pass_quality"] = {
        item["detection_pass"]: item["detection_set_quality"]
        for item in attempts
    }
    selected["detection_pass_errors"] = {
        item["detection_pass"]: item["pass_error"]
        for item in attempts if item.get("pass_error")
    }
    completed_passes = [item for item in attempts if not item.get("pass_error")]
    selected["detection_passes_completed"] = [
        item["detection_pass"] for item in completed_passes
    ]
    all_passes_failed = not completed_passes
    selected["all_detection_passes_failed"] = all_passes_failed
    selected["detection_selection_reason"] = (
        "All block-detection passes crashed; review the recorded pass errors."
        if all_passes_failed
        else "Selected the completed pass with the best count, geometry, contrast, "
             "size consistency, and internal-location quality score."
    )
    selected["phantom_detection_status"] = phantom_detection_status
    selected.setdefault("phantom_center", {
        "x": round(phantom_geometry[0], 2),
        "y": round(phantom_geometry[1], 2),
    })
    selected.setdefault("phantom_radius", round(phantom_geometry[2], 2))
    selected.setdefault("thresholds", {})
    selected.setdefault("components_considered", 0)
    selected.setdefault("components_rejected", 0)
    selected.setdefault("merge_groups", [])
    selected.setdefault("merge_kernel_size", None)
    selected.setdefault("median_block_side_px", None)
    selected["pixel_spacing"] = {
        "row_mm": pixel_spacing[0] if pixel_spacing else None,
        "column_mm": pixel_spacing[1] if pixel_spacing else None,
        "used_for_detection": False,
        "note": (
            "Preserved for debug; current geometry is normalized to detected phantom radius."
        ),
    }
    count = len(selected["candidates"])
    quality = selected["detection_set_quality"]
    selected["analysis_review_status"] = (
        "error" if all_passes_failed
        else "detected" if (
            6 <= count <= 8
            and quality >= 0.62
            and phantom_detection_status == "detected"
            and not selected.get("layout_needs_review", False)
        )
        else "partial" if count >= 3 and quality >= 0.42
        else "needs_review" if count
        else "not_found"
    )
    for candidate in selected["candidates"]:
        if candidate["needs_review"]:
            candidate["preliminary_visibility"] = "needs_review"
            candidate["reason"] = (
                f"{candidate['reason']} Geometry or local contrast needs review; "
                "formal measurement is not implemented."
            )
    return selected


def _select_module4_display_window(
    raw: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> dict:
    """Choose the best display-only window from safe Module 4 candidates."""
    yy, xx = np.indices(raw.shape)
    finite = np.isfinite(raw)
    interior_mask = finite & (np.hypot(xx - cx, yy - cy) <= radius * 0.86)
    outside_mask = finite & (np.hypot(xx - cx, yy - cy) >= radius * 1.04)
    values = raw[interior_mask]
    if values.size < 100:
        return {
            "window_width": 700.0,
            "window_level": 150.0,
            "window_method": "module4_safe_fixed_fallback",
            "window_quality_score": 0.0,
            "candidate_windows": [],
            "window_note": (
                "Fallback display window used because phantom statistics were incomplete. "
                "Visualization only; analysis uses raw CT pixels."
            ),
        }

    p05, p10, p50, p90, p985, p995 = np.percentile(
        values,
        (5.0, 10.0, 50.0, 90.0, 98.5, 99.5),
    )
    spread = max(20.0, float(p90 - p10))
    candidates = [
        ("body_centered", max(480.0, spread * 7.0), float(p50 + spread * 0.35)),
        (
            "bright_tail_controlled",
            min(900.0, max(520.0, float(p985 - (p50 - max(180.0, spread * 2.0))))),
            0.0,
        ),
        (
            "contrast_enhanced_phantom",
            min(760.0, max(420.0, spread * 5.5)),
            float(p50 + spread * 0.20),
        ),
        ("safe_fixed", 700.0, 150.0),
    ]
    adjusted = []
    for name, width, level in candidates:
        width = float(max(350.0, min(1000.0, width)))
        if name == "bright_tail_controlled":
            lower = float(p50 - max(180.0, spread * 2.0))
            level = lower + width / 2.0
        lower = level - width / 2.0
        body_gray = _clamp01((float(p50) - lower) / width)
        bright_gray = _clamp01((float(p985) - lower) / width)
        outside_values = raw[outside_mask]
        outside_median = (
            float(np.median(outside_values))
            if outside_values.size
            else float(p05 - width)
        )
        outside_gray = _clamp01((outside_median - lower) / width)
        body_score = _clamp01(1.0 - abs(body_gray - 0.42) / 0.32)
        bright_score = _clamp01(1.0 - abs(bright_gray - 0.88) / 0.28)
        outside_score = _clamp01(1.0 - outside_gray / 0.18)
        contrast_score = _clamp01((bright_gray - body_gray) / 0.32)
        quality = _clamp01(
            0.34 * body_score
            + 0.27 * bright_score
            + 0.22 * outside_score
            + 0.17 * contrast_score
        )
        adjusted.append({
            "method": name,
            "window_width": round(width, 2),
            "window_level": round(float(level), 2),
            "quality_score": round(quality, 4),
            "body_gray": round(body_gray, 4),
            "bright_gray": round(bright_gray, 4),
            "outside_gray": round(outside_gray, 4),
        })
    best = max(adjusted, key=lambda item: item["quality_score"])
    return {
        "window_width": best["window_width"],
        "window_level": best["window_level"],
        "window_method": f"module4_{best['method']}",
        "window_quality_score": best["quality_score"],
        "candidate_windows": adjusted,
        "window_note": (
            "Adaptive Module 4 display window selected for visualization only; "
            "detection and review metrics use raw CT pixels."
        ),
    }


def generate_module4_block_overlay(
    slice_pixels: np.ndarray,
    detections: list[dict],
    window_width: float = 400.0,
    window_level: float = 40.0,
    photometric: str = "",
    title: str = "Module 4 blocks - measurement pending",
) -> str:
    overlay = window_pixels_to_image(
        slice_pixels,
        window_width,
        window_level,
        photometric,
    )
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        final_roi = detection.get("final_roi", detection)
        if not final_roi.get("draw_on_overlay", False):
            continue
        final_corners = final_roi.get(
            "final_corners", detection.get("rotated_box", [])
        )
        points = [(point["x"], point["y"]) for point in final_corners]
        draw.line(points + [points[0]], fill="#22d3ee", width=3)
        inner = final_roi.get("inner_roi", detection["inner_roi"])
        half_w, half_h = inner["width"] / 2, inner["height"] / 2
        angle = math.radians(inner["angle_degrees"])
        inner_points = []
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
            inner_points.append((
                inner["center"]["x"] + dx * math.cos(angle) - dy * math.sin(angle),
                inner["center"]["y"] + dx * math.sin(angle) + dy * math.cos(angle),
            ))
        draw.line(inner_points + [inner_points[0]], fill="#a5f3fc", width=1)
        draw.text(
            (points[0][0] + 3, max(0, points[0][1] - 14)),
            detection.get("display_label", detection["id"]),
            fill="#67e8f9",
        )
    draw.text((12, 12), title, fill="#fde047")
    return image_to_base64(overlay)


def fit_module4_slot_template_fallback(
    slice_pixels: np.ndarray,
    phantom_center: tuple[float, float],
    phantom_radius: float,
    expected_center: tuple[float, float],
    expected_side: float,
    crop_side: int,
    target_id: str,
    target_slot: str,
) -> dict:
    """Attempt a review-only square fit when component localization is weak."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    finite = np.isfinite(raw)
    fill = float(np.median(raw[finite]))
    working = np.where(finite, raw, fill)
    half_crop = crop_side // 2
    x0 = max(0, int(round(expected_center[0])) - half_crop)
    y0 = max(0, int(round(expected_center[1])) - half_crop)
    x1 = min(raw.shape[1], x0 + crop_side)
    y1 = min(raw.shape[0], y0 + crop_side)
    crop = working[y0:y1, x0:x1]
    background_size = max(5, int(round(expected_side * 0.35)) | 1)
    background = ndimage.median_filter(crop, size=background_size)
    residual = crop - background
    residual_median = float(np.median(residual))
    noise = max(
        1.0,
        float(np.median(np.abs(residual - residual_median)) * 1.4826),
    )
    local_threshold = max(
        residual_median + 2.25 * noise,
        float(np.percentile(residual, 74.0)),
    )
    threshold_mask = residual >= local_threshold
    merge_size = max(3, min(9, int(round(expected_side * 0.14)) | 1))
    merged_mask = ndimage.binary_closing(
        threshold_mask,
        structure=np.ones((merge_size, merge_size), dtype=bool),
        iterations=2,
    )
    merged_mask = ndimage.binary_fill_holes(merged_mask)
    empty_component_mask = np.zeros(crop.shape, dtype=bool)
    template_fit = fit_single_block_square_template(
        crop,
        residual,
        empty_component_mask,
        start_center=(expected_center[0] - x0, expected_center[1] - y0),
        expected_side=expected_side,
        expected_angle=45.0,
        noise=noise,
        crop_origin=(x0, y0),
        phantom_center=phantom_center,
        phantom_radius=phantom_radius,
    )
    best = template_fit.get("best")
    reasonable = bool(
        best
        and best["template_score"] >= 0.30
        and (
            best["boundary_edge_score"] >= 0.24
            or best["interior_fill_score"] >= 0.24
        )
    )
    debug = {
        "debug_target_id": target_id,
        "debug_target_slot": target_slot,
        "crop_bounds": {
            "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
        },
        "template_only_fallback": True,
        "local_threshold": round(local_threshold, 3),
        "components_before_merge": int(ndimage.label(threshold_mask)[1]),
        "components_after_merge": int(ndimage.label(merged_mask)[1]),
        "local_crop_image": image_to_base64(window_pixels_to_image(
            crop,
            max(
                250.0,
                float(np.percentile(crop, 98.0) - np.percentile(crop, 5.0)),
            ),
            float(
                (np.percentile(crop, 98.0) + np.percentile(crop, 5.0)) / 2.0
            ),
        )),
        "threshold_mask_image": _debug_mask_image(threshold_mask),
        "merged_mask_image": _debug_mask_image(merged_mask),
        "template_attempted": True,
        "template_search_center_range": template_fit["search_center_range"],
        "template_search_side_range": template_fit["search_side_range"],
        "template_search_angle_range": template_fit["search_angle_range"],
        "top_template_trials": template_fit["top_trials"],
        "template_trial_count": template_fit.get("trial_count", 0),
        "best_template_score": (
            round(best["template_score"], 4) if best else None
        ),
        "failure_reason": None if reasonable else (
            template_fit.get("failure_reason")
            or "Slot-centered template evidence was too weak for a review box."
        ),
    }
    if not reasonable:
        return {**debug, "success": False, "candidate": None}
    center_x = best["center_x"] + x0
    center_y = best["center_y"] + y0
    side = best["side_px"]
    angle = best["angle_degrees"]
    inner_ratio = 0.58
    inner_roi = {
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "width": round(side * inner_ratio, 2),
        "height": round(side * inner_ratio, 2),
        "angle_degrees": round(angle, 3),
    }
    metrics = _module4_metrics_from_rotated_inner_roi(
        working, center_x, center_y, side * inner_ratio, angle, noise,
    )
    candidate = {
        "id": target_id,
        "target_slot": target_slot,
        "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "bbox": {
            "x": max(0, int(round(center_x - side / 2.0))),
            "y": max(0, int(round(center_y - side / 2.0))),
            "width": int(round(side)),
            "height": int(round(side)),
        },
        "rotated_box": _square_points(center_x, center_y, side, angle),
        "inner_roi": inner_roi,
        "angle_degrees": round(angle, 3),
        "side_px": round(side, 3),
        "box_fit_quality": round(best["template_score"], 4),
        "fit_quality": round(best["template_score"], 4),
        "assignment_confidence": 0.0,
        "geometry_source": "module4_block_square_template_slot_fallback",
        "template_score": round(best["template_score"], 4),
        "boundary_edge_score": round(best["boundary_edge_score"], 4),
        "edge_symmetry_score": round(best["edge_symmetry_score"], 4),
        "interior_fill_score": round(best["interior_fill_score"], 4),
        "outside_leakage_penalty": round(best["outside_leakage_penalty"], 4),
        "mask_coverage_score": None,
        "center_proximity_score": round(best["center_proximity_score"], 4),
        "template_center": {"x": round(center_x, 2), "y": round(center_y, 2)},
        "template_side_px": round(side, 3),
        "template_angle_degrees": round(angle, 3),
        "selected_component_center": None,
        "selected_component_bbox": None,
        "selected_component_bbox_size": None,
        "center_delta_from_component": None,
        "expected_slot_center": {
            "x": round(expected_center[0], 2),
            "y": round(expected_center[1], 2),
        },
        "center_delta_from_slot": round(math.hypot(
            center_x - expected_center[0],
            center_y - expected_center[1],
        ), 3),
        "draw_on_overlay": False,
        "needs_review": True,
        "reason": (
            "Review-only perfect-square fit from slot-centered image evidence; "
            "no unique component localization was available."
        ),
        **metrics,
    }
    candidate["geometry_reason"] = candidate["reason"]
    candidate.update(_module4_draft_roi_scoring(candidate))
    candidate["reason"] = candidate["draft_score_reason"]
    fitted_crop = window_pixels_to_image(
        crop,
        max(
            250.0,
            float(np.percentile(crop, 98.0) - np.percentile(crop, 5.0)),
        ),
        float((np.percentile(crop, 98.0) + np.percentile(crop, 5.0)) / 2.0),
    )
    fitted_draw = ImageDraw.Draw(fitted_crop)
    fitted_points = [
        (point["x"] - x0, point["y"] - y0)
        for point in candidate["rotated_box"]
    ]
    fitted_draw.line(
        fitted_points + [fitted_points[0]],
        fill="#22d3ee",
        width=2,
    )
    return {
        **debug,
        "success": True,
        "candidate": candidate,
        "final_rotated_box": candidate["rotated_box"],
        "inner_roi": inner_roi,
        "fit_quality": round(best["template_score"], 4),
        "fitted_crop_image": image_to_base64(fitted_crop),
    }


def _analyze_module4_high_contrast_slice_legacy(
    slice_pixels: np.ndarray,
    window_width: float = 400.0,
    window_level: float = 40.0,
    photometric: str = "",
    pixel_spacing: tuple[float | None, float | None] | None = None,
    debug_show_reference_targets: bool = False,
) -> dict:
    raw = np.asarray(slice_pixels, dtype=np.float32)
    try:
        cx, cy, radius = _estimate_phantom_geometry(raw)
        phantom_detection_status = "detected"
    except Exception:
        height, width = raw.shape
        cx, cy, radius = (
            (width - 1) / 2.0,
            (height - 1) / 2.0,
            min(height, width) * 0.42,
        )
        phantom_detection_status = "fallback_needs_review"
    rough_side = max(12.0, radius * 0.14)
    crop_side = int(round(max(110.0, min(140.0, radius * 0.52))))
    slot_definitions = (
        ("B8", "upper_left", -135.0),
        ("B7", "left", 180.0),
        ("B6", "lower_left", 135.0),
        ("B5", "bottom", 90.0),
        ("B4", "lower_right", 45.0),
        ("B1", "top", -90.0),
        ("B2", "upper_right", -45.0),
        ("B3", "right", 0.0),
    )
    target_localizations = []
    for target_id, target_slot, sector_angle in slot_definitions:
        angle_radians = math.radians(sector_angle)
        ideal_center = (
            cx + radius * 0.46 * math.cos(angle_radians),
            cy + radius * 0.46 * math.sin(angle_radians),
        )
        target_localizations.append(localize_module4_target(
            raw,
            phantom_center=(cx, cy),
            phantom_radius=radius,
            ideal_center=ideal_center,
            expected_side=rough_side,
            target_id=target_id,
            target_slot=target_slot,
            sector_angle_degrees=sector_angle,
            sector_half_width_degrees=(
                50.0
                if target_id == "B6"
                else 45.0
                if target_id in {"B8", "B7", "B5"}
                else 42.0
                if target_id == "B4"
                else 34.0
            ),
            radial_minimum=(
                0.16
                if target_id == "B6"
                else 0.20
                if target_id in {"B8", "B7", "B5"}
                else 0.23
            ),
            radial_maximum=(
                0.84
                if target_id == "B6"
                else 0.80
                if target_id in {"B8", "B7", "B5"}
                else 0.76
            ),
            minimum_aspect_ratio=0.26 if target_id == "B6" else 0.34,
        ))

    # Adjacent angular sectors overlap intentionally. Assign accepted
    # slot/component proposals globally so a losing slot can try its next-best
    # component instead of disappearing after one duplicate conflict.
    proposals = []
    for localization in target_localizations:
        for candidate in localization["sector_candidates_considered"]:
            if candidate["accepted"]:
                proposals.append({
                    "localization": localization,
                    "candidate": candidate,
                    "score": candidate["localization_score"],
                })
        localization["success"] = False
        localization["selected_candidate"] = None
        localization["detected_component_center"] = None
        localization["duplicate_component_assignment"] = None
    claimed_centers = []
    assigned_targets = set()
    duplicate_conflicts = {item["target_id"]: [] for item in target_localizations}
    assignment_priority = {
        "B8": 3, "B7": 3, "B6": 3, "B5": 3,
        "B4": 2,
        "B1": 1, "B2": 1, "B3": 1,
    }
    for proposal in sorted(
        proposals,
        key=lambda item: (
            assignment_priority[item["localization"]["target_id"]],
            -item["candidate"]["sector_angular_error_degrees"],
            item["score"],
        ),
        reverse=True,
    ):
        localization = proposal["localization"]
        target_id = localization["target_id"]
        if target_id in assigned_targets:
            continue
        center = proposal["candidate"]["center"]
        duplicate = next((
            claim for claim in claimed_centers
            if math.hypot(
                center["x"] - claim["center"]["x"],
                center["y"] - claim["center"]["y"],
            ) < rough_side * 0.65
        ), None)
        if duplicate:
            duplicate_conflicts[target_id].append(duplicate["target_id"])
            continue
        localization["success"] = True
        localization["selected_candidate"] = proposal["candidate"]
        localization["detected_component_center"] = center
        localization["final_crop_center"] = center
        localization["selected_localization_reason"] = (
            "Selected the strongest non-duplicate component assigned to "
            f"the {localization['target_slot']} sector."
        )
        assigned_targets.add(target_id)
        claimed_centers.append({"target_id": target_id, "center": center})
    for localization in target_localizations:
        conflicts = sorted(set(duplicate_conflicts[localization["target_id"]]))
        if conflicts:
            localization["duplicate_component_assignment"] = {
                "conflicting_target_ids": conflicts,
                "reason": (
                    "Higher-scoring slots claimed overlapping components; "
                    "remaining candidates were considered."
                ),
            }
        if not localization["success"]:
            localization["final_crop_center"] = localization[
                "original_ideal_rough_center"
            ]
            localization["selected_localization_reason"] = (
                "No unique accepted component remained; slot-centered "
                "template fallback will be attempted."
            )

    candidates = []
    target_debug = []
    missing_targets = []
    weak_targets = []
    detected_targets = []
    slot_lifecycle = []
    for localization in target_localizations:
        target_id = localization["target_id"]
        target_slot = localization["target_slot"]
        target_crop_side = max(crop_side, 160) if target_id == "B6" else crop_side
        crop_center = localization["final_crop_center"]
        if localization["success"]:
            block_debug = fit_single_module4_block_roi(
                raw,
                phantom_center=(cx, cy),
                phantom_radius=radius,
                rough_center=(float(crop_center["x"]), float(crop_center["y"])),
                rough_expected_side=rough_side,
                rough_expected_angle=45.0,
                target_id=target_id,
                target_slot=target_slot,
                crop_side_override=target_crop_side,
                localization_debug=localization,
            )
            template_mode = "localized_component"
        else:
            block_debug = fit_module4_slot_template_fallback(
                raw,
                phantom_center=(cx, cy),
                phantom_radius=radius,
                expected_center=(float(crop_center["x"]), float(crop_center["y"])),
                expected_side=rough_side,
                crop_side=target_crop_side,
                target_id=target_id,
                target_slot=target_slot,
            )
            block_debug["localization"] = localization
            template_mode = "slot_centered_fallback"
        candidate = block_debug.get("candidate")
        if candidate is None:
            missing_targets.append(target_id)
        else:
            candidate["analysis_priority"] = (
                "primary"
                if target_id in {"B8", "B7", "B6", "B5"}
                else "secondary"
                if target_id == "B4"
                else "reference"
            )
            candidate["geometry_status"] = "approximate_review_roi"
            candidate["normal_overlay_allowed"] = (
                target_id in {"B8", "B7", "B6", "B5", "B4"}
                or debug_show_reference_targets
            )
            # In preliminary-review mode, a localized component is sufficient
            # to show an approximate guide even when the perfect-square fit
            # remains review quality. Slot-only fallbacks stay hidden.
            if localization["success"]:
                candidate["draw_on_overlay"] = bool(
                    candidate["normal_overlay_allowed"]
                    and (
                    candidate.get("mask_coverage_score") is not None
                    and candidate["mask_coverage_score"] >= 0.55
                    and candidate.get("interior_fill_score", 0.0) >= 0.15
                    and candidate.get("outside_leakage_penalty", 1.0) <= 0.65
                    )
                )
                if target_id == "B6":
                    b6_standard_draw_gate = candidate["draw_on_overlay"]
                    candidate["draw_on_overlay"] = True
                    if not b6_standard_draw_gate:
                        candidate["needs_review"] = True
                        candidate["reason"] = (
                            "B6 localized; preliminary geometry needs review. "
                            f'Original template assessment: {candidate["reason"]}'
                        )
            else:
                candidate["draw_on_overlay"] = False
            candidates.append(candidate)
            if candidate["needs_review"]:
                weak_targets.append(target_id)
            else:
                detected_targets.append(target_id)
        if (
            target_id not in {"B1", "B2", "B3", "B6"}
            and candidate
            and not candidate["needs_review"]
        ):
            block_debug["local_crop_image"] = None
            block_debug["threshold_mask_image"] = None
            block_debug["merged_mask_image"] = None
            block_debug["fitted_crop_image"] = None
            localization["localization_mask_image"] = None
        target_debug.append(block_debug)
        accepted_candidates = [
            item for item in localization["sector_candidates_considered"]
            if item["accepted"]
        ]
        final_status = (
            "missing"
            if candidate is None
            else "needs_review"
            if candidate["needs_review"]
            else "detected"
        )
        slot_lifecycle.append({
            "id": target_id,
            "target_slot": target_slot,
            "slot_attempted": True,
            "expected_center": localization["original_ideal_rough_center"],
            "sector_center_angle_deg": localization["sector_angle_degrees"],
            "sector_range_deg": [
                localization["sector_angle_degrees"]
                - localization["sector_half_width_degrees"],
                localization["sector_angle_degrees"]
                + localization["sector_half_width_degrees"],
            ],
            "radial_range": localization["radial_search_range"],
            "raw_candidates_found": localization["candidate_count"],
            "filtered_candidates_found": len(accepted_candidates),
            "candidate_rejection_reasons": [
                {
                    "center": item["center"],
                    "reason": item["rejection_reason"],
                }
                for item in localization["sector_candidates_considered"]
                if not item["accepted"]
            ],
            "selected_component_found": localization["success"],
            "selected_component_center": localization.get(
                "detected_component_center"
            ),
            "selected_component_reason": localization[
                "selected_localization_reason"
            ],
            "template_attempted": True,
            "template_mode": template_mode,
            "template_score": (
                candidate.get("template_score") if candidate else None
            ),
            "draw_on_overlay": bool(
                candidate and candidate.get("draw_on_overlay", False)
            ),
            "normal_overlay_allowed": (
                target_id in {"B8", "B7", "B6", "B5", "B4"}
                or debug_show_reference_targets
            ),
            "template_fit_quality": (
                "review"
                if candidate and candidate["needs_review"]
                else "accepted"
                if candidate
                else "insufficient"
            ),
            "final_status": final_status,
            "final_reason": (
                candidate["reason"]
                if candidate
                else block_debug.get("failure_reason")
            ),
            "duplicate_conflict_with": (
                localization.get("duplicate_component_assignment") or {}
            ).get("conflicting_target_ids"),
        })

    good_target_count = sum(not item["needs_review"] for item in candidates)
    candidate_by_id = {item["id"]: item for item in candidates}
    primary_ids = ("B8", "B7", "B6", "B5")
    all_primary_localized = all(
        candidate_by_id.get(target_id, {}).get("selected_component_center")
        is not None
        for target_id in primary_ids
    )
    b4_candidate = candidate_by_id.get("B4")
    if b4_candidate and not all_primary_localized:
        b4_candidate["draw_on_overlay"] = False
        b4_candidate["secondary_overlay_deferred"] = True
        b4_candidate["reason"] = (
            f'{b4_candidate["reason"]} Secondary B4 overlay was deferred '
            "because at least one primary target was not localized."
        )
        for lifecycle in slot_lifecycle:
            if lifecycle["id"] == "B4":
                lifecycle["draw_on_overlay"] = False
                lifecycle["final_reason"] = b4_candidate["reason"]
                lifecycle["secondary_overlay_deferred"] = True
                break
    b6_lifecycle = next(
        item for item in slot_lifecycle if item["id"] == "B6"
    )
    b6_target_debug = next(
        (
            item for item in target_debug
            if item.get("debug_target_id") == "B6"
            or item.get("target_id") == "B6"
        ),
        {},
    )
    if b6_lifecycle["selected_component_found"]:
        if b6_lifecycle["draw_on_overlay"]:
            b6_diagnosis = (
                "B6 localized to a real sector component and is drawn as an "
                "approximate preliminary-review ROI."
            )
        else:
            b6_diagnosis = (
                "B6 localized, but local component/template fitting did not "
                f'produce a drawable ROI: {b6_lifecycle["final_reason"]}'
            )
    elif b6_lifecycle["raw_candidates_found"] == 0:
        b6_diagnosis = (
            "B6 sector thresholding found no high-contrast components."
        )
    elif b6_lifecycle["filtered_candidates_found"] == 0:
        rejection_counts = {}
        for rejection in b6_lifecycle["candidate_rejection_reasons"]:
            reason = rejection["reason"] or "unknown"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        b6_diagnosis = (
            f'B6 found {b6_lifecycle["raw_candidates_found"]} raw candidates, '
            f"but all were rejected: {rejection_counts}."
        )
    elif b6_lifecycle["duplicate_conflict_with"]:
        b6_diagnosis = (
            "B6 accepted candidates existed but conflicted with already "
            "assigned physical components: "
            f'{b6_lifecycle["duplicate_conflict_with"]}.'
        )
    else:
        b6_diagnosis = (
            "B6 accepted component evidence existed but no unique component "
            f'was selected: {b6_lifecycle["selected_component_reason"]}'
        )
    b6_focused_debug = {
        **b6_lifecycle,
        "diagnosis": b6_diagnosis,
        "all_sector_candidates": next(
            item["sector_candidates_considered"]
            for item in target_localizations
            if item["target_id"] == "B6"
        ),
        "crop_image": b6_target_debug.get("local_crop_image"),
        "threshold_mask_image": b6_target_debug.get("threshold_mask_image"),
        "merged_mask_image": b6_target_debug.get("merged_mask_image"),
        "fitted_crop_image": b6_target_debug.get("fitted_crop_image"),
        "template_trials": b6_target_debug.get("top_template_trials", []),
        "best_template_score": b6_target_debug.get("best_template_score"),
    }
    primary_reviewed = sum(
        target_id in candidate_by_id for target_id in primary_ids
    )
    primary_visible = sum(
        candidate_by_id.get(target_id, {}).get("preliminary_visibility")
        == "visible"
        for target_id in primary_ids
    )
    primary_needing_review = sum(
        target_id not in candidate_by_id
        or candidate_by_id[target_id].get("needs_review", True)
        or candidate_by_id[target_id].get("preliminary_visibility")
        in {"weak", "needs_review"}
        for target_id in primary_ids
    )
    draft_result_order = ("B8", "B7", "B6", "B5", "B4", "B1", "B2", "B3")
    priority_by_id = {
        target_id: (
            "primary"
            if target_id in primary_ids
            else "secondary"
            if target_id == "B4"
            else "reference"
        )
        for target_id in draft_result_order
    }
    slot_by_id = {
        target_id: target_slot
        for target_id, target_slot, _ in slot_definitions
    }
    draft_roi_results = []
    for target_id in draft_result_order:
        if target_id in candidate_by_id:
            draft_roi_results.append(candidate_by_id[target_id])
        else:
            lifecycle = next(
                item for item in slot_lifecycle if item["id"] == target_id
            )
            draft_roi_results.append({
                "id": target_id,
                "analysis_priority": priority_by_id[target_id],
                "target_slot": slot_by_id[target_id],
                "roi_available": False,
                "roi_score": None,
                "draft_status": "missing",
                "preliminary_pass_fail": "missing",
                "geometry_status": "missing",
                "draw_on_overlay": False,
                "normal_overlay_allowed": lifecycle[
                    "normal_overlay_allowed"
                ],
                "reason": lifecycle["final_reason"]
                or "Target ROI not available.",
                "formal_measurement_status": "pending",
            })
    primary_draft_results = [
        item for item in draft_roi_results
        if item["id"] in primary_ids
    ]
    primary_targets_scored = sum(
        item["draft_status"] != "missing" for item in primary_draft_results
    )
    primary_targets_passed = sum(
        item["draft_status"] == "draft_pass" for item in primary_draft_results
    )
    primary_targets_failed = sum(
        item["draft_status"] == "draft_fail" for item in primary_draft_results
    )
    primary_targets_missing = sum(
        item["draft_status"] == "missing" for item in primary_draft_results
    )
    primary_targets_review = sum(
        item["draft_status"] == "needs_review"
        for item in primary_draft_results
    )
    overall_draft_status = (
        "needs_review"
        if primary_targets_missing
        else "draft_pass"
        if primary_targets_passed == 4
        else "draft_fail"
        if primary_targets_failed >= 2
        else "needs_review"
    )
    overall_draft_result = {
        "overall_draft_status": overall_draft_status,
        "primary_targets_scored": primary_targets_scored,
        "primary_targets_passed": primary_targets_passed,
        "primary_targets_failed": primary_targets_failed,
        "primary_targets_review": primary_targets_review,
        "primary_targets_missing": primary_targets_missing,
        "formal_measurement_status": "pending",
        "formal_measurement_note": (
            "Draft ROI scoring only. Formal lp/cm measurement is not "
            "implemented."
        ),
    }
    if good_target_count >= 7:
        analysis_review_status = "detected"
    elif good_target_count >= 4:
        analysis_review_status = "partial"
    elif candidates:
        analysis_review_status = "needs_review"
    else:
        analysis_review_status = "not_found"
    display_window = _select_module4_display_window(raw, cx, cy, radius)
    auto_width = display_window["window_width"]
    auto_level = display_window["window_level"]
    overlay = (
        None
        if not candidates
        else generate_module4_block_overlay(
            slice_pixels,
            candidates,
            window_width=auto_width,
            window_level=auto_level,
            photometric=photometric,
            title="Module 4 draft ROI scoring - formal measurement pending",
        )
    )
    return {
        "candidates": candidates,
        "phantom_center": {"x": round(cx, 2), "y": round(cy, 2)},
        "phantom_radius": round(radius, 2),
        "phantom_detection_status": phantom_detection_status,
        "analysis_review_status": analysis_review_status,
        "square_detection_status": analysis_review_status,
        "squares_detected": len(candidates),
        "square_candidates": candidates,
        "missing_targets": missing_targets,
        "weak_targets": weak_targets,
        "detected_targets": detected_targets,
        "slot_lifecycle": slot_lifecycle,
        "stage_summary": {
            "slots_attempted": 8,
            "slots_localized": sum(
                item["selected_component_found"] for item in slot_lifecycle
            ),
            "slots_template_fit_attempted": sum(
                item["template_attempted"] for item in slot_lifecycle
            ),
            "slots_accepted": len(detected_targets),
            "slots_weak": len(weak_targets),
            "slots_missing": len(missing_targets),
        },
        "measurement_status": "pending",
        "formal_measurement_status": "pending",
        "draft_roi_results": draft_roi_results,
        "overall_draft_result": overall_draft_result,
        "preliminary_review_summary": {
            "primary_targets": list(primary_ids),
            "primary_targets_reviewed": primary_reviewed,
            "primary_targets_visible": primary_visible,
            "primary_targets_needing_review": primary_needing_review,
            "all_primary_targets_localized": all_primary_localized,
            "measurement_status": "pending",
            "note": (
                "Preliminary Module 4 review metrics. Formal line-pair "
                "measurement is not implemented yet."
            ),
        },
        "all_block_debug_enabled": True,
        "debug_show_reference_targets": debug_show_reference_targets,
        "target_processing_order": [
            target_id for target_id, _, _ in slot_definitions
        ],
        "target_debug": target_debug,
        "b6_focused_debug": b6_focused_debug,
        "thresholds": {
            "per_target": [
                {
                    "target_id": item.get("debug_target_id"),
                    "local_residual": item.get("local_threshold"),
                    "noise_sigma": item.get("noise_sigma"),
                }
                for item in target_debug
                if item.get("local_threshold") is not None
            ],
        },
        "components_considered": sum(
            item.get("components_after_merge", 0) for item in target_debug
        ),
        "components_rejected": sum(
            max(0, item.get("components_after_merge", 0) - int(bool(item.get("candidate"))))
            for item in target_debug
        ),
        "merge_groups": [],
        "merge_kernel_size": None,
        "median_block_side_px": (
            round(float(np.median([item["side_px"] for item in candidates])), 3)
            if candidates else None
        ),
        "pixel_spacing": {
            "row_mm": pixel_spacing[0] if pixel_spacing else None,
            "column_mm": pixel_spacing[1] if pixel_spacing else None,
            "used_for_detection": False,
        },
        "overlay_image": overlay,
        "selected_slice_image": image_to_base64(
            window_pixels_to_image(
                slice_pixels,
                auto_width,
                auto_level,
                photometric,
            )
        ),
        "window_width": round(auto_width, 2),
        "window_level": round(auto_level, 2),
        "display_window_method": display_window["window_method"],
        "display_window_quality_score": display_window["window_quality_score"],
        "display_window_candidates": display_window["candidate_windows"],
        "display_window_note": display_window["window_note"],
    }


def detect_module4_phantom_geometry(slice_pixels: np.ndarray) -> dict:
    """Detect the Module 4 phantom body from raw CT/HU pixels."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(
            f"Module 4 phantom detection requires 2D CT pixels, got {raw.shape}."
        )
    height, width = raw.shape
    finite = np.isfinite(raw)
    if int(np.sum(finite)) < 100:
        return {
            "phantom_center": {"x": (width - 1) / 2.0, "y": (height - 1) / 2.0},
            "phantom_radius": min(height, width) * 0.42,
            "phantom_area": 0,
            "phantom_detection_confidence": 0.0,
            "phantom_detection_status": "needs_review",
            "phantom_detection_method": "raw_hu_largest_connected_phantom_body",
            "phantom_failure_reason": "Not enough finite raw CT pixels.",
        }

    border_width = max(2, int(round(min(height, width) * 0.06)))
    border_values = np.concatenate((
        raw[:border_width].ravel(),
        raw[-border_width:].ravel(),
        raw[:, :border_width].ravel(),
        raw[:, -border_width:].ravel(),
    ))
    border_values = border_values[np.isfinite(border_values)]
    central_values = raw[
        int(height * 0.25):int(height * 0.75),
        int(width * 0.25):int(width * 0.75),
    ]
    central_values = central_values[np.isfinite(central_values)]
    if not border_values.size or not central_values.size:
        return {
            "phantom_center": {"x": (width - 1) / 2.0, "y": (height - 1) / 2.0},
            "phantom_radius": min(height, width) * 0.42,
            "phantom_area": 0,
            "phantom_detection_confidence": 0.0,
            "phantom_detection_status": "needs_review",
            "phantom_detection_method": "raw_hu_largest_connected_phantom_body",
            "phantom_failure_reason": "Background or central phantom samples were empty.",
        }

    background_hu = float(np.median(border_values))
    interior_hu = float(np.median(central_values))
    contrast_hu = abs(interior_hu - background_hu)
    threshold_hu = background_hu + (interior_hu - background_hu) * 0.35
    body = finite & (
        (raw >= threshold_hu) if interior_hu >= background_hu
        else (raw <= threshold_hu)
    )
    close_size = max(3, int(round(min(height, width) * 0.012)) | 1)
    body = ndimage.binary_closing(
        body,
        structure=np.ones((close_size, close_size), dtype=bool),
    )
    body = ndimage.binary_fill_holes(body)
    labels, count = ndimage.label(body)
    if count < 1:
        component = np.zeros_like(body)
    else:
        sizes = np.asarray(
            ndimage.sum(body, labels, range(1, count + 1)),
            dtype=float,
        )
        component = labels == int(np.argmax(sizes) + 1)

    yy, xx = np.nonzero(component)
    area = int(xx.size)
    fallback_center = {"x": (width - 1) / 2.0, "y": (height - 1) / 2.0}
    fallback_radius = min(height, width) * 0.42
    if area < max(100, int(height * width * 0.03)):
        return {
            "phantom_center": fallback_center,
            "phantom_radius": round(fallback_radius, 3),
            "phantom_area": area,
            "phantom_detection_confidence": 0.0,
            "phantom_detection_status": "needs_review",
            "phantom_detection_method": "raw_hu_largest_connected_phantom_body",
            "phantom_failure_reason": "Largest thresholded component was too small.",
            "phantom_threshold_hu": round(threshold_hu, 3),
            "phantom_background_hu": round(background_hu, 3),
            "phantom_interior_hu": round(interior_hu, 3),
        }

    center_x = float(np.mean(xx))
    center_y = float(np.mean(yy))
    radius = float(math.sqrt(area / math.pi))
    frame_area_ratio = area / float(height * width)
    center_offset = math.hypot(
        center_x - (width - 1) / 2.0,
        center_y - (height - 1) / 2.0,
    ) / max(min(height, width), 1)
    expected_area_score = _clamp01(1.0 - abs(frame_area_ratio - 0.50) / 0.42)
    center_score = _clamp01(1.0 - center_offset / 0.28)
    contrast_score = _clamp01(contrast_hu / 500.0)
    confidence = _clamp01(
        0.45 * expected_area_score + 0.30 * center_score + 0.25 * contrast_score
    )
    status = "detected" if confidence >= 0.55 else "needs_review"
    reason = None if status == "detected" else (
        "Phantom body was found, but its area, centering, or background "
        "contrast was weaker than expected."
    )
    return {
        "phantom_center": {
            "x": round(center_x, 3),
            "y": round(center_y, 3),
        },
        "phantom_radius": round(radius, 3),
        "phantom_area": area,
        "phantom_detection_confidence": round(confidence, 4),
        "phantom_detection_status": status,
        "phantom_detection_method": "raw_hu_largest_connected_phantom_body",
        "phantom_failure_reason": reason,
        "phantom_threshold_hu": round(threshold_hu, 3),
        "phantom_background_hu": round(background_hu, 3),
        "phantom_interior_hu": round(interior_hu, 3),
    }


def detect_module4_bottom_pins(
    slice_pixels: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
) -> dict:
    """Detect two compact high-contrast anchors in the lower phantom arc."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    center_x = float(phantom_center["x"])
    center_y = float(phantom_center["y"])
    radius = max(float(phantom_radius), 1.0)
    yy, xx = np.indices(raw.shape)
    finite = np.isfinite(raw)
    radial = np.hypot(xx - center_x, yy - center_y) / radius
    angle = (np.degrees(np.arctan2(yy - center_y, xx - center_x)) + 360.0) % 360.0
    search_mask = (
        finite
        & (radial >= 0.55)
        & (radial <= 0.96)
        & (angle >= 48.0)
        & (angle <= 132.0)
    )
    method = "raw_hu_lower_peripheral_compact_blob_pair"
    if int(np.sum(search_mask)) < 50:
        return {
            "bottom_pin_candidates": [],
            "selected_bottom_pins": [],
            "pin_midpoint": None,
            "pin_detection_confidence": 0.0,
            "pin_detection_status": "failed",
            "pin_detection_method": method,
            "pin_failure_reason": "Lower-peripheral phantom search region was empty.",
        }

    fill = float(np.median(raw[finite]))
    working = np.where(finite, raw, fill)
    filter_size = max(5, int(round(radius * 0.045)) | 1)
    local_background = ndimage.median_filter(working, size=filter_size)
    residual = working - local_background
    search_residual = residual[search_mask]
    residual_median = float(np.median(search_residual))
    noise_sigma = max(
        1.0,
        float(np.median(np.abs(search_residual - residual_median)) * 1.4826),
    )
    residual_threshold = max(
        residual_median + 3.5 * noise_sigma,
        float(np.percentile(search_residual, 96.5)),
    )
    bright_mask = search_mask & (residual >= residual_threshold)
    labels, count = ndimage.label(bright_mask)
    objects = ndimage.find_objects(labels)
    minimum_area = max(2, int(round(radius * radius * 0.00004)))
    maximum_area = max(minimum_area + 1, int(round(radius * radius * 0.0030)))
    candidates = []

    for label_id in range(1, count + 1):
        bounds = objects[label_id - 1]
        if bounds is None:
            continue
        region = labels[bounds] == label_id
        area = int(np.sum(region))
        local_y, local_x = np.nonzero(region)
        if not local_x.size:
            continue
        x = float(bounds[1].start + np.mean(local_x))
        y = float(bounds[0].start + np.mean(local_y))
        radial_ratio = math.hypot(x - center_x, y - center_y) / radius
        angle_degrees = (
            math.degrees(math.atan2(y - center_y, x - center_x)) + 360.0
        ) % 360.0
        height, width = region.shape
        aspect = min(width, height) / max(width, height)
        contrast = float(np.median(residual[bounds][region]))
        rejection_reasons = []
        if area < minimum_area:
            rejection_reasons.append("too_small")
        if area > maximum_area:
            rejection_reasons.append("too_large_insert_like")
        if aspect < 0.45:
            rejection_reasons.append("not_compact")
        if not 0.55 <= radial_ratio <= 0.96:
            rejection_reasons.append("outside_lower_peripheral_radial_range")
        if not 48.0 <= angle_degrees <= 132.0:
            rejection_reasons.append("outside_bottom_sector")
        compact_score = _clamp01((aspect - 0.35) / 0.65)
        radial_score = _clamp01(1.0 - abs(radial_ratio - 0.78) / 0.23)
        bottom_score = _clamp01(1.0 - _angle_error(angle_degrees, 90.0) / 44.0)
        contrast_score = _clamp01((contrast / noise_sigma - 2.5) / 8.0)
        area_center = math.sqrt(minimum_area * maximum_area)
        area_score = _clamp01(
            1.0 - abs(math.log(max(area, 1) / max(area_center, 1.0))) / 2.5
        )
        candidate_score = _clamp01(
            0.25 * compact_score
            + 0.22 * radial_score
            + 0.18 * bottom_score
            + 0.22 * contrast_score
            + 0.13 * area_score
        )
        candidates.append({
            "x": round(x, 3),
            "y": round(y, 3),
            "area": area,
            "contrast": round(contrast, 3),
            "radial_ratio": round(radial_ratio, 4),
            "angle_degrees": round(angle_degrees, 3),
            "candidate_score": round(candidate_score, 4),
            "selected": False,
            "rejection_reason": (
                ", ".join(rejection_reasons) if rejection_reasons else None
            ),
        })

    eligible = [
        candidate for candidate in candidates
        if candidate["rejection_reason"] is None
    ]
    pair_rankings = []
    for first_index, first in enumerate(eligible):
        for second in eligible[first_index + 1:]:
            separation = math.hypot(
                second["x"] - first["x"],
                second["y"] - first["y"],
            ) / radius
            if not 0.06 <= separation <= 0.42:
                continue
            midpoint_x = (first["x"] + second["x"]) / 2.0
            midpoint_y = (first["y"] + second["y"]) / 2.0
            midpoint_angle = (
                math.degrees(math.atan2(
                    midpoint_y - center_y,
                    midpoint_x - center_x,
                )) + 360.0
            ) % 360.0
            symmetry = _clamp01(
                1.0 - abs(
                    _angle_error(first["angle_degrees"], 90.0)
                    - _angle_error(second["angle_degrees"], 90.0)
                ) / 28.0
            )
            vertical_alignment = _clamp01(
                1.0 - abs(first["y"] - second["y"]) / (radius * 0.12)
            )
            midpoint_bottom = _clamp01(
                1.0 - _angle_error(midpoint_angle, 90.0) / 24.0
            )
            separation_score = _clamp01(1.0 - abs(separation - 0.18) / 0.18)
            score = _clamp01(
                0.35 * ((first["candidate_score"] + second["candidate_score"]) / 2.0)
                + 0.22 * symmetry
                + 0.18 * vertical_alignment
                + 0.15 * midpoint_bottom
                + 0.10 * separation_score
            )
            pair_rankings.append((score, first, second, midpoint_x, midpoint_y))

    selected = []
    midpoint = None
    confidence = 0.0
    if pair_rankings:
        confidence, first, second, midpoint_x, midpoint_y = max(
            pair_rankings, key=lambda item: item[0]
        )
        if confidence >= 0.48:
            first["selected"] = True
            second["selected"] = True
            selected = [first, second]
            midpoint = {
                "x": round(midpoint_x, 3),
                "y": round(midpoint_y, 3),
            }
    for candidate in candidates:
        if not candidate["selected"] and candidate["rejection_reason"] is None:
            candidate["rejection_reason"] = "not_selected_by_best_pin_pair"

    if len(selected) == 2:
        status = "detected" if confidence >= 0.62 else "needs_review"
        failure_reason = (
            None if status == "detected"
            else "Two pins were selected, but pair confidence is marginal."
        )
    else:
        status = "failed"
        failure_reason = (
            "No eligible lower-peripheral component pair met the pin geometry "
            "and confidence requirements."
        )
    return {
        "bottom_pin_candidates": candidates,
        "selected_bottom_pins": selected,
        "pin_midpoint": midpoint,
        "pin_detection_confidence": round(float(confidence), 4),
        "pin_detection_status": status,
        "pin_detection_method": method,
        "pin_failure_reason": failure_reason,
        "pin_detection_thresholds": {
            "search_angle_degrees": [48.0, 132.0],
            "search_radial_ratio": [0.55, 0.96],
            "minimum_area": minimum_area,
            "maximum_area": maximum_area,
            "residual_threshold_hu": round(residual_threshold, 3),
            "noise_sigma_hu": round(noise_sigma, 3),
        },
    }


def calculate_module4_pin_angle(
    phantom_center: dict,
    pin_midpoint: dict | None,
    fallback_angle_degrees: float = MODULE4_FALLBACK_PIN_ANGLE_DEGREES,
) -> dict:
    """Return a screen-coordinate polar angle for later geometry placement."""
    fallback_used = pin_midpoint is None
    if fallback_used:
        angle_degrees = float(fallback_angle_degrees)
        orientation_status = "needs_review"
    else:
        angle_degrees = (
            math.degrees(math.atan2(
                float(pin_midpoint["y"]) - float(phantom_center["y"]),
                float(pin_midpoint["x"]) - float(phantom_center["x"]),
            )) + 360.0
        ) % 360.0
        orientation_status = "detected"
    return {
        "pin_angle_degrees": round(angle_degrees, 3),
        "orientation_status": orientation_status,
        "fallback_angle_used": fallback_used,
        "fallback_angle_degrees": float(fallback_angle_degrees),
        "angle_convention": (
            "Degrees clockwise in image coordinates: 0=right, 90=down, "
            "180=left, 270=up; compatible with x=cx+R*cos(theta), "
            "y=cy+R*sin(theta)."
        ),
        "y_axis_convention": "Image y increases downward.",
    }


def place_module4_geometry_rois(
    phantom_center: dict,
    phantom_radius: float,
    pin_angle_degrees: float,
    fallback_angle_used: bool,
    insert_radius_ratio: float = MODULE4_INSERT_RADIUS_RATIO,
    phi_offset_degrees: float = MODULE4_PHI_OFFSET_DEGREES,
    roi_side_ratio: float = MODULE4_ROI_SIDE_RATIO,
    roi_angle_offset_degrees: float = MODULE4_ROI_ANGLE_OFFSET_DEGREES,
    target_evidence_scores: dict[str, float] | None = None,
    calibration_status: str = "fallback",
) -> dict:
    """Place all eight Module 4 ROIs from pin-anchored phantom geometry."""
    center_x = float(phantom_center["x"])
    center_y = float(phantom_center["y"])
    radius = float(phantom_radius)
    insert_radius = float(insert_radius_ratio) * radius
    roi_side = float(roi_side_ratio) * radius
    orientation_adjustment = (
        float(pin_angle_degrees)
        - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
        + float(phi_offset_degrees)
    )
    roi_angle = (
        orientation_adjustment + float(roi_angle_offset_degrees)
    ) % 360.0
    definitions = (
        ("B1", "top", 270.0, "reference"),
        ("B2", "upper_right", 315.0, "reference"),
        ("B3", "right", 0.0, "reference"),
        ("B4", "lower_right", 45.0, "secondary"),
        ("B5", "bottom", 90.0, "primary"),
        ("B6", "lower_left", 135.0, "primary"),
        ("B7", "left", 180.0, "primary"),
        ("B8", "upper_left", 225.0, "primary"),
    )
    target_rois = []
    for target_id, target_slot, nominal_angle, priority in definitions:
        final_angle = (nominal_angle + orientation_adjustment) % 360.0
        theta = math.radians(final_angle)
        target_x = center_x + insert_radius * math.cos(theta)
        target_y = center_y + insert_radius * math.sin(theta)
        roi_corners = _square_points(
            target_x,
            target_y,
            roi_side,
            roi_angle,
        )
        inner_side = roi_side * 0.72
        inner_corners = _square_points(
            target_x,
            target_y,
            inner_side,
            roi_angle,
        )
        draw_on_overlay = True
        geometry_status = (
            "geometry_needs_review"
            if fallback_angle_used or calibration_status != "calibrated"
            else "geometry_placed"
        )
        evidence_score = (
            target_evidence_scores or {}
        ).get(target_id)
        if fallback_angle_used:
            reason = "ROI placed using fallback/review orientation; measurement pending."
        elif calibration_status != "calibrated":
            reason = "Geometry ROI placed using review/fallback calibration; measurement pending."
        elif evidence_score is not None and evidence_score < 0.12:
            reason = (
                f"Geometry ROI placed; {target_id} image evidence is weak. "
                "The ROI was retained and measurement remains pending."
            )
        else:
            reason = "ROI placed from globally calibrated phantom geometry; measurement pending."
        target_rois.append({
            "id": target_id,
            "nominal_lp_cm": MODULE4_NOMINAL_LP_CM_BY_ID[target_id],
            "display_label": (
                f"{MODULE4_NOMINAL_LP_CM_BY_ID[target_id]} lp/cm"
            ),
            "priority": priority,
            "analysis_priority": priority,
            "target_slot": target_slot,
            "geometry_source": "phantom_geometry_pin_anchored_roi",
            "center": {
                "x": round(target_x, 3),
                "y": round(target_y, 3),
            },
            "pin_angle_degrees": round(float(pin_angle_degrees), 3),
            "nominal_target_angle": nominal_angle,
            "final_target_angle": round(final_angle, 3),
            "angle_degrees": round(final_angle, 3),
            "phi_offset_degrees": round(float(phi_offset_degrees), 3),
            "roi_side_px": round(roi_side, 3),
            "roi_angle_degrees": round(roi_angle, 3),
            "roi_corners": roi_corners,
            "inner_roi_corners": inner_corners,
            "roi_available": True,
            "geometry_status": geometry_status,
            "evidence_score": (
                round(float(evidence_score), 4)
                if evidence_score is not None else None
            ),
            "reason": reason,
            "draw_on_overlay": draw_on_overlay,
            "normal_overlay_allowed": draw_on_overlay,
            # Compatibility fields used by the existing overlay helper.
            "rotated_box": roi_corners,
            "inner_roi": {
                "center": {
                    "x": round(target_x, 3),
                    "y": round(target_y, 3),
                },
                "width": round(inner_side, 3),
                "height": round(inner_side, 3),
                "angle_degrees": round(roi_angle, 3),
            },
        })
    return {
        "insert_radius_ratio": round(float(insert_radius_ratio), 4),
        "insert_radius_px": round(insert_radius, 3),
        "phi_offset_degrees": round(float(phi_offset_degrees), 3),
        "roi_side_ratio": round(float(roi_side_ratio), 4),
        "roi_side_px": round(roi_side, 3),
        "roi_angle_offset_degrees": round(
            float(roi_angle_offset_degrees), 3
        ),
        "roi_angle_degrees": round(roi_angle, 3),
        "orientation_adjustment_degrees": round(orientation_adjustment, 3),
        "target_centers": [
            {
                "id": target["id"],
                "target_slot": target["target_slot"],
                "center": target["center"],
                "nominal_target_angle": target["nominal_target_angle"],
                "final_target_angle": target["final_target_angle"],
            }
            for target in target_rois
        ],
        "target_rois": target_rois,
    }


def _module4_geometry_evidence_score(
    raw: np.ndarray,
    center_x: float,
    center_y: float,
    roi_side_px: float,
    phantom_scale_hu: float,
) -> float:
    """Score raw-HU structure near one geometry-predicted target center."""
    half = max(4, int(round(roi_side_px * 0.58)))
    x0 = max(0, int(round(center_x)) - half)
    x1 = min(raw.shape[1], int(round(center_x)) + half + 1)
    y0 = max(0, int(round(center_y)) - half)
    y1 = min(raw.shape[0], int(round(center_y)) + half + 1)
    patch = raw[y0:y1, x0:x1]
    finite = np.isfinite(patch)
    if patch.size < 49 or int(np.sum(finite)) < patch.size * 0.8:
        return 0.0
    values = patch[finite]
    p10, p90 = np.percentile(values, [10.0, 90.0])
    peak_to_valley = float(p90 - p10)
    fill = float(np.median(values))
    working = np.where(finite, patch, fill)
    smooth = ndimage.gaussian_filter(working, sigma=max(0.8, roi_side_px * 0.035))
    high_frequency = np.abs(working - smooth)
    positive_residual = np.maximum(working - smooth, 0.0)
    gradient_y, gradient_x = np.gradient(smooth)
    gradient = np.hypot(gradient_x, gradient_y)
    local_dynamic = _clamp01(peak_to_valley / max(phantom_scale_hu * 0.30, 1.0))
    texture = _clamp01(
        float(np.percentile(high_frequency, 90.0))
        / max(phantom_scale_hu * 0.08, 1.0)
    )
    edges = _clamp01(
        float(np.percentile(gradient, 85.0))
        / max(phantom_scale_hu * 0.12, 1.0)
    )
    # Reward structured variation, but require more than a single extreme pixel.
    occupied = _clamp01(
        float(np.mean(high_frequency >= np.percentile(high_frequency, 75.0))) / 0.25
    )
    patch_height, patch_width = working.shape
    middle_x = patch_width // 2
    middle_y = patch_height // 2
    left_evidence = float(np.mean(high_frequency[:, :max(middle_x, 1)]))
    right_evidence = float(np.mean(high_frequency[:, middle_x:]))
    top_evidence = float(np.mean(high_frequency[:max(middle_y, 1), :]))
    bottom_evidence = float(np.mean(high_frequency[middle_y:, :]))
    left_right_balance = _clamp01(
        1.0 - abs(left_evidence - right_evidence)
        / max(left_evidence + right_evidence, 1e-6)
    )
    top_bottom_balance = _clamp01(
        1.0 - abs(top_evidence - bottom_evidence)
        / max(top_evidence + bottom_evidence, 1e-6)
    )
    residual_threshold = float(np.percentile(positive_residual, 70.0))
    target_mask = positive_residual >= residual_threshold
    fill_score = _clamp01(float(np.mean(target_mask)) / 0.38)
    weights = np.where(target_mask, positive_residual, 0.0)
    if float(np.sum(weights)) > 0.0:
        local_y, local_x = np.indices(weights.shape)
        centroid_x = float(np.sum(local_x * weights) / np.sum(weights))
        centroid_y = float(np.sum(local_y * weights) / np.sum(weights))
        centroid_delta = math.hypot(
            centroid_x - (patch_width - 1) / 2.0,
            centroid_y - (patch_height - 1) / 2.0,
        )
        centroid_score = _clamp01(
            1.0 - centroid_delta / max(min(patch_height, patch_width) * 0.32, 1.0)
        )
    else:
        centroid_score = 0.0
    opposite_edge_symmetry = 0.5 * (
        left_right_balance + top_bottom_balance
    )
    return round(_clamp01(
        0.23 * local_dynamic
        + 0.15 * texture
        + 0.13 * edges
        + 0.05 * occupied
        + 0.11 * left_right_balance
        + 0.11 * top_bottom_balance
        + 0.10 * centroid_score
        + 0.06 * fill_score
        + 0.06 * opposite_edge_symmetry
    ), 5)


def _module4_local_target_center(
    raw: np.ndarray,
    geometry_center: dict,
    roi_side_px: float,
) -> dict:
    """Estimate a balanced local target centroid without moving the ROI."""
    center_x = float(geometry_center["x"])
    center_y = float(geometry_center["y"])
    half = max(8, int(round(roi_side_px * 0.72)))
    x0 = max(0, int(round(center_x)) - half)
    x1 = min(raw.shape[1], int(round(center_x)) + half + 1)
    y0 = max(0, int(round(center_y)) - half)
    y1 = min(raw.shape[0], int(round(center_y)) + half + 1)
    patch = raw[y0:y1, x0:x1]
    finite = np.isfinite(patch)
    if patch.size < 49 or int(np.sum(finite)) < patch.size * 0.8:
        return {
            "center": {"x": round(center_x, 3), "y": round(center_y, 3)},
            "method": "balanced_positive_residual_centroid",
            "confidence": 0.0,
            "score": 0.0,
            "reason": "Local target window contained insufficient finite pixels.",
        }
    fill = float(np.median(patch[finite]))
    working = np.where(finite, patch, fill)
    background = ndimage.gaussian_filter(
        working,
        sigma=max(2.0, roi_side_px * 0.22),
    )
    positive = np.maximum(working - background, 0.0)
    threshold = float(np.percentile(positive, 68.0))
    mask = positive >= threshold
    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    # Cap weights so one bright edge cannot pull the centroid by itself.
    cap = max(float(np.percentile(positive[mask], 88.0)), 1e-6)
    local_y, local_x = np.indices(positive.shape)
    patch_center_x = center_x - x0
    patch_center_y = center_y - y0
    distance = np.hypot(local_x - patch_center_x, local_y - patch_center_y)
    spatial_weight = np.exp(
        -0.5 * (distance / max(roi_side_px * 0.62, 1.0)) ** 2
    )
    weights = np.where(mask, np.minimum(positive, cap) * spatial_weight, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return {
            "center": {"x": round(center_x, 3), "y": round(center_y, 3)},
            "method": "balanced_positive_residual_centroid",
            "confidence": 0.0,
            "score": 0.0,
            "reason": "No balanced positive-residual support was found.",
        }
    centroid_x = float(x0 + np.sum(local_x * weights) / total)
    centroid_y = float(y0 + np.sum(local_y * weights) / total)
    delta = math.hypot(centroid_x - center_x, centroid_y - center_y)
    if delta > 12.0:
        scale = 12.0 / delta
        centroid_x = center_x + (centroid_x - center_x) * scale
        centroid_y = center_y + (centroid_y - center_y) * scale
        delta = 12.0
    left = float(np.sum(weights[:, :max(weights.shape[1] // 2, 1)]))
    right = float(np.sum(weights[:, weights.shape[1] // 2:]))
    top = float(np.sum(weights[:max(weights.shape[0] // 2, 1), :]))
    bottom = float(np.sum(weights[weights.shape[0] // 2:, :]))
    balance = 0.5 * (
        _clamp01(1.0 - abs(left - right) / max(left + right, 1e-6))
        + _clamp01(1.0 - abs(top - bottom) / max(top + bottom, 1e-6))
    )
    confidence = _clamp01(
        0.65 * balance
        + 0.35 * _clamp01(1.0 - delta / max(roi_side_px * 0.65, 1.0))
    )
    return {
        "center": {
            "x": round(centroid_x, 3),
            "y": round(centroid_y, 3),
        },
        "method": "balanced_positive_residual_centroid",
        "confidence": round(confidence, 4),
        "score": round(confidence, 4),
        "reason": (
            "Weighted centroid of capped positive raw-HU residual with "
            "spatial and opposite-side balance constraints; diagnostic "
            "displacement is capped at 12 pixels."
        ),
    }


def calibrate_module4_geometry_rois(
    slice_pixels: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
    pin_angle_degrees: float,
) -> dict:
    """Calibrate one rigid radius/phi model from raw-HU target evidence."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    center_x = float(phantom_center["x"])
    center_y = float(phantom_center["y"])
    radius = float(phantom_radius)
    roi_side = MODULE4_ROI_SIDE_RATIO * radius
    finite = np.isfinite(raw)
    yy, xx = np.indices(raw.shape)
    interior = finite & (np.hypot(xx - center_x, yy - center_y) <= radius * 0.88)
    interior_values = raw[interior]
    if interior_values.size < 200:
        return {
            "enabled": True,
            "method": "bounded_global_radius_phi_search",
            "default_insert_radius_ratio": MODULE4_INSERT_RADIUS_RATIO,
            "calibrated_insert_radius_ratio": MODULE4_INSERT_RADIUS_RATIO,
            "default_phi_offset_degrees": MODULE4_PHI_OFFSET_DEGREES,
            "calibrated_phi_offset_degrees": MODULE4_PHI_OFFSET_DEGREES,
            "search_radius_range": [0.55, 0.78],
            "search_phi_range": [-15.0, 15.0],
            "calibration_score": 0.0,
            "calibration_confidence": 0.0,
            "calibration_status": "fallback",
            "calibration_reason": "Insufficient finite phantom pixels for calibration.",
            "target_evidence_scores": {
                target_id: 0.0
                for target_id in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
            },
            "candidate_count": 0,
        }
    phantom_scale_hu = max(
        10.0,
        float(np.percentile(interior_values, 95.0) - np.percentile(interior_values, 5.0)),
    )
    target_angles = {
        "B8": 225.0,
        "B7": 180.0,
        "B6": 135.0,
        "B5": 90.0,
        "B4": 45.0,
    }
    base_orientation = (
        float(pin_angle_degrees) - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
    )
    trials = []
    for radius_step in range(55, 79):
        ratio = radius_step / 100.0
        insert_radius = ratio * radius
        for phi in range(-15, 16):
            scores = {}
            for target_id, nominal_angle in target_angles.items():
                theta = math.radians(nominal_angle + base_orientation + phi)
                target_x = center_x + insert_radius * math.cos(theta)
                target_y = center_y + insert_radius * math.sin(theta)
                scores[target_id] = _module4_geometry_evidence_score(
                    raw,
                    target_x,
                    target_y,
                    roi_side,
                    phantom_scale_hu,
                )
            ordered = sorted(scores.values())
            robust_score = float(np.mean(ordered[1:-1]))
            trials.append({
                "insert_radius_ratio": ratio,
                "phi_offset_degrees": float(phi),
                "score": robust_score,
                "target_scores": scores,
            })
    best = max(trials, key=lambda item: item["score"])
    trial_scores = np.asarray([trial["score"] for trial in trials], dtype=float)
    baseline = float(np.median(trial_scores))
    spread = max(float(np.std(trial_scores)), 0.01)
    advantage = _clamp01((best["score"] - baseline) / (2.5 * spread))
    absolute_quality = _clamp01((best["score"] - 0.10) / 0.35)
    confidence = _clamp01(0.55 * absolute_quality + 0.45 * advantage)
    if best["score"] >= 0.18 and confidence >= 0.35 and advantage >= 0.15:
        status = "calibrated"
        reason = (
            "Selected the rigid radius/phi hypothesis with the strongest "
            "trimmed-mean raw-HU evidence across B8, B7, B6, B5, and B4."
        )
        selected_ratio = best["insert_radius_ratio"]
        selected_phi = best["phi_offset_degrees"]
    elif best["score"] >= 0.10 and advantage >= 0.05:
        status = "needs_review"
        reason = (
            "A global radius/phi hypothesis was selected, but its absolute or "
            "relative image-evidence confidence is weak."
        )
        selected_ratio = best["insert_radius_ratio"]
        selected_phi = best["phi_offset_degrees"]
    else:
        status = "fallback"
        reason = (
            "Global calibration evidence was insufficient; default geometry "
            "constants were retained."
        )
        selected_ratio = MODULE4_INSERT_RADIUS_RATIO
        selected_phi = MODULE4_PHI_OFFSET_DEGREES
    all_target_angles = {
        "B1": 270.0,
        "B2": 315.0,
        "B3": 0.0,
        "B4": 45.0,
        "B5": 90.0,
        "B6": 135.0,
        "B7": 180.0,
        "B8": 225.0,
    }
    selected_scores = {}
    selected_radius = selected_ratio * radius
    for target_id, nominal_angle in all_target_angles.items():
        theta = math.radians(
            nominal_angle + base_orientation + selected_phi
        )
        selected_scores[target_id] = _module4_geometry_evidence_score(
            raw,
            center_x + selected_radius * math.cos(theta),
            center_y + selected_radius * math.sin(theta),
            roi_side,
            phantom_scale_hu,
        )
    return {
        "enabled": True,
        "method": "bounded_global_radius_phi_search",
        "default_insert_radius_ratio": MODULE4_INSERT_RADIUS_RATIO,
        "calibrated_insert_radius_ratio": round(float(selected_ratio), 4),
        "default_phi_offset_degrees": MODULE4_PHI_OFFSET_DEGREES,
        "calibrated_phi_offset_degrees": round(float(selected_phi), 3),
        "search_radius_range": [0.55, 0.78],
        "search_phi_range": [-15.0, 15.0],
        "search_radius_step": 0.01,
        "search_phi_step_degrees": 1.0,
        "calibration_score": round(float(best["score"]), 4),
        "calibration_confidence": round(float(confidence), 4),
        "calibration_status": status,
        "calibration_reason": reason,
        "target_evidence_scores": selected_scores,
        "candidate_count": len(trials),
        "robust_score_method": (
            "trimmed_mean_middle_three_of_B8_B7_B6_B5_B4"
        ),
    }


def calibrate_module4_geometry_fast(
    slice_pixels: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
    pin_angle_degrees: float,
) -> dict:
    """Run a capped radius-first calibration for normal interactive use."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    radius_values = np.append(np.arange(0.55, 0.78, 0.02), 0.78)
    phi_values = np.arange(-6.0, 6.01, 2.0)

    def robust_layout_score(
        radius_ratio: float,
        phi_offset: float,
    ) -> tuple[float, dict[str, float], list[str]]:
        scores = _module4_layout_target_evidence_scores(
            raw,
            phantom_center,
            phantom_radius,
            pin_angle_degrees,
            radius_ratio,
            phi_offset,
        )
        ordered = sorted(
            scores.items(), key=lambda item: item[1]
        )
        trimmed = ordered[1:-1] if len(ordered) > 4 else ordered
        robust_score = float(np.median([
            float(score) for _, score in trimmed
        ]))
        targets_used = [target_id for target_id, _ in trimmed]
        return robust_score, scores, targets_used

    radius_trials = []
    for ratio in radius_values:
        score, scores, targets_used = robust_layout_score(
            float(ratio), 0.0
        )
        radius_trials.append({
            "ratio": float(ratio),
            "score": score,
            "scores": scores,
            "targets_used": targets_used,
        })
    best_radius = max(radius_trials, key=lambda item: item["score"])

    phi_trials = []
    for phi in phi_values:
        score, scores, targets_used = robust_layout_score(
            best_radius["ratio"], float(phi)
        )
        phi_trials.append({
            "phi": float(phi),
            "score": score,
            "scores": scores,
            "targets_used": targets_used,
        })
    best_phi = max(phi_trials, key=lambda item: item["score"])
    all_trial_scores = np.asarray(
        [trial["score"] for trial in radius_trials]
        + [trial["score"] for trial in phi_trials],
        dtype=float,
    )
    baseline = float(np.median(all_trial_scores))
    spread = max(float(np.std(all_trial_scores)), 0.01)
    advantage = _clamp01(
        (float(best_phi["score"]) - baseline) / (2.5 * spread)
    )
    absolute_quality = _clamp01(
        (float(best_phi["score"]) - 0.10) / 0.35
    )
    confidence = _clamp01(
        0.55 * absolute_quality + 0.45 * advantage
    )
    selected_ratio = float(best_radius["ratio"])
    selected_phi = float(best_phi["phi"])
    return {
        "enabled": True,
        "method": "fast_radius_then_tiny_phi_raw_hu_evidence",
        "default_insert_radius_ratio": MODULE4_INSERT_RADIUS_RATIO,
        "calibrated_insert_radius_ratio": round(selected_ratio, 4),
        "fast_calibrated_insert_radius_ratio": round(selected_ratio, 4),
        "default_phi_offset_degrees": MODULE4_PHI_OFFSET_DEGREES,
        "calibrated_phi_offset_degrees": round(selected_phi, 3),
        "fast_calibrated_phi_offset_degrees": round(selected_phi, 3),
        "search_radius_range": [0.55, 0.78],
        "search_radius_step": 0.02,
        "search_phi_range": [-6.0, 6.0],
        "search_phi_step_degrees": 2.0,
        "calibration_score": round(float(best_phi["score"]), 4),
        "calibration_confidence": round(confidence, 4),
        "calibration_status": (
            "calibrated" if confidence >= 0.35 else "needs_review"
        ),
        "calibration_reason": (
            "Selected a radius from 13 capped hypotheses, then performed a "
            "seven-hypothesis phi check at that radius using robust median "
            "raw-HU evidence across all eight targets."
        ),
        "target_evidence_scores": best_phi["scores"],
        "candidate_count": len(radius_trials) + len(phi_trials),
        "robust_score_method": "trimmed_all_8_target_median",
        "fast_radius_calibration_enabled": True,
        "fast_radius_calibration_method": (
            "coarse_radius_search_then_tiny_phi_check"
        ),
        "fast_radius_search_range": [0.55, 0.78],
        "fast_radius_search_step": 0.02,
        "fast_radius_score": round(float(best_radius["score"]), 4),
        "fast_radius_confidence": round(confidence, 4),
        "fast_radius_targets_used": best_radius["targets_used"],
        "fast_phi_calibration_enabled": True,
    }


def correct_module4_geometry_from_primary_centers(
    slice_pixels: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
    provisional_rois: list[dict],
    pre_correction_radius_ratio: float,
    pre_correction_phi_degrees: float,
) -> dict:
    """Convert median primary centroid deltas into one bounded global correction."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    center_x = float(phantom_center["x"])
    center_y = float(phantom_center["y"])
    radius = max(float(phantom_radius), 1.0)
    primary_diagnostics = {}
    radial_deltas = []
    tangential_deltas = []
    for target in provisional_rois:
        if target["priority"] != "primary":
            continue
        local = _module4_local_target_center(
            raw,
            target["center"],
            target["roi_side_px"],
        )
        geometry_x = float(target["center"]["x"])
        geometry_y = float(target["center"]["y"])
        local_x = float(local["center"]["x"])
        local_y = float(local["center"]["y"])
        delta_x = local_x - geometry_x
        delta_y = local_y - geometry_y
        distance = max(math.hypot(
            geometry_x - center_x,
            geometry_y - center_y,
        ), 1.0)
        radial_x = (geometry_x - center_x) / distance
        radial_y = (geometry_y - center_y) / distance
        tangent_x, tangent_y = -radial_y, radial_x
        radial_delta = delta_x * radial_x + delta_y * radial_y
        tangential_delta = delta_x * tangent_x + delta_y * tangent_y
        radial_deltas.append(radial_delta)
        tangential_deltas.append(tangential_delta)
        primary_diagnostics[target["id"]] = {
            "original_geometry_center": target["center"],
            "local_target_center": local["center"],
            "local_target_center_method": local["method"],
            "local_target_center_confidence": local["confidence"],
            "local_target_score": local["score"],
            "target_center_confidence": local["confidence"],
            "local_target_score": local["score"],
            "target_center_confidence": local["confidence"],
            "center_to_local_target_delta_px": round(
                math.hypot(delta_x, delta_y), 3
            ),
            "radial_delta_px": round(radial_delta, 3),
            "tangential_delta_px": round(tangential_delta, 3),
            "reason": local["reason"],
        }
    median_radial = float(np.median(radial_deltas)) if radial_deltas else 0.0
    median_tangential = (
        float(np.median(tangential_deltas)) if tangential_deltas else 0.0
    )
    radius_correction_ratio = max(-0.05, min(0.05, median_radial / radius))
    insert_radius_px = max(pre_correction_radius_ratio * radius, 1.0)
    phi_correction = math.degrees(math.atan2(
        median_tangential,
        insert_radius_px,
    ))
    phi_correction = max(-6.0, min(6.0, phi_correction))
    return {
        "pre_correction_radius_ratio": round(
            float(pre_correction_radius_ratio), 4
        ),
        "post_correction_radius_ratio": round(
            float(pre_correction_radius_ratio + radius_correction_ratio), 4
        ),
        "radius_correction_px": round(
            radius_correction_ratio * radius, 3
        ),
        "radius_correction_ratio": round(radius_correction_ratio, 4),
        "pre_correction_phi_degrees": round(
            float(pre_correction_phi_degrees), 3
        ),
        "post_correction_phi_degrees": round(
            float(pre_correction_phi_degrees + phi_correction), 3
        ),
        "phi_correction_degrees": round(phi_correction, 3),
        "median_primary_radial_delta_px_before": round(median_radial, 3),
        "median_primary_tangential_delta_px_before": round(
            median_tangential, 3
        ),
        "primary_local_target_diagnostics": primary_diagnostics,
        "radius_correction_bound_ratio": 0.05,
        "phi_correction_bound_degrees": 6.0,
        "method": "median_primary_balanced_centroid_global_correction",
    }


def _module4_layout_target_evidence_scores(
    raw: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
    pin_angle_degrees: float,
    insert_radius_ratio: float,
    phi_offset_degrees: float,
    roi_side_ratio: float = MODULE4_ROI_SIDE_RATIO,
) -> dict[str, float]:
    """Evaluate all fixed target centers for one rigid geometry layout."""
    center_x = float(phantom_center["x"])
    center_y = float(phantom_center["y"])
    radius = float(phantom_radius)
    yy, xx = np.indices(raw.shape)
    finite = np.isfinite(raw)
    interior = finite & (np.hypot(xx - center_x, yy - center_y) <= radius * 0.88)
    values = raw[interior]
    phantom_scale_hu = (
        max(10.0, float(np.percentile(values, 95.0) - np.percentile(values, 5.0)))
        if values.size >= 200 else 100.0
    )
    orientation = (
        float(pin_angle_degrees)
        - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
        + float(phi_offset_degrees)
    )
    insert_radius = float(insert_radius_ratio) * radius
    scores = {}
    for target_id, nominal_angle in (
        ("B1", 270.0), ("B2", 315.0), ("B3", 0.0), ("B4", 45.0),
        ("B5", 90.0), ("B6", 135.0), ("B7", 180.0), ("B8", 225.0),
    ):
        theta = math.radians(nominal_angle + orientation)
        scores[target_id] = _module4_geometry_evidence_score(
            raw,
            center_x + insert_radius * math.cos(theta),
            center_y + insert_radius * math.sin(theta),
            float(roi_side_ratio) * radius,
            phantom_scale_hu,
        )
    return scores


def fit_module4_global_ring_geometry(
    slice_pixels: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
    pin_angle_degrees: float,
    current_radius_ratio: float,
    current_phi_degrees: float,
    current_rois: list[dict],
) -> dict:
    """Fit one robust ring center, radius, and phi to confident local centers."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    phantom_x = float(phantom_center["x"])
    phantom_y = float(phantom_center["y"])
    radius = max(float(phantom_radius), 1.0)
    center_limit = 0.08 * radius
    nominal_angles = {
        "B8": 225.0,
        "B7": 180.0,
        "B6": 135.0,
        "B5": 90.0,
        "B4": 45.0,
    }
    roi_by_id = {target["id"]: target for target in current_rois}
    local_diagnostics = {}
    used = []
    excluded = []
    for target_id in ("B8", "B7", "B6", "B5", "B4"):
        target = roi_by_id.get(target_id)
        if not target:
            excluded.append({
                "id": target_id,
                "reason": "Geometry ROI was unavailable for ring fitting.",
            })
            continue
        local = _module4_local_target_center(
            raw,
            target["center"],
            target["roi_side_px"],
        )
        minimum_confidence = 0.35 if target_id != "B4" else 0.45
        diagnostic = {
            "id": target_id,
            "priority": target["priority"],
            "pre_fit_center": target["center"],
            "local_target_center": local["center"],
            "local_target_score": local["score"],
            "target_center_confidence": local["confidence"],
            "local_target_center_method": local["method"],
        }
        local_diagnostics[target_id] = diagnostic
        if local["confidence"] >= minimum_confidence:
            used.append(diagnostic)
        else:
            excluded.append({
                "id": target_id,
                "reason": (
                    f"Local target-center confidence {local['confidence']:.3f} "
                    f"was below the {minimum_confidence:.2f} fit threshold."
                ),
            })

    def prediction_error(
        ring_x: float,
        ring_y: float,
        radius_ratio: float,
        phi_degrees: float,
        targets: list[dict],
    ) -> list[float]:
        insert_radius = radius_ratio * radius
        base_orientation = (
            float(pin_angle_degrees)
            - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
            + phi_degrees
        )
        errors = []
        for item in targets:
            theta = math.radians(
                nominal_angles[item["id"]] + base_orientation
            )
            predicted_x = ring_x + insert_radius * math.cos(theta)
            predicted_y = ring_y + insert_radius * math.sin(theta)
            local = item["local_target_center"]
            errors.append(math.hypot(
                predicted_x - float(local["x"]),
                predicted_y - float(local["y"]),
            ))
        return errors

    pre_errors = prediction_error(
        phantom_x,
        phantom_y,
        float(current_radius_ratio),
        float(current_phi_degrees),
        used,
    )
    pre_median = float(np.median(pre_errors)) if pre_errors else None
    primary_used_count = sum(
        item["priority"] == "primary" for item in used
    )
    if primary_used_count < 3:
        return {
            "enabled": True,
            "method": "global_ring_center_radius_phi_fit",
            "phantom_center": {
                "x": round(phantom_x, 3),
                "y": round(phantom_y, 3),
            },
            "calibrated_ring_center": {
                "x": round(phantom_x, 3),
                "y": round(phantom_y, 3),
            },
            "center_offset_x_px": 0.0,
            "center_offset_y_px": 0.0,
            "center_offset_magnitude_px": 0.0,
            "center_offset_limit_px": round(center_limit, 3),
            "radius_ratio_before": round(float(current_radius_ratio), 4),
            "radius_ratio_after": round(float(current_radius_ratio), 4),
            "phi_before": round(float(current_phi_degrees), 3),
            "phi_after": round(float(current_phi_degrees), 3),
            "pre_fit_median_center_error_px": (
                round(pre_median, 3) if pre_median is not None else None
            ),
            "post_fit_median_center_error_px": (
                round(pre_median, 3) if pre_median is not None else None
            ),
            "targets_used_for_fit": [item["id"] for item in used],
            "targets_excluded_from_fit": excluded,
            "fit_status": "fallback",
            "fit_reason": (
                "Fewer than three confident primary local target centers were "
                "available; B4 was not allowed to replace a missing primary "
                "and phantom-centered geometry was retained."
            ),
            "local_target_diagnostics": local_diagnostics,
        }

    best = None
    for radius_step in range(110, 157):
        ratio = radius_step / 200.0
        insert_radius = ratio * radius
        for phi_step in range(-30, 31):
            phi = phi_step / 2.0
            base_orientation = (
                float(pin_angle_degrees)
                - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
                + phi
            )
            center_x_votes = []
            center_y_votes = []
            for item in used:
                theta = math.radians(
                    nominal_angles[item["id"]] + base_orientation
                )
                local = item["local_target_center"]
                center_x_votes.append(
                    float(local["x"]) - insert_radius * math.cos(theta)
                )
                center_y_votes.append(
                    float(local["y"]) - insert_radius * math.sin(theta)
                )
            ring_x = float(np.median(center_x_votes))
            ring_y = float(np.median(center_y_votes))
            offset_x = max(-center_limit, min(center_limit, ring_x - phantom_x))
            offset_y = max(-center_limit, min(center_limit, ring_y - phantom_y))
            ring_x = phantom_x + offset_x
            ring_y = phantom_y + offset_y
            errors = prediction_error(ring_x, ring_y, ratio, phi, used)
            ordered = sorted(errors)
            median_error = float(np.median(ordered))
            trimmed_error = (
                float(np.mean(ordered[1:-1]))
                if len(ordered) > 3 else float(np.mean(ordered))
            )
            offset_magnitude = math.hypot(offset_x, offset_y)
            center_penalty = 0.20 * offset_magnitude / max(center_limit, 1.0)
            boundary_penalty = (
                0.25 if ratio in {0.55, 0.78} else 0.0
            ) + (0.25 if abs(phi) == 15.0 else 0.0)
            objective = (
                0.62 * median_error
                + 0.38 * trimmed_error
                + center_penalty
                + boundary_penalty
            )
            trial = {
                "ring_x": ring_x,
                "ring_y": ring_y,
                "ratio": ratio,
                "phi": phi,
                "errors": errors,
                "median_error": median_error,
                "objective": objective,
            }
            if best is None or trial["objective"] < best["objective"]:
                best = trial

    post_median = float(best["median_error"])
    improvement = (
        float(pre_median - post_median)
        if pre_median is not None else 0.0
    )
    offset_x = best["ring_x"] - phantom_x
    offset_y = best["ring_y"] - phantom_y
    if improvement >= 0.5:
        status = "applied"
        reason = (
            "Applied the robust rigid-ring fit because it reduced median "
            "confident-target center error by at least 0.5 pixels."
        )
    elif improvement > 0.1:
        status = "needs_review"
        reason = (
            "A bounded rigid-ring solution was found, but its median-error "
            "improvement was less than 0.5 pixels."
        )
    else:
        status = "fallback"
        reason = (
            "The bounded ring fit did not improve median center error by more "
            "than 0.1 pixels; the pre-fit rigid geometry was retained."
        )
        best = {
            **best,
            "ring_x": phantom_x,
            "ring_y": phantom_y,
            "ratio": float(current_radius_ratio),
            "phi": float(current_phi_degrees),
            "median_error": float(pre_median),
        }
        post_median = float(pre_median)
        offset_x = 0.0
        offset_y = 0.0
    return {
        "enabled": True,
        "method": "global_ring_center_radius_phi_fit",
        "phantom_center": {
            "x": round(phantom_x, 3),
            "y": round(phantom_y, 3),
        },
        "calibrated_ring_center": {
            "x": round(float(best["ring_x"]), 3),
            "y": round(float(best["ring_y"]), 3),
        },
        "center_offset_x_px": round(offset_x, 3),
        "center_offset_y_px": round(offset_y, 3),
        "center_offset_magnitude_px": round(
            math.hypot(offset_x, offset_y), 3
        ),
        "center_offset_limit_px": round(center_limit, 3),
        "radius_ratio_before": round(float(current_radius_ratio), 4),
        "radius_ratio_after": round(float(best["ratio"]), 4),
        "phi_before": round(float(current_phi_degrees), 3),
        "phi_after": round(float(best["phi"]), 3),
        "pre_fit_median_center_error_px": round(float(pre_median), 3),
        "post_fit_median_center_error_px": round(post_median, 3),
        "targets_used_for_fit": [item["id"] for item in used],
        "targets_excluded_from_fit": excluded,
        "fit_status": status,
        "fit_reason": reason,
        "local_target_diagnostics": local_diagnostics,
    }


def _module4_square_edge_metrics(
    gradient: np.ndarray,
    center: dict,
    side_px: float,
    angle_degrees: float,
    local_statistics: dict | None = None,
) -> dict:
    """Return normalized support for each edge and their balanced aggregate."""
    half = float(side_px) / 2.0
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    side_values = []
    side_samples = []
    samples = np.linspace(-half, half, 19)
    for fixed_axis, fixed_value in (("x", -half), ("x", half), ("y", -half), ("y", half)):
        xs, ys = [], []
        for sample in samples:
            dx, dy = (
                (fixed_value, sample)
                if fixed_axis == "x" else (sample, fixed_value)
            )
            xs.append(float(center["x"]) + dx * cosine - dy * sine)
            ys.append(float(center["y"]) + dx * sine + dy * cosine)
        values = ndimage.map_coordinates(
            gradient,
            [np.asarray(ys), np.asarray(xs)],
            order=1,
            mode="nearest",
        )
        side_samples.append(np.asarray(values, dtype=float))
        side_values.append(float(np.mean(values)))
    side_array = np.asarray(side_values, dtype=float)
    if local_statistics:
        local_floor = float(local_statistics["gradient_floor"])
        local_contrast = float(local_statistics["gradient_contrast"])
        local_significant_threshold = float(
            local_statistics["gradient_significant_threshold"]
        )
    else:
        local_radius = max(6, int(math.ceil(float(side_px) * 0.8)))
        center_x = int(round(float(center["x"])))
        center_y = int(round(float(center["y"])))
        x0 = max(0, center_x - local_radius)
        x1 = min(gradient.shape[1], center_x + local_radius + 1)
        y0 = max(0, center_y - local_radius)
        y1 = min(gradient.shape[0], center_y + local_radius + 1)
        local_gradient = np.asarray(gradient[y0:y1, x0:x1], dtype=float)
        finite_local = local_gradient[np.isfinite(local_gradient)]
        local_floor = (
            float(np.percentile(finite_local, 50.0))
            if finite_local.size else 0.0
        )
        local_high = (
            float(np.percentile(finite_local, 95.0))
            if finite_local.size else local_floor + 1.0
        )
        local_contrast = max(local_high - local_floor, 1e-6)
        local_significant_threshold = (
            float(np.percentile(finite_local, 78.0))
            if finite_local.size else local_high
        )
    relative_support = np.maximum(
        (side_array - local_floor) / local_contrast,
        0.0,
    )
    # Smooth ratio remains below one instead of clipping every strong edge.
    normalized_array = relative_support / (1.0 + relative_support)
    coverage_array = np.asarray([
        float(np.mean(values >= local_significant_threshold))
        for values in side_samples
    ])
    strength = float(np.mean(normalized_array))
    opposite_balance = 0.5 * (
        _clamp01(1.0 - abs(normalized_array[0] - normalized_array[1])
                 / max(normalized_array[0] + normalized_array[1], 1e-6))
        + _clamp01(1.0 - abs(normalized_array[2] - normalized_array[3])
                   / max(normalized_array[2] + normalized_array[3], 1e-6))
    )
    lower_quartile = float(np.percentile(normalized_array, 25.0))
    minimum = float(np.min(normalized_array))
    coverage_score = 0.5 * (
        float(np.mean(coverage_array)) + float(np.min(coverage_array))
    )
    boundary_alignment = _clamp01(
        0.35 * lower_quartile
        + 0.25 * minimum
        + 0.20 * strength
        + 0.20 * coverage_score
    )
    saturation_detected = bool(np.all(normalized_array >= 0.995))
    return {
        "left_edge_score_raw": round(float(side_array[0]), 5),
        "right_edge_score_raw": round(float(side_array[1]), 5),
        "top_edge_score_raw": round(float(side_array[2]), 5),
        "bottom_edge_score_raw": round(float(side_array[3]), 5),
        "left_edge_score": round(float(normalized_array[0]), 5),
        "right_edge_score": round(float(normalized_array[1]), 5),
        "top_edge_score": round(float(normalized_array[2]), 5),
        "bottom_edge_score": round(float(normalized_array[3]), 5),
        "left_boundary_coverage": round(float(coverage_array[0]), 5),
        "right_boundary_coverage": round(float(coverage_array[1]), 5),
        "top_boundary_coverage": round(float(coverage_array[2]), 5),
        "bottom_boundary_coverage": round(float(coverage_array[3]), 5),
        "boundary_coverage_score": round(coverage_score, 5),
        "min_edge_score": round(minimum, 5),
        "lower_quartile_edge_score": round(lower_quartile, 5),
        "edge_balance_score": round(opposite_balance, 5),
        "edge_score_saturation_detected": saturation_detected,
        "boundary_alignment_score": round(
            boundary_alignment, 5
        ),
    }


def _module4_square_boundary_alignment(
    gradient: np.ndarray,
    center: dict,
    side_px: float,
    angle_degrees: float,
) -> float:
    """Score balanced gradient support along four predicted square edges."""
    return float(_module4_square_edge_metrics_legacy(
        gradient, center, side_px, angle_degrees
    )["boundary_alignment_score"])


def _module4_angle_consistency_metrics(
    working: np.ndarray,
    gradient: np.ndarray,
    center: dict,
    side_px: float,
    angle_degrees: float,
) -> dict:
    """Evaluate one fixed-center/size angle for the final consistency pass."""
    edge_metrics = _module4_square_edge_metrics(
        gradient, center, side_px, angle_degrees
    )
    half_span = max(4, int(math.ceil(float(side_px) * 0.72)))
    yy, xx = np.mgrid[
        max(0, int(round(float(center["y"]))) - half_span):
        min(working.shape[0], int(round(float(center["y"]))) + half_span + 1),
        max(0, int(round(float(center["x"]))) - half_span):
        min(working.shape[1], int(round(float(center["x"]))) + half_span + 1),
    ]
    angle_radians = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    dx = xx - float(center["x"])
    dy = yy - float(center["y"])
    local_x = dx * cosine + dy * sine
    local_y = -dx * sine + dy * cosine
    half = float(side_px) / 2.0
    inside = (
        (np.abs(local_x) <= 0.82 * half)
        & (np.abs(local_y) <= 0.82 * half)
    )
    outside_ring = (
        (np.maximum(np.abs(local_x), np.abs(local_y)) >= 1.04 * half)
        & (np.maximum(np.abs(local_x), np.abs(local_y)) <= 1.28 * half)
    )
    sample_values = working[yy, xx]
    baseline = float(np.median(sample_values))
    intensity_scale = max(
        float(np.percentile(sample_values, 92.0)) - baseline, 1e-6
    )
    normalized = np.maximum(sample_values - baseline, 0.0) / intensity_scale
    interior_fill = _clamp01(
        float(np.mean(np.clip(normalized[inside], 0.0, 1.0)))
        if np.any(inside) else 0.0
    )
    outside_leakage = _clamp01(
        float(np.mean(np.clip(normalized[outside_ring], 0.0, 1.0)))
        if np.any(outside_ring) else 1.0
    )
    min_coverage = min(
        float(edge_metrics["top_boundary_coverage"]),
        float(edge_metrics["right_boundary_coverage"]),
        float(edge_metrics["bottom_boundary_coverage"]),
        float(edge_metrics["left_boundary_coverage"]),
    )
    score = _clamp01(
        0.30 * float(edge_metrics["lower_quartile_edge_score"])
        + 0.20 * float(edge_metrics["boundary_coverage_score"])
        + 0.20 * float(edge_metrics["edge_balance_score"])
        + 0.15 * interior_fill
        + 0.10 * (1.0 - outside_leakage)
        + 0.05
    )
    return {
        **edge_metrics,
        "interior_fill_score": interior_fill,
        "outside_leakage_score": outside_leakage,
        "min_boundary_coverage": min_coverage,
        "score": score,
    }


def _module4_square_edge_metrics_legacy(
    gradient: np.ndarray,
    center: dict,
    side_px: float,
    angle_degrees: float,
    global_scale: float | None = None,
) -> dict:
    """Preserve the pre-diagnostic local-fit behavior for frozen baselines."""
    half = float(side_px) / 2.0
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    side_values = []
    samples = np.linspace(-half, half, 19)
    for fixed_axis, fixed_value in (
        ("x", -half), ("x", half), ("y", -half), ("y", half)
    ):
        xs, ys = [], []
        for sample in samples:
            dx, dy = (
                (fixed_value, sample)
                if fixed_axis == "x" else (sample, fixed_value)
            )
            xs.append(float(center["x"]) + dx * cosine - dy * sine)
            ys.append(float(center["y"]) + dx * sine + dy * cosine)
        values = ndimage.map_coordinates(
            gradient,
            [np.asarray(ys), np.asarray(xs)],
            order=1,
            mode="nearest",
        )
        side_values.append(float(np.mean(values)))
    side_array = np.asarray(side_values, dtype=float)
    local_scale = (
        float(global_scale)
        if global_scale is not None
        else max(float(np.percentile(gradient, 92.0)), 1.0)
    )
    strength = _clamp01(float(np.median(side_array)) / local_scale * 2.0)
    opposite_balance = 0.5 * (
        _clamp01(1.0 - abs(side_array[0] - side_array[1])
                 / max(side_array[0] + side_array[1], 1e-6))
        + _clamp01(1.0 - abs(side_array[2] - side_array[3])
                   / max(side_array[2] + side_array[3], 1e-6))
    )
    return {
        "boundary_alignment_score": _clamp01(
            0.72 * strength + 0.28 * opposite_balance
        ),
        "edge_balance_score": opposite_balance,
    }


def fit_module4_global_roi_angle(
    slice_pixels: np.ndarray,
    phantom_radius: float,
    pin_angle_degrees: float,
    phi_offset_degrees: float,
    roi_side_ratio: float,
    current_roi_angle_offset: float,
    current_rois: list[dict],
) -> dict:
    """Polish one shared ROI angle/size without changing any target center."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    fill = float(np.median(raw[np.isfinite(raw)]))
    working = np.where(np.isfinite(raw), raw, fill)
    smooth = ndimage.gaussian_filter(working, sigma=1.0)
    gy, gx = np.gradient(smooth)
    gradient = np.hypot(gx, gy)
    radius = max(float(phantom_radius), 1.0)
    initial_side_ratio = float(roi_side_ratio)
    orientation = (
        float(pin_angle_degrees)
        - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
        + float(phi_offset_degrees)
    )
    priorities = {
        "primary": 1.35,
        "secondary": 1.0,
        "reference": 0.85,
    }

    def evaluate(offset: float, side_ratio: float) -> dict:
        angle = orientation + float(offset)
        side_px = float(side_ratio) * radius
        evidence = []
        weighted_total = 0.0
        weight_total = 0.0
        for target in current_rois:
            score = _module4_square_boundary_alignment(
                gradient,
                target["center"],
                side_px,
                angle,
            )
            weight = priorities.get(target.get("priority"), 1.0)
            weighted_total += weight * score
            weight_total += weight
            evidence.append({
                "id": target["id"],
                "priority": target.get("priority"),
                "boundary_alignment": round(float(score), 4),
                "weight": round(float(weight), 3),
            })
        return {
            "offset": float(offset),
            "score": weighted_total / max(weight_total, 1e-6),
            "target_evidence": evidence,
        }

    before = evaluate(float(current_roi_angle_offset), initial_side_ratio)
    angle_low = max(0.0, float(current_roi_angle_offset) - 15.0)
    angle_high = min(90.0, float(current_roi_angle_offset) + 15.0)
    coarse_trials = [
        evaluate(float(offset), initial_side_ratio)
        for offset in np.arange(angle_low, angle_high + 0.001, 1.0)
    ]
    coarse_best = max(coarse_trials, key=lambda item: item["score"])
    fine_low = max(angle_low, coarse_best["offset"] - 2.0)
    fine_high = min(angle_high, coarse_best["offset"] + 2.0)
    fine_trials = [
        evaluate(float(offset), initial_side_ratio)
        for offset in np.arange(fine_low, fine_high + 0.001, 0.25)
    ]
    angle_best = max(fine_trials, key=lambda item: item["score"])
    size_low = max(0.06, initial_side_ratio - 0.03)
    size_high = min(0.24, initial_side_ratio + 0.03)
    size_trials = [
        {
            **evaluate(angle_best["offset"], float(side_ratio)),
            "side_ratio": float(side_ratio),
        }
        for side_ratio in np.arange(size_low, size_high + 0.0001, 0.0025)
    ]
    size_best = max(size_trials, key=lambda item: item["score"])
    size_improvement = float(size_best["score"] - angle_best["score"])
    size_applied = (
        abs(float(size_best["side_ratio"]) - initial_side_ratio) >= 0.001
        and size_improvement > 0.003
    )
    final_side_ratio = (
        float(size_best["side_ratio"])
        if size_applied else initial_side_ratio
    )
    final_angle_low = max(angle_low, angle_best["offset"] - 1.0)
    final_angle_high = min(angle_high, angle_best["offset"] + 1.0)
    final_trials = [
        {
            **evaluate(float(offset), final_side_ratio),
            "side_ratio": final_side_ratio,
        }
        for offset in np.arange(
            final_angle_low, final_angle_high + 0.001, 0.25
        )
    ]
    best = max(final_trials, key=lambda item: item["score"])
    improvement = float(best["score"] - before["score"])
    all_scores = sorted(
        [trial["score"] for trial in coarse_trials + fine_trials],
        reverse=True,
    )
    comparison_score = (
        float(np.median(all_scores))
        if all_scores else float(best["score"])
    )
    separation = max(0.0, float(best["score"]) - comparison_score)
    confidence = _clamp01(
        0.55 * max(0.0, improvement) / 0.08
        + 0.45 * separation / 0.08
    )
    size_confidence = _clamp01(max(0.0, size_improvement) / 0.06)
    if abs(best["offset"] - float(current_roi_angle_offset)) < 0.125:
        status = "retained"
        reason = (
            "The focused angle search confirmed the existing shared ROI angle."
        )
    elif improvement > 0.005:
        status = "applied"
        reason = (
            "Applied the shared angle with the strongest weighted square-boundary "
            "support while preserving every target center."
        )
    else:
        status = "needs_review"
        reason = (
            "The best shared-angle hypothesis improved edge support only weakly; "
            "it was retained as review-quality geometry without moving centers."
        )
    return {
        "enabled": True,
        "method": "bounded_shared_roi_angle_size_polish",
        "angle_convention": (
            "Degrees clockwise in image coordinates; final ROI angle equals "
            "pin-relative orientation plus one shared ROI angle offset."
        ),
        "search_bounds_degrees": [
            round(angle_low, 3), round(angle_high, 3)
        ],
        "coarse_step_degrees": 1.0,
        "fine_step_degrees": 0.25,
        "roi_angle_offset_before": round(float(current_roi_angle_offset), 3),
        "roi_angle_offset_after": round(float(best["offset"]), 3),
        "roi_angle_degrees_before": round(
            orientation + float(current_roi_angle_offset), 3
        ),
        "roi_angle_degrees_after": round(orientation + float(best["offset"]), 3),
        "angle_fit_score": round(float(best["score"]), 4),
        "angle_fit_score_before": round(float(before["score"]), 4),
        "angle_fit_improvement": round(improvement, 4),
        "angle_fit_confidence": round(confidence, 4),
        "angle_fit_status": status,
        "angle_fit_reason": reason,
        "targets_used_for_angle_fit": [
            target["id"] for target in current_rois
        ],
        "pre_fit_angle_error_estimate": round(
            max(0.0, 1.0 - float(before["score"])), 4
        ),
        "post_fit_angle_error_estimate": round(
            max(0.0, 1.0 - float(best["score"])), 4
        ),
        "angle_evidence_per_target": best["target_evidence"],
        "roi_side_ratio_before": round(initial_side_ratio, 4),
        "roi_side_ratio_after": round(float(best["side_ratio"]), 4),
        "roi_side_px_before": round(initial_side_ratio * radius, 3),
        "roi_side_px_after": round(float(best["side_ratio"]) * radius, 3),
        "size_search_bounds": [
            round(size_low, 4), round(size_high, 4)
        ],
        "size_search_step": 0.0025,
        "size_fit_score": round(float(size_best["score"]), 4),
        "size_fit_confidence": round(size_confidence, 4),
        "size_fit_status": (
            "applied" if size_applied else "retained"
        ),
        "center_freeze_enabled": True,
        "max_center_shift_px_after_angle_size_polish": 0.0,
        "centers_changed_by_angle_fit": False,
    }


def fit_module4_geometry_guided_local_squares(
    slice_pixels: np.ndarray,
    phantom_radius: float,
    geometry_rois: list[dict],
    performance_mode: str = "fast",
) -> dict:
    """Fit each expected square only inside its geometry-guided local window."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    finite_raw = np.isfinite(raw)
    fill = float(np.median(raw[finite_raw]))
    working = np.where(finite_raw, raw, fill)
    smooth = ndimage.gaussian_filter(working, sigma=1.0)
    gy, gx = np.gradient(smooth)
    gradient = np.hypot(gx, gy)
    cached_legacy_gradient_scale = max(
        float(np.percentile(gradient, 92.0)), 1.0
    )
    fitted_targets = []
    target_debug = []
    diagnostic_intermediates = {}
    fast_mode = performance_mode != "debug"
    local_detection_ms = 0.0
    micro_refinement_ms = 0.0
    targeted_refinement_ms = 0.0
    total_local_candidates = 0
    total_square_hypotheses = 0
    total_micro_hypotheses = 0
    total_targeted_hypotheses = 0
    caps = {
        "local_candidates": 8 if fast_mode else 24,
        "square_hypotheses": 240 if fast_mode else 700,
        "micro_hypotheses": 200 if fast_mode else 500,
        "targeted_hypotheses": 0 if fast_mode else 400,
    }

    for geometry_target in geometry_rois:
        local_stage_started = time.perf_counter()
        target = dict(geometry_target)
        target_id = target["id"]
        geometry_center = {
            "x": float(target["center"]["x"]),
            "y": float(target["center"]["y"]),
        }
        geometry_side = float(target["roi_side_px"])
        geometry_angle = float(target["roi_angle_degrees"])
        search_radius = max(
            18.0,
            (2.10 if fast_mode else 1.75) * geometry_side,
        )
        shift_limit = min(
            0.08 * float(phantom_radius),
            0.75 * geometry_side,
        )
        x0 = max(0, int(math.floor(geometry_center["x"] - search_radius)))
        x1 = min(raw.shape[1], int(math.ceil(geometry_center["x"] + search_radius + 1)))
        y0 = max(0, int(math.floor(geometry_center["y"] - search_radius)))
        y1 = min(raw.shape[0], int(math.ceil(geometry_center["y"] + search_radius + 1)))
        patch = working[y0:y1, x0:x1]
        local_background = ndimage.gaussian_filter(
            patch,
            sigma=max(2.0, geometry_side * 0.24),
        )
        residual = np.maximum(patch - local_background, 0.0)
        positive_scale = max(float(np.percentile(residual, 95.0)), 1e-6)
        threshold = max(
            float(np.percentile(residual, 72.0)),
            0.18 * positive_scale,
        )
        bright_mask = residual >= threshold
        bright_mask = ndimage.binary_opening(
            bright_mask, structure=np.ones((2, 2), dtype=bool)
        )
        bright_mask = ndimage.binary_closing(
            bright_mask, structure=np.ones((3, 3), dtype=bool), iterations=2
        )
        labels, label_count = ndimage.label(bright_mask)
        components = []
        for label_index in range(1, label_count + 1):
            local_y, local_x = np.where(labels == label_index)
            area = int(local_x.size)
            if area < max(6, int(0.015 * geometry_side ** 2)):
                continue
            center_x = float(x0 + np.mean(local_x))
            center_y = float(y0 + np.mean(local_y))
            shift = math.hypot(
                center_x - geometry_center["x"],
                center_y - geometry_center["y"],
            )
            mean_residual = float(np.mean(residual[local_y, local_x]))
            proximity = _clamp01(1.0 - shift / max(search_radius, 1.0))
            area_score = _clamp01(
                area / max(0.30 * geometry_side ** 2, 1.0)
            )
            contrast_score = _clamp01(mean_residual / positive_scale)
            component_score = (
                0.46 * proximity
                + 0.30 * contrast_score
                + 0.24 * area_score
            )
            components.append({
                "label_index": label_index,
                "center": {"x": center_x, "y": center_y},
                "area": area,
                "bbox": {
                    "x0": int(x0 + np.min(local_x)),
                    "y0": int(y0 + np.min(local_y)),
                    "x1": int(x0 + np.max(local_x) + 1),
                    "y1": int(y0 + np.max(local_y) + 1),
                },
                "center_shift_px": shift,
                "component_score": component_score,
                "contrast_score": contrast_score,
                "area_score": area_score,
            })
        selected = (
            max(
                sorted(
                    components,
                    key=lambda item: item["component_score"],
                    reverse=True,
                )[:caps["local_candidates"]],
                key=lambda item: item["component_score"],
            )
            if components else None
        )
        selected_mask = (
            labels == int(selected["label_index"])
            if selected is not None else np.zeros_like(bright_mask)
        )
        diagnostic_intermediates[target_id] = {
            "bounds": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "raw_crop": np.asarray(patch, dtype=np.float32),
            "residual": np.asarray(residual, dtype=np.float32),
            "threshold_mask": np.asarray(bright_mask, dtype=bool),
            "selected_component_mask": np.asarray(selected_mask, dtype=bool),
            "selected_component": dict(selected) if selected else None,
            "geometry_corners": list(target.get("roi_corners", [])),
            "threshold_value": float(threshold),
            "positive_residual_scale": float(positive_scale),
        }
        local_candidates_evaluated = min(
            len(components), caps["local_candidates"]
        )
        total_local_candidates += local_candidates_evaluated
        square_hypotheses_evaluated = 0
        local_gradient_values = np.asarray(
            gradient[y0:y1, x0:x1], dtype=float
        )
        finite_gradient_values = local_gradient_values[
            np.isfinite(local_gradient_values)
        ]
        gradient_floor = (
            float(np.percentile(finite_gradient_values, 50.0))
            if finite_gradient_values.size else 0.0
        )
        gradient_high = (
            float(np.percentile(finite_gradient_values, 95.0))
            if finite_gradient_values.size else gradient_floor + 1.0
        )
        cached_local_statistics = {
            "gradient_floor": gradient_floor,
            "gradient_contrast": max(
                gradient_high - gradient_floor, 1e-6
            ),
            "gradient_significant_threshold": (
                float(np.percentile(finite_gradient_values, 78.0))
                if finite_gradient_values.size else gradient_high
            ),
        }
        cached_intensity_baseline = float(np.median(patch))
        cached_intensity_high = max(
            float(np.percentile(patch, 92.0))
            - cached_intensity_baseline,
            1e-6,
        )

        def hypothesis_metrics(
            candidate_center: dict,
            side_px: float,
            angle_degrees: float,
        ) -> dict:
            edge_metrics = _module4_square_edge_metrics(
                gradient,
                candidate_center,
                side_px,
                angle_degrees,
                local_statistics=cached_local_statistics,
            )
            boundary = float(edge_metrics["boundary_alignment_score"])
            half_span = max(4, int(math.ceil(side_px * 0.72)))
            yy, xx = np.mgrid[
                max(0, int(round(candidate_center["y"])) - half_span):
                min(raw.shape[0], int(round(candidate_center["y"])) + half_span + 1),
                max(0, int(round(candidate_center["x"])) - half_span):
                min(raw.shape[1], int(round(candidate_center["x"])) + half_span + 1),
            ]
            angle_radians = math.radians(angle_degrees)
            cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
            dx = xx - float(candidate_center["x"])
            dy = yy - float(candidate_center["y"])
            local_x = dx * cosine + dy * sine
            local_y = -dx * sine + dy * cosine
            half = side_px / 2.0
            inside = (
                (np.abs(local_x) <= 0.82 * half)
                & (np.abs(local_y) <= 0.82 * half)
            )
            outside_ring = (
                (np.maximum(np.abs(local_x), np.abs(local_y)) >= 1.04 * half)
                & (np.maximum(np.abs(local_x), np.abs(local_y)) <= 1.28 * half)
            )
            sample_values = working[yy, xx]
            normalized = (
                np.maximum(
                    sample_values - cached_intensity_baseline, 0.0
                )
                / cached_intensity_high
            )
            interior_fill = _clamp01(
                float(np.mean(np.clip(normalized[inside], 0.0, 1.0)))
                if np.any(inside) else 0.0
            )
            outside_leakage = _clamp01(
                float(np.mean(np.clip(normalized[outside_ring], 0.0, 1.0)))
                if np.any(outside_ring) else 1.0
            )
            leakage_score = 1.0 - outside_leakage
            score = _clamp01(
                0.30 * float(edge_metrics["lower_quartile_edge_score"])
                + 0.20 * float(edge_metrics["boundary_coverage_score"])
                + 0.20 * float(edge_metrics["edge_balance_score"])
                + 0.15 * interior_fill
                + 0.10 * leakage_score
                + 0.05
            )
            return {
                "score": score,
                **edge_metrics,
                "interior_fill_score": interior_fill,
                "outside_leakage_score": outside_leakage,
            }

        def legacy_hypothesis_metrics(
            candidate_center: dict,
            side_px: float,
            angle_degrees: float,
        ) -> dict:
            """Reproduce the accepted pre-fix ROI for frozen baseline targets."""
            metrics = hypothesis_metrics(
                candidate_center, side_px, angle_degrees
            )
            legacy_edges = _module4_square_edge_metrics_legacy(
                gradient,
                candidate_center,
                side_px,
                angle_degrees,
                global_scale=cached_legacy_gradient_scale,
            )
            legacy_score = _clamp01(
                0.50 * legacy_edges["boundary_alignment_score"]
                + 0.30 * metrics["interior_fill_score"]
                + 0.20 * (1.0 - metrics["outside_leakage_score"])
            )
            return {
                **metrics,
                **legacy_edges,
                "score": legacy_score,
            }

        best_fit = None
        if selected is not None:
            side_values = (
                np.arange(
                    geometry_side - 6.0,
                    geometry_side + 6.01,
                    2.0,
                )
                if fast_mode
                else geometry_side * np.arange(0.70, 1.301, 0.05)
            )
            angle_values = (
                np.arange(-18.0, 18.01, 3.0)
                if fast_mode else np.arange(-20.0, 20.01, 2.0)
            )
            for side_px in side_values:
                for angle_delta in angle_values:
                    if square_hypotheses_evaluated >= caps["square_hypotheses"]:
                        break
                    square_hypotheses_evaluated += 1
                    angle_degrees = geometry_angle + float(angle_delta)
                    metrics = legacy_hypothesis_metrics(
                        selected["center"], side_px, angle_degrees
                    )
                    combined = (
                        0.82 * metrics["score"]
                        + 0.18 * selected["component_score"]
                    )
                    trial = {
                        **metrics,
                        "combined_score": combined,
                        "side_px": side_px,
                        "angle_degrees": angle_degrees,
                    }
                    if best_fit is None or combined > best_fit["combined_score"]:
                        best_fit = trial
                if square_hypotheses_evaluated >= caps["square_hypotheses"]:
                    break
            coarse_angle = best_fit["angle_degrees"]
            coarse_side = best_fit["side_px"]
            fine_sides = (
                np.arange(coarse_side - 2.0, coarse_side + 2.01, 1.0)
                if fast_mode
                else np.arange(
                    max(0.65 * geometry_side, coarse_side - 0.08 * geometry_side),
                    min(1.35 * geometry_side, coarse_side + 0.08 * geometry_side) + 0.001,
                    max(0.015 * geometry_side, 0.5),
                )
            )
            fine_angles = (
                np.arange(coarse_angle - 2.0, coarse_angle + 2.01, 1.0)
                if fast_mode
                else np.arange(coarse_angle - 2.0, coarse_angle + 2.01, 0.5)
            )
            for side_px in fine_sides:
                for angle_degrees in fine_angles:
                    if square_hypotheses_evaluated >= caps["square_hypotheses"]:
                        break
                    square_hypotheses_evaluated += 1
                    metrics = legacy_hypothesis_metrics(
                        selected["center"], float(side_px), float(angle_degrees)
                    )
                    combined = (
                        0.82 * metrics["score"]
                        + 0.18 * selected["component_score"]
                    )
                    if combined > best_fit["combined_score"]:
                        best_fit = {
                            **metrics,
                            "combined_score": combined,
                            "side_px": float(side_px),
                            "angle_degrees": float(angle_degrees),
                        }
                if square_hypotheses_evaluated >= caps["square_hypotheses"]:
                    break
        total_square_hypotheses += square_hypotheses_evaluated
        local_detection_ms += (
            time.perf_counter() - local_stage_started
        ) * 1000.0

        accepted = bool(
            selected is not None
            and best_fit is not None
            and selected["center_shift_px"] <= shift_limit
            and best_fit["combined_score"] >= 0.36
            and best_fit["edge_balance_score"] >= 0.24
        )
        micro_applied = False
        micro_reason = "Local square fallback was retained; micro-refinement was not run."
        micro_pre_center = (
            dict(selected["center"]) if accepted else dict(geometry_center)
        )
        micro_post_center = dict(micro_pre_center)
        micro_pre_side = (
            float(best_fit["side_px"]) if accepted else geometry_side
        )
        micro_post_side = micro_pre_side
        micro_pre_angle = (
            float(best_fit["angle_degrees"]) if accepted else geometry_angle
        )
        micro_post_angle = micro_pre_angle
        micro_pre_metrics = (
            legacy_hypothesis_metrics(
                micro_pre_center, micro_pre_side, micro_pre_angle
            )
            if accepted else None
        )
        micro_best = None
        pre_trial = None
        refinement_eval_counts = {"micro": 0, "targeted": 0}
        refinement_phase = "micro"
        micro_refinement_skipped_reason = ""
        targeted_refinement_skipped_reason = ""

        def micro_trial(
            trial_center: dict,
            trial_side: float,
            trial_angle: float,
        ) -> dict:
            refinement_eval_counts[refinement_phase] += 1
            metrics = legacy_hypothesis_metrics(
                trial_center, trial_side, trial_angle
            )
            center_delta = math.hypot(
                float(trial_center["x"]) - float(micro_pre_center["x"]),
                float(trial_center["y"]) - float(micro_pre_center["y"]),
            )
            center_stability = _clamp01(1.0 - center_delta / 3.0)
            leakage_quality = 1.0 - metrics["outside_leakage_score"]
            score = _clamp01(
                0.58 * metrics["boundary_alignment_score"]
                + 0.16 * metrics["edge_balance_score"]
                + 0.13 * metrics["interior_fill_score"]
                + 0.09 * leakage_quality
                + 0.04 * center_stability
            )
            return {
                **metrics,
                "square_score": score,
                "center": {
                    "x": float(trial_center["x"]),
                    "y": float(trial_center["y"]),
                },
                "side_px": float(trial_side),
                "angle_degrees": float(trial_angle),
            }

        initial_fit_already_good = bool(
            accepted
            and best_fit["combined_score"] >= 0.97
            and best_fit["interior_fill_score"] >= 0.90
            and best_fit["outside_leakage_score"] <= 0.01
            and min(
                best_fit["top_boundary_coverage"],
                best_fit["right_boundary_coverage"],
                best_fit["bottom_boundary_coverage"],
                best_fit["left_boundary_coverage"],
            ) >= 0.85
        )
        micro_stage_started = time.perf_counter()
        if accepted and not (fast_mode and initial_fit_already_good):
            pre_trial = micro_trial(
                micro_pre_center, micro_pre_side, micro_pre_angle
            )
            micro_best = pre_trial
            # Stage 1: bounded center search at the current local-square shape.
            micro_center_limit = 2 if fast_mode else 3
            for dx in range(-micro_center_limit, micro_center_limit + 1):
                for dy in range(-micro_center_limit, micro_center_limit + 1):
                    if math.hypot(dx, dy) > micro_center_limit + 1e-6:
                        continue
                    if refinement_eval_counts["micro"] >= caps["micro_hypotheses"]:
                        break
                    trial = micro_trial(
                        {
                            "x": micro_pre_center["x"] + dx,
                            "y": micro_pre_center["y"] + dy,
                        },
                        micro_pre_side,
                        micro_pre_angle,
                    )
                    if trial["square_score"] > micro_best["square_score"]:
                        micro_best = trial
                if refinement_eval_counts["micro"] >= caps["micro_hypotheses"]:
                    break
            center_best = dict(micro_best["center"])
            # Stage 2: shared local center is fixed while side/angle are polished.
            micro_side_limit = 3 if fast_mode else 5
            micro_angle_values = (
                np.arange(-2.0, 2.01, 1.0)
                if fast_mode else np.arange(-3.0, 3.01, 0.5)
            )
            for side_delta in range(-micro_side_limit, micro_side_limit + 1):
                for angle_delta in micro_angle_values:
                    if refinement_eval_counts["micro"] >= caps["micro_hypotheses"]:
                        break
                    trial = micro_trial(
                        center_best,
                        micro_pre_side + side_delta,
                        micro_pre_angle + float(angle_delta),
                    )
                    if trial["square_score"] > micro_best["square_score"]:
                        micro_best = trial
                if refinement_eval_counts["micro"] >= caps["micro_hypotheses"]:
                    break
            # Stage 3: a small joint pass avoids coordinate-search artifacts.
            if not fast_mode:
                joint_center = dict(micro_best["center"])
                joint_side = float(micro_best["side_px"])
                joint_angle = float(micro_best["angle_degrees"])
                for dx in (-1.0, 0.0, 1.0):
                    for dy in (-1.0, 0.0, 1.0):
                        candidate_center = {
                            "x": joint_center["x"] + dx,
                            "y": joint_center["y"] + dy,
                        }
                        if math.hypot(
                            candidate_center["x"] - micro_pre_center["x"],
                            candidate_center["y"] - micro_pre_center["y"],
                        ) > 3.0 + 1e-6:
                            continue
                        for side_delta in (-1.0, 0.0, 1.0):
                            candidate_side = joint_side + side_delta
                            if abs(candidate_side - micro_pre_side) > 5.0 + 1e-6:
                                continue
                            for angle_delta in (-0.5, 0.0, 0.5):
                                if refinement_eval_counts["micro"] >= caps["micro_hypotheses"]:
                                    break
                                candidate_angle = joint_angle + angle_delta
                                if abs(candidate_angle - micro_pre_angle) > 3.0 + 1e-6:
                                    continue
                                trial = micro_trial(
                                    candidate_center,
                                    candidate_side,
                                    candidate_angle,
                                )
                                if trial["square_score"] > micro_best["square_score"]:
                                    micro_best = trial
            if micro_best["square_score"] > pre_trial["square_score"] + 0.002:
                micro_applied = True
                micro_post_center = dict(micro_best["center"])
                micro_post_side = float(micro_best["side_px"])
                micro_post_angle = float(micro_best["angle_degrees"])
                micro_reason = (
                    "Applied bounded local micro-refinement using balanced "
                    "four-edge support, interior fill, outside leakage, and "
                    "center stability."
                )
            else:
                micro_best = pre_trial
                micro_reason = (
                    "The bounded micro-search did not materially improve the "
                    "current local-square boundary fit; the detected ROI was retained."
                )
        elif accepted:
            micro_refinement_skipped_reason = "initial fit already good"
            micro_reason = "Micro-refinement skipped because the initial fit is already good."
            pre_trial = micro_trial(
                micro_pre_center, micro_pre_side, micro_pre_angle
            )
            micro_best = pre_trial
        micro_refinement_ms += (
            time.perf_counter() - micro_stage_started
        ) * 1000.0
        total_micro_hypotheses += refinement_eval_counts["micro"]
        final_center = (
            micro_post_center if accepted else geometry_center
        )
        final_side = (
            micro_post_side if accepted else geometry_side
        )
        final_angle = (
            micro_post_angle if accepted else geometry_angle
        )
        pre_targeted_center = dict(final_center)
        pre_targeted_side = float(final_side)
        pre_targeted_angle = float(final_angle)
        post_targeted_center = dict(pre_targeted_center)
        post_targeted_side = pre_targeted_side
        post_targeted_angle = pre_targeted_angle
        targeted_applied = False
        frozen_good_target = target_id not in MODULE4_TARGETED_POLISH_IDS
        targeted_reason = (
            "Frozen because current ROI is acceptable."
            if frozen_good_target
            else "Targeted refinement retained the current ROI."
        )

        def improved_targeted_metrics(
            trial_center: dict,
            trial_side: float,
            trial_angle: float,
        ) -> dict:
            metrics = hypothesis_metrics(
                trial_center, trial_side, trial_angle
            )
            center_delta = math.hypot(
                float(trial_center["x"]) - float(pre_targeted_center["x"]),
                float(trial_center["y"]) - float(pre_targeted_center["y"]),
            )
            center_stability = _clamp01(1.0 - center_delta / 2.0)
            square_score = _clamp01(
                0.30 * metrics["lower_quartile_edge_score"]
                + 0.20 * metrics["boundary_coverage_score"]
                + 0.20 * metrics["edge_balance_score"]
                + 0.15 * metrics["interior_fill_score"]
                + 0.10 * (1.0 - metrics["outside_leakage_score"])
                + 0.05 * center_stability
            )
            return {
                **metrics,
                "square_score": square_score,
                "center": {
                    "x": float(trial_center["x"]),
                    "y": float(trial_center["y"]),
                },
                "side_px": float(trial_side),
                "angle_degrees": float(trial_angle),
            }

        targeted_before = (
            improved_targeted_metrics(
                pre_targeted_center,
                pre_targeted_side,
                pre_targeted_angle,
            )
            if accepted else None
        )
        targeted_best = targeted_before
        targeted_candidate = targeted_before
        targeted_candidate_score_improved = False
        containment_gate_passed = False
        targeted_rejected = False
        targeted_rejection_reason = ""
        containment_gate_failures = []

        targeted_stage_started = time.perf_counter()
        if (
            target_id in MODULE4_TARGETED_POLISH_IDS
            and accepted
            and targeted_before is not None
            and not fast_mode
        ):
            reference_targets = [
                item for item in fitted_targets
                if item["id"] in {"B1", "B2", "B3", "B4", "B5"}
            ]
            reference_angles = [
                float(item["roi_angle_degrees"])
                for item in reference_targets
            ]
            reference_sides = [
                float(item["roi_side_px"])
                for item in reference_targets
            ]
            reference_angle_low = (
                min(reference_angles) - 1.0
                if reference_angles else pre_targeted_angle - 3.0
            )
            reference_angle_high = (
                max(reference_angles) + 1.0
                if reference_angles else pre_targeted_angle + 3.0
            )
            reference_side_median = (
                float(np.median(reference_sides))
                if reference_sides else pre_targeted_side
            )

            def targeted_trial(
                trial_center: dict,
                trial_side: float,
                trial_angle: float,
            ) -> dict:
                refinement_eval_counts["targeted"] += 1
                trial = improved_targeted_metrics(
                    trial_center, trial_side, trial_angle
                )
                size_growth = max(
                    0.0,
                    float(trial_side) - pre_targeted_side,
                )
                oversized_penalty = 0.025 * size_growth / 3.0
                angle_outlier = max(
                    reference_angle_low - float(trial_angle),
                    float(trial_angle) - reference_angle_high,
                    0.0,
                )
                reference_angle_penalty = 0.012 * angle_outlier
                reference_size_penalty = 0.012 * max(
                    0.0,
                    float(trial_side) - (reference_side_median + 3.0),
                )
                return {
                    **trial,
                    "targeted_score": _clamp01(
                        trial["square_score"]
                        - oversized_penalty
                        - reference_angle_penalty
                        - reference_size_penalty
                    ),
                }

            targeted_before = targeted_trial(
                pre_targeted_center,
                pre_targeted_side,
                pre_targeted_angle,
            )
            targeted_best = targeted_before
            targeted_center = dict(pre_targeted_center)
            if target_id == "B7":
                # B7 remains an angle outlier. Continue angle-only toward the
                # frozen B1-B5 reference range; never move its center or side.
                angle_trials = np.arange(
                    max(reference_angle_low, pre_targeted_angle - 10.0),
                    min(pre_targeted_angle, reference_angle_high + 2.0) + 0.01,
                    0.5,
                )
                side_deltas = (0.0,)
            else:
                angle_trials = (
                    pre_targeted_angle
                    + np.arange(-3.0, 3.01, 0.5)
                )
                side_deltas = range(-3, 4)
            for side_delta in side_deltas:
                for trial_angle in angle_trials:
                    if (
                        refinement_eval_counts["targeted"]
                        >= caps["targeted_hypotheses"]
                    ):
                        break
                    trial = targeted_trial(
                        targeted_center,
                        pre_targeted_side + side_delta,
                        float(trial_angle),
                    )
                    if trial["targeted_score"] > targeted_best["targeted_score"]:
                        targeted_best = trial
                if (
                    refinement_eval_counts["targeted"]
                    >= caps["targeted_hypotheses"]
                ):
                    break
            targeted_candidate = targeted_best
            targeted_candidate_score_improved = bool(
                targeted_candidate["targeted_score"]
                > targeted_before["targeted_score"] + 0.002
            )
            pre_min_coverage = min(
                float(targeted_before["top_boundary_coverage"]),
                float(targeted_before["right_boundary_coverage"]),
                float(targeted_before["bottom_boundary_coverage"]),
                float(targeted_before["left_boundary_coverage"]),
            )
            post_min_coverage = min(
                float(targeted_candidate["top_boundary_coverage"]),
                float(targeted_candidate["right_boundary_coverage"]),
                float(targeted_candidate["bottom_boundary_coverage"]),
                float(targeted_candidate["left_boundary_coverage"]),
            )
            if (
                float(targeted_candidate["interior_fill_score"])
                < float(targeted_before["interior_fill_score"]) - 0.005
            ):
                containment_gate_failures.append(
                    "interior fill decreased by more than 0.005"
                )
            if (
                float(targeted_candidate["outside_leakage_score"])
                > float(targeted_before["outside_leakage_score"]) + 0.002
            ):
                containment_gate_failures.append(
                    "outside leakage increased by more than 0.002"
                )
            if (
                float(targeted_candidate["lower_quartile_edge_score"])
                < float(targeted_before["lower_quartile_edge_score"])
            ):
                containment_gate_failures.append(
                    "lower-quartile edge support decreased"
                )
            if post_min_coverage < pre_min_coverage:
                containment_gate_failures.append(
                    "minimum boundary coverage decreased"
                )
            containment_gate_passed = not containment_gate_failures
            if targeted_candidate_score_improved and containment_gate_passed:
                targeted_applied = True
                post_targeted_center = dict(targeted_candidate["center"])
                post_targeted_side = float(targeted_candidate["side_px"])
                post_targeted_angle = float(targeted_candidate["angle_degrees"])
                targeted_reason = (
                    "Applied targeted raw-HU polish after combined-score and "
                    "containment non-regression checks passed."
                )
            elif targeted_candidate_score_improved:
                targeted_rejected = True
                targeted_rejection_reason = (
                    "Score-improving candidate rejected: "
                    + "; ".join(containment_gate_failures)
                    + "."
                )
                targeted_reason = targeted_rejection_reason
                targeted_best = targeted_before
            else:
                targeted_best = targeted_before
                targeted_reason = (
                    "The targeted bounded search did not materially improve "
                    "the current local ROI, so it was retained."
                )
        elif (
            target_id in MODULE4_TARGETED_POLISH_IDS
            and accepted
            and fast_mode
        ):
            targeted_refinement_skipped_reason = "fast mode / initial fit acceptable"
            targeted_reason = "Targeted refinement skipped in fast mode."
        elif target_id in MODULE4_TARGETED_POLISH_IDS and not accepted:
            targeted_reason = (
                "Target remains a geometry fallback; targeted refinement was "
                "not forced without a confident local-square detection."
            )
        targeted_refinement_ms += (
            time.perf_counter() - targeted_stage_started
        ) * 1000.0
        total_targeted_hypotheses += refinement_eval_counts["targeted"]

        final_center = post_targeted_center
        final_side = post_targeted_side
        final_angle = post_targeted_angle
        final_fit_metrics = targeted_best if accepted else None
        pre_targeted_min_coverage = (
            min(
                float(targeted_before["top_boundary_coverage"]),
                float(targeted_before["right_boundary_coverage"]),
                float(targeted_before["bottom_boundary_coverage"]),
                float(targeted_before["left_boundary_coverage"]),
            )
            if targeted_before else 0.0
        )
        candidate_min_coverage = (
            min(
                float(targeted_candidate["top_boundary_coverage"]),
                float(targeted_candidate["right_boundary_coverage"]),
                float(targeted_candidate["bottom_boundary_coverage"]),
                float(targeted_candidate["left_boundary_coverage"]),
            )
            if targeted_candidate else 0.0
        )
        fitted_corners = _square_points(
            final_center["x"], final_center["y"], final_side, final_angle
        )
        inner_corners = _square_points(
            final_center["x"], final_center["y"], final_side * 0.72, final_angle
        )
        center_shift = math.hypot(
            final_center["x"] - geometry_center["x"],
            final_center["y"] - geometry_center["y"],
        )
        if accepted:
            roi_source = "local_square_detected"
            geometry_status = "local_square_detected"
            confidence = (
                "high" if best_fit["combined_score"] >= 0.62 else "moderate"
            )
            reason = (
                "Geometry-guided local raw-HU search detected and fitted the "
                "bright square insert inside its bounded target window. "
                f"{micro_reason}"
            )
        else:
            roi_source = "geometry_fallback_needs_review"
            geometry_status = "geometry_fallback_needs_review"
            confidence = "needs_review"
            reason = "Expected target region searched; local square detection weak."
        performance_cap_reached = bool(
            len(components) > caps["local_candidates"]
            or square_hypotheses_evaluated >= caps["square_hypotheses"]
            or refinement_eval_counts["micro"] >= caps["micro_hypotheses"]
            or (
                not fast_mode
                and refinement_eval_counts["targeted"]
                >= caps["targeted_hypotheses"]
            )
        )
        if performance_cap_reached:
            confidence = "needs_review"
            reason = f"{reason} Performance cap reached; best current ROI retained."

        target.update({
            "geometry_center": {
                "x": round(geometry_center["x"], 3),
                "y": round(geometry_center["y"], 3),
            },
            "detected_center": (
                {
                    "x": round(float(selected["center"]["x"]), 3),
                    "y": round(float(selected["center"]["y"]), 3),
                }
                if selected is not None else None
            ),
            "final_center": {
                "x": round(float(final_center["x"]), 3),
                "y": round(float(final_center["y"]), 3),
            },
            "center": {
                "x": round(float(final_center["x"]), 3),
                "y": round(float(final_center["y"]), 3),
            },
            "center_shift_px": round(center_shift, 3),
            "center_shift_limit_px": round(shift_limit, 3),
            "roi_source": roi_source,
            "fitted_side_px": round(final_side, 3),
            "roi_side_px": round(final_side, 3),
            "fitted_angle_degrees": round(final_angle, 3),
            "roi_angle_degrees": round(final_angle, 3),
            "fitted_corners": fitted_corners,
            "roi_corners": fitted_corners,
            "rotated_box": fitted_corners,
            "inner_roi_corners": inner_corners,
            "inner_roi": {
                "center": {
                    "x": round(float(final_center["x"]), 3),
                    "y": round(float(final_center["y"]), 3),
                },
                "width": round(final_side * 0.72, 3),
                "height": round(final_side * 0.72, 3),
                "angle_degrees": round(final_angle, 3),
            },
            "local_search_radius_px": round(search_radius, 3),
            "search_window_bounds": {
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            },
            "local_search_window": {
                "center": {
                    "x": round(geometry_center["x"], 3),
                    "y": round(geometry_center["y"], 3),
                },
                "radius_px": round(search_radius, 3),
                "bounds": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            },
            "local_square_score": round(
                float(best_fit["combined_score"]) if best_fit else 0.0, 4
            ),
            "target_changed": targeted_applied,
            "frozen_good_target": frozen_good_target,
            "pre_targeted_center": {
                "x": round(float(pre_targeted_center["x"]), 3),
                "y": round(float(pre_targeted_center["y"]), 3),
            },
            "post_targeted_center": {
                "x": round(float(post_targeted_center["x"]), 3),
                "y": round(float(post_targeted_center["y"]), 3),
            },
            "targeted_center_shift_px": round(math.hypot(
                float(post_targeted_center["x"])
                - float(pre_targeted_center["x"]),
                float(post_targeted_center["y"])
                - float(pre_targeted_center["y"]),
            ), 3),
            "pre_targeted_side_px": round(pre_targeted_side, 3),
            "post_targeted_side_px": round(post_targeted_side, 3),
            "targeted_side_change_px": round(
                post_targeted_side - pre_targeted_side, 3
            ),
            "pre_targeted_angle_degrees": round(pre_targeted_angle, 3),
            "post_targeted_angle_degrees": round(post_targeted_angle, 3),
            "targeted_angle_change_degrees": round(
                post_targeted_angle - pre_targeted_angle, 3
            ),
            "targeted_fit_score_before": round(
                float(targeted_before["targeted_score"])
                if targeted_before and "targeted_score" in targeted_before
                else float(targeted_before["square_score"])
                if targeted_before else 0.0,
                4,
            ),
            "targeted_fit_score_after": round(
                float(targeted_best["targeted_score"])
                if targeted_best and "targeted_score" in targeted_best
                else float(targeted_best["square_score"])
                if targeted_best else 0.0,
                4,
            ),
            "targeted_candidate_fit_score": round(
                float(targeted_candidate["targeted_score"])
                if targeted_candidate
                and "targeted_score" in targeted_candidate
                else float(targeted_candidate["square_score"])
                if targeted_candidate else 0.0,
                4,
            ),
            "targeted_refinement_applied": targeted_applied,
            "targeted_refinement_reason": targeted_reason,
            "micro_refinement_skipped_reason": (
                micro_refinement_skipped_reason
            ),
            "targeted_refinement_skipped_reason": (
                targeted_refinement_skipped_reason
            ),
            "local_candidates_evaluated": local_candidates_evaluated,
            "square_hypotheses_evaluated": square_hypotheses_evaluated,
            "micro_hypotheses_evaluated": refinement_eval_counts["micro"],
            "targeted_hypotheses_evaluated": (
                refinement_eval_counts["targeted"]
            ),
            "performance_cap_reached": performance_cap_reached,
            "pre_targeted_fill": round(
                float(targeted_before["interior_fill_score"])
                if targeted_before else 0.0, 4
            ),
            "post_targeted_fill": round(
                float(targeted_candidate["interior_fill_score"])
                if targeted_candidate else 0.0, 4
            ),
            "fill_delta": round(
                (
                    float(targeted_candidate["interior_fill_score"])
                    - float(targeted_before["interior_fill_score"])
                )
                if targeted_candidate and targeted_before else 0.0,
                4,
            ),
            "pre_targeted_leakage": round(
                float(targeted_before["outside_leakage_score"])
                if targeted_before else 1.0, 4
            ),
            "post_targeted_leakage": round(
                float(targeted_candidate["outside_leakage_score"])
                if targeted_candidate else 1.0, 4
            ),
            "leakage_delta": round(
                (
                    float(targeted_candidate["outside_leakage_score"])
                    - float(targeted_before["outside_leakage_score"])
                )
                if targeted_candidate and targeted_before else 0.0,
                4,
            ),
            "pre_lower_quartile_edge": round(
                float(targeted_before["lower_quartile_edge_score"])
                if targeted_before else 0.0, 4
            ),
            "post_lower_quartile_edge": round(
                float(targeted_candidate["lower_quartile_edge_score"])
                if targeted_candidate else 0.0, 4
            ),
            "lower_quartile_edge_delta": round(
                (
                    float(targeted_candidate["lower_quartile_edge_score"])
                    - float(targeted_before["lower_quartile_edge_score"])
                )
                if targeted_candidate and targeted_before else 0.0,
                4,
            ),
            "pre_min_boundary_coverage": round(
                pre_targeted_min_coverage, 4
            ),
            "post_min_boundary_coverage": round(
                candidate_min_coverage, 4
            ),
            "min_boundary_coverage_delta": round(
                candidate_min_coverage - pre_targeted_min_coverage, 4
            ),
            "containment_gate_passed": containment_gate_passed,
            "targeted_refinement_candidate_score_improved": (
                targeted_candidate_score_improved
            ),
            "targeted_refinement_accepted": targeted_applied,
            "targeted_refinement_rejected": targeted_rejected,
            "targeted_refinement_rejection_reason": (
                targeted_rejection_reason
            ),
            "pre_targeted_interior_fill_score": round(
                float(targeted_before["interior_fill_score"])
                if targeted_before else 0.0,
                4,
            ),
            "post_targeted_interior_fill_score": round(
                float(final_fit_metrics["interior_fill_score"])
                if final_fit_metrics else 0.0,
                4,
            ),
            "pre_targeted_outside_leakage_score": round(
                float(targeted_before["outside_leakage_score"])
                if targeted_before else 1.0,
                4,
            ),
            "post_targeted_outside_leakage_score": round(
                float(final_fit_metrics["outside_leakage_score"])
                if final_fit_metrics else 1.0,
                4,
            ),
            "pre_micro_center": {
                "x": round(float(micro_pre_center["x"]), 3),
                "y": round(float(micro_pre_center["y"]), 3),
            },
            "post_micro_center": {
                "x": round(float(micro_post_center["x"]), 3),
                "y": round(float(micro_post_center["y"]), 3),
            },
            "micro_center_shift_px": round(math.hypot(
                float(micro_post_center["x"]) - float(micro_pre_center["x"]),
                float(micro_post_center["y"]) - float(micro_pre_center["y"]),
            ), 3),
            "pre_micro_side_px": round(micro_pre_side, 3),
            "post_micro_side_px": round(micro_post_side, 3),
            "micro_side_change_px": round(
                micro_post_side - micro_pre_side, 3
            ),
            "pre_micro_angle_degrees": round(micro_pre_angle, 3),
            "post_micro_angle_degrees": round(micro_post_angle, 3),
            "micro_angle_change_degrees": round(
                micro_post_angle - micro_pre_angle, 3
            ),
            "pre_micro_square_score": round(
                float(pre_trial["square_score"]) if accepted else 0.0, 4
            ),
            "post_micro_square_score": round(
                float(micro_best["square_score"]) if accepted else 0.0, 4
            ),
            "micro_refinement_applied": micro_applied,
            "micro_refinement_reason": micro_reason,
            "pre_micro_corners": _square_points(
                micro_pre_center["x"],
                micro_pre_center["y"],
                micro_pre_side,
                micro_pre_angle,
            ),
            "top_edge_score": round(
                float(final_fit_metrics["top_edge_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "right_edge_score": round(
                float(final_fit_metrics["right_edge_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "bottom_edge_score": round(
                float(final_fit_metrics["bottom_edge_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "left_edge_score": round(
                float(final_fit_metrics["left_edge_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "top_edge_score_raw": round(
                float(final_fit_metrics["top_edge_score_raw"])
                if final_fit_metrics else 0.0, 4
            ),
            "right_edge_score_raw": round(
                float(final_fit_metrics["right_edge_score_raw"])
                if final_fit_metrics else 0.0, 4
            ),
            "bottom_edge_score_raw": round(
                float(final_fit_metrics["bottom_edge_score_raw"])
                if final_fit_metrics else 0.0, 4
            ),
            "left_edge_score_raw": round(
                float(final_fit_metrics["left_edge_score_raw"])
                if final_fit_metrics else 0.0, 4
            ),
            "top_boundary_coverage": round(
                float(final_fit_metrics["top_boundary_coverage"])
                if final_fit_metrics else 0.0, 4
            ),
            "right_boundary_coverage": round(
                float(final_fit_metrics["right_boundary_coverage"])
                if final_fit_metrics else 0.0, 4
            ),
            "bottom_boundary_coverage": round(
                float(final_fit_metrics["bottom_boundary_coverage"])
                if final_fit_metrics else 0.0, 4
            ),
            "left_boundary_coverage": round(
                float(final_fit_metrics["left_boundary_coverage"])
                if final_fit_metrics else 0.0, 4
            ),
            "min_edge_score": round(
                float(final_fit_metrics["min_edge_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "lower_quartile_edge_score": round(
                float(final_fit_metrics["lower_quartile_edge_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "edge_score_saturation_detected": bool(
                final_fit_metrics
                and final_fit_metrics["edge_score_saturation_detected"]
            ),
            "edge_balance_score": round(
                float(final_fit_metrics["edge_balance_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "interior_fill_score": round(
                float(final_fit_metrics["interior_fill_score"])
                if final_fit_metrics else 0.0, 4
            ),
            "outside_leakage_score": round(
                float(final_fit_metrics["outside_leakage_score"])
                if final_fit_metrics else 1.0, 4
            ),
            "location_confidence": confidence,
            "location_reason": reason,
            "reason": reason,
            "geometry_status": geometry_status,
            "draw_on_overlay": True,
            "normal_overlay_allowed": True,
        })
        final_roi_stage = (
            "targeted_refinement"
            if targeted_applied
            else "micro_refinement"
            if micro_applied
            else "local_square_detection"
            if accepted
            else "geometry_fallback"
        )
        target["final_roi_stage"] = final_roi_stage
        target["final_roi"] = {
            "id": target_id,
            "roi_source": roi_source,
            "final_center": target["final_center"],
            "final_side_px": target["roi_side_px"],
            "final_angle_degrees": target["roi_angle_degrees"],
            "final_corners": target["roi_corners"],
            "inner_roi_corners": target["inner_roi_corners"],
            "inner_roi": target["inner_roi"],
            "draw_on_overlay": True,
            "location_confidence": confidence,
            "geometry_status": geometry_status,
            "reason": reason,
            "stage": final_roi_stage,
        }
        target_debug.append({
            "id": target_id,
            "component_count": len(components),
            "selected_component": selected,
            "accepted": accepted,
            "roi_source": roi_source,
            "search_window_bounds": target["search_window_bounds"],
        })
        fitted_targets.append(target)

    angle_consistency_started = time.perf_counter()
    strong_angle_targets = [
        target for target in fitted_targets
        if (
            target["roi_source"] == "local_square_detected"
            and target["location_confidence"] == "high"
            and float(target["min_edge_score"]) >= 0.35
            and float(target["interior_fill_score"]) >= 0.90
            and float(target["outside_leakage_score"]) <= 0.01
            and min(
                float(target["top_boundary_coverage"]),
                float(target["right_boundary_coverage"]),
                float(target["bottom_boundary_coverage"]),
                float(target["left_boundary_coverage"]),
            ) >= 0.85
        )
    ]
    shared_angle_selection_reason = (
        "Median of strong, high-confidence local-square detections."
    )
    if len(strong_angle_targets) < 3:
        strong_angle_targets = [
            target for target in fitted_targets
            if (
                target["roi_source"] == "local_square_detected"
                and target["location_confidence"] == "high"
            )
        ]
        shared_angle_selection_reason = (
            "Median of high-confidence local-square detections; fewer than "
            "three targets passed the strict evidence gate."
        )
    if len(strong_angle_targets) < 3:
        strong_angle_targets = [
            target for target in fitted_targets
            if target["roi_source"] == "local_square_detected"
        ]
        shared_angle_selection_reason = (
            "Median of all local-square detections; fewer than three targets "
            "passed the confidence gate."
        )
    if not strong_angle_targets:
        strong_angle_targets = list(fitted_targets)
        shared_angle_selection_reason = (
            "Median of geometry fallbacks because no local-square detection "
            "was accepted."
        )

    shared_angle = float(np.median([
        float(target["roi_angle_degrees"])
        for target in strong_angle_targets
    ]))
    shared_angle_deviations = [
        abs(float(target["roi_angle_degrees"]) - shared_angle)
        for target in strong_angle_targets
    ]
    shared_angle_mad = float(np.median(shared_angle_deviations))
    shared_angle_confidence = _clamp01(
        (1.0 - min(shared_angle_mad / 5.0, 1.0))
        * min(len(strong_angle_targets) / 5.0, 1.0)
    )
    shared_angle_source_targets = [
        target["id"] for target in strong_angle_targets
    ]
    shared_angle_confidence_threshold = 0.80
    shared_angle_override_enabled = (
        shared_angle_confidence >= shared_angle_confidence_threshold
    )
    shared_angle_override_rejected_reason = (
        None
        if shared_angle_override_enabled
        else (
            "Shared angle confidence "
            f"{shared_angle_confidence:.3f} is below the "
            f"{shared_angle_confidence_threshold:.2f} application threshold; "
            "each target retains its own local/micro-refined angle."
        )
    )

    for target in fitted_targets:
        center = {
            "x": float(target["final_center"]["x"]),
            "y": float(target["final_center"]["y"]),
        }
        side_px = float(target["roi_side_px"])
        pre_angle = float(target["roi_angle_degrees"])
        if shared_angle_override_enabled:
            shared_metrics = _module4_angle_consistency_metrics(
                working, gradient, center, side_px, shared_angle
            )
            chosen_angle = shared_angle
            chosen_metrics = shared_metrics
            local_delta = max(
                -1.0, min(1.0, pre_angle - shared_angle)
            )
        else:
            shared_metrics = None
            chosen_angle = pre_angle
            chosen_metrics = _module4_angle_consistency_metrics(
                working, gradient, center, side_px, pre_angle
            )
            local_delta = 0.0

        if shared_angle_override_enabled and abs(local_delta) > 1e-6:
            local_angle = shared_angle + local_delta
            local_metrics = _module4_angle_consistency_metrics(
                working, gradient, center, side_px, local_angle
            )
            containment_preserved = (
                float(local_metrics["interior_fill_score"])
                >= float(shared_metrics["interior_fill_score"]) - 0.005
                and float(local_metrics["outside_leakage_score"])
                <= float(shared_metrics["outside_leakage_score"]) + 0.002
                and float(local_metrics["lower_quartile_edge_score"])
                >= float(shared_metrics["lower_quartile_edge_score"])
                and float(local_metrics["min_boundary_coverage"])
                >= float(shared_metrics["min_boundary_coverage"])
            )
            if (
                float(local_metrics["score"])
                > float(shared_metrics["score"]) + 0.002
                and containment_preserved
            ):
                chosen_angle = local_angle
                chosen_metrics = local_metrics

        applied_local_delta = chosen_angle - shared_angle
        angle_changed = abs(chosen_angle - pre_angle) > 1e-6
        if not shared_angle_override_enabled:
            consistency_reason = shared_angle_override_rejected_reason
        elif abs(applied_local_delta) > 1e-6:
            consistency_reason = (
                "A bounded local angle deviation improved fit while preserving "
                "fill, leakage, lower-quartile edge support, and minimum "
                "boundary coverage."
            )
        elif angle_changed:
            consistency_reason = (
                "The shared median angle replaced the target-specific angle; "
                "no bounded local deviation passed every containment gate."
            )
        else:
            consistency_reason = (
                "The target angle already matched the shared median angle."
            )

        corners = _square_points(
            center["x"], center["y"], side_px, chosen_angle
        )
        inner_roi = target["inner_roi"]
        inner_side = float(
            inner_roi.get(
                "side_px",
                min(
                    float(inner_roi.get("width", side_px * 0.72)),
                    float(inner_roi.get("height", side_px * 0.72)),
                ),
            )
        )
        inner_corners = _square_points(
            center["x"], center["y"], inner_side, chosen_angle
        )
        target.update({
            "pre_consistency_angle_degrees": round(pre_angle, 3),
            "shared_roi_angle_degrees": round(shared_angle, 3),
            "local_angle_delta_from_shared": round(applied_local_delta, 3),
            "final_angle_degrees": round(chosen_angle, 3),
            "angle_consistency_applied": angle_changed,
            "angle_consistency_reason": consistency_reason,
            "shared_angle_override_enabled": (
                shared_angle_override_enabled
            ),
            "shared_angle_override_applied": angle_changed,
            "shared_angle_override_rejected_reason": (
                shared_angle_override_rejected_reason
            ),
            "pre_shared_angle_degrees": round(pre_angle, 3),
            "post_shared_angle_degrees": round(chosen_angle, 3),
            "roi_angle_degrees": round(chosen_angle, 3),
            "fitted_angle_degrees": round(chosen_angle, 3),
            "roi_corners": corners,
            "fitted_corners": corners,
            "rotated_box": corners,
            "inner_roi_corners": inner_corners,
            "angle_consistency_fit_score": round(
                float(chosen_metrics["score"]), 5
            ),
        })
        target["inner_roi"]["angle_degrees"] = round(chosen_angle, 3)
        target["inner_roi"]["corners"] = inner_corners
        target["pre_angle_consistency_final_roi_stage"] = (
            target["final_roi_stage"]
        )
        target["pre_angle_consistency_corners"] = list(
            target["final_roi"]["final_corners"]
        )
        if shared_angle_override_enabled:
            target["final_roi_stage"] = "angle_consistency"
        target["final_roi"].update({
            "final_angle_degrees": round(chosen_angle, 3),
            "final_corners": corners,
            "inner_roi_corners": inner_corners,
            "inner_roi": target["inner_roi"],
            "stage": target["final_roi_stage"],
        })

    angle_consistency_ms = (
        time.perf_counter() - angle_consistency_started
    ) * 1000.0

    return {
        "enabled": True,
        "method": "geometry_guided_local_square_detection",
        "target_rois": fitted_targets,
        "target_debug": target_debug,
        "local_detections": sum(
            target["roi_source"] == "local_square_detected"
            for target in fitted_targets
        ),
        "needs_review": sum(
            target["roi_source"] == "geometry_fallback_needs_review"
            for target in fitted_targets
        ),
        "micro_refinements_applied": sum(
            target["micro_refinement_applied"]
            for target in fitted_targets
        ),
        "max_micro_center_shift_px": round(max(
            target["micro_center_shift_px"] for target in fitted_targets
        ), 3),
        "max_micro_side_change_px": round(max(
            abs(target["micro_side_change_px"]) for target in fitted_targets
        ), 3),
        "max_micro_angle_change_degrees": round(max(
            abs(target["micro_angle_change_degrees"])
            for target in fitted_targets
        ), 3),
        "targeted_problem_target_ids": sorted(
            MODULE4_TARGETED_POLISH_IDS
        ),
        "targeted_targets_changed": [
            target["id"] for target in fitted_targets
            if target["targeted_refinement_applied"]
        ],
        "frozen_good_targets": [
            target["id"] for target in fitted_targets
            if target["frozen_good_target"]
        ],
        "shared_roi_angle_degrees": round(shared_angle, 3),
        "shared_angle_source_targets": shared_angle_source_targets,
        "shared_angle_confidence": round(shared_angle_confidence, 4),
        "shared_angle_target_count": len(strong_angle_targets),
        "shared_angle_selection_reason": shared_angle_selection_reason,
        "shared_angle_confidence_threshold": (
            shared_angle_confidence_threshold
        ),
        "shared_angle_override_enabled": shared_angle_override_enabled,
        "shared_angle_override_applied": any(
            target["shared_angle_override_applied"]
            for target in fitted_targets
        ),
        "shared_angle_override_rejected_reason": (
            shared_angle_override_rejected_reason
        ),
        "all_targets_retained": len(fitted_targets) == 8,
        "b6_drawn": any(
            target["id"] == "B6" and target["draw_on_overlay"]
            for target in fitted_targets
        ),
        "final_roi_stage_used_by_overlay": "target.final_roi.stage",
        "final_roi_corner_field_used_by_overlay": "target.final_roi.final_corners",
        "table_roi_stage_used": "target.final_roi",
        "debug_overlay_stage_used": (
            "intermediate geometry/local/micro/targeted diagnostics plus "
            "target.final_roi as the final cyan box"
        ),
        "performance_mode": performance_mode,
        "stage_timings_ms": {
            "local_square_detection_ms": round(local_detection_ms, 2),
            "micro_refinement_ms": round(micro_refinement_ms, 2),
            "targeted_refinement_ms": round(targeted_refinement_ms, 2),
            "angle_consistency_ms": round(angle_consistency_ms, 2),
        },
        "hypothesis_counts": {
            "local_candidates_evaluated": total_local_candidates,
            "square_hypotheses_evaluated": total_square_hypotheses,
            "micro_hypotheses_evaluated": total_micro_hypotheses,
            "targeted_hypotheses_evaluated": total_targeted_hypotheses,
        },
        "performance_caps": caps,
        "_diagnostic_intermediates": diagnostic_intermediates,
    }


def fit_module4_global_geometry_full(
    slice_pixels: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
    pin_angle_degrees: float,
    current_ring_center: dict,
    current_radius_ratio: float,
    current_phi_degrees: float,
    current_roi_side_ratio: float,
    current_roi_angle_offset: float,
    current_rois: list[dict],
) -> dict:
    """Fit one shared center/radius/phi/side/angle geometry for all targets."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    phantom_x = float(phantom_center["x"])
    phantom_y = float(phantom_center["y"])
    radius = max(float(phantom_radius), 1.0)
    center_limit = 0.08 * radius
    nominal_angles = {
        "B1": 270.0, "B2": 315.0, "B3": 0.0, "B4": 45.0,
        "B5": 90.0, "B6": 135.0, "B7": 180.0, "B8": 225.0,
    }
    roi_by_id = {target["id"]: target for target in current_rois}
    used, excluded, local_debug = [], [], {}
    for target_id in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"):
        target = roi_by_id[target_id]
        local = _module4_local_target_center(
            raw, target["center"], target["roi_side_px"]
        )
        threshold = 0.35 if target["priority"] == "primary" else 0.42
        item = {
            "id": target_id,
            "priority": target["priority"],
            "pre_fit_center": target["center"],
            "local_target_center": local["center"],
            "local_target_score": local["score"],
            "target_center_confidence": local["confidence"],
            "local_target_center_method": local["method"],
        }
        local_debug[target_id] = item
        if local["confidence"] >= threshold:
            used.append(item)
        else:
            excluded.append({
                "id": target_id,
                "reason": (
                    f"Local target-center confidence {local['confidence']:.3f} "
                    f"was below the {threshold:.2f} full-fit threshold."
                ),
            })

    def errors_for(
        ring_x: float,
        ring_y: float,
        ratio: float,
        phi: float,
    ) -> list[float]:
        insert_radius = ratio * radius
        orientation = (
            float(pin_angle_degrees)
            - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
            + phi
        )
        errors = []
        for item in used:
            theta = math.radians(
                nominal_angles[item["id"]] + orientation
            )
            local = item["local_target_center"]
            errors.append(math.hypot(
                ring_x + insert_radius * math.cos(theta) - float(local["x"]),
                ring_y + insert_radius * math.sin(theta) - float(local["y"]),
            ))
        return errors

    pre_errors = errors_for(
        float(current_ring_center["x"]),
        float(current_ring_center["y"]),
        float(current_radius_ratio),
        float(current_phi_degrees),
    )
    pre_median = float(np.median(pre_errors)) if pre_errors else None
    primary_used = sum(item["priority"] == "primary" for item in used)
    if primary_used < 3 or len(used) < 4:
        return {
            "enabled": True,
            "method": "global_ring_center_radius_phi_size_angle_fit",
            "phantom_center": phantom_center,
            "calibrated_ring_center": current_ring_center,
            "center_offset_x_px": round(float(current_ring_center["x"]) - phantom_x, 3),
            "center_offset_y_px": round(float(current_ring_center["y"]) - phantom_y, 3),
            "center_offset_magnitude_px": round(math.hypot(
                float(current_ring_center["x"]) - phantom_x,
                float(current_ring_center["y"]) - phantom_y,
            ), 3),
            "radius_ratio_before": current_radius_ratio,
            "radius_ratio_after": current_radius_ratio,
            "phi_before": current_phi_degrees,
            "phi_after": current_phi_degrees,
            "roi_side_ratio_before": current_roi_side_ratio,
            "roi_side_ratio_after": current_roi_side_ratio,
            "roi_angle_offset_before": current_roi_angle_offset,
            "roi_angle_offset_after": current_roi_angle_offset,
            "pre_fit_median_center_error_px": (
                round(pre_median, 3) if pre_median is not None else None
            ),
            "post_fit_median_center_error_px": (
                round(pre_median, 3) if pre_median is not None else None
            ),
            "targets_used_for_fit": [item["id"] for item in used],
            "targets_excluded_from_fit": excluded,
            "fit_status": "fallback",
            "fit_reason": (
                "The full fit requires at least three confident primaries and "
                "four confident targets overall; the prior rigid geometry was retained."
            ),
            "local_target_diagnostics": local_debug,
        }

    best_ring = None
    for radius_step in range(55, 79):
        ratio = radius_step / 100.0
        insert_radius = ratio * radius
        for phi_step in range(-30, 31):
            phi = phi_step / 2.0
            orientation = (
                float(pin_angle_degrees)
                - MODULE4_FALLBACK_PIN_ANGLE_DEGREES + phi
            )
            x_votes, y_votes = [], []
            for item in used:
                theta = math.radians(
                    nominal_angles[item["id"]] + orientation
                )
                local = item["local_target_center"]
                x_votes.append(float(local["x"]) - insert_radius * math.cos(theta))
                y_votes.append(float(local["y"]) - insert_radius * math.sin(theta))
            ring_x = phantom_x + max(
                -center_limit,
                min(center_limit, float(np.median(x_votes)) - phantom_x),
            )
            ring_y = phantom_y + max(
                -center_limit,
                min(center_limit, float(np.median(y_votes)) - phantom_y),
            )
            errors = errors_for(ring_x, ring_y, ratio, phi)
            ordered = sorted(errors)
            median_error = float(np.median(ordered))
            trimmed = (
                float(np.mean(ordered[1:-1]))
                if len(ordered) > 3 else float(np.mean(ordered))
            )
            offset_penalty = 0.20 * math.hypot(
                ring_x - phantom_x, ring_y - phantom_y
            ) / max(center_limit, 1.0)
            boundary_penalty = (
                (0.25 if ratio in {0.55, 0.78} else 0.0)
                + (0.25 if abs(phi) == 15.0 else 0.0)
            )
            objective = (
                0.62 * median_error + 0.38 * trimmed
                + offset_penalty + boundary_penalty
            )
            trial = {
                "ring_x": ring_x, "ring_y": ring_y, "ratio": ratio,
                "phi": phi, "median_error": median_error,
                "objective": objective,
            }
            if best_ring is None or objective < best_ring["objective"]:
                best_ring = trial

    fill = float(np.median(raw[np.isfinite(raw)]))
    working = np.where(np.isfinite(raw), raw, fill)
    smooth = ndimage.gaussian_filter(working, sigma=1.0)
    gy, gx = np.gradient(smooth)
    gradient = np.hypot(gx, gy)
    orientation = (
        float(pin_angle_degrees)
        - MODULE4_FALLBACK_PIN_ANGLE_DEGREES
        + best_ring["phi"]
    )
    fitted_centers = {}
    for item in used:
        theta = math.radians(
            nominal_angles[item["id"]] + orientation
        )
        fitted_centers[item["id"]] = {
            "x": best_ring["ring_x"] + best_ring["ratio"] * radius * math.cos(theta),
            "y": best_ring["ring_y"] + best_ring["ratio"] * radius * math.sin(theta),
        }

    def boundary_layout_score(side_ratio: float, angle_offset: float) -> float:
        scores = [
            _module4_square_boundary_alignment(
                gradient,
                fitted_centers[item["id"]],
                side_ratio * radius,
                orientation + angle_offset,
            )
            for item in used
        ]
        ordered = sorted(scores)
        return (
            float(np.mean(ordered[1:-1]))
            if len(ordered) > 3 else float(np.mean(ordered))
        )

    pre_boundary_score = boundary_layout_score(
        float(current_roi_side_ratio),
        float(current_roi_angle_offset),
    )
    best_shape = None
    for side_step in range(20, 37):
        side_ratio = side_step / 200.0
        for angle_offset in range(35, 56):
            score = boundary_layout_score(side_ratio, float(angle_offset))
            penalty = (
                (0.02 if side_ratio in {0.10, 0.18} else 0.0)
                + (0.02 if angle_offset in {35, 55} else 0.0)
            )
            objective = score - penalty
            if best_shape is None or objective > best_shape["objective"]:
                best_shape = {
                    "side_ratio": side_ratio,
                    "angle_offset": float(angle_offset),
                    "score": score,
                    "objective": objective,
                }

    center_improvement = float(pre_median - best_ring["median_error"])
    boundary_improvement = float(best_shape["score"] - pre_boundary_score)
    if center_improvement >= 0.5 or boundary_improvement >= 0.03:
        status = "applied"
        reason = (
            "Applied one robust shared center/radius/phi/side/angle geometry "
            "because center or balanced boundary support improved materially."
        )
    elif center_improvement > 0.1 or boundary_improvement > 0.01:
        status = "needs_review"
        reason = (
            "The full shared-geometry fit improved modestly and remains review quality."
        )
    else:
        status = "fallback"
        reason = (
            "The full fit did not materially improve center or boundary support; "
            "the prior rigid geometry was retained."
        )
        best_ring = {
            **best_ring,
            "ring_x": float(current_ring_center["x"]),
            "ring_y": float(current_ring_center["y"]),
            "ratio": float(current_radius_ratio),
            "phi": float(current_phi_degrees),
            "median_error": float(pre_median),
        }
        best_shape = {
            "side_ratio": float(current_roi_side_ratio),
            "angle_offset": float(current_roi_angle_offset),
            "score": pre_boundary_score,
        }

    return {
        "enabled": True,
        "method": "global_ring_center_radius_phi_size_angle_fit",
        "phantom_center": phantom_center,
        "calibrated_ring_center": {
            "x": round(float(best_ring["ring_x"]), 3),
            "y": round(float(best_ring["ring_y"]), 3),
        },
        "center_offset_x_px": round(best_ring["ring_x"] - phantom_x, 3),
        "center_offset_y_px": round(best_ring["ring_y"] - phantom_y, 3),
        "center_offset_magnitude_px": round(math.hypot(
            best_ring["ring_x"] - phantom_x,
            best_ring["ring_y"] - phantom_y,
        ), 3),
        "radius_ratio_before": round(float(current_radius_ratio), 4),
        "radius_ratio_after": round(float(best_ring["ratio"]), 4),
        "phi_before": round(float(current_phi_degrees), 3),
        "phi_after": round(float(best_ring["phi"]), 3),
        "roi_side_ratio_before": round(float(current_roi_side_ratio), 4),
        "roi_side_ratio_after": round(float(best_shape["side_ratio"]), 4),
        "roi_angle_offset_before": round(float(current_roi_angle_offset), 3),
        "roi_angle_offset_after": round(float(best_shape["angle_offset"]), 3),
        "pre_fit_median_center_error_px": round(float(pre_median), 3),
        "post_fit_median_center_error_px": round(
            float(best_ring["median_error"]), 3
        ),
        "pre_fit_boundary_support": round(pre_boundary_score, 4),
        "post_fit_boundary_support": round(float(best_shape["score"]), 4),
        "targets_used_for_fit": [item["id"] for item in used],
        "targets_excluded_from_fit": excluded,
        "fit_status": status,
        "fit_reason": reason,
        "local_target_diagnostics": local_debug,
    }


def review_module4_geometry_locations(
    slice_pixels: np.ndarray,
    phantom_center: dict,
    phantom_radius: float,
    target_rois: list[dict],
    calibration_status: str,
    orientation_status: str,
) -> dict:
    """Review fixed geometry centers against nearby raw-HU evidence."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    center_x = float(phantom_center["x"])
    center_y = float(phantom_center["y"])
    radius = float(phantom_radius)
    yy, xx = np.indices(raw.shape)
    finite = np.isfinite(raw)
    interior = finite & (np.hypot(xx - center_x, yy - center_y) <= radius * 0.88)
    values = raw[interior]
    phantom_scale_hu = (
        max(10.0, float(np.percentile(values, 95.0) - np.percentile(values, 5.0)))
        if values.size >= 200 else 100.0
    )
    reviewed_targets = []
    radial_deltas = []
    tangential_deltas = []
    primary_current_scores = []
    primary_small_scores = []
    primary_large_scores = []

    for target in target_rois:
        geometry_x = float(target["center"]["x"])
        geometry_y = float(target["center"]["y"])
        side = float(target["roi_side_px"])
        local = _module4_local_target_center(raw, target["center"], side)
        local_x = float(local["center"]["x"])
        local_y = float(local["center"]["y"])
        delta_x = local_x - geometry_x
        delta_y = local_y - geometry_y
        delta = math.hypot(delta_x, delta_y)
        center_score = float(target.get("evidence_score") or 0.0)
        if (
            calibration_status == "calibrated"
            and orientation_status == "detected"
            and delta <= 4.0
            and center_score >= 0.15
        ):
            confidence = "high"
            reason = (
                "Geometry center agrees closely with nearby raw-HU insert evidence."
            )
        elif delta <= 8.0 and center_score >= 0.10:
            confidence = "moderate"
            reason = (
                "Geometry center has nearby supporting evidence, but alignment "
                "or global calibration remains review quality."
            )
        else:
            confidence = "needs_review"
            reason = (
                "Geometry ROI was retained, but nearby evidence is weak or its "
                "balanced local target centroid is displaced from the geometry center."
            )
        reviewed = {
            **target,
            "nominal_angle_degrees": target["nominal_target_angle"],
            "final_angle_degrees": target["final_target_angle"],
            "evidence_score_at_center": round(center_score, 4),
            "local_target_center": local["center"],
            "local_target_center_method": local["method"],
            "local_target_center_confidence": local["confidence"],
            "center_to_local_target_delta_px": round(delta, 3),
            # Compatibility alias for the Step 3.6 UI contract.
            "center_to_local_peak_delta_px": round(delta, 3),
            "location_confidence": confidence,
            "location_reason": reason,
            "radial_delta_px": None,
            "tangential_delta_px": None,
        }
        reviewed_targets.append(reviewed)
        if target["priority"] == "primary":
            distance = max(math.hypot(
                geometry_x - center_x,
                geometry_y - center_y,
            ), 1.0)
            radial_x = (geometry_x - center_x) / distance
            radial_y = (geometry_y - center_y) / distance
            tangent_x, tangent_y = -radial_y, radial_x
            radial_deltas.append(delta_x * radial_x + delta_y * radial_y)
            tangential_deltas.append(delta_x * tangent_x + delta_y * tangent_y)
            reviewed["radial_delta_px"] = round(radial_deltas[-1], 3)
            reviewed["tangential_delta_px"] = round(
                tangential_deltas[-1], 3
            )
            primary_current_scores.append(center_score)
            primary_small_scores.append(_module4_geometry_evidence_score(
                raw, geometry_x, geometry_y, side * 0.82, phantom_scale_hu
            ))
            primary_large_scores.append(_module4_geometry_evidence_score(
                raw, geometry_x, geometry_y, side * 1.18, phantom_scale_hu
            ))

    median_radial = float(np.median(radial_deltas)) if radial_deltas else 0.0
    median_tangential = (
        float(np.median(tangential_deltas)) if tangential_deltas else 0.0
    )
    if not radial_deltas or calibration_status == "fallback":
        radius_assessment = "needs_review"
    elif median_radial > 3.0:
        radius_assessment = "too_inward"
    elif median_radial < -3.0:
        radius_assessment = "too_outward"
    else:
        radius_assessment = "ok"
    if not tangential_deltas or orientation_status != "detected":
        phi_assessment = "needs_review"
    elif median_tangential > 2.5:
        phi_assessment = "clockwise_shift_needed"
    elif median_tangential < -2.5:
        phi_assessment = "counterclockwise_shift_needed"
    else:
        phi_assessment = "ok"
    current_size_score = (
        float(np.median(primary_current_scores))
        if primary_current_scores else 0.0
    )
    small_size_score = (
        float(np.median(primary_small_scores))
        if primary_small_scores else 0.0
    )
    large_size_score = (
        float(np.median(primary_large_scores))
        if primary_large_scores else 0.0
    )
    if not primary_current_scores or max(
        current_size_score, small_size_score, large_size_score
    ) < 0.10:
        size_assessment = "needs_review"
    elif large_size_score > current_size_score + 0.06:
        size_assessment = "too_small"
    elif small_size_score > current_size_score + 0.06:
        size_assessment = "too_large"
    else:
        size_assessment = "ok"
    recommended_side_ratio = None
    if max(current_size_score, small_size_score, large_size_score) >= 0.20:
        if size_assessment == "too_small":
            recommended_side_ratio = round(
                MODULE4_ROI_SIDE_RATIO * 1.18, 4
            )
        elif size_assessment == "too_large":
            recommended_side_ratio = round(
                MODULE4_ROI_SIDE_RATIO * 0.82, 4
            )
    # The current evidence diagnostic is intentionally rotation-invariant.
    # A reliable global angle assessment requires a dedicated edge-orientation
    # review and must not be inferred from noisy local peaks.
    angle_assessment = "needs_review"
    assessments = [
        radius_assessment,
        phi_assessment,
        size_assessment,
        angle_assessment,
    ]
    if all(item == "ok" for item in assessments):
        recommended_action = (
            "Geometry location is internally consistent; proceed to manual "
            "overlay confirmation before adding measurements."
        )
    else:
        recommended_action = (
            "Review the normal/debug overlays and the aggregated radius, phi, "
            "size, and angle diagnostics before starting measurement work."
        )
    confidence_order = {"high": 2, "moderate": 1, "needs_review": 0}
    primary_confidences = [
        target["location_confidence"]
        for target in reviewed_targets
        if target["priority"] == "primary"
    ]
    overall_confidence = (
        min(primary_confidences, key=lambda item: confidence_order[item])
        if primary_confidences else "needs_review"
    )
    return {
        "target_rois": reviewed_targets,
        "location_confidence": overall_confidence,
        "geometry_calibration_review": {
            "radius_assessment": radius_assessment,
            "phi_assessment": phi_assessment,
            "roi_size_assessment": size_assessment,
            "roi_angle_assessment": angle_assessment,
            "recommended_roi_side_ratio": recommended_side_ratio,
            "recommended_roi_angle_offset_degrees": None,
            "recommended_next_action": recommended_action,
            "median_primary_radial_delta_px": round(median_radial, 3),
            "median_primary_tangential_delta_px": round(median_tangential, 3),
            "roi_size_evidence": {
                "smaller_side_median": round(small_size_score, 4),
                "current_side_median": round(current_size_score, 4),
                "larger_side_median": round(large_size_score, 4),
            },
        },
    }


def generate_module4_location_debug_overlay(
    slice_pixels: np.ndarray,
    geometry: dict,
    window_width: float,
    window_level: float,
    photometric: str = "",
) -> str:
    """Draw geometry diagnostics separately from the normal result overlay."""
    image = window_pixels_to_image(
        slice_pixels, window_width, window_level, photometric
    )
    draw = ImageDraw.Draw(image)
    center = geometry["phantom_center"]
    center_xy = (float(center["x"]), float(center["y"]))
    ring_center = geometry.get("calibrated_ring_center", center)
    ring_center_xy = (
        float(ring_center["x"]),
        float(ring_center["y"]),
    )
    radius = float(geometry["phantom_radius"])
    insert_radius = float(geometry["insert_radius_px"])
    draw.ellipse(
        (
            center_xy[0] - radius,
            center_xy[1] - radius,
            center_xy[0] + radius,
            center_xy[1] + radius,
        ),
        outline="#60a5fa",
        width=2,
    )
    draw.ellipse(
        (
            ring_center_xy[0] - insert_radius,
            ring_center_xy[1] - insert_radius,
            ring_center_xy[0] + insert_radius,
            ring_center_xy[1] + insert_radius,
        ),
        outline="#facc15",
        width=2,
    )
    draw.line(
        (
            center_xy[0] - 5, center_xy[1],
            center_xy[0] + 5, center_xy[1],
        ),
        fill="#ffffff",
        width=2,
    )
    draw.line(
        (center_xy, ring_center_xy),
        fill="#34d399",
        width=2,
    )
    draw.ellipse(
        (
            ring_center_xy[0] - 4,
            ring_center_xy[1] - 4,
            ring_center_xy[0] + 4,
            ring_center_xy[1] + 4,
        ),
        outline="#34d399",
        width=2,
    )
    draw.line(
        (
            center_xy[0], center_xy[1] - 5,
            center_xy[0], center_xy[1] + 5,
        ),
        fill="#ffffff",
        width=2,
    )
    for pin in geometry.get("selected_bottom_pins", []):
        pin_x, pin_y = float(pin["x"]), float(pin["y"])
        draw.ellipse(
            (pin_x - 4, pin_y - 4, pin_x + 4, pin_y + 4),
            outline="#f97316",
            width=2,
        )
    midpoint = geometry.get("pin_midpoint")
    if midpoint:
        draw.line(
            (
                center_xy,
                (float(midpoint["x"]), float(midpoint["y"])),
            ),
            fill="#f97316",
            width=2,
        )
    original_centers = geometry.get("pre_global_correction_centers", {})
    local_centers_before = geometry.get(
        "primary_local_target_centers_before_correction", {}
    )
    for target_id, original in original_centers.items():
        ox, oy = float(original["x"]), float(original["y"])
        draw.ellipse(
            (ox - 3, oy - 3, ox + 3, oy + 3),
            outline="#c084fc",
            width=2,
        )
        local = local_centers_before.get(target_id)
        if local:
            lx, ly = float(local["x"]), float(local["y"])
            draw.line((ox, oy, lx, ly), fill="#f472b6", width=2)
            draw.ellipse(
                (lx - 3, ly - 3, lx + 3, ly + 3),
                fill="#f472b6",
            )
    for target in geometry.get("target_rois", []):
        final_roi = target.get("final_roi", target)
        target_center = final_roi.get("final_center", target["center"])
        tx, ty = float(target_center["x"]), float(target_center["y"])
        geometry_center = target.get("geometry_center", target_center)
        gx, gy = float(geometry_center["x"]), float(geometry_center["y"])
        bounds = target.get("search_window_bounds")
        if bounds:
            draw.rectangle(
                (
                    float(bounds["x0"]), float(bounds["y0"]),
                    float(bounds["x1"]), float(bounds["y1"]),
                ),
                outline="#a78bfa",
                width=1,
            )
        draw.ellipse((gx - 3, gy - 3, gx + 3, gy + 3), outline="#facc15", width=2)
        detected_center = target.get("detected_center")
        if detected_center:
            dx, dy = (
                float(detected_center["x"]),
                float(detected_center["y"]),
            )
            draw.line((gx, gy, dx, dy), fill="#fb7185", width=2)
            draw.ellipse((dx - 3, dy - 3, dx + 3, dy + 3), fill="#fb7185")
        pre_micro_corners = target.get("pre_micro_corners")
        if pre_micro_corners:
            pre_points = [
                (float(point["x"]), float(point["y"]))
                for point in pre_micro_corners
            ]
            draw.line(
                pre_points + [pre_points[0]],
                fill="#f472b6",
                width=1,
            )
        if target.get("targeted_refinement_applied"):
            targeted_center = target["pre_targeted_center"]
            targeted_points = _square_points(
                float(targeted_center["x"]),
                float(targeted_center["y"]),
                float(target["pre_targeted_side_px"]),
                float(target["pre_targeted_angle_degrees"]),
            )
            targeted_xy = [
                (float(point["x"]), float(point["y"]))
                for point in targeted_points
            ]
            draw.line(
                targeted_xy + [targeted_xy[0]],
                fill="#fde047",
                width=1,
            )
        draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill="#22d3ee")
        draw.text(
            (tx + 4, ty - 10),
            target.get("display_label", target["id"]),
            fill="#67e8f9",
        )
        if final_roi.get("draw_on_overlay", target.get("draw_on_overlay")):
            points = [
                (float(point["x"]), float(point["y"]))
                for point in final_roi.get(
                    "final_corners", target["roi_corners"]
                )
            ]
            draw.line(points + [points[0]], fill="#22d3ee", width=2)
        if target["priority"] == "primary" and target.get("local_target_center"):
            local = target["local_target_center"]
            lx, ly = float(local["x"]), float(local["y"])
            draw.line((tx, ty, lx, ly), fill="#fb7185", width=1)
            draw.ellipse((lx - 2, ly - 2, lx + 2, ly + 2), fill="#fb7185")
    draw.text((12, 12), "Module 4 location debug", fill="#fde047")
    return image_to_base64(image)


def phantom_geometry_pin_anchored_roi(
    slice_pixels: np.ndarray,
    pixel_spacing: tuple[float | None, float | None] | None = None,
    performance_mode: str = "fast",
) -> dict:
    """Prepare geometry anchors, then fit all eight squares in local windows."""
    raw = np.asarray(slice_pixels, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(
            f"Module 4 geometry preparation requires 2D CT pixels, got {raw.shape}."
        )

    phantom_started = time.perf_counter()
    phantom = detect_module4_phantom_geometry(raw)
    phantom_geometry_ms = (time.perf_counter() - phantom_started) * 1000.0
    pins_started = time.perf_counter()
    pins = detect_module4_bottom_pins(
        raw,
        phantom["phantom_center"],
        phantom["phantom_radius"],
    )
    angle = calculate_module4_pin_angle(
        phantom["phantom_center"],
        (
            pins["pin_midpoint"]
            if pins["pin_detection_status"] == "detected"
            else None
        ),
    )
    bottom_pin_detection_ms = (time.perf_counter() - pins_started) * 1000.0
    calibration_started = time.perf_counter()
    pre_calibration = place_module4_geometry_rois(
        phantom["phantom_center"],
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        angle["fallback_angle_used"],
    )
    if (
        performance_mode != "debug"
        and phantom["phantom_detection_status"] == "detected"
        and pins["pin_detection_status"] == "detected"
    ):
        fast_radius_started = time.perf_counter()
        calibration = calibrate_module4_geometry_fast(
            raw,
            phantom["phantom_center"],
            phantom["phantom_radius"],
            angle["pin_angle_degrees"],
        )
        fast_radius_calibration_ms = (
            time.perf_counter() - fast_radius_started
        ) * 1000.0
        geometry_calibration_ms = (
            time.perf_counter() - calibration_started
        ) * 1000.0
        calibration.update({
            "calibrated_roi_side_ratio": MODULE4_ROI_SIDE_RATIO,
            "calibrated_roi_angle_offset_degrees": (
                MODULE4_ROI_ANGLE_OFFSET_DEGREES
            ),
            "geometry_calibration_fast_mode": True,
            "geometry_calibration_skipped_or_capped": True,
            "geometry_calibration_skip_reason": (
                "Replaced the expensive full global search with a capped "
                "radius-first and tiny-phi calibration in fast mode."
            ),
            "fast_radius_calibration_ms": round(
                fast_radius_calibration_ms, 2
            ),
        })
        placements = place_module4_geometry_rois(
            phantom["phantom_center"],
            phantom["phantom_radius"],
            angle["pin_angle_degrees"],
            angle["fallback_angle_used"],
            insert_radius_ratio=calibration[
                "fast_calibrated_insert_radius_ratio"
            ],
            phi_offset_degrees=calibration[
                "fast_calibrated_phi_offset_degrees"
            ],
            target_evidence_scores=calibration[
                "target_evidence_scores"
            ],
            calibration_status=calibration["calibration_status"],
        )
        local_square_fit = fit_module4_geometry_guided_local_squares(
            raw,
            phantom["phantom_radius"],
            placements["target_rois"],
            performance_mode=performance_mode,
        )
        placements["target_rois"] = local_square_fit["target_rois"]
        calibrated_ring_center = dict(phantom["phantom_center"])
        confidence_order = {"high": 2, "moderate": 1, "needs_review": 0}
        location_confidence = min(
            (
                target["location_confidence"]
                for target in placements["target_rois"]
                if target["priority"] == "primary"
            ),
            key=lambda item: confidence_order[item],
            default="needs_review",
        )
        centers = {
            target["id"]: target["center"]
            for target in placements["target_rois"]
        }
        geometry_location_refinement = {
            "enabled": False,
            "method": "skipped_fast_mode",
            "iterations": 0,
            "iteration_details": [],
            "pre_refinement_insert_radius_ratio": (
                calibration["fast_calibrated_insert_radius_ratio"]
            ),
            "post_refinement_insert_radius_ratio": (
                calibration["fast_calibrated_insert_radius_ratio"]
            ),
            "pre_refinement_phi_offset_degrees": (
                calibration["fast_calibrated_phi_offset_degrees"]
            ),
            "post_refinement_phi_offset_degrees": (
                calibration["fast_calibrated_phi_offset_degrees"]
            ),
            "radius_correction_px": 0.0,
            "radius_correction_ratio": 0.0,
            "phi_correction_degrees": 0.0,
            "refinement_status": "skipped_fast_mode",
            "refinement_reason": calibration[
                "geometry_calibration_skip_reason"
            ],
        }
        ring_center_fit = {
            "fit_status": "skipped_fast_mode",
            "calibrated_ring_center": calibrated_ring_center,
            "center_offset_magnitude_px": 0.0,
            "radius_ratio_before": (
                calibration["fast_calibrated_insert_radius_ratio"]
            ),
            "radius_ratio_after": (
                calibration["fast_calibrated_insert_radius_ratio"]
            ),
            "phi_before": calibration[
                "fast_calibrated_phi_offset_degrees"
            ],
            "phi_after": calibration[
                "fast_calibrated_phi_offset_degrees"
            ],
            "pre_fit_median_center_error_px": None,
            "post_fit_median_center_error_px": None,
            "targets_used_for_fit": [],
            "targets_excluded_from_fit": [],
        }
        full_fit = {
            **ring_center_fit,
            "roi_side_ratio_before": MODULE4_ROI_SIDE_RATIO,
            "roi_side_ratio_after": MODULE4_ROI_SIDE_RATIO,
            "roi_angle_offset_before": MODULE4_ROI_ANGLE_OFFSET_DEGREES,
            "roi_angle_offset_after": MODULE4_ROI_ANGLE_OFFSET_DEGREES,
        }
        angle_fit = {
            "angle_fit_status": "skipped_fast_mode",
            "angle_fit_confidence": None,
            "angle_fit_score": None,
            "targets_used_for_angle_fit": [],
            "center_freeze_enabled": True,
            "center_freeze_verified": True,
            "roi_angle_offset_before": MODULE4_ROI_ANGLE_OFFSET_DEGREES,
            "roi_angle_offset_after": MODULE4_ROI_ANGLE_OFFSET_DEGREES,
            "roi_side_ratio_before": MODULE4_ROI_SIDE_RATIO,
            "roi_side_ratio_after": MODULE4_ROI_SIDE_RATIO,
            "roi_side_px_before": round(
                MODULE4_ROI_SIDE_RATIO
                * float(phantom["phantom_radius"]),
                3,
            ),
            "roi_side_px_after": round(
                MODULE4_ROI_SIDE_RATIO
                * float(phantom["phantom_radius"]),
                3,
            ),
            "size_fit_status": "skipped_fast_mode",
            "size_fit_confidence": None,
            "max_center_shift_px_after_angle_size_polish": 0.0,
        }
        calibration_review = {
            "radius_assessment": "fast_mode_local_fit",
            "phi_assessment": "fast_mode_local_fit",
            "roi_size_assessment": "local_square_detection",
            "roi_angle_assessment": "local_square_detection",
            "recommended_next_action": (
                "Review local square ROIs; use debug mode only when full "
                "global calibration diagnostics are required."
            ),
        }
        geometry_status = (
            "prepared"
            if local_square_fit["needs_review"] == 0 else "needs_review"
        )
        return {
            "geometry_source": "phantom_geometry_pin_anchored_roi",
            "geometry_step": 4,
            "geometry_status": geometry_status,
            "geometry_stage": "geometry_guided_local_square_detection",
            "geometry_reason": (
                "Fast mode used phantom/pin anchored default placement, then "
                "fit all eight squares inside their local windows."
            ),
            **phantom,
            **pins,
            **angle,
            **placements,
            "geometry_calibration": calibration,
            "geometry_location_refinement": geometry_location_refinement,
            "geometry_ring_center_fit": ring_center_fit,
            "geometry_full_fit": full_fit,
            "geometry_angle_fit": angle_fit,
            "geometry_local_square_fit": local_square_fit,
            "calibrated_ring_center": calibrated_ring_center,
            "geometry_calibration_review": calibration_review,
            "location_confidence": location_confidence,
            "pre_calibration_centers": centers,
            "post_calibration_centers": centers,
            "center_shift_px_by_target": {
                target_id: 0.0 for target_id in centers
            },
            "pre_global_correction_centers": centers,
            "primary_local_target_centers_before_correction": {},
            "normal_overlay_targets": [
                "B1", "B2", "B3", "B4",
                "B5", "B6", "B7", "B8",
            ],
            "bottom_pin_anchoring_status": "implemented",
            "target_roi_placement_status": "implemented",
            "module4_scoring_status": "automatic_preliminary_available",
            "image_evidence_role": "bounded_local_square_location_only",
            "legacy_image_locator_active": False,
            "legacy_path_active": False,
            "geometry_calibration_fast_mode": True,
            "geometry_calibration_skipped_or_capped": True,
            "geometry_calibration_skip_reason": calibration[
                "geometry_calibration_skip_reason"
            ],
            "pixel_spacing": {
                "row_mm": pixel_spacing[0] if pixel_spacing else None,
                "column_mm": pixel_spacing[1] if pixel_spacing else None,
                "used_for_geometry": False,
            },
            "performance": {
                "phantom_geometry_ms": round(phantom_geometry_ms, 2),
                "bottom_pin_detection_ms": round(
                    bottom_pin_detection_ms, 2
                ),
                "geometry_calibration_ms": round(
                    geometry_calibration_ms, 2
                ),
                "fast_radius_calibration_ms": round(
                    fast_radius_calibration_ms, 2
                ),
                **local_square_fit["stage_timings_ms"],
            },
        }
    try:
        calibration = calibrate_module4_geometry_rois(
            raw,
            phantom["phantom_center"],
            phantom["phantom_radius"],
            angle["pin_angle_degrees"],
        )
    except Exception as exc:
        calibration = {
            "enabled": True,
            "method": "bounded_global_radius_phi_search",
            "default_insert_radius_ratio": MODULE4_INSERT_RADIUS_RATIO,
            "calibrated_insert_radius_ratio": MODULE4_INSERT_RADIUS_RATIO,
            "default_phi_offset_degrees": MODULE4_PHI_OFFSET_DEGREES,
            "calibrated_phi_offset_degrees": MODULE4_PHI_OFFSET_DEGREES,
            "search_radius_range": [0.55, 0.78],
            "search_phi_range": [-15.0, 15.0],
            "calibration_score": 0.0,
            "calibration_confidence": 0.0,
            "calibration_status": "fallback",
            "calibration_reason": f"Calibration failed safely: {exc}",
            "target_evidence_scores": {
                target_id: 0.0
                for target_id in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
            },
            "candidate_count": 0,
        }
    provisional_placements = place_module4_geometry_rois(
        phantom["phantom_center"],
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        angle["fallback_angle_used"],
        insert_radius_ratio=calibration["calibrated_insert_radius_ratio"],
        phi_offset_degrees=calibration["calibrated_phi_offset_degrees"],
        target_evidence_scores=calibration["target_evidence_scores"],
        calibration_status=calibration["calibration_status"],
    )
    pre_refinement_ratio = float(
        calibration["calibrated_insert_radius_ratio"]
    )
    pre_refinement_phi = float(calibration["calibrated_phi_offset_degrees"])
    current_ratio = pre_refinement_ratio
    current_phi = pre_refinement_phi
    current_placements = provisional_placements
    refinement_passes = []
    for iteration_index in range(1, 3):
        correction = correct_module4_geometry_from_primary_centers(
            raw,
            phantom["phantom_center"],
            phantom["phantom_radius"],
            current_placements["target_rois"],
            current_ratio,
            current_phi,
        )
        correction["iteration"] = iteration_index
        refinement_passes.append(correction)
        current_ratio = float(correction["post_correction_radius_ratio"])
        current_phi = float(correction["post_correction_phi_degrees"])
        final_evidence_scores = _module4_layout_target_evidence_scores(
            raw,
            phantom["phantom_center"],
            phantom["phantom_radius"],
            angle["pin_angle_degrees"],
            current_ratio,
            current_phi,
        )
        current_placements = place_module4_geometry_rois(
            phantom["phantom_center"],
            phantom["phantom_radius"],
            angle["pin_angle_degrees"],
            angle["fallback_angle_used"],
            insert_radius_ratio=current_ratio,
            phi_offset_degrees=current_phi,
            target_evidence_scores=final_evidence_scores,
            calibration_status=calibration["calibration_status"],
        )
        if (
            abs(correction["median_primary_radial_delta_px_before"]) < 0.75
            and abs(correction["median_primary_tangential_delta_px_before"]) < 0.75
        ):
            break
    placements = current_placements
    calibration["calibrated_insert_radius_ratio"] = round(current_ratio, 4)
    calibration["calibrated_phi_offset_degrees"] = round(current_phi, 3)
    calibration["target_evidence_scores"] = final_evidence_scores
    pre_ring_fit_placements = placements
    ring_center_fit = fit_module4_global_ring_geometry(
        raw,
        phantom["phantom_center"],
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        current_ratio,
        current_phi,
        placements["target_rois"],
    )
    calibrated_ring_center = ring_center_fit["calibrated_ring_center"]
    current_ratio = float(ring_center_fit["radius_ratio_after"])
    current_phi = float(ring_center_fit["phi_after"])
    final_evidence_scores = _module4_layout_target_evidence_scores(
        raw,
        calibrated_ring_center,
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        current_ratio,
        current_phi,
    )
    placements = place_module4_geometry_rois(
        calibrated_ring_center,
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        angle["fallback_angle_used"],
        insert_radius_ratio=current_ratio,
        phi_offset_degrees=current_phi,
        target_evidence_scores=final_evidence_scores,
        calibration_status=calibration["calibration_status"],
    )
    calibration["calibrated_insert_radius_ratio"] = round(current_ratio, 4)
    calibration["calibrated_phi_offset_degrees"] = round(current_phi, 3)
    calibration["target_evidence_scores"] = final_evidence_scores
    pre_full_fit_placements = placements
    full_fit = fit_module4_global_geometry_full(
        raw,
        phantom["phantom_center"],
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        calibrated_ring_center,
        current_ratio,
        current_phi,
        MODULE4_ROI_SIDE_RATIO,
        MODULE4_ROI_ANGLE_OFFSET_DEGREES,
        placements["target_rois"],
    )
    calibrated_ring_center = full_fit["calibrated_ring_center"]
    current_ratio = float(full_fit["radius_ratio_after"])
    current_phi = float(full_fit["phi_after"])
    current_side_ratio = float(full_fit["roi_side_ratio_after"])
    current_angle_offset = float(full_fit["roi_angle_offset_after"])
    angle_fit_placements = place_module4_geometry_rois(
        calibrated_ring_center,
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        angle["fallback_angle_used"],
        insert_radius_ratio=current_ratio,
        phi_offset_degrees=current_phi,
        roi_side_ratio=current_side_ratio,
        roi_angle_offset_degrees=current_angle_offset,
        target_evidence_scores=final_evidence_scores,
        calibration_status=calibration["calibration_status"],
    )
    angle_fit = fit_module4_global_roi_angle(
        raw,
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        current_phi,
        current_side_ratio,
        current_angle_offset,
        angle_fit_placements["target_rois"],
    )
    current_angle_offset = float(angle_fit["roi_angle_offset_after"])
    current_side_ratio = float(angle_fit["roi_side_ratio_after"])
    final_evidence_scores = _module4_layout_target_evidence_scores(
        raw,
        calibrated_ring_center,
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        current_ratio,
        current_phi,
        roi_side_ratio=current_side_ratio,
    )
    placements = place_module4_geometry_rois(
        calibrated_ring_center,
        phantom["phantom_radius"],
        angle["pin_angle_degrees"],
        angle["fallback_angle_used"],
        insert_radius_ratio=current_ratio,
        phi_offset_degrees=current_phi,
        roi_side_ratio=current_side_ratio,
        roi_angle_offset_degrees=current_angle_offset,
        target_evidence_scores=final_evidence_scores,
        calibration_status=calibration["calibration_status"],
    )
    pre_polish_centers = {
        target["id"]: target["center"]
        for target in angle_fit_placements["target_rois"]
    }
    max_polish_center_shift = max(
        math.hypot(
            float(target["center"]["x"])
            - float(pre_polish_centers[target["id"]]["x"]),
            float(target["center"]["y"])
            - float(pre_polish_centers[target["id"]]["y"]),
        )
        for target in placements["target_rois"]
    )
    angle_fit["max_center_shift_px_after_angle_size_polish"] = round(
        max_polish_center_shift, 6
    )
    angle_fit["center_freeze_verified"] = max_polish_center_shift <= 1e-6
    calibration["calibrated_insert_radius_ratio"] = round(current_ratio, 4)
    calibration["calibrated_phi_offset_degrees"] = round(current_phi, 3)
    calibration["calibrated_roi_side_ratio"] = round(current_side_ratio, 4)
    calibration["calibrated_roi_angle_offset_degrees"] = round(
        current_angle_offset, 3
    )
    calibration["target_evidence_scores"] = final_evidence_scores
    location_review = review_module4_geometry_locations(
        raw,
        calibrated_ring_center,
        phantom["phantom_radius"],
        placements["target_rois"],
        calibration["calibration_status"],
        angle["orientation_status"],
    )
    first_correction = refinement_passes[0]
    location_review["geometry_calibration_review"].update({
        "median_primary_radial_delta_px_before": first_correction[
            "median_primary_radial_delta_px_before"
        ],
        "median_primary_radial_delta_px_after": location_review[
            "geometry_calibration_review"
        ]["median_primary_radial_delta_px"],
        "median_primary_tangential_delta_px_before": first_correction[
            "median_primary_tangential_delta_px_before"
        ],
        "median_primary_tangential_delta_px_after": location_review[
            "geometry_calibration_review"
        ]["median_primary_tangential_delta_px"],
    })
    total_radius_ratio_correction = current_ratio - pre_refinement_ratio
    total_phi_correction = current_phi - pre_refinement_phi
    if not refinement_passes:
        refinement_status = "fallback"
        refinement_reason = "No global location-refinement pass completed."
    elif location_review["location_confidence"] == "needs_review":
        refinement_status = "needs_review"
        refinement_reason = (
            "Global radius/phi refinement completed, but at least one primary "
            "target still has weak or displaced local-center evidence."
        )
    else:
        refinement_status = "applied"
        refinement_reason = (
            "Applied bounded median-primary radial/tangential corrections to "
            "one rigid B1-B8 geometry layout."
        )
    geometry_location_refinement = {
        "enabled": True,
        "method": "local_target_center_global_radius_phi_correction",
        "iterations": len(refinement_passes),
        "iteration_details": refinement_passes,
        "pre_refinement_insert_radius_ratio": round(
            pre_refinement_ratio, 4
        ),
        "post_refinement_insert_radius_ratio": round(current_ratio, 4),
        "pre_refinement_phi_offset_degrees": round(pre_refinement_phi, 3),
        "post_refinement_phi_offset_degrees": round(current_phi, 3),
        "radius_correction_px": round(
            total_radius_ratio_correction * float(phantom["phantom_radius"]),
            3,
        ),
        "radius_correction_ratio": round(
            total_radius_ratio_correction, 4
        ),
        "phi_correction_degrees": round(total_phi_correction, 3),
        "median_radial_delta_px_before": first_correction[
            "median_primary_radial_delta_px_before"
        ],
        "median_radial_delta_px_after": location_review[
            "geometry_calibration_review"
        ]["median_primary_radial_delta_px_after"],
        "median_tangential_delta_px_before": first_correction[
            "median_primary_tangential_delta_px_before"
        ],
        "median_tangential_delta_px_after": location_review[
            "geometry_calibration_review"
        ]["median_primary_tangential_delta_px_after"],
        "refinement_status": refinement_status,
        "refinement_reason": refinement_reason,
    }
    placements["target_rois"] = location_review["target_rois"]
    provisional_by_id = {
        target["id"]: target
        for target in provisional_placements["target_rois"]
    }
    initial_primary_debug = first_correction[
        "primary_local_target_diagnostics"
    ]
    pre_ring_by_id = {
        target["id"]: target
        for target in pre_ring_fit_placements["target_rois"]
    }
    ring_fit_debug = ring_center_fit.get("local_target_diagnostics", {})
    excluded_ring_targets = {
        item["id"]
        for item in ring_center_fit.get("targets_excluded_from_fit", [])
    }
    pre_full_by_id = {
        target["id"]: target
        for target in pre_full_fit_placements["target_rois"]
    }
    full_fit_debug = full_fit.get("local_target_diagnostics", {})
    excluded_full_targets = {
        item["id"]
        for item in full_fit.get("targets_excluded_from_fit", [])
    }
    for target in placements["target_rois"]:
        target["pre_refinement_center"] = provisional_by_id[
            target["id"]
        ]["center"]
        target["post_refinement_center"] = target["center"]
        before_debug = initial_primary_debug.get(target["id"])
        target["center_to_local_target_delta_px_before"] = (
            before_debug["center_to_local_target_delta_px"]
            if before_debug else None
        )
        target["center_to_local_target_delta_px_after"] = target[
            "center_to_local_target_delta_px"
        ]
        target["pre_ring_fit_center"] = pre_ring_by_id[
            target["id"]
        ]["center"]
        target["post_ring_fit_center"] = target["center"]
        ring_debug = ring_fit_debug.get(target["id"])
        target["center_delta_before_px"] = (
            round(math.hypot(
                float(pre_ring_by_id[target["id"]]["center"]["x"])
                - float(ring_debug["local_target_center"]["x"]),
                float(pre_ring_by_id[target["id"]]["center"]["y"])
                - float(ring_debug["local_target_center"]["y"]),
            ), 3)
            if ring_debug else None
        )
        target["center_delta_after_px"] = target[
            "center_to_local_target_delta_px"
        ]
        target["pre_fit_center"] = pre_full_by_id[target["id"]]["center"]
        target["post_fit_center"] = target["center"]
        full_debug = full_fit_debug.get(target["id"])
        target["center_delta_before_px"] = (
            round(math.hypot(
                float(pre_full_by_id[target["id"]]["center"]["x"])
                - float(full_debug["local_target_center"]["x"]),
                float(pre_full_by_id[target["id"]]["center"]["y"])
                - float(full_debug["local_target_center"]["y"]),
            ), 3)
            if full_debug else target["center_delta_before_px"]
        )
        target["center_delta_after_px"] = target[
            "center_to_local_target_delta_px"
        ]
        if target["id"] in excluded_ring_targets:
            target["location_confidence"] = "needs_review"
            target["location_reason"] = (
                f'{target["id"]} remained geometry-placed and drawn, but its '
                "local target center was too weak to drive the global ring fit."
            )
        if target["id"] in excluded_full_targets:
            target["location_confidence"] = "needs_review"
            target["location_reason"] = (
                f'{target["id"]} remains drawn from the final shared geometry, '
                "but its local evidence was too weak to drive the full fit."
            )
        if full_fit["fit_status"] != "applied":
            target["geometry_status"] = "geometry_needs_review"
    geometry_calibration_ms = (
        time.perf_counter() - calibration_started
    ) * 1000.0
    local_square_fit = fit_module4_geometry_guided_local_squares(
        raw,
        phantom["phantom_radius"],
        placements["target_rois"],
        performance_mode=performance_mode,
    )
    placements["target_rois"] = local_square_fit["target_rois"]
    confidence_order = {"high": 2, "moderate": 1, "needs_review": 0}
    location_review["location_confidence"] = min(
        (
            target["location_confidence"]
            for target in placements["target_rois"]
            if target["priority"] == "primary"
        ),
        key=lambda item: confidence_order[item],
        default="needs_review",
    )
    pre_centers = {
        target["id"]: target["center"]
        for target in pre_calibration["target_rois"]
    }
    post_centers = {
        target["id"]: target["center"]
        for target in placements["target_rois"]
    }
    center_shifts = {
        target_id: round(math.hypot(
            post_centers[target_id]["x"] - pre_center["x"],
            post_centers[target_id]["y"] - pre_center["y"],
        ), 3)
        for target_id, pre_center in pre_centers.items()
    }
    if (
        phantom["phantom_detection_status"] == "detected"
        and pins["pin_detection_status"] == "detected"
        and calibration["calibration_status"] == "calibrated"
        and full_fit["fit_status"] == "applied"
        and local_square_fit["needs_review"] == 0
    ):
        geometry_status = "prepared"
        geometry_reason = (
            "Phantom envelope and two bottom-pin orientation anchors detected; "
            "geometry-guided local searches retained all eight target ROIs."
        )
    else:
        geometry_status = "needs_review"
        geometry_reason = (
            "All eight target ROIs were retained, but phantom, pin, calibration, "
            "shared geometry, or local square confidence requires review."
        )
    return {
        "geometry_source": "phantom_geometry_pin_anchored_roi",
        "geometry_step": 4,
        "geometry_status": geometry_status,
        "geometry_stage": "geometry_guided_local_square_detection",
        "geometry_reason": geometry_reason,
        **phantom,
        **pins,
        **angle,
        **placements,
        "geometry_calibration": calibration,
        "geometry_location_refinement": geometry_location_refinement,
        "geometry_ring_center_fit": ring_center_fit,
        "geometry_full_fit": full_fit,
        "geometry_angle_fit": angle_fit,
        "geometry_local_square_fit": local_square_fit,
        "calibrated_ring_center": calibrated_ring_center,
        "geometry_calibration_review": location_review[
            "geometry_calibration_review"
        ],
        "location_confidence": location_review["location_confidence"],
        "pre_calibration_centers": pre_centers,
        "post_calibration_centers": post_centers,
        "center_shift_px_by_target": center_shifts,
        "pre_global_correction_centers": {
            target["id"]: target["center"]
            for target in pre_full_fit_placements["target_rois"]
        },
        "primary_local_target_centers_before_correction": {
            target_id: diagnostic["local_target_center"]
            for target_id, diagnostic in full_fit_debug.items()
        },
        "normal_overlay_targets": [
            "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"
        ],
        "bottom_pin_anchoring_status": "implemented",
        "target_roi_placement_status": "implemented",
        "module4_scoring_status": "pending",
        "image_evidence_role": "bounded_local_square_location_only",
        "legacy_image_locator_active": False,
        "legacy_path_active": False,
        "pixel_spacing": {
            "row_mm": pixel_spacing[0] if pixel_spacing else None,
            "column_mm": pixel_spacing[1] if pixel_spacing else None,
            "used_for_geometry": False,
        },
        "performance": {
            "phantom_geometry_ms": round(phantom_geometry_ms, 2),
            "bottom_pin_detection_ms": round(bottom_pin_detection_ms, 2),
            "geometry_calibration_ms": round(geometry_calibration_ms, 2),
            **local_square_fit["stage_timings_ms"],
        },
    }


def _export_module4_fit_diagnostics(
    raw: np.ndarray,
    geometry: dict,
    window_width: float,
    window_level: float,
    photometric: str,
) -> dict:
    """Write a diagnostic-only comparison of every ROI fitting stage."""
    output_dir = Path(__file__).resolve().parents[1] / "debug_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_fit = geometry["geometry_local_square_fit"]
    intermediates = local_fit.get("_diagnostic_intermediates", {})
    targets = {
        target["id"]: target for target in local_fit["target_rois"]
    }
    display_image = window_pixels_to_image(
        raw, window_width, window_level, photometric
    ).convert("RGB")
    colors = {
        "geometry": "#9ca3af",
        "local": "#fde047",
        "micro": "#f472b6",
        "targeted": "#fb923c",
        "final": "#22d3ee",
        "component": "#22c55e",
    }

    def normalized_image(values: np.ndarray) -> Image.Image:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return Image.new("RGB", (max(values.shape[1], 1), max(values.shape[0], 1)))
        low, high = np.percentile(finite, [1.0, 99.0])
        scaled = np.clip(
            (values - float(low)) / max(float(high - low), 1e-6),
            0.0,
            1.0,
        )
        return Image.fromarray(np.uint8(scaled * 255.0), mode="L").convert("RGB")

    def mask_image(mask: np.ndarray, selected: np.ndarray) -> Image.Image:
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb[mask] = (160, 160, 160)
        rgb[selected] = (34, 197, 94)
        return Image.fromarray(rgb, mode="RGB")

    def local_points(points: list[dict], bounds: dict) -> list[tuple[float, float]]:
        return [
            (
                float(point["x"]) - float(bounds["x0"]),
                float(point["y"]) - float(bounds["y0"]),
            )
            for point in (points or [])
        ]

    def draw_polygon(
        image: Image.Image,
        points: list[dict],
        bounds: dict,
        color: str,
        width: int = 2,
    ) -> None:
        xy = local_points(points, bounds)
        if len(xy) >= 3:
            ImageDraw.Draw(image).line(xy + [xy[0]], fill=color, width=width)

    def polygon_mask(
        shape: tuple[int, int],
        points: list[dict],
        bounds: dict,
    ) -> np.ndarray:
        canvas = Image.new("1", (shape[1], shape[0]), 0)
        xy = local_points(points, bounds)
        if len(xy) >= 3:
            ImageDraw.Draw(canvas).polygon(xy, fill=1)
        return np.asarray(canvas, dtype=bool)

    def labelled_panel(image: Image.Image, label: str) -> Image.Image:
        panel = Image.new("RGB", (200, 220), "#07111f")
        fitted = image.copy()
        fitted.thumbnail((188, 184), Image.Resampling.NEAREST)
        panel.paste(
            fitted,
            ((panel.width - fitted.width) // 2, 28),
        )
        ImageDraw.Draw(panel).text((8, 7), label, fill="#e5eef9")
        return panel

    records = []
    row_images = []
    html_rows = []
    order = [f"B{index}" for index in range(1, 9)]
    for target_id in order:
        target = targets[target_id]
        artifact = intermediates[target_id]
        bounds = artifact["bounds"]
        crop_box = (
            int(bounds["x0"]), int(bounds["y0"]),
            int(bounds["x1"]), int(bounds["y1"]),
        )
        display_crop = display_image.crop(crop_box)
        residual_panel = normalized_image(artifact["residual"])
        masks_panel = mask_image(
            artifact["threshold_mask"],
            artifact["selected_component_mask"],
        )
        selected = artifact["selected_component"]
        final_roi = target["final_roi"]
        final_corners = final_roi["final_corners"]

        final_display = display_crop.copy()
        draw_polygon(final_display, final_corners, bounds, colors["final"], 2)

        stages_display = display_crop.copy()
        draw_polygon(
            stages_display, artifact["geometry_corners"], bounds,
            colors["geometry"], 1,
        )
        draw_polygon(
            stages_display, target.get("pre_micro_corners", []), bounds,
            colors["local"], 1,
        )
        micro_corners = _square_points(
            float(target["post_micro_center"]["x"]),
            float(target["post_micro_center"]["y"]),
            float(target["post_micro_side_px"]),
            float(target["post_micro_angle_degrees"]),
        )
        draw_polygon(stages_display, micro_corners, bounds, colors["micro"], 1)
        targeted_corners = (
            target.get("pre_angle_consistency_corners", [])
            if target.get("targeted_refinement_applied") else []
        )
        draw_polygon(
            stages_display, targeted_corners, bounds, colors["targeted"], 1
        )
        draw_polygon(stages_display, final_corners, bounds, colors["final"], 2)

        contour_panel = display_crop.copy()
        selected_mask = artifact["selected_component_mask"]
        contour = selected_mask & ~ndimage.binary_erosion(selected_mask)
        contour_y, contour_x = np.where(contour)
        contour_draw = ImageDraw.Draw(contour_panel)
        for px, py in zip(contour_x.tolist(), contour_y.tolist()):
            contour_draw.point((px, py), fill=colors["component"])
        if selected and selected.get("bbox"):
            bbox = selected["bbox"]
            contour_draw.rectangle(
                (
                    bbox["x0"] - bounds["x0"],
                    bbox["y0"] - bounds["y0"],
                    bbox["x1"] - bounds["x0"],
                    bbox["y1"] - bounds["y0"],
                ),
                outline=colors["component"],
                width=1,
            )

        component_compare = contour_panel.copy()
        draw_polygon(
            component_compare, final_corners, bounds, colors["final"], 2
        )

        final_mask = polygon_mask(
            selected_mask.shape, final_corners, bounds
        )
        intersection = int(np.count_nonzero(final_mask & selected_mask))
        component_area = int(np.count_nonzero(selected_mask))
        final_area = int(np.count_nonzero(final_mask))
        component_inside_final = (
            intersection / component_area if component_area else None
        )
        final_occupied_by_component = (
            intersection / final_area if final_area else None
        )
        windowed_array = np.asarray(display_crop.convert("L"), dtype=np.uint8)
        visible_threshold = float(np.percentile(windowed_array, 85.0))
        visible_white_mask = windowed_array >= visible_threshold
        visible_overlap = int(np.count_nonzero(final_mask & visible_white_mask))
        visible_inside_final = (
            visible_overlap / int(np.count_nonzero(visible_white_mask))
            if np.any(visible_white_mask) else None
        )

        edge_scores = {
            name: target.get(f"{name}_edge_score")
            for name in ("top", "right", "bottom", "left")
        }
        boundary_coverage = {
            name: target.get(f"{name}_boundary_coverage")
            for name in ("top", "right", "bottom", "left")
        }
        record = {
            "id": target_id,
            "roi_source": target.get("roi_source"),
            "final_roi": {
                "stage": final_roi.get("stage"),
                "final_center": final_roi.get("final_center"),
                "final_side_px": final_roi.get("final_side_px"),
                "final_angle_degrees": final_roi.get(
                    "final_angle_degrees"
                ),
                "final_corners": final_corners,
                "inner_roi_corners": final_roi.get("inner_roi_corners"),
            },
            "geometry_center": target.get("geometry_center"),
            "detected_center": target.get("selected_component_center"),
            "final_center": final_roi.get("final_center"),
            "final_side_px": final_roi.get("final_side_px"),
            "final_angle_degrees": final_roi.get("final_angle_degrees"),
            "final_corners": final_corners,
            "inner_roi_corners": final_roi.get("inner_roi_corners"),
            "search_window_bounds": bounds,
            "selected_component_found": selected is not None,
            "selected_component_area": (
                selected.get("area") if selected else None
            ),
            "selected_component_center": (
                selected.get("center") if selected else None
            ),
            "selected_component_bbox": (
                selected.get("bbox") if selected else None
            ),
            "fitted_rect_width_px": target.get("fitted_side_px"),
            "fitted_rect_height_px": target.get("fitted_side_px"),
            "fitted_rect_angle_degrees": target.get(
                "fitted_angle_degrees"
            ),
            "local_square_score": target.get("local_square_score"),
            "edge_scores": edge_scores,
            "boundary_coverage": boundary_coverage,
            "fill": target.get("interior_fill_score"),
            "leakage": target.get("outside_leakage_score"),
            "micro": {
                key: value for key, value in target.items()
                if "micro" in key
            },
            "targeted": {
                key: value for key, value in target.items()
                if "targeted" in key
            },
            "final_roi_corner_field_used_by_overlay": (
                local_fit["final_roi_corner_field_used_by_overlay"]
            ),
            "diagnostic_comparison": {
                "component_pixels_inside_final_roi_ratio": (
                    round(component_inside_final, 5)
                    if component_inside_final is not None else None
                ),
                "final_roi_pixels_occupied_by_component_ratio": (
                    round(final_occupied_by_component, 5)
                    if final_occupied_by_component is not None else None
                ),
                "visible_white_pixels_inside_final_roi_ratio": (
                    round(visible_inside_final, 5)
                    if visible_inside_final is not None else None
                ),
                "threshold_value_raw_residual": round(
                    artifact["threshold_value"], 5
                ),
                "display_white_threshold": visible_threshold,
            },
        }
        records.append(record)

        panels = [
            labelled_panel(final_display, "Display + final ROI"),
            labelled_panel(stages_display, "All ROI stages"),
            labelled_panel(residual_panel, "Positive residual"),
            labelled_panel(masks_panel, "Threshold / selected mask"),
            labelled_panel(contour_panel, "Component contour / bbox"),
            labelled_panel(component_compare, "Final ROI vs component"),
        ]
        row = Image.new("RGB", (sum(p.width for p in panels), 245), "#020617")
        ImageDraw.Draw(row).text((8, 222), target_id, fill="#ffffff")
        cursor = 0
        for panel in panels:
            row.paste(panel, (cursor, 0))
            cursor += panel.width
        row_images.append(row)
        html_rows.append(
            f"<section><h2>{html_lib.escape(target_id)}</h2>"
            f"<img src='{image_to_base64(row)}' "
            f"alt='{target_id} diagnostic row'>"
            f"<pre>{html_lib.escape(json.dumps(record, indent=2))}</pre>"
            "</section>"
        )

    def group_summary(ids: list[str]) -> dict:
        group = [record for record in records if record["id"] in ids]

        def average(path: tuple[str, ...]) -> float | None:
            values = []
            for record in group:
                value = record
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                if isinstance(value, (int, float)):
                    values.append(float(value))
            return round(float(np.mean(values)), 5) if values else None

        return {
            "targets": ids,
            "mean_final_side_px": average(("final_side_px",)),
            "mean_final_angle_degrees": average(("final_angle_degrees",)),
            "mean_fill": average(("fill",)),
            "mean_leakage": average(("leakage",)),
            "mean_component_area": average(("selected_component_area",)),
            "mean_component_inside_final_roi_ratio": average((
                "diagnostic_comparison",
                "component_pixels_inside_final_roi_ratio",
            )),
            "mean_final_roi_component_occupancy": average((
                "diagnostic_comparison",
                "final_roi_pixels_occupied_by_component_ratio",
            )),
            "mean_visible_white_inside_final_roi_ratio": average((
                "diagnostic_comparison",
                "visible_white_pixels_inside_final_roi_ratio",
            )),
        }

    comparison = {
        "good_looking_targets": group_summary(
            ["B1", "B2", "B3", "B4", "B5"]
        ),
        "problem_looking_targets": group_summary(["B6", "B7", "B8"]),
        "interpretation_note": (
            "Component/final overlap compares the raw-HU residual component "
            "with the cyan final ROI. Visible-white overlap is a diagnostic "
            "comparison against the windowed display and is not used to fit."
        ),
    }
    payload = {
        "analysis": "Module 4 fit diagnostics",
        "diagnostics_only": True,
        "normal_overlay_source": "final_roi.final_corners",
        "display_window": {
            "window_width": round(float(window_width), 3),
            "window_level": round(float(window_level), 3),
            "visualization_only": True,
        },
        "legend": colors,
        "targets": records,
        "good_vs_bad_comparison": comparison,
    }
    json_path = output_dir / "module4_fit_diagnostics.json"
    html_path = output_dir / "module4_fit_diagnostics.html"
    png_path = output_dir / "module4_fit_diagnostics.png"
    json_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    composite = Image.new(
        "RGB",
        (max(row.width for row in row_images), sum(row.height for row in row_images)),
        "#020617",
    )
    cursor_y = 0
    for row in row_images:
        composite.paste(row, (0, cursor_y))
        cursor_y += row.height
    composite.save(png_path)
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Module 4 Fit Diagnostics</title><style>"
        "body{font-family:system-ui;background:#07111f;color:#e5eef9;margin:24px}"
        "section{border-top:1px solid #334155;padding:18px 0}"
        "img{width:100%;max-width:1200px;background:#020617}"
        "pre{white-space:pre-wrap;background:#0f172a;padding:12px;border-radius:8px}"
        "</style></head><body><h1>Module 4 Fit Diagnostics</h1>"
        "<p>Rows in the PNG are ordered B1-B8. Colors: geometry gray, local "
        "yellow, micro pink, targeted orange, final cyan, component green.</p>"
        f"<pre>{html_lib.escape(json.dumps(comparison, indent=2))}</pre>"
        + "".join(html_rows)
        + "</body></html>",
        encoding="utf-8",
    )
    return {
        "status": "written",
        "json": str(json_path),
        "html": str(html_path),
        "png": str(png_path),
    }


def extract_module4_preliminary_roi_data(
    raw_pixels: np.ndarray,
    targets: list[dict],
) -> dict:
    """Read raw-HU review data from fixed final ROIs without changing them."""
    raw = np.asarray(raw_pixels, dtype=np.float32)
    finite_raw = np.isfinite(raw)
    finite_fill = (
        float(np.median(raw[finite_raw])) if np.any(finite_raw) else 0.0
    )
    working = np.where(finite_raw, raw, finite_fill)
    extraction_started = time.perf_counter()
    extraction_warnings = []

    def corners_for(target: dict, inner: bool) -> list[dict]:
        final_roi = target.get("final_roi") or {}
        key = "inner_roi_corners" if inner else "final_corners"
        corners = final_roi.get(key) or target.get(key) or []
        if len(corners) >= 3:
            return corners
        if not inner:
            return []
        inner_roi = final_roi.get("inner_roi") or target.get("inner_roi") or {}
        center = inner_roi.get("center") or final_roi.get("final_center")
        width = inner_roi.get("width", inner_roi.get("side_px"))
        height = inner_roi.get("height", width)
        angle = inner_roi.get(
            "angle_degrees", final_roi.get("final_angle_degrees")
        )
        if center and width is not None and height is not None and angle is not None:
            return _square_points(
                float(center["x"]),
                float(center["y"]),
                min(float(width), float(height)),
                float(angle),
            )
        return []

    def polygon_mask(corners: list[dict]) -> np.ndarray | None:
        if len(corners) < 3:
            return None
        mask_image = Image.new("1", (raw.shape[1], raw.shape[0]), 0)
        points = [
            (float(point["x"]), float(point["y"])) for point in corners
        ]
        ImageDraw.Draw(mask_image).polygon(points, fill=1)
        return np.asarray(mask_image, dtype=bool)

    valid_targets = []
    for target in targets:
        target_id = target.get("id", "unknown")
        inner_corners = corners_for(target, True)
        outer_corners = corners_for(target, False)
        inner_mask = polygon_mask(inner_corners)
        outer_mask = polygon_mask(outer_corners)
        if inner_mask is None:
            reason = "Final inner ROI corners were missing or invalid."
            target["roi_data"] = {
                "roi_data_status": "missing",
                "sample_count": 0,
                "outer_sample_count": int(np.count_nonzero(outer_mask))
                if outer_mask is not None else 0,
                "mean_hu": None, "median_hu": None, "std_hu": None,
                "min_hu": None, "max_hu": None, "p05_hu": None,
                "p95_hu": None, "peak_to_valley_hu": None,
                "snr_like": None,
                "snr_like_definition": "peak_to_valley_hu / std_hu",
                "reason": reason,
            }
            target["profile_data"] = {
                "profile_data_status": "missing",
                "horizontal_profile_length": 0,
                "vertical_profile_length": 0,
                "best_profile_direction": "needs_review",
                "profile_peak_to_valley_hu": None,
                "stripe_peaks_found": 0,
                "periodicity_score": 0.0,
                "profile_quality": 0.0,
                "average_period_px": None,
                "period_consistency": 0.0,
                "reason": reason,
            }
            target["preliminary_visibility"] = "missing"
            extraction_warnings.append(f"{target_id}: {reason}")
            continue
        finite_mask = inner_mask & np.isfinite(raw)
        values = raw[finite_mask]
        outer_count = int(np.count_nonzero(
            outer_mask & np.isfinite(raw)
        )) if outer_mask is not None else 0
        if values.size < 16:
            reason = "Inner ROI contained fewer than 16 finite raw-HU samples."
            status = "needs_review"
            extraction_warnings.append(f"{target_id}: {reason}")
        else:
            reason = (
                "Raw CT/HU statistics were extracted from the fixed final "
                "inner ROI polygon."
            )
            status = "available"
        if values.size:
            p05, p95 = np.percentile(values, [5.0, 95.0])
            std_hu = float(np.std(values))
            peak_to_valley = float(p95 - p05)
            target["roi_data"] = {
                "roi_data_status": status,
                "sample_count": int(values.size),
                "outer_sample_count": outer_count,
                "mean_hu": round(float(np.mean(values)), 3),
                "median_hu": round(float(np.median(values)), 3),
                "std_hu": round(std_hu, 3),
                "min_hu": round(float(np.min(values)), 3),
                "max_hu": round(float(np.max(values)), 3),
                "p05_hu": round(float(p05), 3),
                "p95_hu": round(float(p95), 3),
                "peak_to_valley_hu": round(peak_to_valley, 3),
                "snr_like": round(
                    peak_to_valley / max(std_hu, 1e-6), 4
                ),
                "snr_like_definition": "peak_to_valley_hu / std_hu",
                "reason": reason,
            }
        else:
            target["roi_data"] = {
                "roi_data_status": "missing", "sample_count": 0,
                "outer_sample_count": outer_count,
                "mean_hu": None, "median_hu": None, "std_hu": None,
                "min_hu": None, "max_hu": None, "p05_hu": None,
                "p95_hu": None, "peak_to_valley_hu": None,
                "snr_like": None,
                "snr_like_definition": "peak_to_valley_hu / std_hu",
                "reason": "No finite raw-HU samples were inside the inner ROI.",
            }
            target["preliminary_visibility"] = "missing"
            extraction_warnings.append(
                f"{target_id}: no finite raw-HU samples were available."
            )
            continue
        valid_targets.append((target, inner_corners))

    roi_data_extraction_ms = (
        time.perf_counter() - extraction_started
    ) * 1000.0
    profile_started = time.perf_counter()
    for target, inner_corners in valid_targets:
        roi_data = target["roi_data"]
        final_roi = target.get("final_roi") or {}
        center = final_roi.get("final_center") or target.get("final_center")
        angle_degrees = final_roi.get(
            "final_angle_degrees", target.get("roi_angle_degrees")
        )
        side_lengths = []
        for index, point in enumerate(inner_corners):
            following = inner_corners[(index + 1) % len(inner_corners)]
            side_lengths.append(math.hypot(
                float(following["x"]) - float(point["x"]),
                float(following["y"]) - float(point["y"]),
            ))
        inner_side = float(np.median(side_lengths)) if side_lengths else 0.0
        if not center or angle_degrees is None or inner_side < 4.0:
            reason = "Final ROI-local sampling geometry was incomplete."
            target["profile_data"] = {
                "profile_data_status": "needs_review",
                "horizontal_profile_length": 0,
                "vertical_profile_length": 0,
                "best_profile_direction": "needs_review",
                "profile_peak_to_valley_hu": None,
                "stripe_peaks_found": 0,
                "periodicity_score": 0.0,
                "profile_quality": 0.0,
                "average_period_px": None,
                "period_consistency": 0.0,
                "reason": reason,
            }
            target["preliminary_visibility"] = "needs_review"
            extraction_warnings.append(f'{target["id"]}: {reason}')
            continue
        sample_size = max(8, int(round(inner_side)))
        coordinates = np.linspace(
            -(inner_side - 1.0) / 2.0,
            (inner_side - 1.0) / 2.0,
            sample_size,
        )
        local_y, local_x = np.meshgrid(coordinates, coordinates, indexing="ij")
        angle = math.radians(float(angle_degrees))
        image_x = (
            float(center["x"]) + local_x * math.cos(angle)
            - local_y * math.sin(angle)
        )
        image_y = (
            float(center["y"]) + local_x * math.sin(angle)
            + local_y * math.cos(angle)
        )
        sampled = ndimage.map_coordinates(
            working, [image_y, image_x], order=1, mode="nearest"
        )
        horizontal_profile = np.mean(sampled, axis=0)
        vertical_profile = np.mean(sampled, axis=1)
        horizontal_evidence = _profile_periodicity(horizontal_profile)
        vertical_evidence = _profile_periodicity(vertical_profile)
        direction, evidence, profile = max(
            (
                ("horizontal", horizontal_evidence, horizontal_profile),
                ("vertical", vertical_evidence, vertical_profile),
            ),
            key=lambda item: (
                item[1]["periodicity_score"]
                + item[1]["peak_score"]
                + item[1]["period_consistency"]
            ),
        )
        profile_quality = _clamp01(
            0.55 * float(evidence["periodicity_score"])
            + 0.25 * float(evidence["peak_score"])
            + 0.20 * float(evidence["period_consistency"])
        )
        profile_p05, profile_p95 = np.percentile(profile, [5.0, 95.0])
        enough_samples = int(roi_data["sample_count"]) >= 25
        peaks_found = int(evidence["peaks_found"])
        periodicity = float(evidence["periodicity_score"])
        if (
            enough_samples and profile_quality >= 0.68
            and periodicity >= 0.45 and peaks_found >= 3
        ):
            visibility = "visible"
        elif (
            enough_samples and (
                profile_quality >= 0.40
                or periodicity >= 0.25
                or peaks_found >= 2
            )
        ):
            visibility = "partial"
        elif enough_samples:
            visibility = "weak"
        else:
            visibility = "needs_review"
        profile_reason = (
            "Preliminary ROI-local directional profile summary; no lp/cm or "
            "pass/fail interpretation is applied."
        )
        target["profile_data"] = {
            "profile_data_status": "available" if enough_samples else "needs_review",
            "horizontal_profile_length": int(horizontal_profile.size),
            "vertical_profile_length": int(vertical_profile.size),
            "best_profile_direction": direction,
            "profile_peak_to_valley_hu": round(
                float(profile_p95 - profile_p05), 3
            ),
            "stripe_peaks_found": peaks_found,
            "periodicity_score": round(periodicity, 4),
            "profile_quality": round(profile_quality, 4),
            "average_period_px": evidence["average_period_px"],
            "period_consistency": round(
                float(evidence["period_consistency"]), 4
            ),
            "reason": profile_reason,
        }
        target["preliminary_visibility"] = visibility

    profile_data_extraction_ms = (
        time.perf_counter() - profile_started
    ) * 1000.0
    by_id = {target.get("id"): target for target in targets}

    def summarize(target_ids: list[str], primary: bool = False) -> dict:
        selected = [by_id[target_id] for target_id in target_ids if target_id in by_id]
        counts = {
            label: sum(
                target.get("preliminary_visibility") == label
                for target in selected
            )
            for label in ("visible", "partial", "weak", "needs_review")
        }
        with_data = sum(
            int(target.get("roi_data", {}).get("sample_count", 0)) > 0
            for target in selected
        )
        if primary:
            return {
                "primary_targets": target_ids,
                "primary_with_data": with_data,
                "primary_visible": counts["visible"],
                "primary_partial": counts["partial"],
                "primary_weak": counts["weak"],
                "primary_needing_review": counts["needs_review"]
                + sum(target.get("preliminary_visibility") == "missing" for target in selected),
            }
        return {
            "targets_total": len(target_ids),
            "targets_with_data": with_data,
            "targets_visible": counts["visible"],
            "targets_partial": counts["partial"],
            "targets_weak": counts["weak"],
            "targets_needing_review": counts["needs_review"]
            + sum(target.get("preliminary_visibility") == "missing" for target in selected),
            "measurement_status": "pending",
            "formal_lp_cm_status": "not_implemented",
            "pass_fail_status": "not_implemented",
        }

    return {
        "module4_roi_data_summary": summarize(
            [f"B{index}" for index in range(1, 9)]
        ),
        "primary_roi_data_summary": summarize(
            ["B8", "B7", "B6", "B5"], primary=True
        ),
        "extraction_warnings": extraction_warnings,
        "roi_data_extraction_ms": round(roi_data_extraction_ms, 2),
        "profile_data_extraction_ms": round(profile_data_extraction_ms, 2),
    }


def generate_module4_preliminary_graph_data(
    raw_pixels: np.ndarray,
    targets: list[dict],
    pixel_spacing: tuple[float | None, float | None] | None = None,
) -> dict:
    """Generate compact graph/support data from fixed final inner ROIs."""
    analysis_started = time.perf_counter()
    profile_graph_generation_ms = 0.0
    fft_graph_generation_ms = 0.0
    std_contrast_generation_ms = 0.0
    raw = np.asarray(raw_pixels, dtype=np.float32)
    finite_raw = np.isfinite(raw)
    fill = float(np.median(raw[finite_raw])) if np.any(finite_raw) else 0.0
    working = np.where(finite_raw, raw, fill)

    def finite_spacing(value: float | None) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) and numeric > 0.0 else None

    row_spacing = finite_spacing(pixel_spacing[0]) if pixel_spacing else None
    column_spacing = finite_spacing(pixel_spacing[1]) if pixel_spacing else None

    def compact_points(values: np.ndarray, key: str) -> tuple[list[dict], bool]:
        count = int(values.size)
        if count <= 80:
            indices = np.arange(count, dtype=int)
            downsampled = False
        else:
            indices = np.unique(np.linspace(0, count - 1, 80).round().astype(int))
            downsampled = True
        return [
            {"x" if key == "hu" else "frequency": int(index), key: round(float(values[index]), 5)}
            for index in indices
        ], downsampled

    def polygon_mask(corners: list[dict]) -> np.ndarray | None:
        if len(corners) < 3:
            return None
        image = Image.new("1", (raw.shape[1], raw.shape[0]), 0)
        ImageDraw.Draw(image).polygon(
            [(float(point["x"]), float(point["y"])) for point in corners],
            fill=1,
        )
        return np.asarray(image, dtype=bool)

    for target in targets:
        target_id = target.get("id")
        final_roi = target.get("final_roi") or {}
        inner_corners = (
            final_roi.get("inner_roi_corners")
            or target.get("inner_roi_corners")
            or []
        )
        outer_corners = (
            final_roi.get("final_corners")
            or target.get("roi_corners")
            or []
        )
        expected_lp_cm = MODULE4_EXPECTED_LP_CM.get(target_id)
        warnings = []
        if expected_lp_cm is None:
            warnings.append("expected_lp_cm_mapping_unknown")

        profile_started = time.perf_counter()
        center = final_roi.get("final_center") or target.get("final_center")
        angle_degrees = final_roi.get(
            "final_angle_degrees", target.get("roi_angle_degrees")
        )
        side_lengths = []
        for index, point in enumerate(inner_corners):
            following = inner_corners[(index + 1) % len(inner_corners)]
            side_lengths.append(math.hypot(
                float(following["x"]) - float(point["x"]),
                float(following["y"]) - float(point["y"]),
            ))
        inner_side = float(np.median(side_lengths)) if side_lengths else 0.0
        chosen_profile = None
        best_direction = target.get("profile_data", {}).get(
            "best_profile_direction", "needs_review"
        )
        if center and angle_degrees is not None and inner_side >= 4.0:
            sample_size = max(8, int(round(inner_side)))
            coordinates = np.linspace(
                -(inner_side - 1.0) / 2.0,
                (inner_side - 1.0) / 2.0,
                sample_size,
            )
            local_y, local_x = np.meshgrid(
                coordinates, coordinates, indexing="ij"
            )
            angle_radians = math.radians(float(angle_degrees))
            image_x = (
                float(center["x"]) + local_x * math.cos(angle_radians)
                - local_y * math.sin(angle_radians)
            )
            image_y = (
                float(center["y"]) + local_x * math.sin(angle_radians)
                + local_y * math.cos(angle_radians)
            )
            sampled = ndimage.map_coordinates(
                working, [image_y, image_x], order=1, mode="nearest"
            )
            horizontal_profile = np.mean(sampled, axis=0)
            vertical_profile = np.mean(sampled, axis=1)
            chosen_profile = (
                vertical_profile
                if best_direction == "vertical" else horizontal_profile
            )
        if chosen_profile is None or chosen_profile.size < 8:
            profile_graph = {
                "status": "missing",
                "x_label": "Position across ROI",
                "y_label": "HU",
                "points": [],
                "point_count": 0,
                "downsampled": False,
                "reason": "Fixed inner ROI profile sampling was unavailable.",
            }
            warnings.append("profile_graph_missing")
        else:
            profile_points, profile_downsampled = compact_points(
                chosen_profile, "hu"
            )
            profile_graph = {
                "status": "available",
                "x_label": "Position across ROI",
                "y_label": "HU",
                "points": profile_points,
                "point_count": int(len(profile_points)),
                "source_point_count": int(chosen_profile.size),
                "downsampled": profile_downsampled,
                "profile_direction": best_direction,
                "reason": "Averaged ROI-local profile from the fixed inner ROI.",
            }
        profile_graph_generation_ms += (
            time.perf_counter() - profile_started
        ) * 1000.0

        fft_started = time.perf_counter()
        if chosen_profile is None or chosen_profile.size < 8:
            fft_graph = {
                "status": "missing", "x_label": "Frequency",
                "x_unit": "cycles/pixel",
                "y_label": "Normalized FFT power",
                "expected_lp_cm": expected_lp_cm,
                "measured_peak_lp_cm": None,
                "measured_peak_frequency": None,
                "fft_snr": None, "noise_floor": None,
                "peak_power": None, "points": [], "point_count": 0,
                "downsampled": False,
                "reason": "A usable fixed-ROI profile was unavailable.",
            }
        else:
            profile_x = np.arange(chosen_profile.size, dtype=float)
            trend = np.polyval(
                np.polyfit(profile_x, chosen_profile.astype(float), 1),
                profile_x,
            )
            detrended = chosen_profile.astype(float) - trend
            windowed = detrended * np.hanning(detrended.size)
            fft_values = np.fft.rfft(windowed)
            raw_power = np.abs(fft_values) ** 2
            frequencies = np.fft.rfftfreq(detrended.size, d=1.0)
            raw_power[0] = 0.0
            meaningful = np.arange(1, raw_power.size, dtype=int)
            peak_index = (
                int(meaningful[np.argmax(raw_power[meaningful])])
                if meaningful.size else 0
            )
            peak_raw_power = float(raw_power[peak_index]) if peak_index else 0.0
            noise_values = raw_power[meaningful]
            noise_floor_raw = (
                float(np.median(noise_values)) if noise_values.size else 0.0
            )
            normalized_power = raw_power / max(peak_raw_power, 1e-12)
            effective_spacing_mm = None
            if row_spacing is not None and column_spacing is not None:
                if best_direction == "vertical":
                    effective_spacing_mm = math.hypot(
                        math.sin(angle_radians) * column_spacing,
                        math.cos(angle_radians) * row_spacing,
                    )
                else:
                    effective_spacing_mm = math.hypot(
                        math.cos(angle_radians) * column_spacing,
                        math.sin(angle_radians) * row_spacing,
                    )
            if effective_spacing_mm is not None:
                graph_frequencies = frequencies / effective_spacing_mm * 10.0
                x_unit = "lp/cm"
                measured_peak_lp_cm = round(
                    float(graph_frequencies[peak_index]), 5
                ) if peak_index else None
            else:
                graph_frequencies = frequencies
                x_unit = "cycles/pixel"
                measured_peak_lp_cm = None
                warnings.append("pixel_spacing_missing")
            graph_indices = meaningful
            if graph_indices.size > 80:
                graph_indices = np.unique(
                    np.linspace(1, raw_power.size - 1, 80)
                    .round().astype(int)
                )
                fft_downsampled = True
            else:
                fft_downsampled = False
            fft_points = [
                {
                    "frequency": round(float(graph_frequencies[index]), 6),
                    "power": round(float(normalized_power[index]), 6),
                }
                for index in graph_indices
            ]
            fft_graph = {
                "status": "available" if meaningful.size else "needs_review",
                "x_label": "Frequency", "x_unit": x_unit,
                "y_label": "Normalized FFT power",
                "expected_lp_cm": expected_lp_cm,
                "measured_peak_lp_cm": measured_peak_lp_cm,
                "measured_peak_frequency": (
                    round(float(graph_frequencies[peak_index]), 6)
                    if peak_index else None
                ),
                "fft_snr": round(
                    peak_raw_power / max(noise_floor_raw, 1e-12), 4
                ) if peak_index else None,
                "noise_floor": round(
                    noise_floor_raw / max(peak_raw_power, 1e-12), 6
                ) if peak_index else None,
                "peak_power": round(
                    float(normalized_power[peak_index]), 6
                ) if peak_index else None,
                "points": fft_points,
                "point_count": len(fft_points),
                "source_point_count": int(meaningful.size),
                "downsampled": fft_downsampled,
                "reason": (
                    "DC/trend-removed Hann-window FFT for graph review only; "
                    "no frequency decision was applied."
                ),
            }
        fft_graph_generation_ms += (
            time.perf_counter() - fft_started
        ) * 1000.0

        std_started = time.perf_counter()
        inner_mask = polygon_mask(inner_corners)
        outer_mask = polygon_mask(outer_corners)
        background_values = np.asarray([], dtype=float)
        if inner_mask is not None and outer_mask is not None:
            background_values = raw[
                outer_mask & ~inner_mask & np.isfinite(raw)
            ]
        roi_data = target.get("roi_data") or {}
        std_hu = roi_data.get("std_hu")
        peak_to_valley_hu = roi_data.get("peak_to_valley_hu")
        if background_values.size >= 16 and std_hu is not None:
            background_std_hu = float(np.std(background_values))
            std_ratio = float(std_hu) / max(background_std_hu, 1e-6)
            peak_noise_ratio = (
                float(peak_to_valley_hu) / max(background_std_hu, 1e-6)
                if peak_to_valley_hu is not None else None
            )
            contrast_energy = _clamp01(
                (float(std_hu) - background_std_hu)
                / max(float(std_hu) + background_std_hu, 1e-6)
            )
            background_reliable = background_std_hu < 0.80 * max(float(std_hu), 1e-6)
            if background_reliable:
                std_status = "available"
                background_noise_status = "available"
                std_reason = (
                    "Background noise estimated from the fixed outer-ROI minus "
                    "inner-ROI ring; development support data only."
                )
            else:
                std_status = "needs_review"
                background_noise_status = "needs_review"
                std_reason = (
                    "Background estimate appears unreliable for STD support."
                )
                warnings.append("background_noise_unreliable")
        else:
            background_std_hu = None
            std_ratio = None
            peak_noise_ratio = None
            contrast_energy = None
            std_status = "needs_review" if std_hu is not None else "missing"
            background_noise_status = std_status
            std_reason = (
                "A safe fixed outer-minus-inner background ring estimate was "
                "not available."
            )
            warnings.append("background_noise_estimate_unavailable")
        std_contrast_data = {
            "status": std_status,
            "background_noise_status": background_noise_status,
            "background_noise_reason": std_reason,
            "std_hu": std_hu,
            "background_std_hu": (
                round(background_std_hu, 4)
                if background_std_hu is not None else None
            ),
            "std_ratio": round(std_ratio, 4) if std_ratio is not None else None,
            "peak_to_valley_hu": peak_to_valley_hu,
            "peak_to_valley_noise_ratio": (
                round(peak_noise_ratio, 4)
                if peak_noise_ratio is not None else None
            ),
            "contrast_energy_score": (
                round(contrast_energy, 4)
                if contrast_energy is not None else None
            ),
            "reason": std_reason,
        }
        std_contrast_generation_ms += (
            time.perf_counter() - std_started
        ) * 1000.0

        analysis_status = (
            "missing" if profile_graph["status"] == "missing"
            else "needs_review" if warnings else "available"
        )
        target["module4_preliminary_analysis"] = {
            "target": target_id,
            "expected_lp_cm": expected_lp_cm,
            "analysis_status": analysis_status,
            "warnings": sorted(set(warnings)),
            "profile_graph": profile_graph,
            "fft_graph": fft_graph,
            "std_contrast_data": std_contrast_data,
        }

    primary_ids = ["B8", "B7", "B6", "B5"]
    by_id = {target.get("id"): target for target in targets}
    module4_graph_data = {
        "primary_peak_valley": [
            {
                "target": target_id,
                "value": by_id.get(target_id, {}).get(
                    "roi_data", {}
                ).get("peak_to_valley_hu"),
            }
            for target_id in primary_ids
        ],
        "primary_profile_quality": [
            {
                "target": target_id,
                "periodicity": by_id.get(target_id, {}).get(
                    "profile_data", {}
                ).get("periodicity_score"),
                "profile_quality": by_id.get(target_id, {}).get(
                    "profile_data", {}
                ).get("profile_quality"),
            }
            for target_id in primary_ids
        ],
        "per_target_graphs_available": all(
            target.get("module4_preliminary_analysis", {}).get(
                "profile_graph", {}
            ).get("status") == "available"
            for target in targets
        ),
        "targets_with_profile_graphs": [
            target["id"] for target in targets
            if target.get("module4_preliminary_analysis", {}).get(
                "profile_graph", {}
            ).get("status") == "available"
        ],
        "targets_with_fft_graphs": [
            target["id"] for target in targets
            if target.get("module4_preliminary_analysis", {}).get(
                "fft_graph", {}
            ).get("status") == "available"
        ],
    }
    return {
        "module4_graph_data": module4_graph_data,
        "module4_preliminary_analysis_ms": round(
            (time.perf_counter() - analysis_started) * 1000.0, 2
        ),
        "profile_graph_generation_ms": round(
            profile_graph_generation_ms, 2
        ),
        "fft_graph_generation_ms": round(fft_graph_generation_ms, 2),
        "std_contrast_generation_ms": round(
            std_contrast_generation_ms, 2
        ),
    }


def generate_module4_preliminary_votes(
    targets: list[dict],
    pixel_spacing: tuple[float | None, float | None] | None = None,
) -> dict:
    """Combine existing Module 4 evidence into non-clinical review votes."""
    adaptive_config = MODULE4_ADAPTIVE_THRESHOLD_CONFIG

    def finite(value) -> float | None:
        return _module4_safe_float(value)

    def robust_values(values) -> list[float]:
        return [value for value in (finite(item) for item in values) if value is not None]

    def median_mad(values: list[float]) -> tuple[float | None, float | None]:
        return _module4_safe_median(values), _module4_safe_mad(values)

    def clamp_with_flag(value: float, minimum: float, maximum: float) -> tuple[float, bool]:
        clamped = _module4_clamp(value, minimum, maximum)
        if clamped is None:
            clamped = float(minimum)
        return clamped, not math.isclose(clamped, float(value), rel_tol=1e-9, abs_tol=1e-9)

    fft_snr_values = robust_values(
        (target.get("module4_preliminary_analysis") or {})
        .get("fft_graph", {}).get("fft_snr") for target in targets
    )
    profile_snr_values = []
    profile_quality_values = robust_values(
        (target.get("profile_data") or {}).get("profile_quality")
        for target in targets
    )
    background_std_values = robust_values(
        (target.get("module4_preliminary_analysis") or {})
        .get("std_contrast_data", {}).get("background_std_hu")
        for target in targets
        if (target.get("module4_preliminary_analysis") or {})
        .get("std_contrast_data", {}).get("background_noise_status") == "available"
    )
    roi_std_values = robust_values(
        (target.get("roi_data") or {}).get("std_hu") for target in targets
    )
    primary_peak_values = robust_values(
        (target.get("roi_data") or {}).get("peak_to_valley_hu")
        for target in targets if target.get("id") in {"B8", "B7", "B6", "B5"}
    )
    for target in targets:
        analysis = target.get("module4_preliminary_analysis") or {}
        peak_valley = finite(
            (target.get("profile_data") or {}).get("profile_peak_to_valley_hu")
        )
        background_std = finite(
            (analysis.get("std_contrast_data") or {}).get("background_std_hu")
        )
        if peak_valley is not None and background_std is not None:
            profile_snr_values.append(peak_valley / max(background_std, 1e-6))

    fft_median, fft_mad = median_mad(fft_snr_values)
    profile_snr_median, profile_snr_mad = median_mad(profile_snr_values)
    profile_quality_median, _ = median_mad(profile_quality_values)
    background_std_median, _ = median_mad(background_std_values)
    roi_std_median, roi_std_mad = median_mad(roi_std_values)
    primary_peak_median, primary_peak_mad = median_mad(primary_peak_values)

    fft_raw = fft_median * 0.35 if len(fft_snr_values) >= 3 else 3.0
    fft_snr_threshold, fft_guardrail = clamp_with_flag(
        fft_raw,
        adaptive_config["fft"]["fft_snr_min_guardrail"],
        adaptive_config["fft"]["fft_snr_max_guardrail"],
    )
    fft_score_raw = (
        _clamp01(fft_median / 6.0) * 0.55
        if fft_median is not None else 0.50
    )
    fft_score_threshold, fft_score_guardrail = clamp_with_flag(
        fft_score_raw,
        adaptive_config["fft"]["fft_score_min_guardrail"],
        adaptive_config["fft"]["fft_score_max_guardrail"],
    )

    profile_snr_raw = profile_snr_median * 0.35 if len(profile_snr_values) >= 3 else 2.0
    profile_snr_threshold, profile_snr_guardrail = clamp_with_flag(
        profile_snr_raw,
        adaptive_config["profile"]["profile_snr_min_guardrail"],
        adaptive_config["profile"]["profile_snr_max_guardrail"],
    )
    profile_score_raw = (
        profile_quality_median * 0.55
        if profile_quality_median is not None else 0.50
    )
    profile_score_threshold, profile_score_guardrail = clamp_with_flag(
        profile_score_raw,
        adaptive_config["profile"]["profile_score_min_guardrail"],
        adaptive_config["profile"]["profile_score_max_guardrail"],
    )

    noise_fraction = (
        background_std_median / max(roi_std_median, 1e-6)
        if background_std_median is not None and roi_std_median is not None else None
    )
    if background_std_median is not None:
        std_ratio_raw = 2.0
        std_ratio_threshold, std_ratio_guardrail = clamp_with_flag(
            std_ratio_raw,
            adaptive_config["std_support"]["std_ratio_min_guardrail"],
            adaptive_config["std_support"]["std_ratio_max_guardrail"],
        )
        peak_noise_raw = 3.0
        peak_noise_threshold, peak_noise_guardrail = clamp_with_flag(
            peak_noise_raw,
            adaptive_config["std_support"]["peak_to_valley_noise_min_guardrail"],
            adaptive_config["std_support"]["peak_to_valley_noise_max_guardrail"],
        )
        std_score_threshold = adaptive_config["std_support"]["std_score_min_guardrail"]
    else:
        std_ratio_threshold = None
        peak_noise_threshold = None
        std_score_threshold = None
        std_ratio_guardrail = peak_noise_guardrail = False
    primary_profile_quality_values = robust_values(
        (target.get("profile_data") or {}).get("profile_quality")
        for target in targets if target.get("id") in {"B8", "B7", "B6", "B5"}
    )
    primary_strength = (
        float(np.median(primary_profile_quality_values))
        if primary_profile_quality_values else None
    )
    combined_raw = 0.60 if background_std_median is None else 0.55
    combined_threshold, combined_guardrail = clamp_with_flag(
        combined_raw,
        adaptive_config["combined"]["combined_score_min_guardrail"],
        adaptive_config["combined"]["combined_score_max_guardrail"],
    )

    noise_context_status = (
        "missing"
        if not targets or not (roi_std_values or profile_snr_values or fft_snr_values)
        else "available" if background_std_median is not None
        else "needs_review"
    )
    noise_context = {
        "status": noise_context_status,
        "background_std_hu": round(background_std_median, 4) if background_std_median is not None else None,
        "background_noise_source": "fixed outer-ROI minus inner-ROI rings" if background_std_median is not None else "unavailable",
        "roi_std_median": round(roi_std_median, 4) if roi_std_median is not None else None,
        "roi_std_mad": round(roi_std_mad, 4) if roi_std_mad is not None else None,
        "primary_peak_valley_median": round(primary_peak_median, 4) if primary_peak_median is not None else None,
        "primary_peak_valley_mad": round(primary_peak_mad, 4) if primary_peak_mad is not None else None,
        "profile_snr_median": round(profile_snr_median, 4) if profile_snr_median is not None else None,
        "profile_snr_mad": round(profile_snr_mad, 4) if profile_snr_mad is not None else None,
        "fft_snr_median": round(fft_median, 4) if fft_median is not None else None,
        "fft_snr_mad": round(fft_mad, 4) if fft_mad is not None else None,
        "profile_quality_median": round(profile_quality_median, 4) if profile_quality_median is not None else None,
        "profile_quality_mad": round(_module4_safe_mad(profile_quality_values), 4) if profile_quality_values else None,
        "background_noise_status": (
            "available" if background_std_median is not None
            else "missing" if noise_context_status == "missing"
            else "needs_review"
        ),
        "reason": "Existing fixed-ROI noise and signal distributions; no new background detector was used.",
    }
    adaptive_fft_thresholds = {
        "dynamic_fft_snr_threshold": round(fft_snr_threshold, 4),
        "dynamic_fft_score_threshold": round(fft_score_threshold, 4),
        "max_frequency_error_percent": adaptive_config["fft"]["max_frequency_error_percent"],
        "method": "35% of run median FFT SNR with fallback and development guardrails.",
        "guardrails_applied": fft_guardrail or fft_score_guardrail,
        "reason": "Scan-aware FFT review threshold; frequency validation still requires mapping and pixel spacing.",
    }
    adaptive_profile_thresholds = {
        "dynamic_profile_snr_threshold": round(profile_snr_threshold, 4),
        "dynamic_profile_score_threshold": round(profile_score_threshold, 4),
        "min_peak_count": adaptive_config["profile"]["min_peak_count"],
        "min_valley_count": adaptive_config["profile"]["min_valley_count"],
        "max_spacing_error_percent": adaptive_config["profile"]["max_spacing_error_percent"],
        "method": "35% of median profile SNR and 55% of median profile quality with guardrails.",
        "guardrails_applied": profile_snr_guardrail or profile_score_guardrail,
        "reason": "Scan-aware profile review thresholds; spacing validation still requires mapping.",
    }
    adaptive_std_thresholds = {
        "status": (
            "available" if background_std_median is not None
            else noise_context["status"]
        ),
        "dynamic_std_ratio_threshold": round(std_ratio_threshold, 4) if std_ratio_threshold is not None else None,
        "dynamic_peak_to_valley_noise_threshold": round(peak_noise_threshold, 4) if peak_noise_threshold is not None else None,
        "dynamic_std_score_threshold": round(std_score_threshold, 4) if std_score_threshold is not None else None,
        "background_noise_status": noise_context["background_noise_status"],
        "background_std_hu": noise_context["background_std_hu"],
        "method": "Background-to-ROI noise fraction with development guardrails.",
        "guardrails_applied": std_ratio_guardrail or peak_noise_guardrail,
        "reason": "STD support remains review-only when background noise is unavailable.",
    }
    adaptive_combined_thresholds = {
        "dynamic_combined_score_threshold": round(combined_threshold, 4),
        "weights": {
            "fft": adaptive_config["combined"]["fft_weight"],
            "profile": adaptive_config["combined"]["profile_weight"],
            "std": adaptive_config["combined"]["std_weight"],
        },
        "method": "Moderate scan-aware threshold selected from primary evidence and clamped.",
        "guardrails_applied": combined_guardrail,
        "reason": "Missing mapping raises review behavior rather than forcing a frequency decision.",
    }

    def threshold_margin(actual, threshold, reverse: bool = False) -> float | None:
        return (
            _module4_safe_error_margin(threshold, actual)
            if reverse else _module4_safe_margin(actual, threshold)
        )

    spacing_values = [finite(value) for value in (pixel_spacing or ())]
    valid_spacing = [value for value in spacing_values if value and value > 0.0]
    effective_spacing_mm = (
        float(np.mean(valid_spacing)) if len(valid_spacing) == 2 else None
    )
    target_specific_guardrails = MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG[
        "guardrails"
    ]

    def target_family(target_id: str) -> str:
        for family, target_ids in MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG[
            "target_families"
        ].items():
            if target_id in target_ids:
                return family
        return "reference_high_frequency"

    target_signal_contexts = []
    for target in targets:
        target_id = target.get("id", "unknown")
        analysis = target.get("module4_preliminary_analysis") or {}
        fft_graph = analysis.get("fft_graph") or {}
        std_data = analysis.get("std_contrast_data") or {}
        roi_data = target.get("roi_data") or {}
        profile = target.get("profile_data") or {}
        local_profile_snr = None
        local_peak_valley = finite(profile.get("profile_peak_to_valley_hu"))
        local_background_std = finite(std_data.get("background_std_hu"))
        if local_peak_valley is not None and local_background_std is not None:
            local_profile_snr = local_peak_valley / max(local_background_std, 1e-6)
        local_fft_snr = finite(fft_graph.get("fft_snr"))
        local_fft_score = (
            _clamp01(local_fft_snr / 6.0)
            if local_fft_snr is not None else None
        )
        local_profile_quality = finite(profile.get("profile_quality"))
        snr_like = finite(roi_data.get("snr_like"))
        internal_support_score = (
            _clamp01(
                0.60 * _clamp01((snr_like or 0.0) / 3.0)
                + 0.40 * (local_profile_quality or 0.0)
            )
            if snr_like is not None or local_profile_quality is not None
            else None
        )
        preliminary_components = [
            (0.45, local_fft_score),
            (0.35, local_profile_quality),
            (0.20, internal_support_score),
        ]
        available_components = [
            (weight, value) for weight, value in preliminary_components
            if value is not None
        ]
        local_combined = (
            sum(weight * value for weight, value in available_components)
            / sum(weight for weight, _ in available_components)
            if available_components else None
        )
        final_roi = target.get("final_roi") or {}
        target_signal_contexts.append({
            "target": target_id,
            "nominal_display_lp_cm": MODULE4_NOMINAL_LP_CM_BY_ID.get(target_id),
            "role": (
                "reference"
                if target_family(target_id) == "reference_high_frequency"
                else target_family(target_id)
            ),
            "threshold_family": target_family(target_id),
            "roi_source": final_roi.get("roi_source", target.get("roi_source")),
            "geometry_confidence": target.get(
                "location_confidence", final_roi.get("location_confidence")
            ),
            "peak_to_valley_hu": roi_data.get("peak_to_valley_hu"),
            "std_hu": roi_data.get("std_hu"),
            "profile_quality": local_profile_quality,
            "periodicity": profile.get("periodicity_score"),
            "profile_snr": round(local_profile_snr, 4) if local_profile_snr is not None else None,
            "profile_peak_count": int(profile.get("stripe_peaks_found") or 0),
            "profile_valley_count": max(0, int(profile.get("stripe_peaks_found") or 0) - 1),
            "profile_spacing_px": profile.get("average_period_px"),
            "profile_spacing_mm": (
                round(float(profile["average_period_px"]) * effective_spacing_mm, 4)
                if finite(profile.get("average_period_px")) is not None
                and effective_spacing_mm is not None else None
            ),
            "fft_snr": local_fft_snr,
            "fft_score": round(local_fft_score, 4) if local_fft_score is not None else None,
            "fft_noise_floor": fft_graph.get("noise_floor"),
            "measured_fft_peak_lp_cm": fft_graph.get("measured_peak_lp_cm"),
            "internal_contrast_support": {
                "status": "available" if internal_support_score is not None else "missing",
                "snr_like": snr_like,
                "peak_to_valley_hu": roi_data.get("peak_to_valley_hu"),
                "std_hu": roi_data.get("std_hu"),
                "support_score": round(internal_support_score, 4) if internal_support_score is not None else None,
                "reason": (
                    "Uses target-local contrast only because background noise estimate is unreliable."
                    if std_data.get("background_noise_status") != "available"
                    else "Target-local contrast support accompanies the available background estimate."
                ),
            },
            "local_combined_score": round(local_combined, 4) if local_combined is not None else None,
            "reason": "Target-local evidence for development threshold adaptation; nominal class is display-only.",
        })

    def add_rank(field: str, rank_field: str) -> None:
        ranked = sorted(
            [context for context in target_signal_contexts if finite(context.get(field)) is not None],
            key=lambda context: finite(context[field]), reverse=True,
        )
        for rank, context in enumerate(ranked, start=1):
            context[rank_field] = rank
        for context in target_signal_contexts:
            context.setdefault(rank_field, None)

    add_rank("peak_to_valley_hu", "local_rank_by_peak_valley")
    add_rank("profile_quality", "local_rank_by_profile_quality")
    add_rank("local_combined_score", "local_rank_by_combined_score")
    context_by_target = {
        context["target"]: context for context in target_signal_contexts
    }
    raw_target_by_id = {target.get("id"): target for target in targets}
    family_contexts = {
        family: [
            context for context in target_signal_contexts
            if context["threshold_family"] == family
        ]
        for family in MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG["target_families"]
    }
    target_adaptive_thresholds = []
    target_thresholds_by_id = {}
    for context in target_signal_contexts:
        target_id = context["target"]
        family = context["threshold_family"]
        family_rows = family_contexts.get(family, [])
        family_fft_values = sorted(
            value for value in (
                finite(row.get("fft_snr")) for row in family_rows
            ) if value is not None and value >= 0.0
        )
        family_fft_median = _module4_safe_median(family_fft_values)
        lower_half_count = max(1, int(math.ceil(len(family_fft_values) / 2.0)))
        lower_half_fft_values = family_fft_values[:lower_half_count]
        lower_half_family_fft_median = _module4_safe_median(
            lower_half_fft_values
        )
        family_fft_min = family_fft_values[0] if family_fft_values else None
        family_fft_max = family_fft_values[-1] if family_fft_values else None
        fft_outlier_spread = bool(
            family_fft_median is not None
            and family_fft_max is not None
            and (
                family_fft_max / max(family_fft_median, 1e-6) > 20.0
                or family_fft_max / max(family_fft_min or 0.0, 1e-6) > 100.0
            )
        )
        family_reference_fft_snr = (
            lower_half_family_fft_median
            if fft_outlier_spread else family_fft_median
        )
        family_profile_snr_median = _module4_safe_median(
            row.get("profile_snr") for row in family_rows
        )
        family_profile_quality_median = _module4_safe_median(
            row.get("profile_quality") for row in family_rows
        )
        family_combined_median = _module4_safe_median(
            row.get("local_combined_score") for row in family_rows
        )
        family_internal_support_median = _module4_safe_median(
            (row.get("internal_contrast_support") or {}).get("support_score")
            for row in family_rows
        )
        local_noise_floor = finite(context.get("fft_noise_floor")) or 0.0
        own_fft_snr = finite(context.get("fft_snr"))
        fft_snr_candidates = [
            value for value in (
                own_fft_snr * 0.75 if own_fft_snr is not None else None,
                math.sqrt(own_fft_snr * family_reference_fft_snr) * 0.75
                if own_fft_snr is not None
                and family_reference_fft_snr is not None else None,
                family_reference_fft_snr * 0.75
                if family_reference_fft_snr is not None else None,
            ) if value is not None
        ]
        fft_snr_target = min(fft_snr_candidates) if fft_snr_candidates else 2.5
        target_fft_snr_threshold = _module4_clamp(
            fft_snr_target,
            target_specific_guardrails["fft_snr_min"],
            target_specific_guardrails["fft_snr_max"],
        )
        fft_guardrail_hit = not math.isclose(
            target_fft_snr_threshold, fft_snr_target, rel_tol=0.0, abs_tol=1e-9
        )
        fft_threshold_debug = {
            "target": target_id,
            "actual_fft_snr": own_fft_snr,
            "family": family,
            "family_fft_snr_values": [round(value, 4) for value in family_fft_values],
            "family_reference_fft_snr": (
                round(family_reference_fft_snr, 4)
                if family_reference_fft_snr is not None else None
            ),
            "outlier_spread_detected": fft_outlier_spread,
            "raw_family_median": (
                round(family_fft_median, 4)
                if family_fft_median is not None else None
            ),
            "lower_half_family_median": (
                round(lower_half_family_fft_median, 4)
                if lower_half_family_fft_median is not None else None
            ),
            "threshold_before_guardrail": round(fft_snr_target, 4),
            "threshold_after_guardrail": round(target_fft_snr_threshold, 4),
            "guardrail_hit": fft_guardrail_hit,
            "reason": (
                "Extreme family FFT spread detected; lower-half family median was used with target-local FFT SNR."
                if fft_outlier_spread else
                "Family FFT spread was stable; family median was used with target-local FFT SNR."
            ),
        }
        target_fft_score_threshold = _module4_clamp(
            0.45 + min(0.10, local_noise_floor * 0.10),
            target_specific_guardrails["fft_score_min"],
            target_specific_guardrails["fft_score_max"],
        )
        own_profile_snr = finite(context.get("profile_snr"))
        profile_snr_candidates = [
            value for value in (
                own_profile_snr * 0.75 if own_profile_snr is not None else None,
                family_profile_snr_median * 0.65 if family_profile_snr_median is not None else None,
            ) if value is not None
        ]
        target_profile_snr_threshold = _module4_clamp(
            min(profile_snr_candidates) if profile_snr_candidates else 0.20,
            target_specific_guardrails["profile_snr_min"],
            target_specific_guardrails["profile_snr_max"],
        )
        profile_score_base = max(
            target_specific_guardrails["profile_score_min"],
            (family_profile_quality_median or 0.45) * 0.55,
        )
        if finite(context.get("profile_quality")) is not None:
            profile_score_base = min(
                profile_score_base,
                max(target_specific_guardrails["profile_score_min"], context["profile_quality"]),
            )
        target_profile_score_threshold = _module4_clamp(
            profile_score_base,
            target_specific_guardrails["profile_score_min"],
            target_specific_guardrails["profile_score_max"],
        )
        peak_threshold, valley_threshold = (
            (4, 3) if family == "primary" else (2, 1)
        )
        internal_support = context["internal_contrast_support"]
        background_available = (
            (raw_target_by_id.get(target_id, {}).get(
                "module4_preliminary_analysis"
            ) or {})
            .get("std_contrast_data", {}).get("background_noise_status") == "available"
        )
        target_std_score_threshold = (
            _module4_clamp(
                0.45 + 0.10 * (1.0 - (internal_support.get("support_score") or 0.0)),
                0.45, 0.75,
            ) if background_available else None
        )
        target_internal_support_threshold = _module4_clamp(
            (family_internal_support_median or 0.40) * 0.75
            * (0.95 + 0.10 * (context.get("profile_quality") or 0.0)),
            0.20, 0.70,
        )
        combined_base = (family_combined_median or 0.56) * 0.80
        combined_modifier = 0.92 + 0.12 * (context.get("profile_quality") or 0.0)
        target_combined_threshold = _module4_clamp(
            combined_base * combined_modifier,
            target_specific_guardrails["combined_score_min"],
            target_specific_guardrails["combined_score_max"],
        )
        target_threshold = {
            "target": target_id,
            "threshold_mode": "target_specific_adaptive",
            "fft": {
                "fft_snr_threshold": round(target_fft_snr_threshold, 4),
                "fft_score_threshold": round(target_fft_score_threshold, 4),
                "frequency_error_percent_threshold": (
                    20.0 if MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING
                    and MODULE4_EXPECTED_LP_CM.get(target_id) is not None else None
                ),
                "method": "Target-local FFT SNR with outlier-resistant family reference and guardrails.",
                "reason": "Frequency interpretation remains review-only while analytical mapping is disabled.",
                "fft_threshold_debug": fft_threshold_debug,
            },
            "profile": {
                "profile_snr_threshold": round(target_profile_snr_threshold, 4),
                "profile_score_threshold": round(target_profile_score_threshold, 4),
                "peak_count_threshold": peak_threshold,
                "valley_count_threshold": valley_threshold,
                "spacing_error_percent_threshold": (
                    25.0 if MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING
                    and MODULE4_EXPECTED_LP_CM.get(target_id) is not None else None
                ),
                "method": "Target-local and family profile evidence with target-family expectations.",
                "reason": "Low profile SNR is review support when pattern quality and peaks remain present.",
            },
            "peak_valley_expectation": {
                "target": target_id,
                "peak_count_threshold": peak_threshold,
                "valley_count_threshold": valley_threshold,
                "reason": "Development family expectation; not formal scoring.",
            },
            "std_support": {
                "std_ratio_threshold": std_ratio_threshold if background_available else None,
                "peak_to_valley_noise_threshold": peak_noise_threshold if background_available else None,
                "std_score_threshold": round(target_std_score_threshold, 4) if target_std_score_threshold is not None else None,
                "internal_contrast_support_threshold": round(
                    target_internal_support_threshold, 4
                ),
                "internal_contrast_support_score": internal_support.get(
                    "support_score"
                ),
                "status": "available" if background_available else "needs_review",
                "method": "Background-normalized support when reliable; otherwise target-local internal contrast support.",
                "reason": internal_support["reason"],
            },
            "combined": {
                "combined_score_threshold": round(target_combined_threshold, 4),
                "method": "Family combined median with target profile-quality adjustment and guardrails.",
                "reason": "Diagnostic strength only while analytical mapping is disabled.",
            },
            "threshold_source_reason": {
                "target": target_id,
                "profile_snr_threshold_reason": (
                    f"{target_id} profile threshold used {target_id} profile_snr, "
                    f"{target_id} profile_quality, {family} family context, and guardrails."
                ),
                "profile_score_threshold_reason": (
                    f"{target_id} profile score threshold used target quality, "
                    f"{family} family quality median, and guardrails."
                ),
                "fft_snr_threshold_reason": (
                    f"{target_id} FFT SNR threshold used target-local FFT SNR, "
                    f"outlier-resistant {family} family context, and guardrails."
                ),
                "fft_score_threshold_reason": (
                    f"{target_id} FFT score threshold used target-local FFT noise floor "
                    "and score guardrails."
                ),
                "combined_score_threshold_reason": (
                    f"{target_id} combined threshold used {family} family combined median, "
                    "target profile quality, and guardrails."
                ),
                "data_used": [
                    "target_profile_snr", "target_profile_quality",
                    "target_peak_valley_hu", "target_fft_snr",
                    "target_fft_noise_floor", "target_internal_contrast_support",
                    f"{family}_family_context", "run_context_guardrails",
                ],
            },
        }
        target_adaptive_thresholds.append(target_threshold)
        target_thresholds_by_id[target_id] = target_threshold
    visibility_votes = []
    module_flags: set[str] = {"physicist_review_required"}
    if not MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING:
        module_flags.add("development_mapping_disabled")
    if not _MODULE4_MAPPING_COMPLETE:
        module_flags.add("expected_mapping_incomplete")
    if MODULE4_REQUIRED_LP_CM is None:
        module_flags.add("required_threshold_not_configured")
    if MODULE4_SUGGESTED_MAPPING_REVIEW["suggested_mapping_available"]:
        module_flags.add("suggested_mapping_not_applied")

    for target in targets:
        target_id = target.get("id", "unknown")
        analysis = target.get("module4_preliminary_analysis") or {}
        fft_graph = analysis.get("fft_graph") or {}
        profile = target.get("profile_data") or {}
        std_data = analysis.get("std_contrast_data") or {}
        roi_data = target.get("roi_data") or {}
        expected_lp_cm = MODULE4_EXPECTED_LP_CM.get(target_id)
        target_threshold = target_thresholds_by_id[target_id]
        target_fft_threshold = target_threshold["fft"]
        target_profile_threshold = target_threshold["profile"]
        target_std_threshold = target_threshold["std_support"]
        target_combined_threshold = target_threshold["combined"]
        review_flags: set[str] = set()

        if target.get("location_confidence") in {"poor", "low", "needs_review"}:
            review_flags.add("roi_geometry_confidence_poor")
        if effective_spacing_mm is None:
            review_flags.add("pixel_spacing_missing")
        if expected_lp_cm is None:
            review_flags.add("expected_lp_cm_mapping_unknown")
        if not MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING:
            review_flags.add("development_mapping_disabled")
        if not _MODULE4_MAPPING_COMPLETE:
            review_flags.add("expected_mapping_incomplete")
        if MODULE4_REQUIRED_LP_CM is None:
            review_flags.add("required_threshold_not_configured")
        review_flags.add("physicist_review_required")
        if MODULE4_SUGGESTED_MAPPING_REVIEW["suggested_mapping_available"]:
            review_flags.add("suggested_mapping_not_applied")

        fft_snr = finite(fft_graph.get("fft_snr"))
        measured_lp_cm = finite(fft_graph.get("measured_peak_lp_cm"))
        frequency_error = (
            abs(measured_lp_cm - expected_lp_cm)
            if measured_lp_cm is not None and expected_lp_cm is not None else None
        )
        frequency_error_percent = (
            100.0 * frequency_error / expected_lp_cm
            if frequency_error is not None and expected_lp_cm else None
        )
        fft_strength = _clamp01((fft_snr or 0.0) / 6.0) if fft_snr is not None else None
        if fft_graph.get("status") == "missing" or fft_snr is None:
            fft_pass = None
            fft_label, fft_status = "Missing", "missing"
            fft_reason = "FFT evidence was unavailable."
            fft_score = None
        elif expected_lp_cm is None or effective_spacing_mm is None or measured_lp_cm is None:
            fft_pass = None
            fft_label, fft_status = "Review", "needs_review"
            fft_reason = "Expected lp/cm mapping or pixel spacing is not configured."
            fft_score = fft_strength
        else:
            frequency_score = _clamp01(1.0 - frequency_error_percent / 40.0)
            fft_score = _clamp01(0.55 * fft_strength + 0.45 * frequency_score)
            fft_pass = bool(
                frequency_error_percent <= target_fft_threshold["frequency_error_percent_threshold"]
                and fft_snr >= target_fft_threshold["fft_snr_threshold"]
                and fft_score >= target_fft_threshold["fft_score_threshold"]
            )
            fft_label, fft_status = ("Pass", "available") if fft_pass else ("Fail", "available")
            fft_reason = "FFT peak frequency and strength passed preliminary thresholds." if fft_pass else "FFT peak frequency or strength did not pass preliminary thresholds."
            if frequency_error_percent > target_fft_threshold["frequency_error_percent_threshold"]:
                review_flags.add("fft_wrong_frequency")
        fft_vote = {
            "status": fft_status, "test_label": fft_label,
            "fft_pass": fft_pass,
            "fft_score": round(fft_score, 4) if fft_score is not None else None,
            "expected_lp_cm": expected_lp_cm,
            "measured_peak_lp_cm": measured_lp_cm,
            "frequency_error_lp_cm": round(frequency_error, 4) if frequency_error is not None else None,
            "frequency_error_percent": round(frequency_error_percent, 3) if frequency_error_percent is not None else None,
            "fft_snr": fft_snr, "reason": fft_reason,
        }

        peak_count = int(profile.get("stripe_peaks_found") or 0)
        valley_count = max(0, peak_count - 1)
        peak_valley = finite(profile.get("profile_peak_to_valley_hu"))
        background_std = finite(std_data.get("background_std_hu"))
        profile_snr = (
            peak_valley / max(background_std, 1e-6)
            if peak_valley is not None and background_std is not None else None
        )
        period_px = finite(profile.get("average_period_px"))
        spacing_mm = (
            period_px * effective_spacing_mm
            if period_px is not None and effective_spacing_mm is not None else None
        )
        expected_spacing_mm = 10.0 / expected_lp_cm if expected_lp_cm else None
        spacing_error = (
            abs(spacing_mm - expected_spacing_mm)
            if spacing_mm is not None and expected_spacing_mm is not None else None
        )
        spacing_error_percent = (
            100.0 * spacing_error / expected_spacing_mm
            if spacing_error is not None and expected_spacing_mm else None
        )
        profile_score = finite(profile.get("profile_quality"))
        if profile.get("profile_data_status") == "missing":
            profile_pass = None
            profile_label, profile_status = "Missing", "missing"
            profile_reason = "Profile evidence was unavailable."
            profile_score = None
        elif expected_lp_cm is None or expected_spacing_mm is None:
            profile_pass = None
            profile_label, profile_status = "Review", "needs_review"
            profile_reason = "Profile peaks were measured, but expected spacing is not configured."
            review_flags.add("expected_spacing_unknown")
        else:
            profile_pass = bool(
                peak_count >= target_profile_threshold["peak_count_threshold"]
                and valley_count >= target_profile_threshold["valley_count_threshold"]
                and profile_snr is not None
                and profile_snr >= target_profile_threshold["profile_snr_threshold"]
                and spacing_error_percent is not None
                and spacing_error_percent <= target_profile_threshold["spacing_error_percent_threshold"]
                and profile_score is not None
                and profile_score >= target_profile_threshold["profile_score_threshold"]
            )
            profile_label, profile_status = ("Pass", "available") if profile_pass else ("Fail", "available")
            profile_reason = "Profile peaks, signal, and spacing passed preliminary thresholds." if profile_pass else "Profile peak, signal, or spacing evidence did not pass preliminary thresholds."
            if spacing_error_percent is not None and spacing_error_percent > target_profile_threshold["spacing_error_percent_threshold"]:
                review_flags.add("profile_spacing_mismatch")
        profile_vote = {
            "status": profile_status, "test_label": profile_label,
            "profile_pass": profile_pass,
            "profile_score": round(profile_score, 4) if profile_score is not None else None,
            "peak_count": peak_count, "valley_count": valley_count,
            "mean_peak_to_valley_hu": peak_valley,
            "peak_spacing_px": period_px,
            "peak_spacing_mm": round(spacing_mm, 4) if spacing_mm is not None else None,
            "expected_spacing_mm": round(expected_spacing_mm, 4) if expected_spacing_mm is not None else None,
            "spacing_error_mm": round(spacing_error, 4) if spacing_error is not None else None,
            "spacing_error_percent": round(spacing_error_percent, 3) if spacing_error_percent is not None else None,
            "profile_snr": round(profile_snr, 4) if profile_snr is not None else None,
            "reason": profile_reason,
        }

        std_hu = finite(std_data.get("std_hu"))
        std_ratio = finite(std_data.get("std_ratio"))
        noise_ratio = finite(std_data.get("peak_to_valley_noise_ratio"))
        contrast_energy = finite(std_data.get("contrast_energy_score"))
        std_components = [
            _clamp01(std_ratio / 4.0) if std_ratio is not None else None,
            _clamp01(noise_ratio / 6.0) if noise_ratio is not None else None,
            contrast_energy,
        ]
        available_std_components = [value for value in std_components if value is not None]
        std_score = float(np.mean(available_std_components)) if available_std_components else None
        if std_data.get("status") == "missing" or std_hu is None:
            std_pass = None
            std_label, std_status = "Missing", "missing"
            std_reason = "STD support evidence was unavailable."
            std_score = None
        elif (
            background_std is None
            or std_data.get("background_noise_status") != "available"
        ):
            std_pass = None
            std_label, std_status = "Review", "needs_review"
            std_reason = std_data.get("background_noise_reason") or "Background noise was unavailable, so STD support cannot vote."
            review_flags.add(
                "background_noise_unreliable"
                if std_data.get("background_noise_status") == "needs_review"
                and background_std is not None
                else "background_noise_unavailable"
            )
        else:
            std_pass = bool(
                std_ratio is not None and std_ratio >= target_std_threshold["std_ratio_threshold"]
                and noise_ratio is not None and noise_ratio >= target_std_threshold["peak_to_valley_noise_threshold"]
                and std_score is not None and std_score >= target_std_threshold["std_score_threshold"]
            )
            std_label, std_status = ("Pass", "available") if std_pass else ("Fail", "available")
            std_reason = "STD and contrast-energy support passed preliminary thresholds." if std_pass else "STD or contrast-energy support did not pass preliminary thresholds."
        std_vote = {
            "status": std_status, "test_label": std_label,
            "std_pass": std_pass,
            "std_score": round(std_score, 4) if std_score is not None else None,
            "std_hu": std_hu, "background_std_hu": background_std,
            "std_ratio": std_ratio, "peak_to_valley_hu": finite(roi_data.get("peak_to_valley_hu")),
            "peak_to_valley_noise_ratio": noise_ratio,
            "contrast_energy_score": contrast_energy, "reason": std_reason,
        }

        vote_values = [fft_pass, profile_pass, std_pass]
        votes_available = sum(value is not None for value in vote_values)
        votes_passed = sum(value is True for value in vote_values)
        weighted_scores = [
            (adaptive_config["combined"]["fft_weight"], fft_score),
            (adaptive_config["combined"]["profile_weight"], profile_score),
            (adaptive_config["combined"]["std_weight"], std_score),
        ]
        available_scores = [(weight, score) for weight, score in weighted_scores if score is not None]
        combined_score = (
            sum(weight * score for weight, score in available_scores)
            / sum(weight for weight, _ in available_scores)
            if available_scores else None
        )
        thresholds_used = {
            "threshold_mode": "target_specific_adaptive",
            "fft_snr_threshold": target_fft_threshold["fft_snr_threshold"],
            "fft_score_threshold": target_fft_threshold["fft_score_threshold"],
            "frequency_error_percent_threshold": target_fft_threshold["frequency_error_percent_threshold"],
            "profile_snr_threshold": target_profile_threshold["profile_snr_threshold"],
            "profile_score_threshold": target_profile_threshold["profile_score_threshold"],
            "peak_count_threshold": target_profile_threshold["peak_count_threshold"],
            "valley_count_threshold": target_profile_threshold["valley_count_threshold"],
            "spacing_error_percent_threshold": target_profile_threshold["spacing_error_percent_threshold"],
            "std_ratio_threshold": target_std_threshold["std_ratio_threshold"],
            "peak_to_valley_noise_threshold": target_std_threshold["peak_to_valley_noise_threshold"],
            "std_score_threshold": target_std_threshold["std_score_threshold"],
            "internal_contrast_support_threshold": target_std_threshold[
                "internal_contrast_support_threshold"
            ],
            "internal_contrast_support_score": target_std_threshold[
                "internal_contrast_support_score"
            ],
            "combined_score_threshold": target_combined_threshold["combined_score_threshold"],
        }
        threshold_margins = {
            "margin_basis": "target_specific_adaptive",
            "fft_snr_margin": threshold_margin(fft_snr, target_fft_threshold["fft_snr_threshold"]),
            "fft_score_margin": threshold_margin(fft_score, target_fft_threshold["fft_score_threshold"]),
            "frequency_error_margin_percent": _module4_safe_error_margin(
                target_fft_threshold["frequency_error_percent_threshold"],
                frequency_error_percent,
            ),
            "profile_snr_margin": threshold_margin(profile_snr, target_profile_threshold["profile_snr_threshold"]),
            "profile_score_margin": threshold_margin(profile_score, target_profile_threshold["profile_score_threshold"]),
            "peak_count_margin": threshold_margin(peak_count, target_profile_threshold["peak_count_threshold"]),
            "valley_count_margin": threshold_margin(valley_count, target_profile_threshold["valley_count_threshold"]),
            "spacing_error_margin_percent": _module4_safe_error_margin(
                target_profile_threshold["spacing_error_percent_threshold"],
                spacing_error_percent,
            ),
            "std_ratio_margin": threshold_margin(std_ratio, target_std_threshold["std_ratio_threshold"]),
            "peak_to_valley_noise_margin": threshold_margin(noise_ratio, target_std_threshold["peak_to_valley_noise_threshold"]),
            "std_score_margin": threshold_margin(std_score, target_std_threshold["std_score_threshold"]),
            "internal_contrast_support_margin": threshold_margin(
                target_std_threshold["internal_contrast_support_score"],
                target_std_threshold["internal_contrast_support_threshold"],
            ),
            "combined_score_margin": threshold_margin(combined_score, target_combined_threshold["combined_score_threshold"]),
        }
        if fft_pass is not None and profile_pass is not None and fft_pass != profile_pass:
            review_flags.add("fft_profile_disagreement")
        if (
            profile_snr is not None
            and profile_snr < target_profile_threshold["profile_snr_threshold"]
            and (
                (profile_score is not None and profile_score >= target_profile_threshold["profile_score_threshold"])
                or peak_count >= target_profile_threshold["peak_count_threshold"]
            )
        ):
            review_flags.add("low_profile_snr_but_pattern_present")
        if std_pass and not fft_pass and not profile_pass:
            review_flags.add("high_std_without_frequency_profile_support")
        if roi_data.get("roi_data_status") == "missing":
            review_flags.add("target_data_missing")
        if combined_score is not None and combined_score < target_combined_threshold["combined_score_threshold"]:
            review_flags.add("combined_score_below_adaptive_threshold")

        major_review_flags = {
            "roi_geometry_confidence_poor", "pixel_spacing_missing",
            "expected_lp_cm_mapping_unknown", "background_noise_unavailable",
            "background_noise_unreliable",
            "development_mapping_disabled", "expected_mapping_incomplete",
            "fft_wrong_frequency", "profile_spacing_mismatch",
            "high_std_without_frequency_profile_support",
            "fft_profile_disagreement", "target_data_missing",
            "combined_score_below_adaptive_threshold",
        }
        if "target_data_missing" in review_flags:
            final_label, target_visible, confidence = "missing", None, "needs_review"
        elif review_flags & major_review_flags or votes_available < 3:
            final_label, target_visible, confidence = "needs_review", None, "needs_review"
        elif votes_passed == 3:
            final_label, target_visible, confidence = "visible", True, "high"
        elif votes_passed == 2:
            final_label, target_visible, confidence = "visible", True, "medium"
        elif votes_passed == 0:
            final_label, target_visible, confidence = "not_visible", False, "low"
        else:
            final_label, target_visible, confidence = "needs_review", None, "low"

        reason = (
            "Automated preliminary review requires physicist review."
            if final_label == "needs_review"
            else f"{votes_passed} of {votes_available} available preliminary tests passed."
        )
        internal_support_score = finite(
            target_std_threshold.get("internal_contrast_support_score")
        )
        combined_above = bool(
            combined_score is not None
            and combined_score >= target_combined_threshold["combined_score_threshold"]
        )
        profile_score_above = bool(
            profile_score is not None
            and profile_score >= target_profile_threshold["profile_score_threshold"]
        )
        profile_snr_above = bool(
            profile_snr is not None
            and profile_snr >= target_profile_threshold["profile_snr_threshold"]
        )
        peak_count_meets = peak_count >= target_profile_threshold["peak_count_threshold"]
        valley_count_meets = valley_count >= target_profile_threshold["valley_count_threshold"]
        internal_contrast_above = bool(
            internal_support_score is not None
            and internal_support_score
            >= target_std_threshold["internal_contrast_support_threshold"]
        )
        fft_snr_above = (
            None if fft_snr is None else
            fft_snr >= target_fft_threshold["fft_snr_threshold"]
        )
        diagnostic_evidence = {
            "combined_above_threshold": combined_above,
            "profile_score_above_threshold": profile_score_above,
            "profile_snr_above_threshold": profile_snr_above,
            "peak_count_meets_expectation": peak_count_meets,
            "valley_count_meets_expectation": valley_count_meets,
            "internal_contrast_above_threshold": internal_contrast_above,
            "fft_snr_above_threshold": fft_snr_above,
        }
        major_support_count = sum((
            profile_score_above and profile_snr_above,
            peak_count_meets and valley_count_meets,
            internal_contrast_above,
            fft_snr_above is True,
        ))
        combined_threshold_value = target_combined_threshold[
            "combined_score_threshold"
        ]
        combined_near = bool(
            combined_score is not None
            and combined_score >= 0.90 * combined_threshold_value
        )
        strong_requirements_met = bool(
            combined_above
            and profile_score_above
            and profile_snr_above
            and peak_count_meets
            and valley_count_meets
            and internal_contrast_above
            and (fft_snr_above is True or fft_snr is None)
        )
        profile_and_counts_weak = bool(
            not profile_score_above
            and (not peak_count_meets or not valley_count_meets)
        )
        contradictory_evidence = bool(
            combined_above and major_support_count < 2
        )
        if combined_score is None or profile_score is None:
            diagnostic_label = "needs_review"
            diagnostic_reason = "Required target-local combined or profile evidence was unavailable."
        elif contradictory_evidence:
            diagnostic_label = "needs_review"
            diagnostic_reason = "Combined evidence conflicts with the available target-local evidence groups."
        elif strong_requirements_met:
            diagnostic_label = "strong"
            diagnostic_reason = "Combined, profile, peak/valley, internal-contrast, and available FFT evidence all support a strong diagnostic signal."
        elif combined_near and major_support_count >= 2:
            diagnostic_label = "moderate"
            diagnostic_reason = "Combined evidence is near its threshold and at least two major evidence groups support the target, but strong criteria are incomplete."
        elif not combined_above or profile_and_counts_weak or major_support_count < 2:
            diagnostic_label = "weak"
            diagnostic_reason = "Combined evidence is below threshold or target-local profile/peak evidence is weak."
        else:
            diagnostic_label = "needs_review"
            diagnostic_reason = "Available evidence is incomplete or contradictory and needs diagnostic review."
        diagnostic_strength = {
            "label": diagnostic_label,
            "score": round(combined_score, 4) if combined_score is not None else None,
            "evidence": diagnostic_evidence,
            "major_support_count": major_support_count,
            "reason": diagnostic_reason,
        }
        visibility_vote = {
            "target": target_id, "expected_lp_cm": expected_lp_cm,
            "votes_passed": votes_passed, "votes_available": votes_available,
            "fft": fft_label.lower(), "profile": profile_label.lower(),
            "std": std_label.lower(),
            "combined_score": round(combined_score, 4) if combined_score is not None else None,
            "target_visible": target_visible,
            "visibility_confidence": confidence,
            "final_preliminary_label": final_label,
            "review_flags": sorted(review_flags), "reason": reason,
        }
        analysis["fft_vote"] = fft_vote
        analysis["profile_vote"] = profile_vote
        analysis["std_vote"] = std_vote
        analysis["visibility_vote"] = visibility_vote
        analysis["thresholds_used"] = thresholds_used
        analysis["threshold_margins"] = threshold_margins
        analysis["target_signal_context"] = context_by_target[target_id]
        analysis["target_adaptive_thresholds"] = target_threshold
        analysis["threshold_source_reason"] = target_threshold[
            "threshold_source_reason"
        ]
        analysis["internal_contrast_support"] = context_by_target[target_id][
            "internal_contrast_support"
        ]
        analysis["target_diagnostic_strength"] = diagnostic_strength
        analysis["target_threshold_review"] = {
            "target": target_id,
            "threshold_mode": "target_specific_adaptive",
            "profile_snr": {
                "actual": profile_snr,
                "threshold": thresholds_used["profile_snr_threshold"],
                "margin": threshold_margins["profile_snr_margin"],
            },
            "profile_score": {
                "actual": profile_score,
                "threshold": thresholds_used["profile_score_threshold"],
                "margin": threshold_margins["profile_score_margin"],
            },
            "fft_snr": {
                "actual": fft_snr,
                "threshold": thresholds_used["fft_snr_threshold"],
                "margin": threshold_margins["fft_snr_margin"],
            },
            "combined_score": {
                "actual": combined_score,
                "threshold": thresholds_used["combined_score_threshold"],
                "margin": threshold_margins["combined_score_margin"],
            },
            "diagnostic_strength": diagnostic_strength,
            "fft_threshold_debug": target_fft_threshold[
                "fft_threshold_debug"
            ],
            "threshold_source_reason": target_threshold[
                "threshold_source_reason"
            ],
        }
        target["module4_preliminary_analysis"] = analysis
        visibility_votes.append(visibility_vote)

    threshold_signature_fields = (
        "fft_snr_threshold", "fft_score_threshold",
        "profile_snr_threshold", "profile_score_threshold",
        "peak_count_threshold", "valley_count_threshold",
        "internal_contrast_support_threshold", "combined_score_threshold",
    )
    threshold_signatures = {}
    for target in targets:
        analysis = target.get("module4_preliminary_analysis") or {}
        used = analysis.get("thresholds_used") or {}
        threshold_signatures[target.get("id", "unknown")] = {
            field: used.get(field) for field in threshold_signature_fields
        }
    identical_fields = []
    different_fields = []
    for field in threshold_signature_fields:
        values = {
            json.dumps(signature.get(field), sort_keys=True)
            for signature in threshold_signatures.values()
        }
        (identical_fields if len(values) <= 1 else different_fields).append(field)
    all_thresholds_identical = (
        bool(threshold_signatures) and not different_fields
    )
    target_specific_enabled = MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG[
        "target_specific_thresholds_enabled"
    ]
    target_specific_threshold_warning = (
        "All target thresholds are identical; target-specific adaptation may not be working."
        if all_thresholds_identical and len(target_adaptive_thresholds) > 1
        else None
    )
    if target_specific_threshold_warning:
        module_flags.add("target_specific_thresholds_not_effective")
    max_guardrail_hits = []
    threshold_too_high_for_actual = []
    threshold_too_low_for_actual = []
    diagnostic_strength_review = []
    outlier_control_applied = False
    for target in targets:
        target_id = target.get("id", "unknown")
        analysis = target.get("module4_preliminary_analysis") or {}
        used = analysis.get("thresholds_used") or {}
        strength = analysis.get("target_diagnostic_strength") or {}
        evidence = strength.get("evidence") or {}
        fft_debug = (
            (analysis.get("target_adaptive_thresholds") or {})
            .get("fft", {}).get("fft_threshold_debug", {})
        )
        outlier_control_applied = bool(
            outlier_control_applied
            or fft_debug.get("outlier_spread_detected")
        )
        adaptive_maxima = {
            "fft_snr_threshold": target_specific_guardrails["fft_snr_max"],
            "fft_score_threshold": target_specific_guardrails["fft_score_max"],
            "profile_snr_threshold": target_specific_guardrails["profile_snr_max"],
            "profile_score_threshold": target_specific_guardrails["profile_score_max"],
            "combined_score_threshold": target_specific_guardrails["combined_score_max"],
        }
        for field, maximum in adaptive_maxima.items():
            value = finite(used.get(field))
            if value is not None and math.isclose(
                value, maximum, rel_tol=0.0, abs_tol=1e-6
            ):
                max_guardrail_hits.append({
                    "target": target_id, "field": field, "value": value,
                })
        actual_fft = finite(fft_debug.get("actual_fft_snr"))
        fft_threshold = finite(used.get("fft_snr_threshold"))
        if (
            actual_fft is not None and fft_threshold is not None
            and fft_threshold > 2.0 * max(actual_fft, 1e-6)
        ):
            threshold_too_high_for_actual.append({
                "target": target_id,
                "actual_fft_snr": actual_fft,
                "fft_snr_threshold": fft_threshold,
                "ratio": round(fft_threshold / max(actual_fft, 1e-6), 4),
            })
        if (
            actual_fft is not None and fft_threshold is not None
            and math.isclose(
                fft_threshold, target_specific_guardrails["fft_snr_min"],
                rel_tol=0.0, abs_tol=1e-6,
            )
            and actual_fft > 2.0 * fft_threshold
        ):
            threshold_too_low_for_actual.append({
                "target": target_id,
                "actual_fft_snr": actual_fft,
                "fft_snr_threshold": fft_threshold,
                "reason": "FFT threshold reached the minimum guardrail despite substantially stronger target-local evidence.",
            })
        combined_actual = finite(
            (analysis.get("visibility_vote") or {}).get("combined_score")
        )
        combined_threshold = finite(used.get("combined_score_threshold"))
        combined_barely_above = bool(
            combined_actual is not None and combined_threshold is not None
            and combined_actual >= combined_threshold
            and combined_actual - combined_threshold
            <= max(0.02, 0.05 * combined_threshold)
        )
        strong_contradiction = bool(
            strength.get("label") == "strong"
            and (
                not evidence.get("profile_score_above_threshold")
                or not evidence.get("peak_count_meets_expectation")
                or not evidence.get("valley_count_meets_expectation")
                or combined_barely_above
            )
        )
        diagnostic_strength_review.append({
            "target": target_id,
            "label": strength.get("label", "needs_review"),
            "strong_label_contradiction": strong_contradiction,
            "combined_barely_above_threshold": combined_barely_above,
            "profile_score_below_threshold": not bool(
                evidence.get("profile_score_above_threshold")
            ),
            "peak_or_valley_expectation_weak": not bool(
                evidence.get("peak_count_meets_expectation")
                and evidence.get("valley_count_meets_expectation")
            ),
            "reason": (
                "Strong label conflicts with weak profile/peak evidence or only a marginal combined-score clearance."
                if strong_contradiction else
                "Diagnostic strength is consistent with its multi-evidence rule."
            ),
        })
    strength_contradictions = [
        row for row in diagnostic_strength_review
        if row["strong_label_contradiction"]
    ]
    if strength_contradictions:
        threshold_quality_status = "failed"
        threshold_quality_reason = "One or more strong diagnostic labels contradict weak target-local evidence."
    elif max_guardrail_hits or threshold_too_high_for_actual or threshold_too_low_for_actual:
        threshold_quality_status = "needs_review"
        threshold_quality_reason = "Automatic thresholds are active, but one or more guardrail or target-relative checks need review."
    else:
        threshold_quality_status = "passed"
        threshold_quality_reason = "Outlier-resistant automatic thresholds and multi-evidence diagnostic labels passed runtime quality checks."
    module4_threshold_quality_review = {
        "status": threshold_quality_status,
        "outlier_control_applied": outlier_control_applied,
        "max_guardrail_hits": max_guardrail_hits,
        "threshold_too_high_for_actual": threshold_too_high_for_actual,
        "threshold_too_low_for_actual": threshold_too_low_for_actual,
        "diagnostic_strength_review": diagnostic_strength_review,
        "reason": threshold_quality_reason,
    }
    if not target_specific_enabled or len(threshold_signatures) < len(targets):
        threshold_validation_status = "needs_review"
        threshold_validation_reason = (
            "Target-specific configuration is disabled or threshold data are incomplete."
        )
    elif all_thresholds_identical and len(targets) > 1:
        threshold_validation_status = "failed"
        threshold_validation_reason = target_specific_threshold_warning
    elif threshold_quality_status != "passed":
        threshold_validation_status = (
            "failed" if threshold_quality_status == "failed" else "needs_review"
        )
        threshold_validation_reason = threshold_quality_reason
    else:
        threshold_validation_status = "passed"
        threshold_validation_reason = (
            "Automatic target-local thresholds differ where signal data differ; "
            "identical fixed guardrail fields are expected."
        )
    module4_target_threshold_validation = {
        "status": threshold_validation_status,
        "target_specific_thresholds_enabled": target_specific_enabled,
        "all_thresholds_identical": all_thresholds_identical,
        "identical_fields": identical_fields,
        "different_fields": different_fields,
        "threshold_signatures": threshold_signatures,
        "uses_run_level_values_only": False,
        "threshold_quality_status": threshold_quality_status,
        "outlier_control_status": (
            "applied" if outlier_control_applied else "not_needed"
        ),
        "diagnostic_strength_status": (
            "needs_review" if strength_contradictions else "passed"
        ),
        "warning": target_specific_threshold_warning,
        "reason": threshold_validation_reason,
    }
    target_threshold_review = [
        (target.get("module4_preliminary_analysis") or {}).get(
            "target_threshold_review", {}
        )
        for target in targets
    ]
    threshold_source_reasons = [
        threshold.get("threshold_source_reason", {})
        for threshold in target_adaptive_thresholds
    ]
    fft_threshold_debug = [
        threshold.get("fft", {}).get("fft_threshold_debug", {})
        for threshold in target_adaptive_thresholds
    ]

    scoring_mapping = {
        row["target"]: row["lp_cm"]
        for row in MODULE4_AUTO_PRELIMINARY_SCORING_CONFIG["target_lp_cm_order"]
    }
    strength_review_by_target = {
        row.get("target"): row for row in diagnostic_strength_review
    }
    preliminary_score_candidates = []
    for target in sorted(
        targets,
        key=lambda item: scoring_mapping.get(item.get("id"), 999),
    ):
        target_id = target.get("id", "unknown")
        lp_cm = scoring_mapping.get(target_id)
        analysis = target.get("module4_preliminary_analysis") or {}
        review = analysis.get("target_threshold_review") or {}
        strength = analysis.get("target_diagnostic_strength") or {}
        evidence = strength.get("evidence") or {}
        fft_vote = analysis.get("fft_vote") or {}
        profile_vote = analysis.get("profile_vote") or {}
        roi_data = target.get("roi_data") or {}
        profile_data = target.get("profile_data") or {}
        thresholds_used = analysis.get("thresholds_used") or {}
        visibility_vote = analysis.get("visibility_vote") or {}
        strength_review = strength_review_by_target.get(target_id, {})
        review_flags = set(visibility_vote.get("review_flags") or [])
        major_contradiction_flags = sorted(review_flags & {
            "roi_geometry_confidence_poor", "fft_wrong_frequency",
            "profile_spacing_mismatch", "high_std_without_frequency_profile_support",
            "fft_profile_disagreement", "target_data_missing",
        })
        strength_label = strength.get("label", "needs_review")
        strong_supported = bool(
            strength_label == "strong"
            and not strength_review.get("strong_label_contradiction")
            and not major_contradiction_flags
        )
        moderate_supported = bool(
            strength_label == "moderate"
            and evidence.get("profile_score_above_threshold") is True
            and evidence.get("profile_snr_above_threshold") is True
            and evidence.get("peak_count_meets_expectation") is True
            and evidence.get("valley_count_meets_expectation") is True
            and evidence.get("internal_contrast_above_threshold") is True
            and evidence.get("fft_snr_above_threshold") in {True, None}
            and not major_contradiction_flags
        )
        evidence_eligible = bool(strong_supported or moderate_supported)
        if lp_cm is None or strength_label == "missing":
            initial_status = "missing"
            evidence_eligible = False
            score_reason = "Required target evidence or preliminary lp/cm assignment is missing."
        elif strong_supported:
            initial_status = "resolved"
            score_reason = "Strong target-local evidence supports preliminary resolution review."
        elif moderate_supported:
            initial_status = "resolved"
            score_reason = "Moderate evidence is supported by profile score, profile SNR, peak/valley counts, internal contrast, and available FFT strength."
        elif strength_label == "weak":
            initial_status = "unresolved"
            score_reason = "Weak profile or target-local evidence does not support preliminary resolution."
        else:
            initial_status = "needs_review"
            score_reason = (
                "Target-local evidence is incomplete or contradictory and needs review."
                if not major_contradiction_flags else
                "Major contradictory evidence requires review: "
                + ", ".join(major_contradiction_flags) + "."
            )
        def ratio(actual, threshold):
            actual_value = _module4_safe_float(actual)
            threshold_value = _module4_safe_float(threshold)
            if actual_value is None or threshold_value is None or threshold_value <= 0:
                return None
            return actual_value / threshold_value

        combined_actual = (review.get("combined_score") or {}).get("actual")
        combined_threshold = (review.get("combined_score") or {}).get("threshold")
        profile_actual = (review.get("profile_score") or {}).get("actual")
        profile_threshold = (review.get("profile_score") or {}).get("threshold")
        profile_snr = (review.get("profile_snr") or {}).get("actual")
        profile_snr_threshold = (review.get("profile_snr") or {}).get("threshold")
        fft_snr = (review.get("fft_snr") or {}).get("actual")
        fft_snr_threshold = (review.get("fft_snr") or {}).get("threshold")
        peak_count = profile_vote.get("peak_count")
        peak_threshold = thresholds_used.get("peak_count_threshold")
        valley_count = profile_vote.get("valley_count")
        valley_threshold = thresholds_used.get("valley_count_threshold")
        contrast_actual = thresholds_used.get("internal_contrast_support_score")
        contrast_threshold = thresholds_used.get("internal_contrast_support_threshold")
        score_components = [
            (ratio(combined_actual, combined_threshold), 0.28),
            (ratio(profile_actual, profile_threshold), 0.18),
            (ratio(profile_snr, profile_snr_threshold), 0.14),
            (ratio(peak_count, peak_threshold), 0.06),
            (ratio(valley_count, valley_threshold), 0.06),
            (ratio(contrast_actual, contrast_threshold), 0.16),
            (ratio(fft_snr, fft_snr_threshold), 0.12),
        ]
        available_components = [
            (min(max(value, 0.0), 1.5) / 1.5, weight)
            for value, weight in score_components if value is not None
        ]
        available_weight = sum(weight for _, weight in available_components)
        auto_evidence_score = (
            sum(value * weight for value, weight in available_components)
            / available_weight if available_weight else None
        )

        def support_label(*values):
            usable = [value for value in values if value is not None]
            if not usable:
                return "review_only"
            minimum = min(usable)
            if minimum >= 1.0:
                return "strong"
            if minimum >= 0.8:
                return "moderate"
            return "weak"

        preliminary_score_candidates.append({
            "target": target_id,
            "lp_cm": lp_cm,
            "display_label": f"{lp_cm} lp/cm · {target_id}" if lp_cm is not None else target_id,
            "diagnostic_strength": strength_label,
            "score_status": initial_status,
            "evidence_eligible": evidence_eligible,
            "counts_for_resolution": False,
            "auto_evidence_score": auto_evidence_score,
            "target_specific_threshold_margin": _module4_safe_margin(
                combined_actual, combined_threshold
            ),
            "profile_support": support_label(
                ratio(profile_actual, profile_threshold),
                ratio(profile_snr, profile_snr_threshold),
            ),
            "contrast_support": support_label(
                ratio(contrast_actual, contrast_threshold)
            ),
            "fft_support": support_label(ratio(fft_snr, fft_snr_threshold)),
            "peak_valley_support": support_label(
                ratio(peak_count, peak_threshold), ratio(valley_count, valley_threshold)
            ),
            "combined_score": combined_actual,
            "combined_threshold": combined_threshold,
            "profile_score": profile_actual,
            "profile_score_threshold": profile_threshold,
            "profile_snr": profile_snr,
            "profile_snr_threshold": profile_snr_threshold,
            "fft_snr": fft_snr,
            "fft_snr_threshold": fft_snr_threshold,
            "fft_score": fft_vote.get("fft_score"),
            "fft_score_threshold": thresholds_used.get("fft_score_threshold"),
            "peak_count": peak_count,
            "peak_count_threshold": peak_threshold,
            "valley_count": valley_count,
            "valley_count_threshold": valley_threshold,
            "internal_contrast_support": contrast_actual,
            "internal_contrast_support_threshold": contrast_threshold,
            "peak_to_valley_hu": roi_data.get("peak_to_valley_hu"),
            "std_hu": roi_data.get("std_hu"),
            "profile_quality": profile_data.get("profile_quality"),
            "periodicity": profile_data.get("periodicity_score"),
            "visibility_label": target.get("preliminary_visibility"),
            "review_flags": sorted(review_flags),
            "major_contradiction_flags": major_contradiction_flags,
            "reason": score_reason,
        })

    # Add an evidence-curve view of peak/valley contrast. This compares each
    # target only with the already-visited, lower-frequency targets; it does not
    # introduce a required lp/cm cutoff or alter signal extraction.
    for index, candidate in enumerate(preliminary_score_candidates):
        current_peak_valley = _module4_safe_float(
            candidate.get("peak_to_valley_hu")
        )
        lower_frequency_values = [
            value for value in (
                _module4_safe_float(previous.get("peak_to_valley_hu"))
                for previous in preliminary_score_candidates[:index]
            ) if value is not None and value > 0
        ]
        lower_frequency_reference = _module4_safe_median(lower_frequency_values)
        peak_valley_ratio = (
            current_peak_valley / lower_frequency_reference
            if current_peak_valley is not None
            and lower_frequency_reference is not None
            and lower_frequency_reference > 0 else None
        )
        if peak_valley_ratio is None:
            peak_valley_support = "review_only"
        elif peak_valley_ratio >= 0.75:
            peak_valley_support = "strong"
        elif peak_valley_ratio >= 0.50:
            peak_valley_support = "moderate"
        else:
            peak_valley_support = "weak"
        candidate["peak_valley_support"] = peak_valley_support
        candidate["peak_valley_ladder_ratio"] = peak_valley_ratio
        candidate["lower_frequency_peak_valley_reference_hu"] = (
            lower_frequency_reference
        )
        if (
            candidate.get("auto_evidence_score") is not None
            and peak_valley_ratio is not None
        ):
            candidate["auto_evidence_score"] = (
                0.85 * candidate["auto_evidence_score"]
                + 0.15 * min(max(peak_valley_ratio, 0.0), 1.0)
            )

    preliminary_target_scores = []
    continuity_broken = False
    resolved_score = None
    next_unresolved_score = None
    for candidate in preliminary_score_candidates:
        score = dict(candidate)
        eligible = bool(score.pop("evidence_eligible", False))
        if continuity_broken:
            score["counts_for_resolution"] = False
            score["continuity_status"] = "not_used_past_break"
            score["reason"] = (
                "Not used to extend resolution because continuity stopped at "
                f"{next_unresolved_score['display_label']}."
            )
        elif eligible:
            score["counts_for_resolution"] = True
            score["score_status"] = "resolved"
            score["continuity_status"] = "resolved_in_sequence"
            resolved_score = score
        else:
            score["counts_for_resolution"] = False
            score["continuity_status"] = "first_unresolved"
            continuity_broken = True
            next_unresolved_score = score
        preliminary_target_scores.append(score)

    resolved_lp_cm = resolved_score.get("lp_cm") if resolved_score else None
    resolved_target = resolved_score.get("target") if resolved_score else None
    next_unresolved_lp_cm = (
        next_unresolved_score.get("lp_cm") if next_unresolved_score else None
    )
    next_unresolved_target = (
        next_unresolved_score.get("target") if next_unresolved_score else None
    )
    resolved_sequence = [
        score["lp_cm"] for score in preliminary_target_scores
        if score["counts_for_resolution"]
    ]
    resolved_moderate = any(
        score["counts_for_resolution"]
        and score["diagnostic_strength"] == "moderate"
        for score in preliminary_target_scores
    )
    preliminary_confidence = (
        "low" if resolved_lp_cm is None
        else "moderate" if resolved_moderate or next_unresolved_score is not None
        else "high"
    )
    if resolved_sequence:
        sequence_text = (
            str(resolved_sequence[0]) if len(resolved_sequence) == 1 else
            ", ".join(str(value) for value in resolved_sequence[:-1])
            + f", and {resolved_sequence[-1]}"
        )
        scoring_reason = (
            f"Automatic evidence-based scoring resolved the {sequence_text} lp/cm targets in order. "
            + (
                f"The {next_unresolved_lp_cm} lp/cm target is "
                f"{next_unresolved_score['diagnostic_strength'].replace('_', ' ')}, "
                "and does not satisfy its target-specific evidence checks, "
                f"so preliminary resolution stops at {resolved_lp_cm} lp/cm."
                if next_unresolved_score else
                f"Preliminary resolution reaches {resolved_lp_cm} lp/cm."
            )
        )
    else:
        scoring_reason = "The first target does not yet support contiguous preliminary resolution."
    previous_candidate = None
    break_drop_summary = {}
    if next_unresolved_score is not None:
        break_index = preliminary_target_scores.index(next_unresolved_score)
        if break_index > 0:
            previous_candidate = preliminary_target_scores[break_index - 1]

        def evidence_drop(field):
            if previous_candidate is None:
                return None
            previous_value = _module4_safe_float(previous_candidate.get(field))
            current_value = _module4_safe_float(next_unresolved_score.get(field))
            if previous_value is None or current_value is None:
                return None
            return previous_value - current_value

        break_drop_summary = {
            "combined_drop": evidence_drop("combined_score"),
            "peak_valley_drop": evidence_drop("peak_to_valley_hu"),
            "profile_quality_drop": evidence_drop("profile_score"),
            "peak_count_drop": evidence_drop("peak_count"),
            "valley_count_drop": evidence_drop("valley_count"),
        }
    module4_auto_resolution_break = {
        "break_detected": next_unresolved_score is not None,
        "break_target": next_unresolved_target,
        "break_lp_cm": next_unresolved_lp_cm,
        "previous_resolved_target": resolved_target,
        "previous_resolved_lp_cm": resolved_lp_cm,
        "break_reason": (
            next_unresolved_score.get("reason") if next_unresolved_score
            else "No evidence break was detected in the configured target order."
        ),
        "evidence_drop_summary": break_drop_summary,
    }
    module4_auto_preliminary_scoring = {
        "scoring_enabled": MODULE4_ENABLE_AUTO_PRELIMINARY_SCORING,
        "scoring_type": MODULE4_AUTO_PRELIMINARY_SCORING_CONFIG["scoring_type"],
        "official_acr_result": False,
        "physicist_review_required": True,
        "manual_required_threshold_used": False,
        "auto_cutoff_used": True,
        "target_scores": preliminary_target_scores,
        "preliminary_resolved_lp_cm": resolved_lp_cm,
        "preliminary_resolved_target": resolved_target,
        "first_unresolved_lp_cm": next_unresolved_lp_cm,
        "first_unresolved_target": next_unresolved_target,
        "resolution_break": module4_auto_resolution_break,
        "confidence": preliminary_confidence,
        "reason": scoring_reason,
        "limitations": [
            "Preliminary review scoring only; physicist review is required.",
            "The cutoff is detected automatically from target-local evidence; no manual required lp/cm threshold is used.",
            "Continuity stops at the first target that does not support resolution.",
        ],
    }

    def build_method_result(method_key, method_name, evaluator, review_only=False):
        target_votes = []
        method_resolved = None
        first_unresolved = None
        continuity_broken = False
        previous = None
        for candidate in preliminary_score_candidates:
            supported, vote_status, vote_reason = evaluator(candidate, previous)
            vote = {
                "target": candidate["target"],
                "lp_cm": candidate["lp_cm"],
                "display_label": candidate["display_label"],
                "vote": vote_status,
                "supports_resolution": bool(supported and not continuity_broken),
                "counts_for_method_resolution": False,
                "reason": vote_reason,
            }
            if continuity_broken:
                vote["vote"] = "NOT_USED_PAST_BREAK"
                vote["reason"] = (
                    "Not used because this method stopped at "
                    f"{first_unresolved['display_label']}."
                )
            elif supported:
                vote["counts_for_method_resolution"] = True
                method_resolved = vote
            else:
                continuity_broken = True
                first_unresolved = vote
                vote["vote"] = "FAIL" if vote_status != "REVIEW" else "REVIEW"
            target_votes.append(vote)
            previous = candidate

        resolved_method_lp = method_resolved.get("lp_cm") if method_resolved else None
        resolved_method_target = method_resolved.get("target") if method_resolved else None
        unresolved_method_lp = first_unresolved.get("lp_cm") if first_unresolved else None
        unresolved_method_target = first_unresolved.get("target") if first_unresolved else None
        if method_resolved is None:
            method_result = "FAIL"
        elif review_only:
            method_result = "REVIEW"
        else:
            method_result = "PASS"
        reason = (
            f"{method_name} supports contiguous resolution through "
            f"{resolved_method_lp} lp/cm ({resolved_method_target})."
            if method_resolved else
            f"{method_name} does not support the first target in the resolution order."
        )
        if first_unresolved:
            reason += (
                f" {first_unresolved['display_label']} is the first unsupported target."
            )
        if review_only:
            reason += (
                " Frequency-location validation remains review-only because the "
                "confirmed analytical mapping is disabled."
            )
        return {
            "method_key": method_key,
            "method_name": method_name,
            "method_result": method_result,
            "resolved_lp_cm": resolved_method_lp,
            "resolved_target": resolved_method_target,
            "first_unresolved_lp_cm": unresolved_method_lp,
            "first_unresolved_target": unresolved_method_target,
            "reason": reason,
            "target_votes": target_votes,
        }

    def profile_method_evaluator(candidate, _previous):
        flags = set(candidate.get("review_flags") or [])
        contradiction = flags & {
            "roi_geometry_confidence_poor", "profile_spacing_mismatch",
            "target_data_missing",
        }
        checks = (
            ratio(candidate.get("profile_score"), candidate.get("profile_score_threshold")),
            ratio(candidate.get("profile_snr"), candidate.get("profile_snr_threshold")),
            ratio(candidate.get("peak_count"), candidate.get("peak_count_threshold")),
            ratio(candidate.get("valley_count"), candidate.get("valley_count_threshold")),
        )
        supported = all(value is not None and value >= 1.0 for value in checks)
        if contradiction:
            return False, "REVIEW", "Profile evidence has a review-only contradiction: " + ", ".join(sorted(contradiction)) + "."
        if supported:
            return True, "PASS", "Profile score, profile SNR, and target-local peak/valley counts support this target."
        return False, "FAIL", "Profile score, profile SNR, or target-local peak/valley counts do not support this target."

    def fft_method_evaluator(candidate, _previous):
        flags = set(candidate.get("review_flags") or [])
        contradiction = flags & {
            "roi_geometry_confidence_poor", "fft_wrong_frequency",
            "target_data_missing",
        }
        snr_ratio = ratio(candidate.get("fft_snr"), candidate.get("fft_snr_threshold"))
        score_ratio = ratio(candidate.get("fft_score"), candidate.get("fft_score_threshold"))
        supported = bool(
            snr_ratio is not None and snr_ratio >= 1.0
            and (score_ratio is None or score_ratio >= 1.0)
        )
        if contradiction:
            return False, "REVIEW", "FFT evidence has a review-only contradiction: " + ", ".join(sorted(contradiction)) + "."
        if supported:
            return True, "PASS", "FFT signal strength supports this target; frequency-location confirmation remains review-only."
        return False, "FAIL", "FFT signal strength does not support this target at its target-specific threshold."

    def contrast_method_evaluator(candidate, previous):
        contrast_ratio = ratio(
            candidate.get("internal_contrast_support"),
            candidate.get("internal_contrast_support_threshold"),
        )
        current_peak_valley = _module4_safe_float(candidate.get("peak_to_valley_hu"))
        previous_peak_valley = (
            _module4_safe_float(previous.get("peak_to_valley_hu"))
            if previous else None
        )
        ladder_ratio = (
            current_peak_valley / previous_peak_valley
            if current_peak_valley is not None and previous_peak_valley
            and previous_peak_valley > 0 else None
        )
        sharp_drop = ladder_ratio is not None and ladder_ratio < 0.35
        supported = bool(
            contrast_ratio is not None and contrast_ratio >= 1.0
            and not sharp_drop
            and candidate.get("visibility_label") not in {"missing", "weak"}
        )
        if supported:
            return True, "PASS", "Internal contrast and the peak/valley evidence curve support this target without a sharp drop."
        if sharp_drop:
            return False, "FAIL", "Peak/valley signal drops sharply from the preceding lower-frequency target."
        return False, "FAIL", "Internal contrast or raw/profile visibility does not support this target."

    profile_method = build_method_result(
        "profile_peak_method", "Profile / peaks", profile_method_evaluator
    )
    fft_method = build_method_result(
        "fft_method", "FFT frequency support", fft_method_evaluator,
        review_only=not MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING,
    )
    contrast_method = build_method_result(
        "contrast_std_method", "Contrast / STD support", contrast_method_evaluator
    )
    method_results = [profile_method, fft_method, contrast_method]
    supporting_methods = [
        method for method in method_results
        if resolved_lp_cm is not None
        and method.get("resolved_lp_cm") is not None
        and method["resolved_lp_cm"] >= resolved_lp_cm
    ]
    methods_passed = sum(method["method_result"] == "PASS" for method in method_results)
    methods_failed = sum(method["method_result"] == "FAIL" for method in method_results)
    methods_review = sum(method["method_result"] == "REVIEW" for method in method_results)
    final_preliminary_result = (
        "PASS" if len(supporting_methods) >= 2
        else "REVIEW" if len(supporting_methods) == 1
        else "FAIL"
    )
    method_vote_reason = (
        f"{len(supporting_methods)} of 3 automatic methods support the "
        f"{resolved_lp_cm} lp/cm cutoff."
        if resolved_lp_cm is not None else
        "No contiguous automatic resolution cutoff is currently supported."
    )
    module4_auto_method_scoring = {
        "scoring_type": "automatic_three_method_review",
        "manual_threshold_used": False,
        "official_acr_result": False,
        "physicist_review_required": True,
        "final_preliminary_result": final_preliminary_result,
        "resolved_lp_cm": resolved_lp_cm,
        "resolved_target": resolved_target,
        "first_unresolved_lp_cm": next_unresolved_lp_cm,
        "first_unresolved_target": next_unresolved_target,
        "confidence": preliminary_confidence,
        "method_vote_summary": {
            "methods_total": 3,
            "methods_passed": methods_passed,
            "methods_failed": methods_failed,
            "methods_review": methods_review,
            "methods_supporting_cutoff": len(supporting_methods),
            "combined_rule": "automatic majority/support review",
            "reason": method_vote_reason,
        },
        "methods": method_results,
    }
    module4_auto_preliminary_scoring["module4_auto_method_scoring"] = (
        module4_auto_method_scoring
    )
    # Compatibility alias for downstream consumers while they transition to the
    # explicit automatic-scoring response key. It contains no manual threshold.
    module4_preliminary_scoring = module4_auto_preliminary_scoring

    mapped_votes = [vote for vote in visibility_votes if vote["expected_lp_cm"] is not None]
    continuity_status = "not_applicable"
    if mapped_votes:
        ordered = sorted(mapped_votes, key=lambda vote: vote["expected_lp_cm"])
        seen_not_visible = False
        for vote in ordered:
            if vote["final_preliminary_label"] == "not_visible":
                seen_not_visible = True
            elif seen_not_visible and vote["final_preliminary_label"] == "visible":
                module_flags.add("non_monotonic_visibility")
        continuity_status = (
            "needs_review"
            if "non_monotonic_visibility" in module_flags else "passed"
        )

    required_lp_cm = MODULE4_EXPECTED_LP_CM_CONFIG["required_lp_cm"]
    if not mapped_votes:
        resolution = {
            "status": "not_available", "resolved_lp_cm": None,
            "resolved_target": None, "highest_visible_target": None,
            "continuity_check_status": continuity_status,
            "required_lp_cm": required_lp_cm, "preliminary_result": "pending",
            "physicist_review_required": True,
            "review_flags": sorted(
                module_flags | {"expected_lp_cm_mapping_unknown"}
            ),
            "reason": "Expected lp/cm mapping is not configured.",
        }
    else:
        visible = [vote for vote in mapped_votes if vote["final_preliminary_label"] == "visible"]
        highest = max(visible, key=lambda vote: vote["expected_lp_cm"]) if visible else None
        flags = sorted(module_flags)
        if required_lp_cm is None:
            preliminary_result = "pending"
            resolution_reason = "Required lp/cm threshold is not configured."
        elif highest is None:
            preliminary_result = "preliminary_fail"
            resolution_reason = "No target was preliminarily visible at the configured threshold."
        elif highest["expected_lp_cm"] >= required_lp_cm:
            preliminary_result = "preliminary_pass"
            resolution_reason = "Automated Preliminary PASS. Physicist review required."
        else:
            preliminary_result = "preliminary_fail"
            resolution_reason = "Automated Preliminary FAIL. Physicist review required."
        resolution = {
            "status": (
                "needs_review"
                if "non_monotonic_visibility" in module_flags
                else "available"
            ),
            "resolved_lp_cm": highest["expected_lp_cm"] if highest else None,
            "resolved_target": highest["target"] if highest else None,
            "highest_visible_target": highest["target"] if highest else None,
            "continuity_check_status": continuity_status,
            "required_lp_cm": required_lp_cm,
            "preliminary_result": preliminary_result,
            "physicist_review_required": True, "review_flags": flags,
            "reason": resolution_reason,
        }

    primary_ids = {"B8", "B7", "B6", "B5"}
    summary = {
        "targets_total": len(visibility_votes),
        "targets_visible": sum(v["final_preliminary_label"] == "visible" for v in visibility_votes),
        "targets_not_visible": sum(v["final_preliminary_label"] == "not_visible" for v in visibility_votes),
        "targets_needing_review": sum(v["final_preliminary_label"] == "needs_review" for v in visibility_votes),
        "targets_missing": sum(v["final_preliminary_label"] == "missing" for v in visibility_votes),
        "primary_visible": sum(v["target"] in primary_ids and v["final_preliminary_label"] == "visible" for v in visibility_votes),
        "primary_needing_review": sum(v["target"] in primary_ids and v["final_preliminary_label"] == "needs_review" for v in visibility_votes),
        "module_review_flags": sorted(module_flags),
        "physicist_review_required": True,
    }

    def diagnostic_stats(rows: list[dict]) -> dict:
        fields = (
            "peak_to_valley_hu", "fft_snr", "profile_snr",
            "profile_quality", "periodicity", "combined_score",
        )
        stats = {}
        for field in fields:
            values = [finite(row.get(field)) for row in rows]
            values = [value for value in values if value is not None]
            stats[field] = {
                "min": round(min(values), 4) if values else None,
                "max": round(max(values), 4) if values else None,
                "median": round(float(np.median(values)), 4) if values else None,
                "count": len(values),
            }
        return stats

    vote_by_target = {vote["target"]: vote for vote in visibility_votes}
    diagnostic_targets = []
    for target in targets:
        target_id = target.get("id", "unknown")
        analysis = target.get("module4_preliminary_analysis") or {}
        fft_graph = analysis.get("fft_graph") or {}
        fft_vote = analysis.get("fft_vote") or {}
        profile_vote = analysis.get("profile_vote") or {}
        std_vote = analysis.get("std_vote") or {}
        visibility_vote = vote_by_target.get(target_id, {})
        profile = target.get("profile_data") or {}
        roi_data = target.get("roi_data") or {}
        final_roi = target.get("final_roi") or {}
        diagnostic_targets.append({
            "target": target_id,
            "role": "primary" if target_id in primary_ids else "secondary",
            "visibility_label": target.get("preliminary_visibility", "needs_review"),
            "roi_source": final_roi.get("roi_source", target.get("roi_source")),
            "geometry_confidence": target.get(
                "location_confidence", final_roi.get("location_confidence")
            ),
            "expected_lp_cm": MODULE4_EXPECTED_LP_CM.get(target_id),
            "mapping_status": MODULE4_EXPECTED_LP_CM_CONFIG["status"],
            "mean_hu": roi_data.get("mean_hu"),
            "std_hu": roi_data.get("std_hu"),
            "peak_to_valley_hu": roi_data.get("peak_to_valley_hu"),
            "periodicity": profile.get("periodicity_score"),
            "profile_quality": profile.get("profile_quality"),
            "profile_peak_count": profile_vote.get("peak_count"),
            "profile_valley_count": profile_vote.get("valley_count"),
            "profile_snr": profile_vote.get("profile_snr"),
            "profile_spacing_px": profile_vote.get("peak_spacing_px"),
            "profile_spacing_mm": profile_vote.get("peak_spacing_mm"),
            "measured_fft_peak_frequency": fft_graph.get(
                "measured_peak_frequency"
            ),
            "measured_fft_peak_lp_cm": fft_graph.get("measured_peak_lp_cm"),
            "fft_snr": fft_vote.get("fft_snr"),
            "fft_score": fft_vote.get("fft_score"),
            "std_ratio": std_vote.get("std_ratio"),
            "peak_to_valley_noise_ratio": std_vote.get(
                "peak_to_valley_noise_ratio"
            ),
            "contrast_energy_score": std_vote.get("contrast_energy_score"),
            "current_fft_vote_label": fft_vote.get("test_label", "Missing"),
            "current_profile_vote_label": profile_vote.get(
                "test_label", "Missing"
            ),
            "current_std_vote_label": std_vote.get("test_label", "Missing"),
            "combined_score": visibility_vote.get("combined_score"),
            "review_flags": visibility_vote.get("review_flags", []),
            "thresholds_used": analysis.get("thresholds_used", {}),
            "threshold_margins": analysis.get("threshold_margins", {}),
            "margin_basis": "target_specific_adaptive",
            "target_signal_context": analysis.get("target_signal_context", {}),
            "target_adaptive_thresholds": analysis.get(
                "target_adaptive_thresholds", {}
            ),
            "target_threshold_review": analysis.get(
                "target_threshold_review", {}
            ),
        })
    primary_diagnostics = [
        row for row in diagnostic_targets if row["target"] in primary_ids
    ]
    available_target_count = sum(
        row["peak_to_valley_hu"] is not None for row in diagnostic_targets
    )
    sensitivity_limits = {
        "fft_score_margin": 0.10, "profile_score_margin": 0.10,
        "std_score_margin": 0.10, "combined_score_margin": 0.10,
        "fft_snr_margin": 0.25, "profile_snr_margin": 0.25,
        "internal_contrast_support_margin": 0.05,
        "peak_count_margin": 1.0, "valley_count_margin": 1.0,
        "std_ratio_margin": 0.25, "peak_to_valley_noise_margin": 0.25,
        "frequency_error_margin_percent": 10.0,
        "spacing_error_margin_percent": 10.0,
    }
    closest_candidates = []
    likely_threshold_sensitive_targets = []
    for row in diagnostic_targets:
        margins = row["threshold_margins"]
        sensitive_fields = []
        for field, limit in sensitivity_limits.items():
            value = finite(margins.get(field))
            if value is None:
                continue
            closest_candidates.append({
                "target": row["target"], "field": field,
                "margin": round(value, 4), "absolute_margin": abs(value),
            })
            if abs(value) <= limit:
                sensitive_fields.append(field)
        strength_label = (
            (row.get("target_threshold_review") or {})
            .get("diagnostic_strength", {}).get("label", "needs_review")
        )
        decision_fields = {
            "combined_score_margin", "profile_score_margin",
            "profile_snr_margin", "fft_snr_margin",
            "internal_contrast_support_margin",
            "peak_count_margin", "valley_count_margin",
        }
        meaningful_fields = [
            field for field in sensitive_fields if field in decision_fields
        ]
        if meaningful_fields and strength_label in {
            "strong", "moderate", "weak", "needs_review"
        }:
            high_impact_fields = {
                "combined_score_margin", "profile_score_margin",
                "peak_count_margin", "valley_count_margin",
            }
            sensitivity_importance = (
                "high" if any(
                    field in high_impact_fields for field in meaningful_fields
                ) else "medium"
            )
            likely_threshold_sensitive_targets.append({
                "target": row["target"],
                "diagnostic_strength": strength_label,
                "sensitive_margin_fields": meaningful_fields,
                "sensitivity_importance": sensitivity_importance,
                "reason": "A near-zero decision margin could change this target's diagnostic-strength or review state.",
            })
    closest_candidates.sort(key=lambda item: item["absolute_margin"])
    closest_to_threshold = closest_candidates[:8]
    for item in closest_to_threshold:
        item.pop("absolute_margin", None)

    module4_adaptive_thresholds = {
        "status": (
            "missing" if not targets
            else "needs_review" if noise_context["status"] != "available"
            else "available"
        ),
        "config": MODULE4_ADAPTIVE_THRESHOLD_CONFIG,
        "noise_context": noise_context,
        "fft": {
            "fft_snr_threshold": adaptive_fft_thresholds["dynamic_fft_snr_threshold"],
            "fft_score_threshold": adaptive_fft_thresholds["dynamic_fft_score_threshold"],
            "frequency_error_percent_threshold": adaptive_fft_thresholds["max_frequency_error_percent"],
            "method": adaptive_fft_thresholds["method"],
            "guardrails_applied": adaptive_fft_thresholds["guardrails_applied"],
            "reason": adaptive_fft_thresholds["reason"],
        },
        "profile": {
            "profile_snr_threshold": adaptive_profile_thresholds["dynamic_profile_snr_threshold"],
            "profile_score_threshold": adaptive_profile_thresholds["dynamic_profile_score_threshold"],
            "peak_count_threshold": adaptive_profile_thresholds["min_peak_count"],
            "valley_count_threshold": adaptive_profile_thresholds["min_valley_count"],
            "spacing_error_percent_threshold": adaptive_profile_thresholds["max_spacing_error_percent"],
            "method": adaptive_profile_thresholds["method"],
            "guardrails_applied": adaptive_profile_thresholds["guardrails_applied"],
            "reason": adaptive_profile_thresholds["reason"],
        },
        "std_support": {
            "std_ratio_threshold": adaptive_std_thresholds["dynamic_std_ratio_threshold"],
            "peak_to_valley_noise_threshold": adaptive_std_thresholds["dynamic_peak_to_valley_noise_threshold"],
            "std_score_threshold": adaptive_std_thresholds["dynamic_std_score_threshold"],
            "background_noise_status": adaptive_std_thresholds["background_noise_status"],
            "method": adaptive_std_thresholds["method"],
            "guardrails_applied": adaptive_std_thresholds["guardrails_applied"],
            "reason": adaptive_std_thresholds["reason"],
        },
        "combined": {
            "combined_score_threshold": adaptive_combined_thresholds["dynamic_combined_score_threshold"],
            "weights": adaptive_combined_thresholds["weights"],
            "method": adaptive_combined_thresholds["method"],
            "guardrails_applied": adaptive_combined_thresholds["guardrails_applied"],
            "reason": adaptive_combined_thresholds["reason"],
        },
    }
    threshold_tuning_diagnostics = {
        "status": (
            "missing" if not available_target_count
            else "needs_review" if available_target_count < len(targets)
            else "available"
        ),
        "mapping_enabled": MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING,
        "expected_mapping_configured": _MODULE4_MAPPING_COMPLETE,
        "required_threshold_configured": MODULE4_REQUIRED_LP_CM is not None,
        "targets": diagnostic_targets,
        "primary_targets": ["B8", "B7", "B6", "B5"],
        "summary": {
            "primary_targets": diagnostic_stats(primary_diagnostics),
            "all_targets": diagnostic_stats(diagnostic_targets),
            "targets_with_data": available_target_count,
            "targets_total": len(diagnostic_targets),
        },
        "adaptive_threshold_config": MODULE4_ADAPTIVE_THRESHOLD_CONFIG,
        "adaptive_thresholds": module4_adaptive_thresholds,
        "noise_context": noise_context,
        "closest_to_threshold": closest_to_threshold,
        "likely_threshold_sensitive_targets": likely_threshold_sensitive_targets,
        "target_specific_threshold_config": MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG,
        "target_signal_contexts": target_signal_contexts,
        "target_adaptive_thresholds": target_adaptive_thresholds,
        "target_threshold_review": target_threshold_review,
        "target_specific_threshold_warning": target_specific_threshold_warning,
        "target_threshold_validation": module4_target_threshold_validation,
        "threshold_quality_review": module4_threshold_quality_review,
        "preliminary_scoring": module4_auto_preliminary_scoring,
        "auto_preliminary_scoring": module4_auto_preliminary_scoring,
        "auto_method_scoring": module4_auto_method_scoring,
        "fft_threshold_debug": fft_threshold_debug,
        "threshold_source_reasons": threshold_source_reasons,
        "target_specific_margin_basis": "target_specific_adaptive",
        "reason": (
            "Threshold tuning diagnostics are for development review only. "
            "Do not use as official ACR scoring."
        ),
    }
    return {
        "module4_vote_summary": summary,
        "module4_preliminary_resolution": resolution,
        "module4_expected_lp_cm_config": {
            **MODULE4_EXPECTED_LP_CM_CONFIG,
            "targets": dict(MODULE4_EXPECTED_LP_CM_CONFIG["targets"]),
            "development_mapping_enabled": MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING,
        },
        "module4_analysis_config": {
            **MODULE4_ANALYSIS_CONFIG,
            "expected_lp_cm_mapping": dict(
                MODULE4_ANALYSIS_CONFIG["expected_lp_cm_mapping"]
            ),
        },
        "active_expected_lp_cm_mapping": dict(MODULE4_EXPECTED_LP_CM),
        "suggested_mapping": dict(MODULE4_SUGGESTED_MAPPING_REVIEW),
        "mapping_review": dict(MODULE4_SUGGESTED_MAPPING_REVIEW),
        "module4_mapping_review": dict(MODULE4_SUGGESTED_MAPPING_REVIEW),
        "module4_threshold_tuning_diagnostics": threshold_tuning_diagnostics,
        "MODULE4_ADAPTIVE_THRESHOLD_CONFIG": MODULE4_ADAPTIVE_THRESHOLD_CONFIG,
        "module4_adaptive_threshold_config": MODULE4_ADAPTIVE_THRESHOLD_CONFIG,
        "module4_noise_context": noise_context,
        "module4_adaptive_thresholds": module4_adaptive_thresholds,
        "module4_target_specific_threshold_config": MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG,
        "module4_target_signal_contexts": target_signal_contexts,
        "module4_target_adaptive_thresholds": target_adaptive_thresholds,
        "module4_target_threshold_review": target_threshold_review,
        "module4_target_threshold_validation": module4_target_threshold_validation,
        "module4_threshold_quality_review": module4_threshold_quality_review,
        "module4_auto_preliminary_scoring_config": MODULE4_AUTO_PRELIMINARY_SCORING_CONFIG,
        "module4_preliminary_scoring_config": MODULE4_AUTO_PRELIMINARY_SCORING_CONFIG,
        "module4_auto_preliminary_scoring": module4_auto_preliminary_scoring,
        "module4_auto_method_scoring": module4_auto_method_scoring,
        "module4_preliminary_scoring": module4_preliminary_scoring,
        "module4_fft_threshold_debug": fft_threshold_debug,
        "module4_threshold_source_reasons": threshold_source_reasons,
        "target_specific_threshold_warning": target_specific_threshold_warning,
        "adaptive_fft_thresholds": adaptive_fft_thresholds,
        "adaptive_profile_thresholds": adaptive_profile_thresholds,
        "adaptive_std_thresholds": adaptive_std_thresholds,
        "adaptive_combined_thresholds": adaptive_combined_thresholds,
        "visibility_votes": visibility_votes,
    }


def measure_module4_final_roi_standard_deviation(
    raw_pixels: np.ndarray,
    targets: list[dict],
) -> dict:
    """Measure central bar-pattern HU data without altering approved geometry."""
    raw = np.asarray(raw_pixels, dtype=np.float64)
    height, width = raw.shape

    def polygon_mask(corners: list[dict]) -> np.ndarray | None:
        if not isinstance(corners, list) or len(corners) != 4:
            return None
        points = [(float(point["x"]), float(point["y"])) for point in corners]
        image = Image.new("1", (width, height), 0)
        ImageDraw.Draw(image).polygon(points, fill=1)
        return np.asarray(image, dtype=bool)

    target_masks: dict[str, np.ndarray] = {}
    for target in targets:
        final_roi = target.get("final_roi") or {}
        mask = polygon_mask(final_roi.get("inner_roi_corners", []))
        if mask is not None:
            target_masks[target["id"]] = mask
    finite = np.isfinite(raw)
    measurements = []
    ordered = sorted(
        targets,
        key=lambda target: MODULE4_NOMINAL_LP_CM_BY_ID.get(target["id"], 999),
    )
    for target in ordered:
        lp_cm = MODULE4_NOMINAL_LP_CM_BY_ID[target["id"]]
        mask = target_masks.get(target["id"])
        values = raw[mask & finite] if mask is not None else np.asarray([])
        status = "available"
        reason = (
            "Raw HU pixels measured inside the existing central "
            "final_roi.inner_roi_corners polygon."
        )
        if values.size < 4:
            status = "unavailable"
            reason = "The final ROI did not contain enough finite raw HU samples."

        if values.size:
            p05, p95 = np.percentile(values, [5.0, 95.0])
            std_hu = float(np.std(values, ddof=0))
            percentile_range = float(p95 - p05)
        else:
            p05 = p95 = std_hu = percentile_range = None
        measurements.append({
            "lp_cm": lp_cm,
            "frequency_label": f"{lp_cm} lp/cm",
            "final_roi": target.get("final_roi", {}),
            "std_measurement": {
                "status": status,
                "sample_count": int(values.size),
                "mean_hu": float(np.mean(values)) if values.size else None,
                "median_hu": float(np.median(values)) if values.size else None,
                "std_hu": std_hu,
                "std_ddof": 0,
                "min_hu": float(np.min(values)) if values.size else None,
                "max_hu": float(np.max(values)) if values.size else None,
                "p05_hu": float(p05) if values.size else None,
                "p95_hu": float(p95) if values.size else None,
                "percentile_range_hu": percentile_range,
                "measurement_mask_source": "final_roi.inner_roi_corners",
                "measurement_polygon": target.get("final_roi", {}).get(
                    "inner_roi_corners", []
                ),
                "reason": reason,
            },
        })
    return {
        "std_measurements": measurements,
    }


def generate_module4_std_measurement_debug_overlay(
    raw_pixels: np.ndarray,
    targets: list[dict],
    measurements: list[dict],
    window_width: float,
    window_level: float,
    photometric: str,
) -> str:
    """Show unchanged outer ROIs and the exact central SD polygons."""
    image = window_pixels_to_image(
        raw_pixels, window_width, window_level, photometric
    ).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    samples_by_frequency = {
        item["lp_cm"]: item["std_measurement"]["sample_count"]
        for item in measurements
    }
    for target in targets:
        final_roi = target.get("final_roi") or {}
        outer = final_roi.get("final_corners", [])
        inner = final_roi.get("inner_roi_corners", [])
        if len(outer) == 4:
            outer_points = [
                (float(point["x"]), float(point["y"])) for point in outer
            ]
            draw.line(
                outer_points + [outer_points[0]],
                fill=(34, 211, 238, 255), width=2,
            )
        if len(inner) == 4:
            inner_points = [
                (float(point["x"]), float(point["y"])) for point in inner
            ]
            draw.polygon(inner_points, fill=(236, 72, 153, 38))
            draw.line(
                inner_points + [inner_points[0]],
                fill=(244, 114, 182, 255), width=2,
            )
            lp_cm = MODULE4_NOMINAL_LP_CM_BY_ID[target["id"]]
            sample_count = samples_by_frequency.get(lp_cm, 0)
            label_x = min(point[0] for point in inner_points)
            label_y = min(point[1] for point in inner_points) - 13
            draw.text(
                (label_x, max(0, label_y)),
                f"{lp_cm} lp/cm: n={sample_count}",
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(15, 23, 42, 255),
            )
    return image_to_base64(image.convert("RGB"))


def estimate_module4_automatic_noise(
    raw_pixels: np.ndarray,
    targets: list[dict],
    geometry: dict,
    std_measurements: list[dict],
    patch_size_px: int = 7,
) -> dict:
    """Estimate scan-specific local/global noise from robust raw-HU patches."""
    raw = np.asarray(raw_pixels, dtype=np.float64)
    height, width = raw.shape
    half = patch_size_px // 2
    finite = np.isfinite(raw)
    center = geometry.get("phantom_center") or {}
    radius = float(geometry.get("phantom_radius") or 0.0)
    if radius <= 0 or "x" not in center or "y" not in center:
        return {
            "global_noise_reference": {
                "status": "unavailable", "source": "automatic_uniform_patch_median",
                "patch_size_px": patch_size_px, "patch_count_total": 0,
                "patch_count_accepted": 0, "patch_count_rejected": 0,
                "std_hu": None, "patch_std_min": None, "patch_std_max": None,
                "patch_std_mean": None, "patch_std_median": None,
                "patch_std_mad": None, "patch_std_coefficient_of_variation": None,
                "quality_flags": ["phantom_geometry_missing"],
                "reason": "Phantom geometry is unavailable.",
            },
            "target_noise": [], "_debug_patches": {},
        }

    def polygon_mask(corners: list[dict]) -> np.ndarray:
        image = Image.new("1", (width, height), 0)
        if isinstance(corners, list) and len(corners) == 4:
            points = [(float(p["x"]), float(p["y"])) for p in corners]
            ImageDraw.Draw(image).polygon(points, fill=1)
        return np.asarray(image, dtype=bool)

    target_exclusion = np.zeros(raw.shape, dtype=bool)
    for target in targets:
        final_roi = target.get("final_roi") or {}
        target_exclusion |= polygon_mask(final_roi.get("final_corners", []))
    target_exclusion = ndimage.binary_dilation(
        target_exclusion, iterations=max(3, half)
    )

    yy, xx = np.indices(raw.shape)
    radial = np.hypot(xx - float(center["x"]), yy - float(center["y"]))
    phantom_safe = radial <= radius * 0.82
    pin_exclusion = np.zeros(raw.shape, dtype=bool)
    for pin in geometry.get("selected_bottom_pins", []):
        pin_center = pin.get("center") or pin
        if "x" in pin_center and "y" in pin_center:
            pin_exclusion |= (
                np.hypot(xx - float(pin_center["x"]), yy - float(pin_center["y"]))
                <= max(patch_size_px * 2.0, radius * 0.055)
            )
    valid_base = finite & phantom_safe & ~target_exclusion & ~pin_exclusion
    gx = ndimage.sobel(raw, axis=1, mode="nearest")
    gy = ndimage.sobel(raw, axis=0, mode="nearest")
    gradient = np.hypot(gx, gy)

    def robust_limit(values: list[float], multiplier: float) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        return median, max(mad, abs(median) * 0.02, 1e-6) * multiplier

    def evaluate_patches(centers: list[tuple[int, int]]) -> dict:
        candidates = []
        rejected = []
        for patch_x, patch_y in centers:
            x0, x1 = patch_x - half, patch_x + half + 1
            y0, y1 = patch_y - half, patch_y + half + 1
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                rejected.append((patch_x, patch_y, "image_boundary")); continue
            patch_valid = valid_base[y0:y1, x0:x1]
            if np.count_nonzero(patch_valid) < int(patch_size_px ** 2 * 0.80):
                rejected.append((patch_x, patch_y, "geometric_exclusion")); continue
            values = raw[y0:y1, x0:x1][patch_valid]
            candidates.append({
                "x": patch_x, "y": patch_y,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "gradient": float(np.median(gradient[y0:y1, x0:x1][patch_valid])),
            })
        if not candidates:
            return {"total": len(centers), "accepted": [], "rejected": rejected}
        mean_median, mean_span = robust_limit([p["mean"] for p in candidates], 4.5)
        std_median, std_span = robust_limit([p["std"] for p in candidates], 4.5)
        grad_median, grad_span = robust_limit([p["gradient"] for p in candidates], 3.5)
        accepted = []
        for patch in candidates:
            reason = None
            if abs(patch["mean"] - mean_median) > mean_span:
                reason = "abnormal_mean_hu"
            elif patch["gradient"] > grad_median + grad_span:
                reason = "strong_gradient"
            elif patch["std"] > std_median + std_span:
                reason = "extreme_patch_sd"
            if reason:
                rejected.append((patch["x"], patch["y"], reason))
            else:
                accepted.append(patch)
        return {"total": len(centers), "accepted": accepted, "rejected": rejected}

    def summarize(evaluation: dict, minimum_patches: int, source: str) -> dict:
        accepted = evaluation["accepted"]
        values = np.asarray([patch["std"] for patch in accepted], dtype=np.float64)
        count = int(values.size)
        median = float(np.median(values)) if count else None
        mad = float(np.median(np.abs(values - median))) if count else None
        mean = float(np.mean(values)) if count else None
        cv = float(np.std(values, ddof=0) / mean) if count and mean and mean > 0 else None
        flags = []
        status = "stable"
        if count == 0:
            status = "unavailable"; flags.append("insufficient_local_patches")
        elif count < minimum_patches:
            status = "needs_review"; flags.append("insufficient_local_patches")
        if cv is not None and cv > 0.65:
            status = "needs_review"; flags.append("high_patch_noise_variation")
        if median is not None and median <= 1e-6:
            status = "needs_review"; flags.append("local_noise_near_zero")
        return {
            "status": status, "source": source, "patch_size_px": patch_size_px,
            "patch_count_total": evaluation["total"],
            "patch_count_accepted": count,
            "patch_count_rejected": len(evaluation["rejected"]),
            "std_hu": median,
            "patch_std_min": float(np.min(values)) if count else None,
            "patch_std_max": float(np.max(values)) if count else None,
            "patch_std_mean": mean, "patch_std_median": median,
            "patch_std_mad": mad,
            "patch_std_coefficient_of_variation": cv,
            "quality_flags": flags,
            "reason": (
                "Median population SD from robustly accepted uniform patches."
                if count else "No valid uniform patches were accepted."
            ),
        }

    step = patch_size_px
    global_centers = [
        (x, y)
        for y in range(half, height - half, step)
        for x in range(half, width - half, step)
        if phantom_safe[y, x]
    ]
    global_eval = evaluate_patches(global_centers)
    global_summary = summarize(
        global_eval, 12, "automatic_uniform_patch_median"
    )

    measurement_by_frequency = {item["lp_cm"]: item for item in std_measurements}
    target_noise = []
    debug_local = {}
    for target in targets:
        final_roi = target.get("final_roi") or {}
        target_center = final_roi.get("final_center") or {}
        side = float(final_roi.get("final_side_px") or 0.0)
        angle = math.radians(float(final_roi.get("final_angle_degrees") or 0.0))
        lp_cm = MODULE4_NOMINAL_LP_CM_BY_ID[target["id"]]
        local_centers = []
        if side > 0 and "x" in target_center and "y" in target_center:
            outer_half = side * 2.0 / 2.0
            exclusion_half = side * 1.15 / 2.0
            bound = int(math.ceil(outer_half))
            cx, cy = float(target_center["x"]), float(target_center["y"])
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for y in range(max(half, int(cy - bound)), min(height - half, int(cy + bound) + 1), step):
                for x in range(max(half, int(cx - bound)), min(width - half, int(cx + bound) + 1), step):
                    dx, dy = x - cx, y - cy
                    local_x = cos_a * dx + sin_a * dy
                    local_y = -sin_a * dx + cos_a * dy
                    if max(abs(local_x), abs(local_y)) <= outer_half and max(abs(local_x), abs(local_y)) >= exclusion_half:
                        local_centers.append((x, y))
        local_eval = evaluate_patches(local_centers)
        local_summary = summarize(local_eval, 4, "local_patch_median")
        local_stable = local_summary["status"] == "stable"
        global_available = global_summary["std_hu"] is not None
        selected_noise = local_summary["std_hu"] if local_stable else global_summary["std_hu"]
        noise_source = "local" if local_stable else "global_fallback" if global_available else "unavailable"
        status = "stable" if local_stable else "needs_review" if global_available else "unavailable"
        flags = list(local_summary["quality_flags"])
        if noise_source == "global_fallback": flags.append("global_fallback_used")
        target_measurement = measurement_by_frequency.get(lp_cm, {}).get("std_measurement", {})
        target_std = target_measurement.get("std_hu")
        target_variance = target_std ** 2 if target_std is not None else None
        noise_variance = selected_noise ** 2 if selected_noise is not None else None
        excess = max(target_variance - noise_variance, 0.0) if target_variance is not None and noise_variance is not None else None
        target_noise.append({
            "lp_cm": lp_cm, "frequency_label": f"{lp_cm} lp/cm",
            "local_noise_measurement": {
                "status": status, "noise_source": noise_source,
                "selected_noise_std_hu": selected_noise,
                "local": local_summary,
                "global_fallback": {
                    "status": "available" if global_available else "unavailable",
                    "std_hu": global_summary["std_hu"],
                },
                "normalized": {
                    "target_bar_pattern_std_hu": target_std,
                    "selected_noise_std_hu": selected_noise,
                    "normalized_std_ratio": target_std / selected_noise if target_std is not None and selected_noise and selected_noise > 0 else None,
                    "target_variance": target_variance,
                    "noise_variance": noise_variance,
                    "excess_variance": excess,
                    "normalized_contrast_energy": excess / max(noise_variance, 0.000001) if excess is not None and noise_variance is not None else None,
                },
                "quality_flags": flags,
                "reason": "Local median patch SD used." if local_stable else "Local patches were insufficient or unstable; automatic global median patch SD used." if global_available else "No valid local or global noise baseline is available.",
            },
        })
        debug_local[lp_cm] = local_eval
    return {
        "global_noise_reference": global_summary,
        "target_noise": target_noise,
        "_debug_patches": {"global": global_eval, "local": debug_local},
    }


def generate_module4_noise_debug_overlay(
    raw_pixels: np.ndarray,
    targets: list[dict],
    noise_debug: dict,
    window_width: float,
    window_level: float,
    photometric: str,
    patch_size_px: int = 7,
) -> str:
    """Render accepted/rejected local patches and accepted global patches."""
    image = window_pixels_to_image(
        raw_pixels, window_width, window_level, photometric
    ).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    half = patch_size_px // 2
    for target in targets:
        outer = (target.get("final_roi") or {}).get("final_corners", [])
        if len(outer) == 4:
            points = [(float(p["x"]), float(p["y"])) for p in outer]
            draw.line(points + [points[0]], fill=(34, 211, 238, 255), width=2)
    global_eval = noise_debug.get("global", {})
    for patch in global_eval.get("accepted", []):
        x, y = patch["x"], patch["y"]
        draw.rectangle((x-half, y-half, x+half, y+half), outline=(59, 130, 246, 150), width=1)
    for evaluation in noise_debug.get("local", {}).values():
        for patch in evaluation.get("accepted", []):
            x, y = patch["x"], patch["y"]
            draw.rectangle((x-half, y-half, x+half, y+half), outline=(34, 197, 94, 255), width=2)
        for rejected in evaluation.get("rejected", []):
            x, y = rejected[0], rejected[1]
            draw.line((x-2, y-2, x+2, y+2), fill=(239, 68, 68, 220), width=1)
            draw.line((x-2, y+2, x+2, y-2), fill=(239, 68, 68, 220), width=1)
    return image_to_base64(image.convert("RGB"))


def apply_module4_std_visibility_decisions(
    std_measurements: list[dict],
) -> dict:
    """Apply one fixed provisional SD/noise-ratio evidence rule."""
    def finite_number(value) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    ordered = sorted(std_measurements, key=lambda item: int(item["lp_cm"]))
    visible_frequencies = []
    not_visible_frequencies = []
    review_frequencies = []
    unavailable_frequencies = []

    for item in ordered:
        measurement = item.get("std_measurement") or {}
        noise = item.get("local_noise_measurement") or {}
        normalized = noise.get("normalized") or {}
        target_std = finite_number(measurement.get("std_hu"))
        selected_noise = finite_number(noise.get("selected_noise_std_hu"))
        noise_source = noise.get("noise_source")
        local = noise.get("local") or {}
        flags = list(noise.get("quality_flags") or [])
        if measurement.get("status") == "needs_review":
            flags.append("target_measurement_needs_review")
        decision = "unavailable"
        confidence = "unavailable"
        reason = "The scan-specific normalized Standard Deviation ratio is unavailable."
        ratio = None
        valid_target = (
            measurement.get("status") != "unavailable"
            and target_std is not None
        )
        valid_noise = (
            selected_noise is not None and selected_noise > 0
        )
        if valid_target and valid_noise:
            ratio = target_std / selected_noise
            local_stable = local.get("status") == "stable"
            local_source = noise_source == "local"
            marginal_quality = bool(flags) or measurement.get("status") == "needs_review"
            if (
                ratio >= MODULE4_STD_VISIBLE_RATIO_THRESHOLD
                and local_stable and local_source and not marginal_quality
            ):
                decision = "visible"
                reason = (
                    "The scan-specific normalized SD ratio is above the "
                    "provisional 3.0 visibility threshold."
                )
            elif (
                ratio < MODULE4_STD_REVIEW_RATIO_THRESHOLD
                and local_stable and local_source and not marginal_quality
            ):
                decision = "not_visible"
                reason = (
                    "The scan-specific normalized SD ratio is below the "
                    "provisional 2.5 review boundary."
                )
            else:
                decision = "needs_review"
                if noise_source == "global_fallback":
                    reason = "Automatic global noise fallback was used, so this evidence needs review."
                elif ratio < MODULE4_STD_VISIBLE_RATIO_THRESHOLD and ratio >= MODULE4_STD_REVIEW_RATIO_THRESHOLD:
                    reason = "The normalized SD ratio lies within the provisional review interval."
                else:
                    reason = "The normalized SD ratio is calculable, but measurement quality needs review."

            distance = min(
                abs(ratio - MODULE4_STD_VISIBLE_RATIO_THRESHOLD),
                abs(ratio - MODULE4_STD_REVIEW_RATIO_THRESHOLD),
            )
            patch_cv = local.get("patch_std_coefficient_of_variation")
            if decision in {"visible", "not_visible"}:
                confidence = (
                    "high"
                    if distance >= 0.5
                    and not flags
                    and (patch_cv is None or float(patch_cv) <= 0.45)
                    else "medium"
                )
            elif decision == "needs_review":
                confidence = "low" if noise_source == "global_fallback" or flags else "medium"

        item["std_visibility"] = {
            "method": "standard_deviation_to_scan_specific_noise_ratio",
            "status": "available" if decision != "unavailable" else "unavailable",
            "target_std_hu": target_std if valid_target else None,
            "selected_noise_std_hu": selected_noise if valid_noise else None,
            "noise_source": noise_source or "unavailable",
            "normalized_std_ratio": ratio,
            "visible_ratio_threshold": MODULE4_STD_VISIBLE_RATIO_THRESHOLD,
            "review_ratio_threshold": MODULE4_STD_REVIEW_RATIO_THRESHOLD,
            "threshold_status": MODULE4_STD_THRESHOLD_STATUS,
            "threshold_source": MODULE4_STD_THRESHOLD_SOURCE,
            "decision": decision,
            "confidence": confidence,
            "quality_flags": flags,
            "reason": reason,
        }
        frequency = int(item["lp_cm"])
        if decision == "visible": visible_frequencies.append(frequency)
        elif decision == "not_visible": not_visible_frequencies.append(frequency)
        elif decision == "needs_review": review_frequencies.append(frequency)
        else: unavailable_frequencies.append(frequency)

    highest_continuous = None
    first_nonvisible = None
    break_index = None
    for index, item in enumerate(ordered):
        decision = item["std_visibility"]["decision"]
        if decision == "visible" and break_index is None:
            highest_continuous = int(item["lp_cm"])
        elif break_index is None:
            first_nonvisible = int(item["lp_cm"])
            break_index = index
    nonmonotonic = bool(
        break_index is not None
        and any(
            item["std_visibility"]["decision"] == "visible"
            for item in ordered[break_index + 1:]
        )
    )
    summary_flags = ["nonmonotonic_visibility_pattern"] if nonmonotonic else []
    continuity_status = "needs_review" if nonmonotonic else "valid"
    status = "available" if len(unavailable_frequencies) < len(ordered) else "unavailable"
    return {
        "method": "standard_deviation_to_scan_specific_noise_ratio",
        "status": status,
        "visible_ratio_threshold": MODULE4_STD_VISIBLE_RATIO_THRESHOLD,
        "review_ratio_threshold": MODULE4_STD_REVIEW_RATIO_THRESHOLD,
        "threshold_status": MODULE4_STD_THRESHOLD_STATUS,
        "threshold_source": MODULE4_STD_THRESHOLD_SOURCE,
        "highest_continuously_visible_lp_cm": highest_continuous,
        "first_nonvisible_lp_cm": first_nonvisible,
        "continuity_status": continuity_status,
        "visible_frequencies": visible_frequencies,
        "not_visible_frequencies": not_visible_frequencies,
        "review_frequencies": review_frequencies,
        "unavailable_frequencies": unavailable_frequencies,
        "quality_flags": summary_flags,
        "reason": (
            f"The Standard Deviation evidence method continuously resolves through {highest_continuous} lp/cm using scan-specific noise normalization."
            if highest_continuous is not None
            else "No continuously visible frequency is available from this evidence method."
        ),
    }


def analyze_module4_high_contrast_slice(
    slice_pixels: np.ndarray,
    window_width: float = 400.0,
    window_level: float = 40.0,
    photometric: str = "",
    pixel_spacing: tuple[float | None, float | None] | None = None,
    debug_show_reference_targets: bool = False,
    performance_mode: str = "fast",
    debug_overlay_enabled: bool = False,
) -> dict:
    """Run geometry-guided local square location without measurement."""
    del window_width, window_level, debug_show_reference_targets
    analysis_started = time.perf_counter()
    raw = np.asarray(slice_pixels, dtype=np.float32)
    geometry = phantom_geometry_pin_anchored_roi(
        raw,
        pixel_spacing=pixel_spacing,
        performance_mode=performance_mode,
    )
    # Deprecated scoring experiment removed from the active Module 4 path.
    # Geometry placement and overlay rendering are intentionally the only
    # analysis performed here.
    center = geometry["phantom_center"]
    display_window = _select_module4_display_window(
        raw,
        float(center["x"]),
        float(center["y"]),
        float(geometry["phantom_radius"]),
    )
    auto_width = float(display_window["window_width"])
    auto_level = float(display_window["window_level"])
    diagnostic_export = _export_module4_fit_diagnostics(
        raw,
        geometry,
        auto_width,
        auto_level,
        photometric,
    )
    geometry["geometry_local_square_fit"].pop(
        "_diagnostic_intermediates", None
    )
    selected_slice_image = image_to_base64(
        window_pixels_to_image(raw, auto_width, auto_level, photometric)
    )
    geometry_rois = geometry["target_rois"]
    standard_deviation = measure_module4_final_roi_standard_deviation(
        raw,
        geometry_rois,
    )
    automatic_noise = estimate_module4_automatic_noise(
        raw,
        geometry_rois,
        geometry,
        standard_deviation["std_measurements"],
    )
    noise_by_frequency = {
        item["lp_cm"]: item["local_noise_measurement"]
        for item in automatic_noise["target_noise"]
    }
    for measurement in standard_deviation["std_measurements"]:
        measurement["local_noise_measurement"] = noise_by_frequency.get(
            measurement["lp_cm"]
        )
    std_method_summary = apply_module4_std_visibility_decisions(
        standard_deviation["std_measurements"]
    )
    std_measurement_debug_overlay = (
        generate_module4_std_measurement_debug_overlay(
            raw,
            geometry_rois,
            standard_deviation["std_measurements"],
            auto_width,
            auto_level,
            photometric,
        )
    )
    noise_debug_overlay = (
        generate_module4_noise_debug_overlay(
            raw,
            geometry_rois,
            automatic_noise["_debug_patches"],
            auto_width,
            auto_level,
            photometric,
        )
        if debug_overlay_enabled else None
    )
    overlay_started = time.perf_counter()
    overlay_image = generate_module4_block_overlay(
        raw,
        geometry_rois,
        window_width=auto_width,
        window_level=auto_level,
        photometric=photometric,
        title="Module 4 ROI Location Review",
    )
    overlay_generation_ms = (time.perf_counter() - overlay_started) * 1000.0
    debug_overlay_started = time.perf_counter()
    location_debug_overlay = (
        generate_module4_location_debug_overlay(
            raw,
            geometry,
            window_width=auto_width,
            window_level=auto_level,
            photometric=photometric,
        )
        if debug_overlay_enabled else None
    )
    debug_overlay_generation_ms = (
        time.perf_counter() - debug_overlay_started
    ) * 1000.0
    orientation_needs_review = geometry["orientation_status"] != "detected"
    local_square_fit = geometry["geometry_local_square_fit"]
    local_review_needed = local_square_fit["needs_review"] > 0

    stage_timings = {
        **geometry["performance"],
        "overlay_generation_ms": round(overlay_generation_ms, 2),
        "debug_overlay_generation_ms": round(
            debug_overlay_generation_ms, 2
        ),
    }
    total_module4_analysis_ms = (
        time.perf_counter() - analysis_started
    ) * 1000.0
    slowest_stage = max(stage_timings, key=stage_timings.get)
    performance = {
        "total_ms": round(total_module4_analysis_ms, 2),
        "stage_timings_ms": stage_timings,
        "slowest_stage": slowest_stage,
        "performance_mode": performance_mode,
        "debug_overlay_enabled": debug_overlay_enabled,
        "debug_overlay_generation_status": (
            "generated"
            if debug_overlay_enabled else "skipped_fast_mode"
        ),
        "hypothesis_counts": local_square_fit["hypothesis_counts"],
        "performance_caps": local_square_fit["performance_caps"],
    }
    print(
        "Module4 timing: "
        f"total={performance['total_ms']:.2f}ms, "
        f"slowest={slowest_stage}, "
        f"local={stage_timings['local_square_detection_ms']:.2f}ms, "
        f"micro={stage_timings['micro_refinement_ms']:.2f}ms, "
        f"targeted={stage_timings['targeted_refinement_ms']:.2f}ms, "
        f"overlay={stage_timings['overlay_generation_ms']:.2f}ms"
    )

    return {
        "analysis_path": "geometry_guided_local_square_location",
        "geometry_source": geometry["geometry_source"],
        "geometry": geometry,
        "analysis_review_status": (
            "needs_review"
            if orientation_needs_review or local_review_needed
            else "local_squares_detected"
        ),
        "performance": performance,
        "std_measurements": standard_deviation["std_measurements"],
        "global_noise_reference": automatic_noise[
            "global_noise_reference"
        ],
        "std_method_summary": std_method_summary,
        "std_measurement_debug_overlay": std_measurement_debug_overlay,
        "noise_debug_overlay": noise_debug_overlay,
        "total_module4_analysis_ms": performance["total_ms"],
        "debug_overlay_generation_status": performance[
            "debug_overlay_generation_status"
        ],
        "candidates": geometry_rois,
        "missing_targets": [],
        "weak_targets": (
            [
                target["id"] for target in geometry_rois
                if target["roi_source"] == "geometry_fallback_needs_review"
            ]
        ),
        "detected_targets": [
            target["id"] for target in geometry_rois
            if target["roi_source"] == "local_square_detected"
        ],
        "target_slots": geometry_rois,
        "slot_lifecycle": [],
        "target_debug": [],
        "stage_summary": {
            "active_stage": "geometry_guided_local_square_detection",
            "geometry_source": geometry["geometry_source"],
            "phantom_envelope_prepared": True,
            "bottom_pin_detected": (
                geometry["pin_detection_status"] in {"detected", "needs_review"}
                and len(geometry["selected_bottom_pins"]) == 2
            ),
            "bottom_pin_anchoring_status": "implemented",
            "target_rois_placed": len(geometry_rois),
            "primary_rois_drawn": sum(
                target["priority"] == "primary"
                and target["draw_on_overlay"]
                for target in geometry_rois
            ),
            "target_roi_placement_status": "implemented",
            "geometry_calibration_status": geometry[
                "geometry_calibration"
            ]["calibration_status"],
            "location_confidence": geometry["location_confidence"],
            "image_locator_active": True,
        },
        "phantom_center": center,
        "phantom_radius": geometry["phantom_radius"],
        "phantom_detection_status": geometry["phantom_detection_status"],
        "pixel_spacing": geometry["pixel_spacing"],
        "thresholds": {},
        "components_considered": 0,
        "components_rejected": 0,
        "merge_groups": [],
        "merge_kernel_size": None,
        "median_block_side_px": None,
        "overlay_image": overlay_image,
        "location_debug_overlay": location_debug_overlay,
        "selected_slice_image": selected_slice_image,
        "window_width": round(auto_width, 2),
        "window_level": round(auto_level, 2),
        "display_window_method": display_window["window_method"],
        "display_window_quality_score": display_window["window_quality_score"],
        "display_window_candidates": display_window["candidate_windows"],
        "fit_diagnostics_export": diagnostic_export,
        "display_window_note": (
            f'{display_window["window_note"]} Geometry bounds each local raw-HU '
            "square search; display WW/WL does not affect location evidence."
        ),
        "legacy_path": {
            "active": False,
            "component_blob_localization": False,
            "single_block_b1_debug": False,
            "perfect_square_template_optimization": False,
            "all_8_duplicate_assignment": False,
            "circular_hole_target_rejection": False,
            "b6_special_case": False,
            "slot_centered_fallback": False,
            "jump_line_cluster_fitting": False,
            "draft_roi_scoring": False,
        },
    }
