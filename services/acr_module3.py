import math

import numpy as np
from PIL import ImageDraw
from skimage import filters, measure

from services.dicom_config import ACR_UNIFORMITY_NOT_FOUND, ROBUST_VERSION
from services.dicom_display import window_pixels_to_image, _get_slices_from_stack_or_upload
from services.image_helpers import image_to_base64, normalize_for_display


def _ensure_acr_usable_slice(slice_data):
    info = slice_data.get("info", {})

    if info.get("isColorDicom"):
        raise ValueError(
            "ACR HU uniformity analysis requires original grayscale CT DICOM pixel data. "
            "This file is color/RGB DICOM converted for display, so the app will not generate a fake pass/fail result."
        )


def _detect_phantom_circle(raw_pixels):
    """
    Detect the main circular phantom body.

    The result is used for both the uniformity-module classifier and BB search.
    """
    norm = normalize_for_display(raw_pixels)

    try:
        threshold = filters.threshold_otsu(norm)
        mask = norm > threshold
    except Exception:
        finite_pixels = raw_pixels[np.isfinite(raw_pixels)]

        if finite_pixels.size < 100:
            raise ValueError("Could not detect phantom boundary.")

        threshold = np.percentile(finite_pixels, 25)
        mask = raw_pixels > threshold

    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    if not regions:
        raise ValueError("Could not detect phantom boundary.")

    height, width = raw_pixels.shape
    image_area = height * width

    usable_regions = [
        region
        for region in regions
        if region.area >= image_area * 0.05
    ]

    if not usable_regions:
        usable_regions = regions

    main = max(usable_regions, key=lambda region: region.area)

    cy, cx = main.centroid
    radius = math.sqrt(float(main.area) / math.pi)

    if radius < min(height, width) * 0.12:
        raise ValueError("Detected phantom boundary is too small.")

    return float(cx), float(cy), float(radius)


def _safe_region_centroid(region):
    try:
        return region.centroid_weighted
    except Exception:
        try:
            return region.weighted_centroid
        except Exception:
            return region.centroid


def _safe_region_max(region):
    try:
        return float(region.intensity_max)
    except Exception:
        try:
            return float(region.max_intensity)
        except Exception:
            return 0.0


def _safe_region_mean(region):
    try:
        return float(region.intensity_mean)
    except Exception:
        try:
            return float(region.mean_intensity)
        except Exception:
            return 0.0


def _window_to_unit(raw, window_width, window_level):
    width = float(window_width)

    if width <= 0:
        width = 1.0

    level = float(window_level)
    lower = level - width / 2.0
    upper = level + width / 2.0

    return np.clip(
        (raw - lower) / max(upper - lower, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)


def _circular_mask(shape, cx, cy, radius):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]

    return (
        (xx - float(cx)) ** 2 +
        (yy - float(cy)) ** 2
    ) <= float(radius) ** 2


def _uniformity_signature(raw_pixels, phantom_cx, phantom_cy, phantom_radius):
    """
    Decide whether a slice actually looks like the ACR uniformity module.

    This is the structural fix missing from earlier versions.

    A BB pair is no longer enough to select a slice. Before BB detection, the
    slice must look like a mostly uniform water-equivalent module. Slices with
    large material rods, contrast inserts, line-pair bars, or other structured
    objects are strongly penalized or rejected.
    """
    raw = np.asarray(raw_pixels, dtype=np.float32)
    finite = np.isfinite(raw)

    if np.sum(finite) < 100:
        raise ValueError("Not enough finite pixels for uniformity classification.")

    radius = float(max(phantom_radius, 1.0))
    height, width = raw.shape

    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt(
        (xx - float(phantom_cx)) ** 2 +
        (yy - float(phantom_cy)) ** 2
    )

    # Ignore the shell and external reference holes. The module decision is
    # based on the inner phantom contents.
    interior_mask = finite & (distance <= radius * 0.76)
    analysis_mask = finite & (distance <= radius * 0.68)

    values = raw[analysis_mask]

    if values.size < 200:
        raise ValueError("Not enough phantom interior pixels for uniformity classification.")

    smooth_sigma = max(1.0, min(4.0, radius * 0.010))
    smooth = filters.gaussian(raw, sigma=smooth_sigma, preserve_range=True)

    smooth_values = smooth[analysis_mask]

    median_hu = float(np.median(smooth_values))
    p01 = float(np.percentile(smooth_values, 1))
    p05 = float(np.percentile(smooth_values, 5))
    p95 = float(np.percentile(smooth_values, 95))
    p99 = float(np.percentile(smooth_values, 99))

    robust_range_hu = p95 - p05
    wide_range_hu = p99 - p01

    absolute_deviation = np.abs(smooth_values - median_hu)
    mad = float(np.median(absolute_deviation))
    robust_sigma = max(1.4826 * mad, 0.5)

    # Large rods and inserts differ strongly from the background. Tiny BB dots
    # are too small to survive the component-area requirement below.
    structure_threshold_hu = max(
        28.0,
        robust_sigma * 7.0,
        robust_range_hu * 0.75,
    )

    residual = np.abs(smooth - median_hu)
    structure_mask = (
        interior_mask &
        (residual >= structure_threshold_hu)
    )

    structure_labels = measure.label(structure_mask)
    structure_regions = measure.regionprops(structure_labels)

    interior_area = max(float(np.sum(interior_mask)), 1.0)
    analysis_area = max(float(np.sum(analysis_mask)), 1.0)

    minimum_large_area = max(
        24.0,
        interior_area * 0.00075,
    )
    minimum_medium_area = max(
        12.0,
        interior_area * 0.00030,
    )

    large_components = []
    medium_components = []

    for region in structure_regions:
        area = float(region.area)

        if area >= minimum_medium_area:
            medium_components.append(area)

        if area >= minimum_large_area:
            large_components.append(area)

    large_component_count = len(large_components)
    medium_component_count = len(medium_components)
    largest_component_fraction = (
        max(large_components) / interior_area
        if large_components
        else 0.0
    )

    extreme_fraction = float(np.sum(structure_mask)) / interior_area

    # Sector means detect large structured inserts even when positive and
    # negative materials partly cancel in the overall histogram.
    sector_means = []
    angle = np.arctan2(
        yy - float(phantom_cy),
        xx - float(phantom_cx),
    )

    sector_ring = (
        finite &
        (distance >= radius * 0.18) &
        (distance <= radius * 0.66)
    )

    sector_count = 12

    for sector_index in range(sector_count):
        lower_angle = -math.pi + (2.0 * math.pi * sector_index / sector_count)
        upper_angle = -math.pi + (2.0 * math.pi * (sector_index + 1) / sector_count)

        sector_mask = (
            sector_ring &
            (angle >= lower_angle) &
            (angle < upper_angle)
        )

        sector_values = smooth[sector_mask]

        if sector_values.size >= 20:
            sector_means.append(float(np.mean(sector_values)))

    sector_spread_hu = (
        max(sector_means) - min(sector_means)
        if len(sector_means) >= 4
        else 999.0
    )

    # Edge density catches line-pair bars and material boundaries.
    normalized_smooth = np.clip(
        (smooth - p05) / max(robust_range_hu, 1.0),
        0.0,
        1.0,
    )
    edge_image = filters.sobel(normalized_smooth)
    edge_values = edge_image[analysis_mask]
    edge_density = float(np.mean(edge_values > 0.11)) if edge_values.size else 1.0
    mean_edge_strength = float(np.mean(edge_values)) if edge_values.size else 1.0

    # Build a continuous score for ranking plus hard rejection reasons.
    score = 1.20

    score -= min(0.55, robust_range_hu / 90.0 * 0.55)
    score -= min(0.45, wide_range_hu / 180.0 * 0.45)
    score -= min(0.50, sector_spread_hu / 55.0 * 0.50)
    score -= min(0.65, extreme_fraction / 0.055 * 0.65)
    score -= min(0.80, large_component_count * 0.23)
    score -= min(0.30, max(0, medium_component_count - large_component_count) * 0.04)
    score -= min(0.28, edge_density / 0.075 * 0.28)
    score -= min(0.15, mean_edge_strength / 0.035 * 0.15)

    if (
        large_component_count == 0
        and robust_range_hu <= 45.0
        and sector_spread_hu <= 20.0
        and extreme_fraction <= 0.012
    ):
        score += 0.22

    hard_reject_reasons = []

    if large_component_count >= 2:
        hard_reject_reasons.append(
            f"{large_component_count} large internal insert-like structures detected"
        )

    if robust_range_hu > 115.0:
        hard_reject_reasons.append(
            f"interior HU range too large ({robust_range_hu:.1f} HU)"
        )

    if wide_range_hu > 260.0:
        hard_reject_reasons.append(
            f"wide interior HU range too large ({wide_range_hu:.1f} HU)"
        )

    if extreme_fraction > 0.060:
        hard_reject_reasons.append(
            f"too much structured non-uniform area ({extreme_fraction * 100.0:.1f}%)"
        )

    if sector_spread_hu > 65.0:
        hard_reject_reasons.append(
            f"sector means vary too much ({sector_spread_hu:.1f} HU)"
        )

    if largest_component_fraction > 0.020:
        hard_reject_reasons.append(
            "a large material insert occupies too much of the module"
        )

    hard_rejected = len(hard_reject_reasons) > 0

    return {
        "score": round(float(score), 5),
        "hardRejected": bool(hard_rejected),
        "hardRejectReasons": hard_reject_reasons,
        "medianHu": round(float(median_hu), 3),
        "robustRangeHu": round(float(robust_range_hu), 3),
        "wideRangeHu": round(float(wide_range_hu), 3),
        "robustSigmaHu": round(float(robust_sigma), 3),
        "structureThresholdHu": round(float(structure_threshold_hu), 3),
        "largeComponentCount": int(large_component_count),
        "mediumComponentCount": int(medium_component_count),
        "largestComponentFraction": round(float(largest_component_fraction), 6),
        "structuredAreaFraction": round(float(extreme_fraction), 6),
        "sectorSpreadHu": round(float(sector_spread_hu), 3),
        "edgeDensity": round(float(edge_density), 6),
        "meanEdgeStrength": round(float(mean_edge_strength), 6),
        "analysisPixelCount": int(analysis_area),
        "method": "uniformity-module content classifier",
    }


def _fast_uniformity_prescan(slice_data, slice_index):
    """
    Fast full-stack prescan.

    The image is downsampled before the uniformity signature is calculated.
    This makes scanning hundreds of slices practical.
    """
    _ensure_acr_usable_slice(slice_data)

    raw = np.asarray(slice_data["pixels"], dtype=np.float32)
    height, width = raw.shape

    step = max(1, int(max(height, width) / 256))
    small = raw[::step, ::step]

    phantom_cx, phantom_cy, phantom_radius = _detect_phantom_circle(small)

    signature = _uniformity_signature(
        raw_pixels=small,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    signature.update({
        "sliceIndex": int(slice_index),
        "sliceNumber": int(slice_index) + 1,
        "sliceLabel": slice_data.get("label", ""),
        "sourceName": slice_data.get("sourceName", ""),
        "downsampleStep": int(step),
        "phantomRadiusPixelsPrescan": round(float(phantom_radius), 3),
    })

    return signature


def _make_candidate_score_planes(
    raw,
    search_mask,
    window_width,
    window_level,
    phantom_radius,
    fast_mode=False
):
    values = raw[search_mask & np.isfinite(raw)]

    if values.size < 100:
        raise ValueError(ACR_UNIFORMITY_NOT_FOUND)

    p01 = float(np.percentile(values, 1))
    p99 = float(np.percentile(values, 99))

    raw_unit = np.clip(
        (raw - p01) / max(p99 - p01, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    planes = [
        ("raw_unit", raw_unit),
        ("current_window", _window_to_unit(raw, window_width, window_level)),
        ("ww25_wl0", _window_to_unit(raw, 25, 0)),
        ("ww40_wl0", _window_to_unit(raw, 40, 0)),
        ("ww60_wl0", _window_to_unit(raw, 60, 0)),
        ("ww80_wl0", _window_to_unit(raw, 80, 0)),
        ("ww100_wl0", _window_to_unit(raw, 100, 0)),
        ("ww160_wl0", _window_to_unit(raw, 160, 0)),
        ("ww250_wl20", _window_to_unit(raw, 250, 20)),
        ("ww400_wl40", _window_to_unit(raw, 400, 40)),
    ]

    if fast_mode:
        planes = [
            planes[0],
            planes[1],
            planes[3],
            planes[5],
            planes[7],
        ]

    score_planes = []

    for plane_name, plane in planes:
        # Use two blob scales. This makes the detector robust when the dot size
        # changes with matrix size and field of view.
        small_sigma = max(0.8, min(2.2, phantom_radius * 0.006))
        medium_sigma = max(1.4, min(4.5, phantom_radius * 0.014))

        small_blur = filters.gaussian(
            plane,
            sigma=small_sigma,
            preserve_range=True,
        )
        medium_blur = filters.gaussian(
            plane,
            sigma=medium_sigma,
            preserve_range=True,
        )

        band_pass = np.abs(
            small_blur.astype(np.float32) -
            medium_blur.astype(np.float32)
        )

        band_pass = filters.gaussian(
            band_pass,
            sigma=0.45,
            preserve_range=True,
        )

        band_values = band_pass[search_mask & np.isfinite(band_pass)]

        scale = (
            float(np.percentile(band_values, 99.5))
            if band_values.size
            else 1.0
        )

        if scale <= 0:
            scale = float(np.std(band_values)) if band_values.size else 1.0

        if scale <= 0:
            scale = 1.0

        score_planes.append(
            (
                plane_name,
                np.clip(band_pass / scale, 0.0, 5.0).astype(np.float32),
            )
        )

    return score_planes


def _find_dot_candidates_anywhere(
    raw_pixels,
    phantom_cx,
    phantom_cy,
    phantom_radius,
    window_width,
    window_level,
    fast_mode=False
):
    """
    Find compact dot-like candidates anywhere inside a verified uniformity slice.

    No fixed orientation or location is assumed.
    """
    raw = np.asarray(raw_pixels, dtype=np.float32)
    finite = np.isfinite(raw)

    height, width = raw.shape
    yy, xx = np.ogrid[:height, :width]

    cx = float(phantom_cx)
    cy = float(phantom_cy)
    radius = float(max(phantom_radius, 1.0))

    distance = np.sqrt(
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    )

    # Search only within the phantom body. External alignment holes are outside
    # this radius. The center itself is not excluded because legitimate dots may
    # rotate into different locations.
    search_mask = (
        finite &
        (distance <= radius * 0.84)
    )

    search_values = raw[search_mask]

    if search_values.size < 200:
        raise ValueError(ACR_UNIFORMITY_NOT_FOUND)

    phantom_noise = float(np.std(search_values))

    if phantom_noise <= 0:
        phantom_noise = 1.0

    score_planes = _make_candidate_score_planes(
        raw=raw,
        search_mask=search_mask,
        window_width=window_width,
        window_level=window_level,
        phantom_radius=radius,
        fast_mode=fast_mode,
    )

    raw_candidates = []

    percentiles = [99.98, 99.92, 99.82, 99.65, 99.4, 99.1, 98.7]

    if fast_mode:
        percentiles = [99.92, 99.55, 99.0]

    for plane_name, score_image in score_planes:
        plane_values = score_image[search_mask & np.isfinite(score_image)]

        if plane_values.size < 100:
            continue

        for percentile in percentiles:
            threshold = float(np.percentile(plane_values, percentile))
            threshold = max(threshold, 0.42)

            candidate_mask = (
                search_mask &
                (score_image >= threshold)
            )

            labels = measure.label(candidate_mask)
            regions = measure.regionprops(
                labels,
                intensity_image=score_image,
            )

            for region in regions:
                area = float(region.area)

                maximum_area = max(
                    6.0,
                    math.pi * (radius * 0.038) ** 2,
                )

                if area < 1.0 or area > maximum_area:
                    continue

                min_row, min_col, max_row, max_col = region.bbox
                box_height = max_row - min_row
                box_width = max_col - min_col

                if box_height <= 0 or box_width <= 0:
                    continue

                maximum_box = max(5.0, radius * 0.090)

                if box_height > maximum_box or box_width > maximum_box:
                    continue

                aspect_ratio = max(
                    box_width / max(box_height, 1),
                    box_height / max(box_width, 1),
                )

                if aspect_ratio > 3.5:
                    continue

                fill_ratio = area / max(float(box_height * box_width), 1.0)

                if fill_ratio < 0.12:
                    continue

                candidate_y, candidate_x = _safe_region_centroid(region)

                distance_from_center = math.sqrt(
                    (candidate_x - cx) ** 2 +
                    (candidate_y - cy) ** 2
                )

                if distance_from_center > radius * 0.84:
                    continue

                outer_radius = max(
                    5,
                    int(round(radius * 0.050)),
                )
                inner_radius = max(
                    1,
                    int(round(radius * 0.014)),
                )

                candidate_x_int = int(round(candidate_x))
                candidate_y_int = int(round(candidate_y))

                y0 = max(0, candidate_y_int - outer_radius)
                y1 = min(height, candidate_y_int + outer_radius + 1)
                x0 = max(0, candidate_x_int - outer_radius)
                x1 = min(width, candidate_x_int + outer_radius + 1)

                patch = raw[y0:y1, x0:x1]

                if patch.size < 16:
                    continue

                patch_yy, patch_xx = np.ogrid[y0:y1, x0:x1]
                patch_distance = np.sqrt(
                    (patch_xx - candidate_x) ** 2 +
                    (patch_yy - candidate_y) ** 2
                )

                center_mask = patch_distance <= inner_radius
                ring_mask = (
                    (patch_distance > inner_radius) &
                    (patch_distance <= outer_radius)
                )

                center_values = patch[
                    center_mask &
                    np.isfinite(patch)
                ]
                ring_values = patch[
                    ring_mask &
                    np.isfinite(patch)
                ]

                if center_values.size < 1 or ring_values.size < 8:
                    continue

                center_mean = float(np.mean(center_values))
                ring_median = float(np.median(ring_values))
                signed_contrast = center_mean - ring_median
                local_contrast = abs(signed_contrast)

                if local_contrast < max(
                    0.8,
                    phantom_noise * 0.07,
                ):
                    continue

                region_peak = _safe_region_max(region)
                region_mean = _safe_region_mean(region)

                compactness_score = 1.0 / max(aspect_ratio, 1.0)
                contrast_score = min(
                    local_contrast / max(phantom_noise * 0.65, 1.0),
                    1.0,
                )

                total_score = (
                    0.34 * region_peak +
                    0.16 * region_mean +
                    0.20 * compactness_score +
                    0.24 * contrast_score +
                    0.06 * min(fill_ratio, 1.0)
                )

                raw_candidates.append({
                    "x": float(candidate_x),
                    "y": float(candidate_y),
                    "area": float(area),
                    "meanIntensity": float(region_mean),
                    "maxIntensity": float(region_peak),
                    "rawPeak": float(center_mean),
                    "signedContrast": float(signed_contrast),
                    "localContrast": float(local_contrast),
                    "distanceFromCenter": float(distance_from_center),
                    "edgeGap": float(radius - distance_from_center),
                    "method": f"V15 multi-window blob candidate via {plane_name}",
                    "score": float(total_score),
                    "plane": plane_name,
                    "supportCount": 1,
                })

    if len(raw_candidates) < 2:
        raise ValueError(ACR_UNIFORMITY_NOT_FOUND)

    raw_candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    merge_distance = max(
        2.5,
        radius * 0.020,
    )

    merged_candidates = []

    for candidate in raw_candidates:
        matched = None

        for existing in merged_candidates:
            candidate_distance = math.sqrt(
                (candidate["x"] - existing["x"]) ** 2 +
                (candidate["y"] - existing["y"]) ** 2
            )

            if candidate_distance <= merge_distance:
                matched = existing
                break

        if matched is None:
            merged_candidate = dict(candidate)
            merged_candidate["supportCount"] = 1
            merged_candidate["planes"] = [candidate.get("plane", "")]
            merged_candidates.append(merged_candidate)
        else:
            old_support = int(matched.get("supportCount", 1))
            new_support = old_support + 1

            matched["x"] = (
                matched["x"] * old_support +
                candidate["x"]
            ) / new_support
            matched["y"] = (
                matched["y"] * old_support +
                candidate["y"]
            ) / new_support
            matched["score"] = max(
                float(matched.get("score", 0.0)),
                float(candidate.get("score", 0.0)),
            )
            matched["localContrast"] = max(
                float(matched.get("localContrast", 0.0)),
                float(candidate.get("localContrast", 0.0)),
            )
            matched["supportCount"] = new_support

            plane_name = candidate.get("plane", "")

            if plane_name:
                matched.setdefault("planes", [])

                if plane_name not in matched["planes"]:
                    matched["planes"].append(plane_name)

    supported_candidates = []

    for candidate in merged_candidates:
        support = int(candidate.get("supportCount", 1))
        local_contrast = float(candidate.get("localContrast", 0.0))

        if (
            support >= 2
            or local_contrast >= phantom_noise * 0.45
        ):
            supported_candidates.append(candidate)

    if len(supported_candidates) >= 2:
        merged_candidates = supported_candidates

    merged_candidates.sort(
        key=lambda candidate: (
            float(candidate.get("score", 0.0)) +
            0.085 * min(
                int(candidate.get("supportCount", 1)),
                10,
            ) +
            0.08 * min(
                float(candidate.get("localContrast", 0.0)) /
                max(phantom_noise, 1.0),
                2.0,
            )
        ),
        reverse=True,
    )

    return merged_candidates, phantom_noise


def _detect_two_bright_markers(
    raw_pixels,
    phantom_cx,
    phantom_cy,
    phantom_radius,
    detection_pixels=None,
    window_width=100,
    window_level=0,
    photometric="",
    fast_mode=False,
    row_spacing=None,
    col_spacing=None,
):
    """
    Find the BB pair only after the slice has passed the uniformity-module test.

    No fixed BB orientation or location is assumed.
    """
    candidates, phantom_noise = _find_dot_candidates_anywhere(
        raw_pixels=raw_pixels,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
        window_width=window_width,
        window_level=window_level,
        fast_mode=fast_mode,
    )

    if len(candidates) < 2:
        raise ValueError(ACR_UNIFORMITY_NOT_FOUND)

    radius = float(max(phantom_radius, 1.0))
    center_x = float(phantom_cx)
    center_y = float(phantom_cy)

    usable_candidates = candidates[:36]
    pair_candidates = []

    for first_index in range(len(usable_candidates)):
        for second_index in range(first_index + 1, len(usable_candidates)):
            marker_a = usable_candidates[first_index]
            marker_b = usable_candidates[second_index]

            delta_x = marker_b["x"] - marker_a["x"]
            delta_y = marker_b["y"] - marker_a["y"]

            pair_distance = math.sqrt(
                delta_x * delta_x +
                delta_y * delta_y
            )

            # Very short pairs are usually texture/noise. This is a broad
            # geometric sanity check, not a fixed expected location.
            if pair_distance < max(10.0, radius * 0.20):
                continue

            if pair_distance > radius * 1.45:
                continue

            if (
                marker_a.get("distanceFromCenter", 0.0) > radius * 0.82
                and marker_b.get("distanceFromCenter", 0.0) > radius * 0.82
            ):
                continue

            midpoint_x = (marker_a["x"] + marker_b["x"]) / 2.0
            midpoint_y = (marker_a["y"] + marker_b["y"]) / 2.0

            midpoint_distance = math.sqrt(
                (midpoint_x - center_x) ** 2 +
                (midpoint_y - center_y) ** 2
            )

            support_sum = (
                int(marker_a.get("supportCount", 1)) +
                int(marker_b.get("supportCount", 1))
            )

            minimum_contrast = min(
                float(marker_a.get("localContrast", 0.0)),
                float(marker_b.get("localContrast", 0.0)),
            )

            contrast_score = min(
                minimum_contrast /
                max(phantom_noise * 0.60, 1.0),
                1.0,
            )

            support_score = min(
                support_sum / 14.0,
                1.0,
            )

            separation_score = min(
                pair_distance / max(radius * 0.72, 1.0),
                1.0,
            )

            midpoint_score = max(
                0.0,
                1.0 - midpoint_distance / max(radius * 0.95, 1.0),
            )

            area_similarity = min(
                float(marker_a.get("area", 1.0)),
                float(marker_b.get("area", 1.0)),
            ) / max(
                float(marker_a.get("area", 1.0)),
                float(marker_b.get("area", 1.0)),
                1.0,
            )

            pair_score = (
                float(marker_a.get("score", 0.0)) +
                float(marker_b.get("score", 0.0)) +
                0.38 * support_score +
                0.34 * contrast_score +
                0.18 * separation_score +
                0.08 * midpoint_score +
                0.10 * area_similarity
            )

            pair_candidates.append({
                "markerA": marker_a,
                "markerB": marker_b,
                "distancePixels": float(pair_distance),
                "score": float(pair_score),
                "supportSum": int(support_sum),
                "minimumContrast": float(minimum_contrast),
                "midpointDistance": float(midpoint_distance),
                "areaSimilarity": float(area_similarity),
            })

    if not pair_candidates:
        raise ValueError(
            ACR_UNIFORMITY_NOT_FOUND +
            " No stable, separated BB dot pair was found inside the uniformity module."
        )

    pair_candidates.sort(
        key=lambda pair: pair["score"],
        reverse=True,
    )

    best_pair = pair_candidates[0]
    second_pair = (
        pair_candidates[1]
        if len(pair_candidates) > 1
        else None
    )

    if second_pair is not None:
        score_gap = (
            float(best_pair["score"]) -
            float(second_pair["score"])
        )
        relative_gap = score_gap / max(
            float(best_pair["score"]),
            1.0,
        )

        best_points = {
            (
                round(best_pair["markerA"]["x"]),
                round(best_pair["markerA"]["y"]),
            ),
            (
                round(best_pair["markerB"]["x"]),
                round(best_pair["markerB"]["y"]),
            ),
        }

        second_points = {
            (
                round(second_pair["markerA"]["x"]),
                round(second_pair["markerA"]["y"]),
            ),
            (
                round(second_pair["markerB"]["x"]),
                round(second_pair["markerB"]["y"]),
            ),
        }

        shared_points = len(
            best_points.intersection(second_points)
        )

        if shared_points == 0 and relative_gap < 0.035:
            raise ValueError(
                ACR_UNIFORMITY_NOT_FOUND +
                " Multiple BB pairs were almost equally likely; the app refused to guess."
            )

    marker_a = dict(best_pair["markerA"])
    marker_b = dict(best_pair["markerB"])
    marker_distance = float(best_pair["distancePixels"])

    confidence = max(
        0.0,
        min(
            1.0,
            float(best_pair["score"]) / 3.2,
        ),
    )

    if confidence < 0.42:
        raise ValueError(ACR_UNIFORMITY_NOT_FOUND)

    debug = {
        "confidence": round(float(confidence), 3),
        "candidateCount": int(len(candidates)),
        "method": "V15 uniformity-first multi-WW/WL consensus BB detector",
        "methodSupportCount": int(best_pair.get("supportSum", 0)),
        "pairDistancePixels": round(float(marker_distance), 3),
        "distanceRatioToPhantomRadius": round(
            float(marker_distance / max(radius, 1.0)),
            4,
        ),
        "minimumDotContrast": round(
            float(best_pair.get("minimumContrast", 0.0)),
            3,
        ),
        "areaSimilarity": round(
            float(best_pair.get("areaSimilarity", 0.0)),
            4,
        ),
        "topPairScore": round(
            float(best_pair.get("score", 0.0)),
            5,
        ),
        "secondPairScore": (
            round(float(second_pair.get("score", 0.0)), 5)
            if second_pair
            else None
        ),
        "fixedLocationAssumption": False,
        "uniformityModuleRequired": True,
        "multiWindowSearch": True,
        "ambiguousPairProtection": True,
        "note": (
            "The slice was first verified as a uniformity module. BB candidates "
            "were then searched anywhere inside the phantom using raw pixels, the "
            "viewer WW/WL, and automatic WW/WL planes."
        ),
    }

    return marker_a, marker_b, marker_distance, debug


def _make_circular_roi_mask(shape, cx, cy, radius):
    return _circular_mask(shape, cx, cy, radius)


def _make_circular_roi_mask_mm(
    shape,
    cx,
    cy,
    radius_mm,
    row_spacing,
    col_spacing,
):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]

    return (
        ((xx - cx) * float(col_spacing)) ** 2 +
        ((yy - cy) * float(row_spacing)) ** 2
    ) <= radius_mm ** 2


def _roi_stats(
    raw_pixels,
    name,
    cx,
    cy,
    radius,
    center_mean=None,
    row_spacing=None,
    col_spacing=None,
    radius_mm=None,
    diameter_mm=None,
    target_area_mm2=None,
):
    if (
        row_spacing is not None
        and col_spacing is not None
        and radius_mm is not None
    ):
        mask = _make_circular_roi_mask_mm(
            shape=raw_pixels.shape,
            cx=cx,
            cy=cy,
            radius_mm=radius_mm,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
        )

        radius_x = radius_mm / float(col_spacing)
        radius_y = radius_mm / float(row_spacing)

        actual_area_mm2 = float(
            np.sum(mask) *
            float(row_spacing) *
            float(col_spacing)
        )
    else:
        mask = _make_circular_roi_mask(
            raw_pixels.shape,
            cx,
            cy,
            radius,
        )

        radius_x = radius
        radius_y = radius
        actual_area_mm2 = None

    values = raw_pixels[mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError(f"{name} ROI did not contain pixels.")

    mean_value = float(np.mean(values))
    standard_deviation = float(np.std(values))

    if center_mean is None:
        difference = 0.0
        result = "REFERENCE"
    else:
        difference = mean_value - center_mean
        absolute_difference = abs(difference)

        if absolute_difference <= 5:
            result = "PASS"
        elif absolute_difference <= 7:
            result = "MINOR DEFICIENCY"
        else:
            result = "MAJOR DEFICIENCY"

    return {
        "name": name,
        "cx": int(round(cx)),
        "cy": int(round(cy)),
        "radius": round(float(radius), 2),
        "radiusX": round(float(radius_x), 2),
        "radiusY": round(float(radius_y), 2),
        "targetAreaMm2": (
            round(float(target_area_mm2), 2)
            if target_area_mm2 is not None
            else None
        ),
        "actualAreaMm2": (
            round(float(actual_area_mm2), 2)
            if actual_area_mm2 is not None
            else None
        ),
        "actualAreaPixels": int(np.sum(mask)),
        "radiusMm": (
            round(float(radius_mm), 3)
            if radius_mm is not None
            else None
        ),
        "diameterMm": (
            round(float(diameter_mm), 3)
            if diameter_mm is not None
            else None
        ),
        "mean": round(mean_value, 2),
        "std": round(standard_deviation, 2),
        "diffFromCenter": round(difference, 2),
        "result": result,
    }


def _build_400mm_rois(
    raw_pixels,
    info,
    phantom_cx,
    phantom_cy,
    phantom_radius,
):
    row_spacing = info.get("pixelSpacingRow")
    col_spacing = info.get("pixelSpacingCol")

    target_area_mm2 = 400.0
    roi_radius_mm = math.sqrt(
        target_area_mm2 / math.pi
    )
    roi_diameter_mm = roi_radius_mm * 2.0

    if row_spacing and col_spacing:
        row_spacing = float(row_spacing)
        col_spacing = float(col_spacing)

        roi_radius_x_pixels = roi_radius_mm / col_spacing
        roi_radius_y_pixels = roi_radius_mm / row_spacing
        roi_radius_pixels = (
            roi_radius_x_pixels +
            roi_radius_y_pixels
        ) / 2.0

        average_spacing = (
            row_spacing +
            col_spacing
        ) / 2.0

        phantom_radius_mm = (
            phantom_radius *
            average_spacing
        )

        offset_mm = max(
            roi_radius_mm + 5.0,
            min(
                phantom_radius_mm * 0.65,
                phantom_radius_mm -
                roi_radius_mm -
                15.0,
            ),
        )

        offset_x = offset_mm / col_spacing
        offset_y = offset_mm / row_spacing

        roi_size = {
            "targetAreaMm2": round(
                float(target_area_mm2),
                2,
            ),
            "radiusMm": round(
                float(roi_radius_mm),
                3,
            ),
            "diameterMm": round(
                float(roi_diameter_mm),
                3,
            ),
            "pixelSpacing": [
                row_spacing,
                col_spacing,
            ],
            "note": (
                "Each ROI targets 400 mm² using DICOM PixelSpacing. "
                "Actual area may differ slightly because pixels are discrete."
            ),
        }
    else:
        row_spacing = None
        col_spacing = None
        roi_radius_mm = None
        roi_diameter_mm = None
        target_area_mm2 = None

        roi_radius_pixels = max(
            4.0,
            min(
                phantom_radius * 0.113,
                phantom_radius * 0.18,
            ),
        )

        offset_x = phantom_radius * 0.60
        offset_y = phantom_radius * 0.60

        roi_size = {
            "targetAreaMm2": None,
            "radiusMm": None,
            "diameterMm": None,
            "pixelSpacing": None,
            "note": (
                "PixelSpacing is missing. ROI size is estimated in pixels only; "
                "the app cannot confirm a true 400 mm² ROI."
            ),
        }

    roi_definitions = [
        ("Center", phantom_cx, phantom_cy),
        ("Top", phantom_cx, phantom_cy - offset_y),
        ("Right", phantom_cx + offset_x, phantom_cy),
        ("Bottom", phantom_cx, phantom_cy + offset_y),
        ("Left", phantom_cx - offset_x, phantom_cy),
    ]

    center_roi = _roi_stats(
        raw_pixels=raw_pixels,
        name="Center",
        cx=roi_definitions[0][1],
        cy=roi_definitions[0][2],
        radius=roi_radius_pixels,
        center_mean=None,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        radius_mm=roi_radius_mm,
        diameter_mm=roi_diameter_mm,
        target_area_mm2=target_area_mm2,
    )

    rois = [center_roi]
    center_mean = center_roi["mean"]

    for name, roi_x, roi_y in roi_definitions[1:]:
        rois.append(
            _roi_stats(
                raw_pixels=raw_pixels,
                name=name,
                cx=roi_x,
                cy=roi_y,
                radius=roi_radius_pixels,
                center_mean=center_mean,
                row_spacing=row_spacing,
                col_spacing=col_spacing,
                radius_mm=roi_radius_mm,
                diameter_mm=roi_diameter_mm,
                target_area_mm2=target_area_mm2,
            )
        )

    actual_areas = [
        roi["actualAreaMm2"]
        for roi in rois
        if roi.get("actualAreaMm2") is not None
    ]

    if actual_areas:
        roi_size["actualAreaMinMm2"] = round(
            float(min(actual_areas)),
            2,
        )
        roi_size["actualAreaMaxMm2"] = round(
            float(max(actual_areas)),
            2,
        )
        roi_size["actualAreaMeanMm2"] = round(
            float(np.mean(actual_areas)),
            2,
        )

    return rois, roi_size, roi_radius_pixels


def _analyze_verified_uniformity_slice(
    slices,
    slice_index,
    window_width,
    window_level,
    fast_mode=False,
    make_overlay=True,
):
    slice_index = int(slice_index)

    if slice_index < 0 or slice_index >= len(slices):
        raise ValueError(
            f"Invalid slice index. This file contains {len(slices)} slice(s)."
        )

    selected_slice = slices[slice_index]
    _ensure_acr_usable_slice(selected_slice)

    raw_pixels = selected_slice["pixels"]
    info = selected_slice["info"]

    phantom_cx, phantom_cy, phantom_radius = _detect_phantom_circle(
        raw_pixels
    )

    uniformity_signature = _uniformity_signature(
        raw_pixels=raw_pixels,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    if uniformity_signature["hardRejected"]:
        reasons = "; ".join(
            uniformity_signature["hardRejectReasons"]
        )

        raise ValueError(
            ACR_UNIFORMITY_NOT_FOUND +
            " This slice is not the uniformity module: " +
            reasons
        )

    row_spacing = info.get("pixelSpacingRow")
    col_spacing = info.get("pixelSpacingCol")

    marker_a, marker_b, marker_distance_pixels, detection_debug = (
        _detect_two_bright_markers(
            raw_pixels=raw_pixels,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
            window_width=window_width,
            window_level=window_level,
            photometric=selected_slice.get("photometric", ""),
            fast_mode=fast_mode,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
        )
    )

    marker_distance_mm = None

    if row_spacing and col_spacing:
        delta_x_mm = (
            marker_a["x"] -
            marker_b["x"]
        ) * float(col_spacing)

        delta_y_mm = (
            marker_a["y"] -
            marker_b["y"]
        ) * float(row_spacing)

        marker_distance_mm = math.sqrt(
            delta_x_mm * delta_x_mm +
            delta_y_mm * delta_y_mm
        )

    rois, roi_size, roi_radius_pixels = _build_400mm_rois(
        raw_pixels=raw_pixels,
        info=info,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    worst_absolute_difference = max(
        abs(roi["diffFromCenter"])
        for roi in rois[1:]
    )

    if worst_absolute_difference <= 5:
        final_result = "PASS"
    elif worst_absolute_difference <= 7:
        final_result = "MINOR DEFICIENCY"
    else:
        final_result = "MAJOR DEFICIENCY"

    detection_confidence = float(
        detection_debug.get("confidence", 0.0)
    )

    slice_selection_score = (
        0.72 * float(uniformity_signature["score"]) +
        0.28 * detection_confidence
    )

    detection_debug["sliceSelectionScore"] = round(
        float(slice_selection_score),
        4,
    )
    detection_debug["uniformityScore"] = round(
        float(uniformity_signature["score"]),
        4,
    )

    overlay_base64 = None

    if make_overlay:
        overlay = window_pixels_to_image(
            raw_pixels,
            window_width,
            window_level,
            selected_slice.get("photometric", ""),
        )

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

        draw.line(
            [
                marker_a["x"],
                marker_a["y"],
                marker_b["x"],
                marker_b["y"],
            ],
            fill="cyan",
            width=3,
        )

        for marker in [marker_a, marker_b]:
            marker_x = marker["x"]
            marker_y = marker["y"]
            marker_radius = 6

            draw.ellipse(
                [
                    marker_x - marker_radius,
                    marker_y - marker_radius,
                    marker_x + marker_radius,
                    marker_y + marker_radius,
                ],
                outline="cyan",
                width=3,
            )

        for roi in rois:
            roi_x = roi["cx"]
            roi_y = roi["cy"]
            radius_x = roi.get(
                "radiusX",
                roi["radius"],
            )
            radius_y = roi.get(
                "radiusY",
                roi["radius"],
            )

            color = "lime"

            if roi["result"] == "MINOR DEFICIENCY":
                color = "orange"
            elif roi["result"] == "MAJOR DEFICIENCY":
                color = "red"

            draw.ellipse(
                [
                    roi_x - radius_x,
                    roi_y - radius_y,
                    roi_x + radius_x,
                    roi_y + radius_y,
                ],
                outline=color,
                width=3,
            )

            draw.text(
                (
                    roi_x + radius_x + 4,
                    roi_y - radius_y,
                ),
                roi["name"],
                fill=color,
            )

        overlay_base64 = image_to_base64(overlay)

    return {
        "success": True,
        "analysisType": "ACR CT Module 3 Uniformity",
        "sliceIndex": int(slice_index),
        "sliceLabel": selected_slice["label"],
        "sliceCount": len(slices),
        "windowWidth": float(window_width),
        "windowLevel": float(window_level),
        "windowUsed": {
            "label": "Viewer display only",
            "windowWidth": float(window_width),
            "windowLevel": float(window_level),
            "note": (
                "Slice selection first uses raw-HU uniformity content. "
                "BB detection then uses raw pixels, the viewer WW/WL, and "
                "automatic WW/WL planes. ROI statistics use raw HU."
            ),
        },
        "finalResult": final_result,
        "result": final_result,
        "phantomCenterX": int(round(phantom_cx)),
        "phantomCenterY": int(round(phantom_cy)),
        "phantomRadiusPixels": round(
            float(phantom_radius),
            2,
        ),
        "markerA": {
            "x": round(marker_a["x"], 2),
            "y": round(marker_a["y"], 2),
        },
        "markerB": {
            "x": round(marker_b["x"], 2),
            "y": round(marker_b["y"], 2),
        },
        "markerDistancePixels": round(
            float(marker_distance_pixels),
            2,
        ),
        "markerDistanceMm": (
            round(float(marker_distance_mm), 2)
            if marker_distance_mm is not None
            else None
        ),
        "bbDistance": {
            "pixels": round(
                float(marker_distance_pixels),
                3,
            ),
            "mm": (
                round(float(marker_distance_mm), 3)
                if marker_distance_mm is not None
                else None
            ),
            "note": (
                "Distance is in mm using DICOM PixelSpacing."
                if marker_distance_mm is not None
                else "PixelSpacing missing; distance shown in pixels only."
            ),
        },
        "roiAreaMm2": (
            400.0
            if roi_size.get("targetAreaMm2") is not None
            else None
        ),
        "roiSize": roi_size,
        "roiRadiusPixels": round(
            float(roi_radius_pixels),
            2,
        ),
        "rois": rois,
        "maxAbsDiffFromCenter": round(
            float(worst_absolute_difference),
            3,
        ),
        "criteriaNote": (
            "Each ROI targets 400 mm² when DICOM PixelSpacing is available. "
            "Center ROI is the reference. PASS: all peripheral ROI means are within ±5 HU "
            "of center. MINOR DEFICIENCY: any peripheral difference is >5 HU and ≤7 HU. "
            "MAJOR DEFICIENCY: any peripheral difference is >7 HU."
        ),
        "uniformityModule": uniformity_signature,
        "detection": detection_debug,
        "overlayImage": overlay_base64,
        "image": overlay_base64,
        "imageInfo": info,
        "readerVersion": ROBUST_VERSION,
        "_sliceSelectionScore": float(slice_selection_score),
    }


def _score_auto_candidate(
    slices,
    prescan_candidate,
    window_width,
    window_level,
):
    slice_index = int(
        prescan_candidate["sliceIndex"]
    )
    selected_slice = slices[slice_index]

    raw_pixels = selected_slice["pixels"]
    info = selected_slice.get("info", {})

    phantom_cx, phantom_cy, phantom_radius = _detect_phantom_circle(
        raw_pixels
    )

    full_signature = _uniformity_signature(
        raw_pixels=raw_pixels,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius=phantom_radius,
    )

    if full_signature["hardRejected"]:
        raise ValueError(
            "Full-resolution uniformity classifier rejected this slice."
        )

    marker_a, marker_b, marker_distance, detection_debug = (
        _detect_two_bright_markers(
            raw_pixels=raw_pixels,
            phantom_cx=phantom_cx,
            phantom_cy=phantom_cy,
            phantom_radius=phantom_radius,
            window_width=window_width,
            window_level=window_level,
            photometric=selected_slice.get("photometric", ""),
            fast_mode=True,
            row_spacing=info.get("pixelSpacingRow"),
            col_spacing=info.get("pixelSpacingCol"),
        )
    )

    detection_confidence = float(
        detection_debug.get("confidence", 0.0)
    )
    method_support = int(
        detection_debug.get("methodSupportCount", 0)
    )

    combined_score = (
        0.74 * float(full_signature["score"]) +
        0.26 * detection_confidence +
        min(method_support, 12) * 0.008
    )

    return {
        "sliceIndex": int(slice_index),
        "sliceNumber": int(slice_index) + 1,
        "score": round(float(combined_score), 5),
        "uniformityScore": round(
            float(full_signature["score"]),
            5,
        ),
        "bbConfidence": round(
            float(detection_confidence),
            5,
        ),
        "bbSupportCount": int(method_support),
        "bbDistancePixels": round(
            float(marker_distance),
            3,
        ),
        "largeComponentCount": int(
            full_signature["largeComponentCount"]
        ),
        "mediumComponentCount": int(
            full_signature["mediumComponentCount"]
        ),
        "robustRangeHu": float(
            full_signature["robustRangeHu"]
        ),
        "wideRangeHu": float(
            full_signature["wideRangeHu"]
        ),
        "sectorSpreadHu": float(
            full_signature["sectorSpreadHu"]
        ),
        "structuredAreaFraction": float(
            full_signature["structuredAreaFraction"]
        ),
        "edgeDensity": float(
            full_signature["edgeDensity"]
        ),
        "sliceLabel": selected_slice.get("label", ""),
        "sourceName": selected_slice.get("sourceName", ""),
        "markerA": {
            "x": round(float(marker_a["x"]), 2),
            "y": round(float(marker_a["y"]), 2),
        },
        "markerB": {
            "x": round(float(marker_b["x"]), 2),
            "y": round(float(marker_b["y"]), 2),
        },
    }


def _find_best_acr_uniformity_slice(
    slices,
    window_width,
    window_level,
):
    """
    Three-stage whole-stack selector.

    Stage 1: classify every slice by uniformity-module content.
    Stage 2: BB-check only the best uniformity-like slices.
    Stage 3: run full ROI/overlay analysis once on the best verified slice.

    This prevents insert/linearity/resolution slices from winning merely because
    they contain strong circular or line-like features.
    """
    prescan_results = []
    skipped_color_count = 0
    prescan_failure_count = 0

    for slice_index, slice_data in enumerate(slices):
        try:
            prescan_result = _fast_uniformity_prescan(
                slice_data=slice_data,
                slice_index=slice_index,
            )
            prescan_results.append(prescan_result)
        except ValueError as exc:
            message = str(exc).lower()

            if "color" in message or "rgb" in message:
                skipped_color_count += 1

            prescan_failure_count += 1
        except Exception:
            prescan_failure_count += 1

    if not prescan_results:
        if skipped_color_count == len(slices):
            raise ValueError(
                "ACR HU uniformity analysis requires original grayscale CT DICOM. "
                "All loaded slices are color/RGB DICOM or secondary captures."
            )

        raise ValueError(ACR_UNIFORMITY_NOT_FOUND)

    prescan_results.sort(
        key=lambda candidate: float(candidate["score"]),
        reverse=True,
    )

    non_rejected_prescan = [
        candidate
        for candidate in prescan_results
        if not candidate["hardRejected"]
    ]

    if not non_rejected_prescan:
        top_rejections = [
            {
                "sliceNumber": candidate["sliceNumber"],
                "score": candidate["score"],
                "reasons": candidate["hardRejectReasons"],
                "largeComponentCount": candidate["largeComponentCount"],
                "robustRangeHu": candidate["robustRangeHu"],
            }
            for candidate in prescan_results[:8]
        ]

        raise ValueError(
            ACR_UNIFORMITY_NOT_FOUND +
            " Every scanned slice looked like a non-uniform module. "
            f"Top rejected slices: {top_rejections}"
        )

    # The correct module should rank highly by uniformity content. Only these
    # slices need the more expensive BB search.
    bb_check_limit = min(
        16,
        len(non_rejected_prescan),
    )

    bb_candidates = []
    bb_failure_count = 0

    for prescan_candidate in non_rejected_prescan[:bb_check_limit]:
        try:
            bb_candidate = _score_auto_candidate(
                slices=slices,
                prescan_candidate=prescan_candidate,
                window_width=window_width,
                window_level=window_level,
            )
            bb_candidates.append(bb_candidate)
        except Exception:
            bb_failure_count += 1

    # If the first group had no BB pair, expand once. This keeps normal scans
    # fast but still searches more deeply when needed.
    if (
        not bb_candidates
        and len(non_rejected_prescan) > bb_check_limit
    ):
        expanded_limit = min(
            32,
            len(non_rejected_prescan),
        )

        for prescan_candidate in non_rejected_prescan[
            bb_check_limit:expanded_limit
        ]:
            try:
                bb_candidate = _score_auto_candidate(
                    slices=slices,
                    prescan_candidate=prescan_candidate,
                    window_width=window_width,
                    window_level=window_level,
                )
                bb_candidates.append(bb_candidate)
            except Exception:
                bb_failure_count += 1

    if not bb_candidates:
        top_uniformity_candidates = [
            {
                "sliceNumber": candidate["sliceNumber"],
                "score": candidate["score"],
                "largeComponentCount": candidate["largeComponentCount"],
                "robustRangeHu": candidate["robustRangeHu"],
                "sectorSpreadHu": candidate["sectorSpreadHu"],
            }
            for candidate in non_rejected_prescan[:10]
        ]

        raise ValueError(
            ACR_UNIFORMITY_NOT_FOUND +
            " Uniformity-like slices were found, but no stable BB pair was verified. "
            f"Top uniformity candidates: {top_uniformity_candidates}"
        )

    bb_candidates.sort(
        key=lambda candidate: float(candidate["score"]),
        reverse=True,
    )

    best_candidate = bb_candidates[0]
    selected_index = int(
        best_candidate["sliceIndex"]
    )

    # Full detector + final 400 mm² ROI analysis only once.
    final_result = _analyze_verified_uniformity_slice(
        slices=slices,
        slice_index=selected_index,
        window_width=window_width,
        window_level=window_level,
        fast_mode=False,
        make_overlay=True,
    )

    final_result.pop(
        "_sliceSelectionScore",
        None,
    )

    final_result["autoScan"] = {
        "enabled": True,
        "mode": "V15 uniformity-first three-stage full-stack scan",
        "totalSlicesScanned": len(slices),
        "prescanFailureCount": int(prescan_failure_count),
        "skippedColorSliceCount": int(skipped_color_count),
        "uniformityLikeSliceCount": len(non_rejected_prescan),
        "hardRejectedSliceCount": (
            len(prescan_results) -
            len(non_rejected_prescan)
        ),
        "bbCheckedSliceCount": min(
            len(non_rejected_prescan),
            max(bb_check_limit, 32 if bb_candidates else bb_check_limit),
        ),
        "bbFailureCount": int(bb_failure_count),
        "verifiedCandidateCount": len(bb_candidates),
        "selectedSliceIndex": int(selected_index),
        "selectedSliceNumber": int(selected_index) + 1,
        "selectedScore": round(
            float(best_candidate["score"]),
            5,
        ),
        "selectedUniformityScore": round(
            float(best_candidate["uniformityScore"]),
            5,
        ),
        "selectedBbConfidence": round(
            float(best_candidate["bbConfidence"]),
            5,
        ),
        "candidateSlices": bb_candidates[:12],
        "topUniformityPrescan": [
            {
                "sliceIndex": int(candidate["sliceIndex"]),
                "sliceNumber": int(candidate["sliceNumber"]),
                "score": float(candidate["score"]),
                "hardRejected": bool(candidate["hardRejected"]),
                "hardRejectReasons": candidate["hardRejectReasons"],
                "largeComponentCount": int(candidate["largeComponentCount"]),
                "mediumComponentCount": int(candidate["mediumComponentCount"]),
                "robustRangeHu": float(candidate["robustRangeHu"]),
                "wideRangeHu": float(candidate["wideRangeHu"]),
                "sectorSpreadHu": float(candidate["sectorSpreadHu"]),
                "structuredAreaFraction": float(candidate["structuredAreaFraction"]),
                "edgeDensity": float(candidate["edgeDensity"]),
            }
            for candidate in prescan_results[:15]
        ],
        "note": (
            "V15 first rejects slices containing large rods, contrast inserts, "
            "resolution bars, or excessive HU structure. Only uniformity-like "
            "slices are allowed into the multi-WW/WL BB search."
        ),
    }

    return final_result


def create_acr_module3_analysis(
    slice_index=0,
    stack_id=None,
    uploaded_file=None,
    file_storage=None,
    window_width=100,
    window_level=0,
    auto_scan=False,
):
    if uploaded_file is None and file_storage is not None:
        uploaded_file = file_storage

    slices = _get_slices_from_stack_or_upload(
        stack_id=stack_id,
        uploaded_file=uploaded_file,
    )

    if auto_scan:
        return _find_best_acr_uniformity_slice(
            slices=slices,
            window_width=window_width,
            window_level=window_level,
        )

    result = _analyze_verified_uniformity_slice(
        slices=slices,
        slice_index=int(slice_index),
        window_width=window_width,
        window_level=window_level,
        fast_mode=False,
        make_overlay=True,
    )

    result.pop(
        "_sliceSelectionScore",
        None,
    )

    result["autoScan"] = {
        "enabled": False,
        "selectedSliceIndex": int(slice_index),
        "selectedSliceNumber": int(slice_index) + 1,
        "note": (
            "Selected-slice mode still verifies that the image looks like the "
            "uniformity module before BB distance and ROI analysis."
        ),
    }

    return result
