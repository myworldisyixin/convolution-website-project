"""
Integrated ACR Module 3 full analysis.

Uses the working V4 four-module classifier to:
1. identify the Module 3 range,
2. select the exact strongest two-BB slice,
3. measure the verified BB distance,
4. place five true 400 mm² ROIs using DICOM PixelSpacing,
5. calculate the final uniformity result,
6. generate the final overlay.

This module does not search the full stack for arbitrary bright dots. It only
accepts the already verified best slice from the four-module classifier.
"""

from __future__ import annotations

import base64
from io import BytesIO
from collections import deque
import math
from typing import Any

import numpy as np
from PIL import ImageDraw

from services.acr_module_classifier import (
    CLASSIFIER_VERSION,
    _estimate_phantom_geometry,
    create_acr_module_classification,
)
from services.dicom_display import (
    _get_slices_from_stack_or_upload,
    window_pixels_to_image,
)


FULL_ANALYSIS_VERSION = "ACR_MODULE3_FULL_ANALYSIS_V12_CONNECTED_EDGE_2026_07_05"


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


def _bilinear_sample(
    image: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> np.ndarray:
    """Sample a 2D image at floating-point coordinates without SciPy."""
    height, width = image.shape

    x0 = np.floor(x_coordinates).astype(np.int32)
    y0 = np.floor(y_coordinates).astype(np.int32)

    x0 = np.clip(x0, 0, width - 2)
    y0 = np.clip(y0, 0, height - 2)

    x1 = x0 + 1
    y1 = y0 + 1

    dx = x_coordinates - x0
    dy = y_coordinates - y0

    return (
        (1.0 - dx) * (1.0 - dy) * image[y0, x0]
        + dx * (1.0 - dy) * image[y0, x1]
        + (1.0 - dx) * dy * image[y1, x0]
        + dx * dy * image[y1, x1]
    ).astype(np.float32)


def _moving_average_1d(values: np.ndarray, width: int = 7) -> np.ndarray:
    width = max(1, int(width))

    if width <= 1:
        return values.astype(np.float32, copy=True)

    kernel = np.ones(width, dtype=np.float32) / float(width)

    return np.convolve(
        values.astype(np.float32),
        kernel,
        mode="same",
    ).astype(np.float32)


def _fit_circle_to_points(
    points: np.ndarray,
) -> tuple[float, float, float]:
    if points.ndim != 2 or points.shape[0] < 3:
        raise ValueError("At least three boundary points are required.")

    x_values = points[:, 0].astype(np.float64)
    y_values = points[:, 1].astype(np.float64)

    matrix = np.column_stack([
        x_values,
        y_values,
        np.ones_like(x_values),
    ])
    target = -(x_values ** 2 + y_values ** 2)

    coefficient_x, coefficient_y, coefficient_c = np.linalg.lstsq(
        matrix,
        target,
        rcond=None,
    )[0]

    center_x = -coefficient_x / 2.0
    center_y = -coefficient_y / 2.0

    radius_squared = (
        center_x ** 2
        + center_y ** 2
        - coefficient_c
    )

    if radius_squared <= 0:
        raise ValueError("The fitted phantom circle has an invalid radius.")

    return (
        float(center_x),
        float(center_y),
        float(math.sqrt(radius_squared)),
    )


def _circular_median_smooth(
    values: np.ndarray,
    half_window: int = 3,
) -> np.ndarray:
    """
    Smooth radial boundary measurements while preserving a circular sequence.
    """
    values = np.asarray(values, dtype=np.float64)
    count = values.size

    if count == 0:
        return values

    output = np.empty_like(values)

    for index in range(count):
        neighborhood = [
            values[(index + offset) % count]
            for offset in range(-half_window, half_window + 1)
        ]
        output[index] = float(np.median(neighborhood))

    return output



def _binary_neighbor_filter(mask: np.ndarray) -> np.ndarray:
    """
    Remove isolated threshold pixels without SciPy.

    A pixel remains part of the candidate phantom when at least four pixels in
    its 3x3 neighborhood belong to the same threshold mask.
    """
    binary = np.asarray(mask, dtype=np.uint8)
    padded = np.pad(binary, 1, mode="constant")

    neighborhood_count = np.zeros_like(binary, dtype=np.uint8)

    for y_offset in range(3):
        for x_offset in range(3):
            neighborhood_count += padded[
                y_offset:y_offset + binary.shape[0],
                x_offset:x_offset + binary.shape[1],
            ]

    return neighborhood_count >= 4


def _center_connected_component(
    mask: np.ndarray,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """
    Keep only the threshold component connected to the phantom center.

    This prevents a separate exterior holder, support ring, table, or alignment
    mark from enlarging the yellow phantom outline.
    """
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape

    seed_x = int(round(center_x))
    seed_y = int(round(center_y))

    seed_x = min(max(seed_x, 0), width - 1)
    seed_y = min(max(seed_y, 0), height - 1)

    if not binary[seed_y, seed_x]:
        search_radius = 18

        y0 = max(0, seed_y - search_radius)
        y1 = min(height, seed_y + search_radius + 1)
        x0 = max(0, seed_x - search_radius)
        x1 = min(width, seed_x + search_radius + 1)

        local_y, local_x = np.where(binary[y0:y1, x0:x1])

        if local_x.size == 0:
            raise ValueError(
                "No thresholded phantom pixel was found near the image center."
            )

        global_x = local_x + x0
        global_y = local_y + y0

        nearest = int(np.argmin(
            (global_x - center_x) ** 2
            + (global_y - center_y) ** 2
        ))

        seed_x = int(global_x[nearest])
        seed_y = int(global_y[nearest])

    component = np.zeros_like(binary, dtype=bool)
    queue = deque([(seed_y, seed_x)])
    component[seed_y, seed_x] = True

    while queue:
        y_value, x_value = queue.popleft()

        for y_offset, x_offset in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            neighbor_y = y_value + y_offset
            neighbor_x = x_value + x_offset

            if (
                0 <= neighbor_y < height
                and 0 <= neighbor_x < width
                and binary[neighbor_y, neighbor_x]
                and not component[neighbor_y, neighbor_x]
            ):
                component[neighbor_y, neighbor_x] = True
                queue.append((neighbor_y, neighbor_x))

    return component


def _component_radial_boundary_points(
    component: np.ndarray,
    center_x: float,
    center_y: float,
    angle_count: int = 360,
) -> np.ndarray:
    """
    Find the outside edge of the center-connected component along radial rays.
    """
    height, width = component.shape

    maximum_radius = min(
        center_x,
        center_y,
        width - 1.0 - center_x,
        height - 1.0 - center_y,
    )

    radial_positions = np.arange(
        0.0,
        maximum_radius + 0.5,
        0.5,
        dtype=np.float32,
    )

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        angle_count,
        endpoint=False,
        dtype=np.float64,
    )

    points: list[tuple[float, float]] = []

    for angle in angles:
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))

        x_indices = np.clip(
            np.rint(
                center_x
                + radial_positions * cosine
            ).astype(np.int32),
            0,
            width - 1,
        )
        y_indices = np.clip(
            np.rint(
                center_y
                + radial_positions * sine
            ).astype(np.int32),
            0,
            height - 1,
        )

        values = component[y_indices, x_indices]

        # Start well inside the phantom. The edge is the first sustained run of
        # exterior pixels, not an isolated false pixel caused by image noise.
        search_start = int(len(values) * 0.35)
        boundary_index = None

        for index in range(search_start, len(values) - 6):
            if not np.any(values[index:index + 6]):
                boundary_index = max(0, index - 1)
                break

        if boundary_index is None:
            continue

        radius = float(radial_positions[boundary_index])

        points.append((
            float(center_x + radius * cosine),
            float(center_y + radius * sine),
        ))

    return np.asarray(points, dtype=np.float64)


def _detect_center_connected_phantom_boundary(
    raw: np.ndarray,
    initial_center_x: float,
    initial_center_y: float,
    initial_radius: float,
) -> tuple[
    float,
    float,
    float,
    list[tuple[float, float]],
    dict[str, Any],
]:
    """
    Detect the circular phantom body from the center-connected HU region.

    Unlike V11, this does not choose the first intensity transition along each
    ray. That method could lock onto internal texture or a partial-volume band.
    Instead, this method:

    1. measures the central phantom and exterior background levels,
    2. creates several candidate thresholds between those levels,
    3. keeps only the component connected to the image center,
    4. measures its outside edge along 360 directions,
    5. robustly fits one stable circle to those measured edge points.

    Separate outside rings and holders are excluded because they are not
    connected to the center component.
    """
    image = np.asarray(raw, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            "Connected phantom-edge detection requires a 2D image."
        )

    height, width = image.shape
    yy, xx = np.ogrid[:height, :width]

    center_disk = (
        (xx - float(initial_center_x)) ** 2
        + (yy - float(initial_center_y)) ** 2
    ) <= max(10.0, float(initial_radius) * 0.18) ** 2

    center_values = image[
        center_disk & np.isfinite(image)
    ]

    border_width = max(
        3,
        int(round(min(height, width) * 0.04)),
    )

    border_values = np.concatenate([
        image[:border_width, :].ravel(),
        image[-border_width:, :].ravel(),
        image[:, :border_width].ravel(),
        image[:, -border_width:].ravel(),
    ])
    border_values = border_values[
        np.isfinite(border_values)
    ]

    if center_values.size < 20 or border_values.size < 20:
        raise ValueError(
            "Not enough pixels to measure phantom and exterior levels."
        )

    interior_level = float(np.median(center_values))
    exterior_level = float(np.median(border_values))
    level_difference = interior_level - exterior_level

    if abs(level_difference) < 20.0:
        raise ValueError(
            "The phantom and exterior levels are not sufficiently different."
        )

    threshold_fractions = (
        0.25,
        0.35,
        0.45,
        0.50,
        0.55,
        0.65,
        0.75,
    )

    candidates: list[dict[str, Any]] = []

    for threshold_fraction in threshold_fractions:
        threshold_value = (
            exterior_level
            + threshold_fraction * level_difference
        )

        if level_difference > 0:
            threshold_mask = image >= threshold_value
        else:
            threshold_mask = image <= threshold_value

        threshold_mask &= np.isfinite(image)
        threshold_mask = _binary_neighbor_filter(
            threshold_mask
        )

        try:
            component = _center_connected_component(
                threshold_mask,
                center_x=initial_center_x,
                center_y=initial_center_y,
            )
        except ValueError:
            continue

        component_area = int(np.sum(component))

        if component_area < height * width * 0.06:
            continue

        area_radius = math.sqrt(
            float(component_area) / math.pi
        )

        if not (
            min(height, width) * 0.15
            <= area_radius
            <= min(height, width) * 0.49
        ):
            continue

        boundary_points = _component_radial_boundary_points(
            component=component,
            center_x=initial_center_x,
            center_y=initial_center_y,
            angle_count=360,
        )

        if boundary_points.shape[0] < 280:
            continue

        fitted_center_x, fitted_center_y, fitted_radius = (
            _fit_circle_to_points(boundary_points)
        )

        point_radii = np.sqrt(
            (
                boundary_points[:, 0]
                - fitted_center_x
            ) ** 2
            + (
                boundary_points[:, 1]
                - fitted_center_y
            ) ** 2
        )

        median_point_radius = float(
            np.median(point_radii)
        )
        radial_deviation = np.abs(
            point_radii - median_point_radius
        )
        radial_mad = float(
            np.median(radial_deviation)
        )

        allowed_deviation = max(
            2.5,
            4.0 * 1.4826 * radial_mad,
        )

        inlier_mask = (
            radial_deviation <= allowed_deviation
        )
        inlier_points = boundary_points[inlier_mask]

        if inlier_points.shape[0] < 240:
            continue

        fitted_center_x, fitted_center_y, fitted_radius = (
            _fit_circle_to_points(inlier_points)
        )

        fitted_distances = np.sqrt(
            (
                inlier_points[:, 0]
                - fitted_center_x
            ) ** 2
            + (
                inlier_points[:, 1]
                - fitted_center_y
            ) ** 2
        )

        root_mean_square_error = float(
            np.sqrt(
                np.mean(
                    (
                        fitted_distances
                        - fitted_radius
                    ) ** 2
                )
            )
        )

        center_shift = math.hypot(
            fitted_center_x - initial_center_x,
            fitted_center_y - initial_center_y,
        )

        # The initial estimate is only a broad guide. The lower bound allows a
        # separate outer holder to have inflated that initial estimate.
        if not (
            0.45 * float(initial_radius)
            <= fitted_radius
            <= 1.06 * float(initial_radius)
        ):
            continue

        if center_shift > max(
            12.0,
            fitted_radius * 0.08,
        ):
            continue

        coverage = (
            float(inlier_points.shape[0]) / 360.0
        )

        area_fit_difference = abs(
            area_radius - fitted_radius
        )

        candidate_score = (
            coverage
            - root_mean_square_error / 18.0
            - area_fit_difference / 50.0
            - center_shift / 80.0
        )

        candidates.append({
            "score": float(candidate_score),
            "thresholdFraction": float(
                threshold_fraction
            ),
            "thresholdValue": float(
                threshold_value
            ),
            "centerX": float(
                fitted_center_x
            ),
            "centerY": float(
                fitted_center_y
            ),
            "radius": float(
                fitted_radius
            ),
            "areaRadius": float(
                area_radius
            ),
            "coverage": float(
                coverage
            ),
            "fitRmse": float(
                root_mean_square_error
            ),
            "centerShift": float(
                center_shift
            ),
            "componentArea": int(
                component_area
            ),
            "inlierPointCount": int(
                inlier_points.shape[0]
            ),
        })

    if not candidates:
        raise ValueError(
            "No stable center-connected phantom boundary could be fitted."
        )

    best = max(
        candidates,
        key=lambda candidate: candidate["score"],
    )

    refined_center_x = float(best["centerX"])
    refined_center_y = float(best["centerY"])
    refined_radius = float(best["radius"])

    drawing_angles = np.linspace(
        0.0,
        2.0 * math.pi,
        360,
        endpoint=False,
        dtype=np.float64,
    )

    drawing_points = [
        (
            float(
                refined_center_x
                + refined_radius
                * math.cos(float(angle))
            ),
            float(
                refined_center_y
                + refined_radius
                * math.sin(float(angle))
            ),
        )
        for angle in drawing_angles
    ]

    diagnostics = {
        "method": (
            "center-connected threshold component with robust circle fit"
        ),
        "fallbackUsed": False,
        "initialCenterX": round(
            float(initial_center_x),
            3,
        ),
        "initialCenterY": round(
            float(initial_center_y),
            3,
        ),
        "initialRadiusPixels": round(
            float(initial_radius),
            3,
        ),
        "refinedCenterX": round(
            refined_center_x,
            3,
        ),
        "refinedCenterY": round(
            refined_center_y,
            3,
        ),
        "refinedRadiusPixels": round(
            refined_radius,
            3,
        ),
        "radiusChangePixels": round(
            refined_radius
            - float(initial_radius),
            3,
        ),
        "thresholdFraction": round(
            float(best["thresholdFraction"]),
            3,
        ),
        "thresholdValue": round(
            float(best["thresholdValue"]),
            3,
        ),
        "interiorLevel": round(
            interior_level,
            3,
        ),
        "exteriorLevel": round(
            exterior_level,
            3,
        ),
        "componentAreaPixels": int(
            best["componentArea"]
        ),
        "boundaryPointCount": int(
            best["inlierPointCount"]
        ),
        "angularCoverage": round(
            float(best["coverage"]),
            4,
        ),
        "circleFitRmsePixels": round(
            float(best["fitRmse"]),
            4,
        ),
        "centerShiftPixels": round(
            float(best["centerShift"]),
            4,
        ),
    }

    return (
        refined_center_x,
        refined_center_y,
        refined_radius,
        drawing_points,
        diagnostics,
    )

def _detect_phantom_boundary_from_intensity_change(
    raw: np.ndarray,
    initial_center_x: float,
    initial_center_y: float,
    initial_radius: float,
) -> tuple[float, float, float, list[tuple[float, float]], dict[str, Any]]:
    """
    Detect the visible phantom edge from the first sustained intensity change.

    The previous area-based radius could include the surrounding holder, air
    ring, alignment marks, or other exterior structures and therefore draw a
    yellow circle that was too large. This detector travels outward from the
    phantom interior along many radial lines and selects the first strong,
    sustained transition from phantom material to the exterior.

    No fixed phantom diameter is assumed. The detected radius changes with the
    actual phantom size in each image.
    """
    image = np.asarray(raw, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError("Phantom edge detection requires a 2D image.")

    height, width = image.shape

    initial_center_x = float(initial_center_x)
    initial_center_y = float(initial_center_y)
    initial_radius = float(initial_radius)

    maximum_bound_radius = min(
        initial_center_x,
        initial_center_y,
        width - 1.0 - initial_center_x,
        height - 1.0 - initial_center_y,
    )

    if maximum_bound_radius <= 20:
        raise ValueError("The estimated phantom center is too close to an image edge.")

    # The old radius may be too large, so search substantially inside it and
    # stop before the edge of the matrix.
    search_start = max(
        20.0,
        min(
            initial_radius * 0.48,
            maximum_bound_radius * 0.52,
        ),
    )
    search_end = min(
        maximum_bound_radius * 0.995,
        max(
            initial_radius * 1.08,
            maximum_bound_radius * 0.90,
        ),
    )

    if search_end - search_start < 30:
        raise ValueError("The radial phantom-edge search range is too small.")

    radial_positions = np.arange(
        search_start,
        search_end + 0.5,
        0.5,
        dtype=np.float32,
    )

    yy, xx = np.ogrid[:height, :width]

    central_mask = (
        (xx - initial_center_x) ** 2
        + (yy - initial_center_y) ** 2
    ) <= max(8.0, initial_radius * 0.20) ** 2

    central_values = image[
        central_mask & np.isfinite(image)
    ]

    border_width = max(
        3,
        int(round(min(height, width) * 0.04)),
    )

    border_values = np.concatenate([
        image[:border_width, :].ravel(),
        image[-border_width:, :].ravel(),
        image[:, :border_width].ravel(),
        image[:, -border_width:].ravel(),
    ])
    border_values = border_values[
        np.isfinite(border_values)
    ]

    if central_values.size < 20 or border_values.size < 20:
        raise ValueError("Not enough valid pixels to characterize the phantom edge.")

    interior_level = float(np.median(central_values))
    exterior_level = float(np.median(border_values))

    intensity_direction = (
        1.0
        if interior_level >= exterior_level
        else -1.0
    )

    full_contrast = abs(interior_level - exterior_level)

    # A real phantom-to-exterior change is normally large in CT data, but the
    # lower floor keeps this usable for screenshots or nonstandard rescaling.
    minimum_drop = max(
        35.0,
        full_contrast * 0.16,
    )

    angle_count = 240
    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        angle_count,
        endpoint=False,
        dtype=np.float64,
    )

    accepted_angles: list[float] = []
    accepted_radii: list[float] = []
    accepted_contrasts: list[float] = []

    inside_start = 18
    inside_end = 6
    outside_start = 6
    outside_end = 18

    for angle in angles:
        cos_angle = math.cos(float(angle))
        sin_angle = math.sin(float(angle))

        x_positions = (
            initial_center_x
            + radial_positions * cos_angle
        )
        y_positions = (
            initial_center_y
            + radial_positions * sin_angle
        )

        profile = _bilinear_sample(
            image=image,
            x_coordinates=x_positions,
            y_coordinates=y_positions,
        )
        profile = _moving_average_1d(
            profile,
            width=7,
        )

        candidate_indices: list[int] = []
        candidate_contrasts: list[float] = []

        for index in range(
            inside_start + 2,
            len(radial_positions) - outside_end - 2,
        ):
            inside_value = float(np.median(
                profile[
                    index - inside_start:
                    index - inside_end
                ]
            ))
            outside_value = float(np.median(
                profile[
                    index + outside_start:
                    index + outside_end
                ]
            ))

            sustained_drop = (
                intensity_direction
                * (inside_value - outside_value)
            )

            local_drop = (
                intensity_direction
                * (
                    float(profile[index - 4])
                    - float(profile[index + 4])
                )
            )

            if (
                sustained_drop >= minimum_drop
                and local_drop >= minimum_drop * 0.05
            ):
                candidate_indices.append(index)
                candidate_contrasts.append(sustained_drop)

        if not candidate_indices:
            continue

        strongest_contrast = max(candidate_contrasts)

        # Choose the first strong transition rather than the strongest outer
        # transition. This is what prevents the exterior holder/ring from
        # making the yellow boundary too large.
        required_contrast = max(
            minimum_drop,
            strongest_contrast * 0.34,
        )

        selected_index = None
        selected_contrast = None

        for candidate_index, candidate_contrast in zip(
            candidate_indices,
            candidate_contrasts,
        ):
            if candidate_contrast >= required_contrast:
                selected_index = candidate_index
                selected_contrast = candidate_contrast
                break

        if selected_index is None:
            continue

        accepted_angles.append(float(angle))
        accepted_radii.append(
            float(radial_positions[selected_index])
        )
        accepted_contrasts.append(
            float(selected_contrast)
        )

    if len(accepted_radii) < 80:
        raise ValueError(
            "Too few radial intensity transitions were found to refine the phantom boundary."
        )

    radii_array = np.asarray(
        accepted_radii,
        dtype=np.float64,
    )
    angles_array = np.asarray(
        accepted_angles,
        dtype=np.float64,
    )

    median_radius = float(np.median(radii_array))
    absolute_deviation = np.abs(
        radii_array - median_radius
    )
    mad = float(np.median(absolute_deviation))

    radius_tolerance = max(
        4.0,
        4.5 * 1.4826 * mad,
    )

    robust_mask = (
        absolute_deviation <= radius_tolerance
    )

    robust_radii = radii_array[robust_mask]
    robust_angles = angles_array[robust_mask]

    if robust_radii.size < 60:
        raise ValueError(
            "The detected phantom-edge radii were not sufficiently consistent."
        )

    first_pass_points = np.column_stack([
        initial_center_x
        + robust_radii * np.cos(robust_angles),
        initial_center_y
        + robust_radii * np.sin(robust_angles),
    ])

    refined_center_x, refined_center_y, fitted_radius = (
        _fit_circle_to_points(first_pass_points)
    )

    radial_residuals = np.abs(
        np.sqrt(
            (first_pass_points[:, 0] - refined_center_x) ** 2
            + (first_pass_points[:, 1] - refined_center_y) ** 2
        )
        - fitted_radius
    )

    residual_median = float(
        np.median(radial_residuals)
    )
    residual_mad = float(np.median(
        np.abs(
            radial_residuals
            - residual_median
        )
    ))

    residual_tolerance = max(
        3.0,
        residual_median
        + 4.0 * 1.4826 * residual_mad,
    )

    fit_mask = (
        radial_residuals <= residual_tolerance
    )
    fit_points = first_pass_points[fit_mask]

    if fit_points.shape[0] >= 40:
        refined_center_x, refined_center_y, fitted_radius = (
            _fit_circle_to_points(fit_points)
        )

    # Build a smooth detected-edge contour. This follows the actual color/HU
    # change and is used for drawing the yellow outline.
    contour_angle_count = 240
    contour_angles = np.linspace(
        0.0,
        2.0 * math.pi,
        contour_angle_count,
        endpoint=False,
        dtype=np.float64,
    )

    # Interpolate robust per-angle radii onto the full angle sequence.
    ordered_indices = np.argsort(robust_angles)
    ordered_angles = robust_angles[ordered_indices]
    ordered_radii = robust_radii[ordered_indices]

    wrapped_angles = np.concatenate([
        ordered_angles - 2.0 * math.pi,
        ordered_angles,
        ordered_angles + 2.0 * math.pi,
    ])
    wrapped_radii = np.concatenate([
        ordered_radii,
        ordered_radii,
        ordered_radii,
    ])

    interpolated_radii = np.interp(
        contour_angles,
        wrapped_angles,
        wrapped_radii,
    )
    smoothed_radii = _circular_median_smooth(
        interpolated_radii,
        half_window=4,
    )

    # Recenter the contour around the fitted center while preserving the
    # measured radial shape.
    center_shift_x = (
        refined_center_x - initial_center_x
    )
    center_shift_y = (
        refined_center_y - initial_center_y
    )

    contour_points = [
        (
            float(
                initial_center_x
                + center_shift_x
                + radius * math.cos(float(angle))
            ),
            float(
                initial_center_y
                + center_shift_y
                + radius * math.sin(float(angle))
            ),
        )
        for angle, radius in zip(
            contour_angles,
            smoothed_radii,
        )
    ]

    contour_radius = float(
        np.median(smoothed_radii)
    )

    # Final validation prevents a wildly incorrect refinement from replacing
    # the safer initial result.
    if not (
        0.45 * initial_radius
        <= contour_radius
        <= 1.08 * initial_radius
    ):
        raise ValueError(
            "The intensity-change boundary was not physically consistent with the initial phantom estimate."
        )

    diagnostics = {
        "method": (
            "first sustained radial intensity change with robust circle fit"
        ),
        "initialCenterX": round(
            float(initial_center_x),
            3,
        ),
        "initialCenterY": round(
            float(initial_center_y),
            3,
        ),
        "initialRadiusPixels": round(
            float(initial_radius),
            3,
        ),
        "refinedCenterX": round(
            float(refined_center_x),
            3,
        ),
        "refinedCenterY": round(
            float(refined_center_y),
            3,
        ),
        "refinedRadiusPixels": round(
            float(contour_radius),
            3,
        ),
        "radiusChangePixels": round(
            float(contour_radius - initial_radius),
            3,
        ),
        "validRayCount": int(
            robust_radii.size
        ),
        "medianBoundaryContrast": round(
            float(np.median(accepted_contrasts)),
            3,
        ),
        "interiorLevel": round(
            float(interior_level),
            3,
        ),
        "exteriorLevel": round(
            float(exterior_level),
            3,
        ),
        "minimumAcceptedDrop": round(
            float(minimum_drop),
            3,
        ),
    }

    return (
        float(refined_center_x),
        float(refined_center_y),
        float(contour_radius),
        contour_points,
        diagnostics,
    )


def _elliptical_roi_mask(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    radius_mm: float,
    row_spacing: float,
    col_spacing: float,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]

    return (
        ((xx - float(cx)) * col_spacing) ** 2
        + ((yy - float(cy)) * row_spacing) ** 2
    ) <= radius_mm ** 2


def _roi_measurement(
    raw: np.ndarray,
    name: str,
    cx: float,
    cy: float,
    radius_mm: float,
    row_spacing: float,
    col_spacing: float,
    center_mean: float | None,
) -> dict[str, Any]:
    mask = _elliptical_roi_mask(
        shape=raw.shape,
        cx=cx,
        cy=cy,
        radius_mm=radius_mm,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
    )

    values = raw[mask]
    values = values[np.isfinite(values)]

    if values.size < 10:
        raise ValueError(f"{name} ROI does not contain enough valid pixels.")

    mean_hu = float(np.mean(values))
    std_hu = float(np.std(values))
    actual_area_mm2 = float(values.size * row_spacing * col_spacing)

    if center_mean is None:
        difference = 0.0
        result = "REFERENCE"
    else:
        difference = mean_hu - float(center_mean)
        absolute_difference = abs(difference)

        if absolute_difference <= 5.0:
            result = "PASS"
        elif absolute_difference <= 7.0:
            result = "MINOR DEFICIENCY"
        else:
            result = "MAJOR DEFICIENCY"

    return {
        "name": name,
        "cx": round(float(cx), 3),
        "cy": round(float(cy), 3),
        "radiusMm": round(float(radius_mm), 3),
        "radiusX": round(float(radius_mm / col_spacing), 3),
        "radiusY": round(float(radius_mm / row_spacing), 3),
        "targetAreaMm2": 400.0,
        "actualAreaMm2": round(float(actual_area_mm2), 2),
        "pixelCount": int(values.size),
        "mean": round(float(mean_hu), 2),
        "std": round(float(std_hu), 2),
        "diffFromCenter": round(float(difference), 2),
        "result": result,
    }


def _build_uniformity_rois(
    raw: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius_pixels: float,
    row_spacing: float,
    col_spacing: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_area_mm2 = 400.0
    radius_mm = math.sqrt(target_area_mm2 / math.pi)
    diameter_mm = radius_mm * 2.0

    average_spacing = (row_spacing + col_spacing) / 2.0
    phantom_radius_mm = phantom_radius_pixels * average_spacing

    # Peripheral ROI centers stay inside the phantom and near the periphery.
    offset_mm = max(
        radius_mm + 5.0,
        min(
            phantom_radius_mm * 0.65,
            phantom_radius_mm - radius_mm - 15.0,
        ),
    )

    if offset_mm <= radius_mm:
        raise ValueError(
            "The detected phantom is too small to place five separate 400 mm² ROIs."
        )

    offset_x_pixels = offset_mm / col_spacing
    offset_y_pixels = offset_mm / row_spacing

    definitions = [
        ("Center", phantom_cx, phantom_cy),
        ("Top", phantom_cx, phantom_cy - offset_y_pixels),
        ("Right", phantom_cx + offset_x_pixels, phantom_cy),
        ("Bottom", phantom_cx, phantom_cy + offset_y_pixels),
        ("Left", phantom_cx - offset_x_pixels, phantom_cy),
    ]

    center = _roi_measurement(
        raw=raw,
        name="Center",
        cx=definitions[0][1],
        cy=definitions[0][2],
        radius_mm=radius_mm,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        center_mean=None,
    )

    rois = [center]

    for name, cx, cy in definitions[1:]:
        rois.append(
            _roi_measurement(
                raw=raw,
                name=name,
                cx=cx,
                cy=cy,
                radius_mm=radius_mm,
                row_spacing=row_spacing,
                col_spacing=col_spacing,
                center_mean=float(center["mean"]),
            )
        )

    actual_areas = [float(roi["actualAreaMm2"]) for roi in rois]

    return rois, {
        "targetAreaMm2": target_area_mm2,
        "radiusMm": round(float(radius_mm), 3),
        "diameterMm": round(float(diameter_mm), 3),
        "offsetFromCenterMm": round(float(offset_mm), 3),
        "pixelSpacingRowMm": round(float(row_spacing), 6),
        "pixelSpacingColMm": round(float(col_spacing), 6),
        "actualAreaMinMm2": round(min(actual_areas), 2),
        "actualAreaMaxMm2": round(max(actual_areas), 2),
        "actualAreaMeanMm2": round(float(np.mean(actual_areas)), 2),
    }


def _overall_result(rois: list[dict[str, Any]]) -> tuple[str, float]:
    peripheral = rois[1:]
    worst = max(abs(float(roi["diffFromCenter"])) for roi in peripheral)

    if worst <= 5.0:
        return "PASS", float(worst)

    if worst <= 7.0:
        return "MINOR DEFICIENCY", float(worst)

    return "MAJOR DEFICIENCY", float(worst)


def _validate_verified_target(
    classification: dict[str, Any],
    slices: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    best = classification.get("bestModule3Slice")
    probable = classification.get("probableModule3Group")

    if not probable:
        raise ValueError(
            "The four-module classifier did not identify a confident Module 3 range."
        )

    if not best:
        raise ValueError(
            "Module 3 was identified, but no exact two-BB target slice was found."
        )

    if not best.get("verifiedBestSlice"):
        raise ValueError(
            "A possible Module 3 BB slice was found, but it was not verified strongly enough. "
            "No distance line or uniformity result was generated."
        )

    slice_index = int(best.get("sliceIndex", -1))

    if slice_index < 0 or slice_index >= len(slices):
        raise ValueError("The verified Module 3 slice index is outside the loaded stack.")

    if not (
        int(probable["startSliceIndex"])
        <= slice_index
        <= int(probable["endSliceIndex"])
    ):
        raise ValueError(
            "The selected BB slice is not inside the verified Module 3 range."
        )

    for marker_name in ("markerA", "markerB"):
        marker = best.get(marker_name)

        if not marker:
            raise ValueError(f"The verified result is missing {marker_name} coordinates.")

        for coordinate in ("x", "y"):
            if coordinate not in marker:
                raise ValueError(
                    f"The verified result is missing {marker_name}.{coordinate}."
                )

    return best, slice_index


def create_integrated_module3_analysis(
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

    if classification_result is None:
        classification_result = create_acr_module_classification(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            max_size=160,
        )

    best, slice_index = _validate_verified_target(
        classification=classification_result,
        slices=slices,
    )

    selected = slices[slice_index]
    info = selected.get("info", {})

    if info.get("isColorDicom"):
        raise ValueError(
            "The selected DICOM is color/secondary-capture data. "
            "Raw-HU Module 3 analysis requires original grayscale CT DICOM."
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

    initial_phantom_cx, initial_phantom_cy, initial_phantom_radius = (
        _estimate_phantom_geometry(raw)
    )

    try:
        (
            phantom_cx,
            phantom_cy,
            phantom_radius,
            phantom_boundary_points,
            phantom_boundary_detection,
        ) = _detect_center_connected_phantom_boundary(
            raw=raw,
            initial_center_x=initial_phantom_cx,
            initial_center_y=initial_phantom_cy,
            initial_radius=initial_phantom_radius,
        )
    except Exception as boundary_error:
        # The analysis remains usable if an unusual image cannot support the
        # refined edge search, but the response explicitly reports the fallback.
        phantom_cx = float(initial_phantom_cx)
        phantom_cy = float(initial_phantom_cy)
        phantom_radius = float(initial_phantom_radius)

        phantom_boundary_points = [
            (
                float(
                    phantom_cx
                    + phantom_radius * math.cos(angle)
                ),
                float(
                    phantom_cy
                    + phantom_radius * math.sin(angle)
                ),
            )
            for angle in np.linspace(
                0.0,
                2.0 * math.pi,
                240,
                endpoint=False,
            )
        ]

        phantom_boundary_detection = {
            "method": "initial area estimate fallback",
            "fallbackUsed": True,
            "fallbackReason": str(boundary_error),
            "initialCenterX": round(
                float(initial_phantom_cx),
                3,
            ),
            "initialCenterY": round(
                float(initial_phantom_cy),
                3,
            ),
            "initialRadiusPixels": round(
                float(initial_phantom_radius),
                3,
            ),
            "refinedCenterX": round(
                float(phantom_cx),
                3,
            ),
            "refinedCenterY": round(
                float(phantom_cy),
                3,
            ),
            "refinedRadiusPixels": round(
                float(phantom_radius),
                3,
            ),
            "radiusChangePixels": 0.0,
        }

    marker_a = {
        "x": float(best["markerA"]["x"]),
        "y": float(best["markerA"]["y"]),
    }
    marker_b = {
        "x": float(best["markerB"]["x"]),
        "y": float(best["markerB"]["y"]),
    }

    for marker_name, marker in (("A", marker_a), ("B", marker_b)):
        distance_from_center = math.hypot(
            marker["x"] - phantom_cx,
            marker["y"] - phantom_cy,
        )

        if distance_from_center > phantom_radius * 0.90:
            raise ValueError(
                f"Verified BB marker {marker_name} falls outside the usable phantom interior."
            )

    delta_x_pixels = marker_a["x"] - marker_b["x"]
    delta_y_pixels = marker_a["y"] - marker_b["y"]
    distance_pixels = math.hypot(delta_x_pixels, delta_y_pixels)
    distance_mm = math.hypot(
        delta_x_pixels * col_spacing,
        delta_y_pixels * row_spacing,
    )

    rois, roi_size = _build_uniformity_rois(
        raw=raw,
        phantom_cx=phantom_cx,
        phantom_cy=phantom_cy,
        phantom_radius_pixels=phantom_radius,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
    )

    final_result, worst_difference = _overall_result(rois)

    overlay = window_pixels_to_image(
        raw,
        float(window_width),
        float(window_level),
        selected.get("photometric", "MONOCHROME2"),
    ).convert("RGB")

    draw = ImageDraw.Draw(overlay)

    if len(phantom_boundary_points) >= 3:
        draw.line(
            phantom_boundary_points
            + [phantom_boundary_points[0]],
            fill="yellow",
            width=2,
            joint="curve",
        )
    else:
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

    for marker_name, marker in (("A", marker_a), ("B", marker_b)):
        marker_radius = 6
        draw.ellipse(
            [
                marker["x"] - marker_radius,
                marker["y"] - marker_radius,
                marker["x"] + marker_radius,
                marker["y"] + marker_radius,
            ],
            outline="cyan",
            width=3,
        )
        draw.text(
            (marker["x"] + 8, marker["y"] - 12),
            f"BB {marker_name}",
            fill="cyan",
        )

    line_mid_x = (marker_a["x"] + marker_b["x"]) / 2.0
    line_mid_y = (marker_a["y"] + marker_b["y"]) / 2.0

    draw.text(
        (line_mid_x + 8, line_mid_y + 8),
        f"{distance_mm:.2f} mm",
        fill="cyan",
    )

    for roi in rois:
        cx = float(roi["cx"])
        cy = float(roi["cy"])
        radius_x = float(roi["radiusX"])
        radius_y = float(roi["radiusY"])

        if roi["result"] == "MAJOR DEFICIENCY":
            color = "red"
        elif roi["result"] == "MINOR DEFICIENCY":
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
            (cx + radius_x + 4, cy - radius_y),
            roi["name"],
            fill=color,
        )

    probable = classification_result["probableModule3Group"]

    overlay_data = _image_to_data_url(overlay)

    return {
        "success": True,
        "analysisType": "Automatic ACR Module 3 Full Analysis",
        "analysisVersion": FULL_ANALYSIS_VERSION,
        "classifierVersion": classification_result.get(
            "classifierVersion",
            CLASSIFIER_VERSION,
        ),
        "sliceCount": len(slices),
        "selectedSliceIndex": int(slice_index),
        "selectedSliceNumber": int(slice_index) + 1,
        "selectedSliceLabel": selected.get("label", ""),
        "module3Range": {
            "startSliceIndex": int(probable["startSliceIndex"]),
            "endSliceIndex": int(probable["endSliceIndex"]),
            "startSliceNumber": int(probable["startSliceNumber"]),
            "endSliceNumber": int(probable["endSliceNumber"]),
            "sliceCount": int(probable["sliceCount"]),
        },
        "bbTargetScore": round(float(best.get("targetScore", 0.0)), 2),
        "bbScoreGapToSecond": round(
            float(best.get("scoreGapToSecond", 0.0)),
            2,
        ),
        "adjacentSupportSlices": list(best.get("adjacentSupportSlices", [])),
        "markerA": {
            "x": round(float(marker_a["x"]), 3),
            "y": round(float(marker_a["y"]), 3),
        },
        "markerB": {
            "x": round(float(marker_b["x"]), 3),
            "y": round(float(marker_b["y"]), 3),
        },
        "bbDistance": {
            "pixels": round(float(distance_pixels), 3),
            "mm": round(float(distance_mm), 3),
        },
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
            "boundaryDetection": phantom_boundary_detection,
        },
        "roiSize": roi_size,
        "rois": rois,
        "maxAbsDiffFromCenter": round(float(worst_difference), 3),
        "finalResult": final_result,
        "result": final_result,
        "displayWindow": {
            "windowWidth": float(window_width),
            "windowLevel": float(window_level),
            "note": (
                "WW/WL controls only the displayed overlay. Module sorting, "
                "BB verification, distance, and ROI statistics use raw CT values."
            ),
        },
        "overlayImage": overlay_data,
        "image": overlay_data,
        "classificationSummary": {
            "counts": classification_result.get("counts", {}),
            "processingTimeMs": classification_result.get(
                "processingTimeMs",
                None,
            ),
            "verifiedBestSlice": True,
        },
        "criteriaNote": (
            "Each ROI targets 400 mm² using DICOM PixelSpacing. "
            "The center ROI is the reference. PASS requires every peripheral "
            "mean to be within ±5 HU of the center. Differences >5 HU and ≤7 HU "
            "are MINOR DEFICIENCY; differences >7 HU are MAJOR DEFICIENCY."
        ),
    }
