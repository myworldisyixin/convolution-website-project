from __future__ import annotations

from collections import OrderedDict
import time

from flask import Blueprint, jsonify, request

from services.acr_module3_full_analysis import (
    create_integrated_module3_analysis,
)
from services.acr_module_classifier import (
    create_acr_module_classification,
)
from services.acr_module4_high_contrast import analyze_module4_high_contrast_slice
from services.dicom_display import _get_slices_from_stack_or_upload


module_classifier_bp = Blueprint(
    "module_classifier",
    __name__,
)


_CLASSIFICATION_CACHE = OrderedDict()
_MAX_CACHE_ITEMS = 8
_MODULE_4_KEY = "MODULE_4_HIGH_CONTRAST"


def _uploaded_file():
    return (
        request.files.get("image")
        or request.files.get("file")
        or request.files.get("dicom")
        or request.files.get("dicomFile")
    )


def _remember_classification(stack_id, result):
    if not stack_id:
        return

    key = str(stack_id)
    _CLASSIFICATION_CACHE.pop(key, None)
    _CLASSIFICATION_CACHE[key] = result

    while len(_CLASSIFICATION_CACHE) > _MAX_CACHE_ITEMS:
        _CLASSIFICATION_CACHE.popitem(last=False)


def _cached_classification(stack_id):
    if not stack_id:
        return None

    key = str(stack_id)
    result = _CLASSIFICATION_CACHE.get(key)

    if result is not None:
        _CLASSIFICATION_CACHE.move_to_end(key)

    return result


def _module4_route_candidate(slice_record):
    """Adapt shared classifier evidence for the Module 4 result dashboard."""
    scores = slice_record.get("scores") or {}
    evidence = slice_record.get("module4Evidence") or {}
    reasons = (slice_record.get("reasons") or {}).get(_MODULE_4_KEY) or []
    module4_score = float(scores.get(_MODULE_4_KEY, 0.0) or 0.0) / 100.0
    return {
        "slice_number": int(slice_record.get("sliceNumber", 0) or 0),
        "slice_index": int(slice_record.get("sliceIndex", 0) or 0),
        "predicted_module": slice_record.get("predictionLabel") or "Unknown",
        "prediction_key": slice_record.get("prediction"),
        "module4_score": round(module4_score, 4),
        "base_module4_score": evidence.get("base_module4_score", module4_score),
        "outer_bb_score": evidence.get("outer_4bb_score", 0.0),
        "outer_bbs_detected": evidence.get("outer_bbs_detected", 0),
        "cardinal_markers_found": evidence.get("cardinal_markers_found", {}),
        "detected_outer_bbs": evidence.get("detected_outer_bbs", []),
        "geometry_score": evidence.get("geometry_score", 0.0),
        "final_score": evidence.get("final_module4_score", module4_score),
        "score_cap_applied": evidence.get("score_cap_applied", False),
        "score_cap": evidence.get("score_cap"),
        "cap_reason": evidence.get("cap_reason", ""),
        "selected": False,
        "reason": "; ".join(str(reason) for reason in reasons)
        or "No additional Module 4 classifier reason was recorded.",
    }


@module_classifier_bp.route(
    "/dicom-module-classification",
    methods=["POST"],
)
def dicom_module_classification():
    try:
        stack_id = (
            request.form.get("stack_id")
            or request.form.get("stackId")
        )
        uploaded_file = _uploaded_file()

        try:
            max_size = int(request.form.get("max_size", "160"))
        except Exception:
            max_size = 160

        max_size = max(96, min(max_size, 224))

        result = create_acr_module_classification(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            max_size=max_size,
        )

        _remember_classification(stack_id, result)
        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400


@module_classifier_bp.route(
    "/dicom-module3-full-analysis",
    methods=["POST"],
)
def dicom_module3_full_analysis():
    try:
        stack_id = (
            request.form.get("stack_id")
            or request.form.get("stackId")
        )
        uploaded_file = _uploaded_file()

        try:
            window_width = float(
                request.form.get("window_width", "400")
            )
        except Exception:
            window_width = 400.0

        try:
            window_level = float(
                request.form.get("window_level", "40")
            )
        except Exception:
            window_level = 40.0

        classification = _cached_classification(stack_id)

        if classification is None:
            classification = create_acr_module_classification(
                stack_id=stack_id,
                uploaded_file=uploaded_file,
                max_size=160,
            )
            _remember_classification(stack_id, classification)

        result = create_integrated_module3_analysis(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            window_width=window_width,
            window_level=window_level,
            classification_result=classification,
        )

        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400

# BEGIN MODULE1 CT NUMBER ANALYSIS ROUTE
@module_classifier_bp.route(
    "/dicom-module1-ct-number-analysis",
    methods=["POST"],
)
def dicom_module1_ct_number_analysis():
    try:
        stack_id = (
            request.form.get("stack_id")
            or request.form.get("stackId")
        )
        uploaded_file = _uploaded_file()

        try:
            window_width = float(
                request.form.get("window_width", "400")
            )
        except Exception:
            window_width = 400.0

        try:
            window_level = float(
                request.form.get("window_level", "40")
            )
        except Exception:
            window_level = 40.0

        classification = _cached_classification(stack_id)

        if classification is None:
            classification = create_acr_module_classification(
                stack_id=stack_id,
                uploaded_file=uploaded_file,
                max_size=160,
            )
            _remember_classification(stack_id, classification)

        from services.acr_module1_ct_number import (
            create_module1_ct_number_analysis,
        )

        result = create_module1_ct_number_analysis(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            window_width=window_width,
            window_level=window_level,
            classification_result=classification,
        )

        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400
# END MODULE1 CT NUMBER ANALYSIS ROUTE


@module_classifier_bp.route(
    "/dicom-module4-high-contrast-analysis",
    methods=["POST"],
)
def dicom_module4_high_contrast_analysis():
    """Adapt the shared Split Modules classifier output for Module 4 results."""
    route_started = time.perf_counter()
    slice_selection_started = route_started
    try:
        mode = request.form.get("mode", "selected").strip().lower()
        if mode not in {"selected", "stack"}:
            mode = "selected"
        performance_mode = request.form.get(
            "performance_mode", "fast"
        ).strip().lower()
        if performance_mode not in {"fast", "debug"}:
            performance_mode = "fast"
        debug_overlay_enabled = (
            performance_mode == "debug"
            and request.form.get("debug_overlay", "1").strip().lower()
            not in {"0", "false", "no"}
        )

        stack_id = (
            request.form.get("stack_id")
            or request.form.get("stackId")
        )
        uploaded_file = _uploaded_file()

        try:
            selected_slice_index = int(request.form.get("slice_index", "0"))
        except Exception:
            selected_slice_index = 0
        try:
            window_width = float(request.form.get("window_width", "400"))
            window_level = float(request.form.get("window_level", "40"))
        except Exception:
            window_width, window_level = 400.0, 40.0

        classification = _cached_classification(stack_id)
        if classification is None:
            classification = create_acr_module_classification(
                stack_id=stack_id,
                uploaded_file=uploaded_file,
                max_size=160,
            )
            _remember_classification(stack_id, classification)

        slice_records = classification.get("slices") or []
        adapted = [_module4_route_candidate(record) for record in slice_records]
        if mode == "selected":
            if selected_slice_index < 0 or selected_slice_index >= len(adapted):
                raise ValueError(
                    f"Selected slice index {selected_slice_index} is outside the available stack."
                )
            candidates = [adapted[selected_slice_index]]
            eliminated = []
            selected = candidates[0]
            status = "selected_slice_review"
        else:
            candidates = [
                item for item in adapted
                if item["prediction_key"] == _MODULE_4_KEY
            ]
            candidates.sort(key=lambda item: item["module4_score"], reverse=True)
            eliminated = [
                {
                    **item,
                    "reason": (
                        "Excluded from Module 4 route ranking because the shared "
                        f"classifier predicts {item['predicted_module']}."
                    ),
                }
                for item in adapted
                if item["prediction_key"] != _MODULE_4_KEY
            ]
            selected = candidates[0] if candidates else None
            status = "located" if selected else "not_found"

        if selected:
            selected["selected"] = True
        if selected is None:
            raise ValueError("Split Modules did not predict a Module 4 slice.")
        selected_slice = selected["slice_number"] if selected else None
        selected_index = selected["slice_index"] if selected else None
        confidence = selected["final_score"] if selected else 0.0
        source_slices = _get_slices_from_stack_or_upload(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
        )
        if selected_index < 0 or selected_index >= len(source_slices):
            raise ValueError("Selected Module 4 slice is outside the loaded stack.")
        selected_source = source_slices[selected_index]
        split_module_slice_selection_ms = (
            time.perf_counter() - slice_selection_started
        ) * 1000.0
        selected_info = selected_source.get("info") or {}
        module4_analysis = analyze_module4_high_contrast_slice(
            selected_source["pixels"],
            window_width=window_width,
            window_level=window_level,
            photometric=selected_source.get("photometric", ""),
            pixel_spacing=(
                selected_info.get("pixelSpacingRow"),
                selected_info.get("pixelSpacingCol"),
            ),
            performance_mode=performance_mode,
            debug_overlay_enabled=debug_overlay_enabled,
        )
        raw_squares = module4_analysis["candidates"]
        squares = []
        for target in raw_squares:
            final_roi = target.get("final_roi") or {}
            squares.append({
                "id": target.get("id"),
                "display_label": target.get(
                    "display_label", target.get("id")
                ),
                "roi_source": target.get("roi_source"),
                "final_center": final_roi.get(
                    "final_center", target.get("final_center")
                ),
                "final_side_px": final_roi.get(
                    "final_side_px", target.get("final_side_px")
                ),
                "final_angle_degrees": final_roi.get(
                    "final_angle_degrees",
                    target.get("final_angle_degrees"),
                ),
                "final_corners": final_roi.get(
                    "final_corners", target.get("final_corners")
                ),
                "final_roi": final_roi,
                "location_confidence": target.get(
                    "location_confidence"
                ),
                "geometry_status": target.get("geometry_status"),
                "draw_on_overlay": target.get("draw_on_overlay", True),
            })
        full_geometry = module4_analysis.get("geometry", {})
        if performance_mode == "fast":
            local_fit = full_geometry.get(
                "geometry_local_square_fit", {}
            )
            compact_local_fit = {
                key: value for key, value in local_fit.items()
                if key not in {"target_rois", "target_debug"}
            }
            module4_analysis["geometry"] = {
                "geometry_source": full_geometry.get("geometry_source"),
                "geometry_step": full_geometry.get("geometry_step"),
                "geometry_status": full_geometry.get("geometry_status"),
                "geometry_stage": full_geometry.get("geometry_stage"),
                "geometry_reason": full_geometry.get("geometry_reason"),
                "phantom_center": full_geometry.get("phantom_center"),
                "phantom_radius": full_geometry.get("phantom_radius"),
                "phantom_detection_status": full_geometry.get(
                    "phantom_detection_status"
                ),
                "pin_detection_status": full_geometry.get(
                    "pin_detection_status"
                ),
                "orientation_status": full_geometry.get(
                    "orientation_status"
                ),
                "location_confidence": full_geometry.get(
                    "location_confidence"
                ),
                "geometry_calibration": {
                    key: full_geometry.get(
                        "geometry_calibration", {}
                    ).get(key)
                    for key in (
                        "calibration_status",
                        "calibration_reason",
                        "geometry_calibration_fast_mode",
                        "geometry_calibration_skipped_or_capped",
                        "geometry_calibration_skip_reason",
                        "fast_radius_calibration_enabled",
                        "fast_radius_calibration_method",
                        "default_insert_radius_ratio",
                        "fast_calibrated_insert_radius_ratio",
                        "fast_radius_search_range",
                        "fast_radius_search_step",
                        "fast_radius_score",
                        "fast_radius_confidence",
                        "fast_radius_targets_used",
                        "fast_phi_calibration_enabled",
                        "fast_calibrated_phi_offset_degrees",
                        "fast_radius_calibration_ms",
                    )
                },
                "geometry_calibration_fast_mode": full_geometry.get(
                    "geometry_calibration_fast_mode"
                ),
                "geometry_calibration_skipped_or_capped": (
                    full_geometry.get(
                        "geometry_calibration_skipped_or_capped"
                    )
                ),
                "geometry_calibration_skip_reason": full_geometry.get(
                    "geometry_calibration_skip_reason"
                ),
                "geometry_ring_center_fit": {
                    "fit_status": full_geometry.get(
                        "geometry_ring_center_fit", {}
                    ).get("fit_status")
                },
                "geometry_full_fit": {
                    "fit_status": full_geometry.get(
                        "geometry_full_fit", {}
                    ).get("fit_status")
                },
                "geometry_angle_fit": {
                    "angle_fit_status": full_geometry.get(
                        "geometry_angle_fit", {}
                    ).get("angle_fit_status")
                },
                "geometry_local_square_fit": compact_local_fit,
                "performance": full_geometry.get("performance", {}),
            }
        detection_status = module4_analysis.get("analysis_review_status", "needs_review")
        result = {
            "success": True,
            "implemented": "partial",
            "module": "Module 4",
            "analysis": "High-Contrast Resolution",
            "mode": mode,
            "status": detection_status,
            "location_status": (
                "complete" if len(squares) == 8 else "needs_review"
            ),
            "roi_location_status": detection_status,
            "targets_total": 8,
            "targets_found": len(squares),
            "targets": squares,
            "std_measurements": module4_analysis.get(
                "std_measurements", []
            ),
            "global_noise_reference": module4_analysis.get(
                "global_noise_reference", {}
            ),
            "std_method_summary": module4_analysis.get(
                "std_method_summary", {}
            ),
            "performance": module4_analysis.get("performance", {}),
            "selection_source": (
                "selected_slice"
                if mode == "selected"
                else "split_modules_highest_module4_score"
            ),
            "selected_by": (
                "Selected slice"
                if mode == "selected" else "Split Modules"
            ),
            "selected_slice": selected_slice,
            "slice_index": selected_index,
            "selected_slice_index": selected_index,
            "confidence": confidence,
            "square_detection_status": detection_status,
            "squares_detected": len(squares),
            "square_candidates": squares,
            "geometry_source": module4_analysis.get("geometry_source"),
            "geometry_step": (
                module4_analysis.get("geometry", {}).get("geometry_step")
            ),
            "geometry": module4_analysis.get("geometry", {}),
            "geometry_calibration": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_calibration", {}
                )
            ),
            "geometry_calibration_review": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_calibration_review", {}
                )
            ),
            "geometry_location_refinement": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_location_refinement", {}
                )
            ),
            "geometry_ring_center_fit": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_ring_center_fit", {}
                )
            ),
            "geometry_full_fit": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_full_fit", {}
                )
            ),
            "geometry_local_square_fit": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_local_square_fit", {}
                )
            ),
            "final_roi_stage_used_by_overlay": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_local_square_fit", {}
                ).get("final_roi_stage_used_by_overlay")
            ),
            "final_roi_corner_field_used_by_overlay": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_local_square_fit", {}
                ).get("final_roi_corner_field_used_by_overlay")
            ),
            "table_roi_stage_used": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_local_square_fit", {}
                ).get("table_roi_stage_used")
            ),
            "debug_overlay_stage_used": (
                module4_analysis.get("geometry", {}).get(
                    "geometry_local_square_fit", {}
                ).get("debug_overlay_stage_used")
            ),
            "location_confidence": (
                module4_analysis.get("geometry", {}).get(
                    "location_confidence", "needs_review"
                )
            ),
            "missing_targets": module4_analysis.get("missing_targets", []),
            "weak_targets": module4_analysis.get("weak_targets", []),
            "detected_targets": module4_analysis.get("detected_targets", []),
            "stage_summary": module4_analysis.get("stage_summary", {}),
            "summary": {
                "module_location": (
                    "Used the current selected slice."
                    if mode == "selected"
                    else "Selected highest-scoring Module 4 slice from Split Modules."
                ),
                "selection_method": (
                    "Selected highest-scoring Module 4 slice from Split Modules classifier."
                ),
                "square_detection_note": (
                    "Geometry predicted all eight target regions, then bounded "
                    "local raw-HU searches fitted each visible square insert."
                ),
                "measurement_note": (
                    "No ROI scoring, pass/fail, or lp/cm measurement is "
                    "performed during location review."
                ),
                "review_note": (
                    "All B1-B8 ROIs remain drawn. Weak local fits retain an "
                    "explicit geometry fallback and needs-review status."
                ),
            },
            "candidate_slices": candidates,
            "eliminated_slices": eliminated,
            "display": {
                "overlay_image": module4_analysis["overlay_image"],
                "location_debug_overlay": module4_analysis.get(
                    "location_debug_overlay"
                ),
                "selected_slice_image": module4_analysis["selected_slice_image"],
                "std_measurement_debug_overlay": module4_analysis.get(
                    "std_measurement_debug_overlay"
                ),
                "noise_debug_overlay": module4_analysis.get(
                    "noise_debug_overlay"
                ),
                "window_width": module4_analysis["window_width"],
                "window_level": module4_analysis["window_level"],
                "window_method": module4_analysis["display_window_method"],
                "window_quality_score": module4_analysis["display_window_quality_score"],
                "window_note": module4_analysis["display_window_note"],
            },
            "debug": {
                "classifier_used": True,
                "classifier_version": classification.get("classifierVersion"),
                "outer_4bb_scoring_used": bool(
                    selected and selected.get("outer_bbs_detected") is not None
                ),
                "selection_source": (
                    "selected_slice"
                    if mode == "selected"
                    else "split_modules_highest_module4_score"
                ),
                "module4_outer_bb_runtime_ms": classification.get("module4OuterBbRuntimeMs"),
                "module4_outer_bb_slices_evaluated": classification.get("module4OuterBbSlicesEvaluated"),
                "module4_outer_bb_slices_skipped": classification.get("module4OuterBbSlicesSkipped"),
                "module4_outer_bb_candidate_indices": classification.get("module4OuterBbCandidateIndices", []),
                "selection_reason": selected["reason"] if selected else "The shared classifier produced no Module 4 prediction.",
                "analysis_path": module4_analysis.get("analysis_path"),
                "geometry_source": module4_analysis.get("geometry_source"),
                "geometry": module4_analysis.get("geometry", {}),
                "legacy_path": module4_analysis.get("legacy_path", {}),
                "raw_candidate_count": len(candidates),
                "raw_candidates": candidates,
                "detected_outer_bbs": selected["detected_outer_bbs"] if selected else [],
                "phantom_center": module4_analysis["phantom_center"],
                "phantom_radius": module4_analysis["phantom_radius"],
                "thresholds": module4_analysis["thresholds"],
                "components_considered": module4_analysis["components_considered"],
                "components_rejected": module4_analysis["components_rejected"],
                "merge_groups": module4_analysis.get("merge_groups", []),
                "merge_kernel_size": module4_analysis.get("merge_kernel_size"),
                "median_block_side_px": module4_analysis.get("median_block_side_px"),
                "expected_ring_radius": module4_analysis.get("expected_ring_radius"),
                "internal_ring_radius_px": module4_analysis.get("internal_ring_radius_px"),
                "common_side_px": module4_analysis.get("common_side_px"),
                "side_votes": module4_analysis.get("side_votes", []),
                "global_target_angle_degrees": module4_analysis.get("global_target_angle_degrees"),
                "angle_votes": module4_analysis.get("angle_votes", []),
                "anchor_targets_used": module4_analysis.get("anchor_targets_used", []),
                "strong_anchor_count": module4_analysis.get("strong_anchor_count", 0),
                "orientation_confidence": module4_analysis.get("orientation_confidence"),
                "global_layout_model_used": module4_analysis.get("global_layout_model_used", False),
                "layout_needs_review": module4_analysis.get("layout_needs_review", True),
                "target_slots": module4_analysis.get("target_slots", []),
                "missing_slots": module4_analysis.get("missing_slots", []),
                "weak_slots": module4_analysis.get("weak_slots", []),
                "rejected_candidates": module4_analysis.get("rejected_candidates", []),
                "assignment_scores": module4_analysis.get("assignment_scores", []),
                "global_layout_quality": module4_analysis.get("global_layout_quality"),
                "pixel_spacing": module4_analysis.get("pixel_spacing"),
                "phantom_detection_status": module4_analysis.get("phantom_detection_status"),
                "detection_passes_tried": module4_analysis.get("detection_passes_tried", []),
                "detection_passes_completed": module4_analysis.get("detection_passes_completed", []),
                "all_detection_passes_failed": module4_analysis.get("all_detection_passes_failed", False),
                "selected_detection_pass": module4_analysis.get("selected_detection_pass"),
                "candidates_per_pass": module4_analysis.get("candidates_per_pass", {}),
                "rejection_counts_per_pass": module4_analysis.get("rejection_counts_per_pass", {}),
                "detection_pass_quality": module4_analysis.get("detection_pass_quality", {}),
                "detection_pass_errors": module4_analysis.get("detection_pass_errors", {}),
                "detection_selection_reason": module4_analysis.get("detection_selection_reason"),
                "analysis_review_status": detection_status,
                "all_block_debug_enabled": module4_analysis.get("all_block_debug_enabled", False),
                "debug_show_reference_targets": module4_analysis.get(
                    "debug_show_reference_targets", False
                ),
                "target_processing_order": module4_analysis.get(
                    "target_processing_order", []
                ),
                "target_debug": module4_analysis.get("target_debug", []),
                "b6_focused_debug": module4_analysis.get(
                    "b6_focused_debug", {}
                ),
                "missing_targets": module4_analysis.get("missing_targets", []),
                "weak_targets": module4_analysis.get("weak_targets", []),
                "detected_targets": module4_analysis.get("detected_targets", []),
                "slot_lifecycle": module4_analysis.get("slot_lifecycle", []),
                "stage_summary": module4_analysis.get("stage_summary", {}),
                "rejection_summary": module4_analysis.get("rejection_summary", {}),
                "display_window_method": module4_analysis["display_window_method"],
                "display_window_quality_score": module4_analysis["display_window_quality_score"],
                "display_window_candidates": module4_analysis["display_window_candidates"],
                "display_window_width": module4_analysis["window_width"],
                "display_window_level": module4_analysis["window_level"],
            },
        }
        route_serialization_ms = (
            time.perf_counter() - route_started
        ) * 1000.0 - split_module_slice_selection_ms - float(
            module4_analysis.get("performance", {}).get("total_ms", 0.0)
        )
        result["performance"]["stage_timings_ms"][
            "split_module_slice_selection_ms"
        ] = round(split_module_slice_selection_ms, 2)
        result["performance"]["stage_timings_ms"][
            "route_serialization_ms"
        ] = round(max(route_serialization_ms, 0.0), 2)
        result["performance"]["route_total_ms"] = round(
            (time.perf_counter() - route_started) * 1000.0, 2
        )
        if performance_mode == "fast":
            result["debug"] = {
                "selection_reason": (
                    selected["reason"] if selected
                    else "No Module 4 prediction."
                ),
                "analysis_path": module4_analysis.get("analysis_path"),
                "performance": result["performance"],
                "geometry_source": module4_analysis.get("geometry_source"),
                "final_roi_corner_field_used_by_overlay": result.get(
                    "final_roi_corner_field_used_by_overlay"
                ),
            }
        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "success": False,
            "implemented": "partial",
            "module": "Module 4",
            "analysis": "High-Contrast Resolution",
            "mode": request.form.get("mode", "selected"),
            "status": "error",
            "measurement_status": "not_implemented",
            "message": str(exc),
            "candidate_slices": [],
            "eliminated_slices": [],
            "display": {
                "overlay_image": None,
                "selected_slice_image": None,
            },
            "debug": {
                "classifier_used": True,
                "outer_4bb_scoring_used": False,
                "selection_reason": "Module 4 location request failed.",
                "raw_candidates": [],
            },
        }), 400
