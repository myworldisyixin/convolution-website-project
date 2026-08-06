from __future__ import annotations

from pathlib import Path
import re


REQUIRED_FILES = [
    "templates/partials/script_parts/ui_registry.html",
    "templates/partials/script_parts/ui_feature_manifest.html",
    "templates/partials/script_parts/ui_qol.html",
    "templates/partials/script_parts/features/matrix_manifest.html",
    "templates/partials/script_parts/features/dicom_viewer_manifest.html",
    "templates/partials/script_parts/features/acr_modules_manifest.html",
    "templates/partials/script_groups/core_scripts.html",
    "templates/partials/script_groups/dicom_viewer_scripts.html",
    "templates/partials/script_groups/acr_module_scripts.html",
    "templates/partials/script_groups/matrix_scripts.html",
    "templates/partials/script_parts/ui_result_renderers.html",
    "templates/partials/script_parts/dicom_panels.html",
    "templates/partials/dicom/dicom_tabs.html",
    "templates/partials/tool_cards/matrix_card.html",
    "templates/partials/tool_cards/dicom_roi_card.html",
    "templates/partials/tool_cards/module3_uniformity_card.html",
    "templates/partials/tool_cards/module1_ct_number_card.html",
    "templates/partials/tool_cards/module4_high_contrast_card.html",
    "templates/partials/acr_modules/module3_uniformity_panel.html",
    "templates/partials/acr_modules/module1_ct_number_panel.html",
    "templates/partials/acr_modules/module4_high_contrast_panel.html",
    "templates/partials/script_parts/dicom_module4.html",
    "templates/partials/dicom/panels/info_panel.html",
    "templates/partials/dicom/panels/window_panel.html",
    "templates/partials/dicom/panels/roi_panel.html",
    "templates/partials/dicom/panels/results_panel.html",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def scan_html_files(root: Path) -> list[Path]:
    templates = root / "templates"
    if not templates.exists():
        return []
    return sorted(path for path in templates.rglob("*.html") if "_safe_backups" not in path.parts)


def count_pattern(files: list[Path], pattern: str, flags: int = re.IGNORECASE) -> int:
    total = 0
    rx = re.compile(pattern, flags)
    for path in files:
        total += len(rx.findall(read(path)))
    return total


def collect_ids(files: list[Path]) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    rx = re.compile(r"\bid\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    for path in files:
        # Script partials contain HTML template strings rendered at runtime.
        # They are not static DOM nodes and should not create static-ID warnings.
        if "script_parts" in path.parts:
            continue
        for match in rx.finditer(read(path)):
            ids.setdefault(match.group(1), []).append(str(path))
    return ids


def collect_includes(files: list[Path]) -> dict[str, list[str]]:
    includes: dict[str, list[str]] = {}
    rx = re.compile(r"{%\s*include\s+['\"]([^'\"]+)['\"]\s*%}")
    for path in files:
        for match in rx.finditer(read(path)):
            includes.setdefault(match.group(1), []).append(str(path))
    return includes


def check_js_syntax(root: Path, files: list[Path]) -> list[tuple[str, str]]:
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        return [("WARN", "Node was not found; JS syntax checks skipped.")]

    results: list[tuple[str, str]] = []

    for path in files:
        text = read(path)
        text = re.sub(
            r"{%\s*include\s+['\"][^'\"]+['\"]\s*%}",
            "",
            text,
        )

        if "<script" in text.lower():
            scripts = re.findall(r"<script\b[^>]*>([\s\S]*?)</script>", text, flags=re.IGNORECASE)
            if scripts:
                text = "\n\n".join(scripts)

        if not text.strip():
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)

        try:
            proc = subprocess.run([node, "--check", str(tmp_path)], cwd=str(root), text=True, capture_output=True)
            if proc.returncode != 0:
                results.append(("FAIL", f"JS syntax failed: {path.relative_to(root)}\n{proc.stderr.strip()}"))
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    return results


def main() -> int:
    root = Path.cwd()

    if not (root / "app.py").exists():
        print("ERROR: run from the project root that contains app.py")
        return 2

    html_files = scan_html_files(root)
    script_dir = root / "templates" / "partials" / "script_parts"
    script_files = sorted(script_dir.rglob("*.html")) if script_dir.exists() else []

    messages: list[tuple[str, str]] = []

    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            messages.append(("FAIL", f"Missing required architecture file: {rel}"))

    includes = collect_includes(html_files)
    for include_path, sources in sorted(includes.items()):
        target = root / "templates" / include_path
        if not target.exists():
            messages.append(("FAIL", f"Broken Jinja include: {include_path} used in {len(sources)} file(s)"))

    ids = collect_ids(html_files)
    tab_keys = {key.replace("tab-", "") for key in ids if key.startswith("tab-")}
    panel_keys = {key.replace("panel-", "") for key in ids if key.startswith("panel-")}

    tabs_without_panels = sorted(tab_keys - panel_keys)
    panels_without_tabs = sorted(key for key in panel_keys - tab_keys if key not in {"matrix"})

    if tabs_without_panels:
        messages.append(("FAIL", "Tabs without panels: " + ", ".join(tabs_without_panels)))
    if panels_without_tabs:
        messages.append(("WARN", "Panels without tabs: " + ", ".join(panels_without_tabs)))

    duplicate_ids = {
        key: locations
        for key, locations in ids.items()
        if len(locations) > 1 and not key.startswith("preset-")
    }
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids.keys())[:12])
        messages.append(("WARN", f"Duplicate static ids found: {preview}"))

    expected_tabs = {"info", "window", "roi", "auto", "module1", "module4", "results"}
    expected_panels = {"info", "window", "roi", "auto", "module1", "module4", "results"}
    missing_expected_tabs = sorted(expected_tabs - tab_keys)
    missing_expected_panels = sorted(expected_panels - panel_keys)
    if missing_expected_tabs:
        messages.append(("FAIL", "Missing expected DICOM tabs: " + ", ".join(missing_expected_tabs)))
    if missing_expected_panels:
        messages.append(("FAIL", "Missing expected DICOM panels: " + ", ".join(missing_expected_panels)))

    for required_id in ("panel-module4", "acrModule4Status", "acrModule4ResultArea"):
        locations = ids.get(required_id, [])
        if len(locations) != 1:
            messages.append((
                "FAIL",
                f"Expected exactly one static {required_id}; found {len(locations)}.",
            ))

    module4_panel = root / "templates" / "partials" / "acr_modules" / "module4_high_contrast_panel.html"
    if module4_panel.exists():
        module4_panel_text = read(module4_panel)
        for action_key in ("module4_selected", "module4_stack", "split_modules"):
            action_count = len(re.findall(
                rf"\bdata-action\s*=\s*['\"]{re.escape(action_key)}['\"]",
                module4_panel_text,
            ))
            if action_count != 1:
                messages.append((
                    "FAIL",
                    f"Expected one Module 4 {action_key} button; found {action_count}.",
                ))

    acr_manifest = root / "templates" / "partials" / "script_parts" / "features" / "acr_modules_manifest.html"
    if acr_manifest.exists():
        acr_manifest_text = read(acr_manifest)
        for action_key in (
            "module4_selected", "module4_stack", "clear_module4_results",
        ):
            registration_count = len(re.findall(
                rf"\bkey\s*:\s*['\"]{re.escape(action_key)}['\"]",
                acr_manifest_text,
            ))
            if registration_count != 1:
                messages.append((
                    "FAIL",
                    f"Expected one registered {action_key} action; found {registration_count}.",
                ))
        route_registration_count = len(re.findall(
            r"\bkey\s*:\s*['\"]acr_module4_high_contrast['\"]",
            acr_manifest_text,
        ))
        if route_registration_count != 1:
            messages.append((
                "FAIL",
                "Expected one registered acr_module4_high_contrast route; "
                f"found {route_registration_count}.",
            ))

    module_routes = root / "routes" / "module_classifier_routes.py"
    if module_routes.exists():
        module_routes_text = read(module_routes)
        endpoint_count = module_routes_text.count("/dicom-module4-high-contrast-analysis")
        if endpoint_count != 1:
            messages.append((
                "FAIL",
                "Expected one Module 4 partial-analysis endpoint; "
                f"found {endpoint_count}.",
            ))
        for expected_source in (
            "_cached_classification(stack_id)",
            "_module4_route_candidate(",
            '"split_modules_highest_module4_score"',
            "analyze_module4_high_contrast_slice(",
            '"square_candidates": squares',
            '"location_status": (',
            '"targets_total": 8',
            '"targets_found": len(squares)',
            '"selected_by": (',
            '"std_measurements": module4_analysis.get(',
            '"global_noise_reference": module4_analysis.get(',
            '"std_method_summary": module4_analysis.get(',
        ):
            if expected_source not in module_routes_text:
                messages.append((
                    "FAIL",
                    f"Module 4 partial-location route is missing: {expected_source}",
                ))

        for forbidden_route_scoring in (
            "choose_best_module4_slice(",
            "score_module4_outer_bbs(",
            '"module4_auto_preliminary_scoring": module4_analysis.get(',
            '"module4_auto_method_scoring": module4_analysis.get(',
            '"module4_preliminary_scoring": module4_analysis.get(',
            '"module4_vote_summary": module4_analysis.get(',
            '"module4_graph_data": module4_analysis.get(',
            '"module4_threshold_tuning_diagnostics": module4_analysis.get(',
        ):
            if forbidden_route_scoring in module_routes_text:
                messages.append((
                    "FAIL",
                    f"Module 4 route still performs independent scoring: {forbidden_route_scoring}",
                ))

    module_classifier = root / "services" / "acr_module_classifier.py"
    if not module_classifier.exists():
        messages.append(("FAIL", "Shared module classifier is missing."))
    else:
        module_classifier_text = read(module_classifier)
        for expected_helper in (
            "score_module4_outer_bbs(raw)",
            'slice_record["module4Evidence"] = evidence',
            'outer_score = float(outer["outer_4bb_score"])',
            "def _module4_exact_target_score(",
            "0.25 * base_normalized + 0.75 * outer_normalized",
            "0.78),",
            "0.65),",
            "min(base_normalized * 0.55, 0.55)",
            '"score_cap_applied": bool(score_cap is not None)',
            'scores[MODULE_4] = round(final_score * 100.0, 2)',
            "evaluation_indices = set(priority[:12])",
            'result["module4OuterBbRuntimeMs"]',
            'result["module4OuterBbSlicesSkipped"]',
        ):
            if expected_helper not in module_classifier_text:
                messages.append((
                    "FAIL",
                    f"Shared classifier Module 4 refinement is missing: {expected_helper}",
                ))

    module4_service = root / "services" / "acr_module4_high_contrast.py"
    if not module4_service.exists():
        messages.append(("FAIL", "Module 4 low-level BB detector is missing."))
    else:
        module4_service_text = read(module4_service)
        if "def score_module4_outer_bbs(" not in module4_service_text:
            messages.append(("FAIL", "Module 4 low-level BB detector helper is missing."))
        for geometry_first_requirement in (
            "def detect_module4_phantom_geometry(",
            "def detect_module4_bottom_pins(",
            "def calculate_module4_pin_angle(",
            "def place_module4_geometry_rois(",
            "def calibrate_module4_geometry_rois(",
            "def correct_module4_geometry_from_primary_centers(",
            "def fit_module4_global_ring_geometry(",
            "def fit_module4_global_geometry_full(",
            "def review_module4_geometry_locations(",
            "def generate_module4_location_debug_overlay(",
            "def phantom_geometry_pin_anchored_roi(",
            '"geometry_source": "phantom_geometry_pin_anchored_roi"',
            '"analysis_path": "geometry_guided_local_square_location"',
            '"geometry_step": 4',
            '"legacy_image_locator_active": False',
            '"legacy_path_active": False',
            '"bottom_pin_anchoring_status": "implemented"',
            '"target_roi_placement_status": "implemented"',
            '"module4_scoring_status": "pending"',
            "MODULE4_INSERT_RADIUS_RATIO = 0.46",
            "MODULE4_PHI_OFFSET_DEGREES = 0.0",
            "MODULE4_ROI_SIDE_RATIO = 0.14",
            "MODULE4_ROI_ANGLE_OFFSET_DEGREES = 45.0",
            '"nominal_target_angle":',
            '"final_target_angle":',
            '"roi_corners":',
            '"inner_roi_corners":',
            '"geometry_status": geometry_status',
            '"draw_on_overlay": draw_on_overlay',
            '"method": "bounded_global_radius_phi_search"',
            '"search_radius_range": [0.55, 0.78]',
            '"search_phi_range": [-15.0, 15.0]',
            '"calibrated_insert_radius_ratio":',
            '"calibrated_phi_offset_degrees":',
            '"pre_calibration_centers":',
            '"post_calibration_centers":',
            '"center_shift_px_by_target":',
            '"normal_overlay_targets": [',
            "draw_on_overlay = True",
            '"nominal_angle_degrees":',
            '"final_angle_degrees":',
            '"evidence_score_at_center":',
            '"center_to_local_peak_delta_px":',
            '"location_confidence": confidence',
            '"location_reason": reason',
            '"local_target_center":',
            '"local_target_center_method":',
            '"local_target_score":',
            '"target_center_confidence":',
            '"center_to_local_target_delta_px":',
            '"radial_delta_px":',
            '"tangential_delta_px":',
            '"pre_correction_radius_ratio":',
            '"post_correction_radius_ratio":',
            '"radius_correction_ratio":',
            '"pre_correction_phi_degrees":',
            '"post_correction_phi_degrees":',
            '"phi_correction_degrees":',
            '"radius_correction_bound_ratio": 0.05',
            '"phi_correction_bound_degrees": 6.0',
            '"geometry_location_refinement":',
            '"method": "local_target_center_global_radius_phi_correction"',
            '"iterations": len(refinement_passes)',
            '"pre_refinement_insert_radius_ratio":',
            '"post_refinement_insert_radius_ratio":',
            '"pre_refinement_phi_offset_degrees":',
            '"post_refinement_phi_offset_degrees":',
            'target["center_to_local_target_delta_px_before"]',
            'target["center_to_local_target_delta_px_after"]',
            '"recommended_roi_side_ratio":',
            '"recommended_roi_angle_offset_degrees": None',
            '"method": "global_ring_center_radius_phi_fit"',
            '"geometry_ring_center_fit": ring_center_fit',
            '"calibrated_ring_center": calibrated_ring_center',
            '"center_offset_x_px":',
            '"center_offset_y_px":',
            '"center_offset_magnitude_px":',
            '"center_offset_limit_px":',
            '"radius_ratio_before":',
            '"radius_ratio_after":',
            '"phi_before":',
            '"phi_after":',
            '"pre_fit_median_center_error_px":',
            '"post_fit_median_center_error_px":',
            '"targets_used_for_fit":',
            '"targets_excluded_from_fit":',
            'target["pre_ring_fit_center"]',
            'target["post_ring_fit_center"]',
            'target["center_delta_before_px"]',
            'target["center_delta_after_px"]',
            '"method": "global_ring_center_radius_phi_size_angle_fit"',
            "def fit_module4_global_roi_angle(",
            '"method": "bounded_shared_roi_angle_size_polish"',
            '"fine_step_degrees": 0.25',
            '"center_freeze_enabled": True',
            '"max_center_shift_px_after_angle_size_polish": 0.0',
            '"roi_side_px_before":',
            '"roi_side_px_after":',
            '"centers_changed_by_angle_fit": False',
            '"geometry_angle_fit": angle_fit',
            "def fit_module4_geometry_guided_local_squares(",
            '"method": "geometry_guided_local_square_detection"',
            "def _module4_angle_consistency_metrics(",
            '"shared_roi_angle_degrees":',
            '"shared_angle_source_targets":',
            '"shared_angle_confidence":',
            '"shared_angle_target_count":',
            "shared_angle_confidence_threshold = 0.80",
            '"shared_angle_override_enabled":',
            '"shared_angle_override_applied":',
            '"shared_angle_override_rejected_reason":',
            '"pre_shared_angle_degrees":',
            '"post_shared_angle_degrees":',
            '"pre_consistency_angle_degrees":',
            '"local_angle_delta_from_shared":',
            '"final_angle_degrees":',
            '"angle_consistency_applied":',
            '"angle_consistency_reason":',
            "def _export_module4_fit_diagnostics(",
            '"module4_fit_diagnostics.json"',
            '"module4_fit_diagnostics.html"',
            '"module4_fit_diagnostics.png"',
            '"normal_overlay_source": "final_roi.final_corners"',
            '"fit_diagnostics_export": diagnostic_export',
            '"roi_source": roi_source',
            '"local_square_detected"',
            '"geometry_fallback_needs_review"',
            '"center_shift_limit_px":',
            '"local_search_window":',
            '"local_square_score":',
            '"pre_micro_center":',
            '"post_micro_center":',
            '"micro_center_shift_px":',
            '"micro_side_change_px":',
            '"micro_angle_change_degrees":',
            '"pre_micro_square_score":',
            '"post_micro_square_score":',
            '"micro_refinement_applied":',
            '"top_edge_score":',
            '"right_edge_score":',
            '"bottom_edge_score":',
            '"left_edge_score":',
            '"top_edge_score_raw":',
            '"right_edge_score_raw":',
            '"bottom_edge_score_raw":',
            '"left_edge_score_raw":',
            '"top_boundary_coverage":',
            '"right_boundary_coverage":',
            '"bottom_boundary_coverage":',
            '"left_boundary_coverage":',
            '"min_edge_score":',
            '"lower_quartile_edge_score":',
            '"edge_score_saturation_detected":',
            '"max_micro_center_shift_px":',
            '"max_micro_side_change_px":',
            '"max_micro_angle_change_degrees":',
            'MODULE4_TARGETED_POLISH_IDS = frozenset({"B6", "B7", "B8"})',
            "MODULE4_NOMINAL_LP_CM_BY_ID = {",
            "MODULE4_EXPECTED_LP_CM_CONFIG = {",
            "MODULE4_MANUAL_DEVELOPMENT_LP_CM_MAPPING = {",
            "MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING = False",
            "MODULE4_REQUIRED_LP_CM = None",
            "MODULE4_SUGGESTED_MAPPING_REVIEW = {",
            "MODULE4_ANALYSIS_CONFIG = {",
            'MODULE4_EXPECTED_LP_CM = MODULE4_EXPECTED_LP_CM_CONFIG["targets"]',
            '"B8": 4',
            '"B1": 12',
            '"nominal_lp_cm": MODULE4_NOMINAL_LP_CM_BY_ID[target_id]',
            'detection.get("display_label", detection["id"])',
            '"target_changed": targeted_applied',
            '"frozen_good_target": frozen_good_target',
            '"pre_targeted_center":',
            '"post_targeted_center":',
            '"targeted_center_shift_px":',
            '"targeted_side_change_px":',
            '"targeted_angle_change_degrees":',
            '"targeted_fit_score_before":',
            '"targeted_fit_score_after":',
            '"targeted_refinement_applied": targeted_applied',
            '"targeted_refinement_reason": targeted_reason',
            '"pre_targeted_fill":',
            '"post_targeted_fill":',
            '"fill_delta":',
            '"pre_targeted_leakage":',
            '"post_targeted_leakage":',
            '"leakage_delta":',
            '"pre_lower_quartile_edge":',
            '"post_lower_quartile_edge":',
            '"lower_quartile_edge_delta":',
            '"pre_min_boundary_coverage":',
            '"post_min_boundary_coverage":',
            '"min_boundary_coverage_delta":',
            '"containment_gate_passed": containment_gate_passed',
            '"targeted_refinement_candidate_score_improved":',
            '"targeted_refinement_accepted": targeted_applied',
            '"targeted_refinement_rejected": targeted_rejected',
            '"targeted_refinement_rejection_reason":',
            '"targeted_targets_changed":',
            '"frozen_good_targets":',
            '"geometry_local_square_fit": local_square_fit',
            'target["final_roi"] = {',
            '"final_corners": target["roi_corners"]',
            '"final_roi_stage_used_by_overlay": "target.final_roi.stage"',
            '"final_roi_corner_field_used_by_overlay": "target.final_roi.final_corners"',
            '"performance_mode": performance_mode',
            '"geometry_calibration_fast_mode": True',
            '"geometry_calibration_skipped_or_capped": True',
            '"geometry_calibration_skip_reason":',
            "def calibrate_module4_geometry_fast(",
            '"method": "fast_radius_then_tiny_phi_raw_hu_evidence"',
            '"fast_radius_search_range": [0.55, 0.78]',
            '"fast_radius_search_step": 0.02',
            '"fast_radius_calibration_ms":',
            '"local_candidates_evaluated":',
            '"square_hypotheses_evaluated":',
            '"micro_hypotheses_evaluated":',
            '"targeted_hypotheses_evaluated":',
            '"debug_overlay_generation_status":',
            '"total_module4_analysis_ms": performance["total_ms"]',
            "Module4 timing:",
            "def measure_module4_final_roi_standard_deviation(",
            'final_roi.get("inner_roi_corners", [])',
            '"std_ddof": 0',
            '"measurement_mask_source": "final_roi.inner_roi_corners"',
            "def generate_module4_std_measurement_debug_overlay(",
            "def estimate_module4_automatic_noise(",
            '"source": "automatic_uniform_patch_median"',
            '"noise_source": noise_source',
            '"normalized_std_ratio":',
            '"global_fallback_used"',
            "def generate_module4_noise_debug_overlay(",
            "MODULE4_STD_VISIBLE_RATIO_THRESHOLD = 3.0",
            "MODULE4_STD_REVIEW_RATIO_THRESHOLD = 2.5",
            "def apply_module4_std_visibility_decisions(",
            '"method": "standard_deviation_to_scan_specific_noise_ratio"',
            '"threshold_status": MODULE4_STD_THRESHOLD_STATUS',
            '"highest_continuously_visible_lp_cm": highest_continuous',
            '"nonmonotonic_visibility_pattern"',
            'final_roi = detection.get("final_roi", detection)',
            '"final_corners", detection.get("rotated_box", [])',
            '"geometry_full_fit": full_fit',
            '"roi_side_ratio_before":',
            '"roi_side_ratio_after":',
            '"roi_angle_offset_before":',
            '"roi_angle_offset_after":',
            '"geometry_calibration_review":',
            '"radius_assessment":',
            '"phi_assessment":',
            '"roi_size_assessment":',
            '"roi_angle_assessment":',
            '"bottom_pin_candidates":',
            '"selected_bottom_pins":',
            '"pin_midpoint":',
            '"pin_angle_degrees":',
            '"fallback_angle_used":',
            '"angle_convention":',
            '"y_axis_convention":',
            "def extract_module4_preliminary_roi_data(",
            "def generate_module4_preliminary_graph_data(",
            "def generate_module4_preliminary_votes(",
            "MODULE4_ADAPTIVE_THRESHOLD_CONFIG = {",
            "MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG = {",
            "MODULE4_ENABLE_AUTO_PRELIMINARY_SCORING = True",
            "MODULE4_AUTO_PRELIMINARY_SCORING_CONFIG = {",
            '"scoring_type": "automatic_evidence_based_resolution"',
            '"target_lp_cm_order": [',
            '"continuity_required": True',
            '"stop_at_first_unresolved_target": True',
            '"use_target_specific_thresholds": True',
            '"use_signal_curve_break_detection": True',
            '"status": "development_target_specific"',
            '"target_specific_thresholds_enabled": True',
            '"status": "development_adaptive"',
            '"adaptive_thresholds_enabled": True',
            '"fft_score_max_guardrail": 0.75',
            '"profile_score_max_guardrail": 0.75',
            '"std_score_max_guardrail": 0.75',
            "def _module4_safe_float(value)",
            "def _module4_clamp(value, low, high)",
            "def _module4_safe_median(values)",
            "def _module4_safe_mad(values)",
            "def _module4_safe_margin(actual, threshold)",
            "def _module4_safe_error_margin(max_allowed, actual_error)",
            'target["module4_preliminary_analysis"] = {',
            '"profile_graph": profile_graph',
            '"fft_graph": fft_graph',
            '"std_contrast_data": std_contrast_data',
            '"background_noise_status": background_noise_status',
            '"background_noise_reason": std_reason',
            'warnings.append("background_noise_unreliable")',
            '"module4_graph_data": module4_graph_data',
            '"module4_vote_summary": summary',
            '"module4_preliminary_resolution": resolution',
            '"module4_expected_lp_cm_config": {',
            '"module4_analysis_config": {',
            '"active_expected_lp_cm_mapping": dict(MODULE4_EXPECTED_LP_CM)',
            '"suggested_mapping": dict(MODULE4_SUGGESTED_MAPPING_REVIEW)',
            '"mapping_review": dict(MODULE4_SUGGESTED_MAPPING_REVIEW)',
            '"module4_threshold_tuning_diagnostics": threshold_tuning_diagnostics',
            '"module4_noise_context": noise_context',
            '"module4_adaptive_threshold_config": MODULE4_ADAPTIVE_THRESHOLD_CONFIG',
            '"module4_adaptive_thresholds": module4_adaptive_thresholds',
            '"adaptive_fft_thresholds": adaptive_fft_thresholds',
            '"adaptive_profile_thresholds": adaptive_profile_thresholds',
            '"adaptive_std_thresholds": adaptive_std_thresholds',
            '"adaptive_combined_thresholds": adaptive_combined_thresholds',
            'analysis["thresholds_used"] = thresholds_used',
            'analysis["threshold_margins"] = threshold_margins',
            '"spacing_error_margin_percent": _module4_safe_error_margin(',
            '"closest_to_threshold": closest_to_threshold',
            '"likely_threshold_sensitive_targets": likely_threshold_sensitive_targets',
            '"module4_target_specific_threshold_config": MODULE4_TARGET_SPECIFIC_THRESHOLD_CONFIG',
            '"module4_target_signal_contexts": target_signal_contexts',
            '"module4_target_adaptive_thresholds": target_adaptive_thresholds',
            '"module4_target_threshold_review": target_threshold_review',
            '"module4_target_threshold_validation": module4_target_threshold_validation',
            '"module4_threshold_quality_review": module4_threshold_quality_review',
            '"module4_preliminary_scoring": module4_preliminary_scoring',
            '"module4_auto_preliminary_scoring": module4_auto_preliminary_scoring',
            '"module4_auto_method_scoring": module4_auto_method_scoring',
            '"scoring_type": "automatic_three_method_review"',
            '"final_preliminary_result": final_preliminary_result',
            '"method_vote_summary": {',
            '"method_key": method_key',
            '"profile_peak_method", "Profile / peaks"',
            '"fft_method", "FFT frequency support"',
            '"contrast_std_method", "Contrast / STD support"',
            '"module4_fft_threshold_debug": fft_threshold_debug',
            '"module4_threshold_source_reasons": threshold_source_reasons',
            '"threshold_source_reason": {',
            '"threshold_signatures": threshold_signatures',
            '"identical_fields": identical_fields',
            '"different_fields": different_fields',
            '"internal_contrast_support_threshold":',
            '"target_specific_thresholds_not_effective"',
            '"outlier_spread_detected": fft_outlier_spread',
            '"lower_half_family_median":',
            '"family_reference_fft_snr":',
            'analysis["target_diagnostic_strength"] = diagnostic_strength',
            '"sensitivity_importance": sensitivity_importance',
            '"threshold_quality_status": threshold_quality_status',
            '"outlier_control_status":',
            '"diagnostic_strength_status":',
            '"threshold_mode": "target_specific_adaptive"',
            '"margin_basis": "target_specific_adaptive"',
            '"internal_contrast_support": {',
            '"target_specific_threshold_warning": target_specific_threshold_warning',
            "def diagnostic_stats(rows: list[dict]) -> dict:",
            '"primary_targets": diagnostic_stats(primary_diagnostics)',
            '"all_targets": diagnostic_stats(diagnostic_targets)',
            '"development_mapping_enabled": MODULE4_ENABLE_DEVELOPMENT_LP_CM_MAPPING',
            '"required_lp_cm": required_lp_cm',
            '"visibility_votes": visibility_votes',
            'analysis["fft_vote"] = fft_vote',
            'analysis["profile_vote"] = profile_vote',
            'analysis["std_vote"] = std_vote',
            'analysis["visibility_vote"] = visibility_vote',
            '"physicist_review_required": True',
            '"expected_lp_cm_mapping_unknown"',
            '"pixel_spacing_missing"',
            '"profile_graph_generation_ms":',
            '"fft_graph_generation_ms":',
            '"std_contrast_generation_ms":',
            '"roi_data_status":',
            '"profile_data_status":',
            '"module4_roi_data_summary":',
            '"primary_roi_data_summary":',
            '"roi_data_extraction_ms":',
            '"profile_data_extraction_ms":',
        ):
            if geometry_first_requirement not in module4_service_text:
                messages.append((
                    "FAIL",
                    "Module 4 geometry-first Step 4 preparation is missing: "
                    f"{geometry_first_requirement}",
                ))
        active_module4_analysis = module4_service_text.rsplit(
            "\ndef analyze_module4_high_contrast_slice(", 1
        )[-1]
        for inactive_legacy_call in (
            "localize_module4_target(",
            "fit_single_module4_block_roi(",
            "fit_module4_slot_template_fallback(",
            "fit_module4_global_block_layout(",
            "fit_module4_jump_line_roi(",
            "_module4_draft_roi_scoring(",
        ):
            if inactive_legacy_call in active_module4_analysis:
                messages.append((
                    "FAIL",
                    "Module 4 active Step 4 path still calls legacy geometry: "
                    f"{inactive_legacy_call}",
                ))
        for analysis_helper in (
            "def detect_module4_square_blocks(",
            "def generate_module4_block_overlay(",
            "def analyze_module4_high_contrast_slice(",
            "def localize_module4_target(",
            "def fit_single_block_square_template(",
            "def fit_module4_slot_template_fallback(",
            "def fit_single_module4_block_roi(",
            "target_localizations.append(localize_module4_target(",
            "block_debug = fit_single_module4_block_roi(",
            "module4_block_square_template_optimized",
            "all_block_debug_enabled",
            "sector_candidates_considered",
            "detected_component_center",
            "crop_side_override=target_crop_side",
            '"selected_component_bbox":',
            '"side_margin_ratio":',
            '"final_side_px":',
            '"inner_roi_size_ratio":',
            '"final_square_is_template_fit":',
            '"best_template_score_breakdown":',
            '"top_template_trials":',
            '"missing_targets": missing_targets',
            '"weak_targets": weak_targets',
            '"detected_targets": detected_targets',
            '"slot_lifecycle": slot_lifecycle',
            '"slots_template_fit_attempted":',
            '"draw_on_overlay": draw_on_overlay',
            '"edge_symmetry_score":',
            '"micro_refinement_applied":',
            '"center_delta_from_component":',
            'candidate["analysis_priority"]',
            'candidate["geometry_status"] = "approximate_review_roi"',
            '"preliminary_review_summary":',
            '"normal_overlay_allowed":',
            '"target_processing_order":',
            '"candidate_rejection_reasons":',
            'target_crop_side = max(crop_side, 160) if target_id == "B6"',
            '"secondary_overlay_deferred"',
            '"all_primary_targets_localized":',
            '"b6_focused_debug": b6_focused_debug',
            '"all_sector_candidates":',
            '"B6 localized; preliminary geometry needs review.',
            "b6_bright_square_override",
            "def _module4_draft_roi_scoring(",
            '"draft_roi_results": draft_roi_results',
            '"overall_draft_result": overall_draft_result',
            '"formal_measurement_status": "pending"',
            '"period_consistency":',
            '"best_profile_direction":',
            '"score_breakdown": breakdown',
            '"rotated_box": rotated_box',
            '"inner_roi": {',
            '"periodicity_score":',
            '"peak_to_valley":',
            '"window_width": round(auto_width',
            "def _fit_square_target(",
            "def _safe_boolean_values(",
            "def estimate_module4_stripe_orientation(",
            "def fit_module4_block_from_profiles(",
            "def fit_module4_global_block_layout(",
            "def refine_module4_target_center(",
            "def fit_module4_jump_line_roi(",
            "def _jump_cluster_envelope(",
            "jump_cluster_envelope_perpendicular_roi_fit",
            "jump_cluster_found",
            "jump_peaks_found",
            "selected_outer_jump_lines",
            "perpendicular_extent_px",
            "envelope_fit_score",
            "def _module4_metrics_from_rotated_inner_roi(",
            "global_template_raw_hu_coarse_to_fine_alignment",
            "expected_ring_radius",
            "missing_slots",
            "assignment_confidence",
            "global_target_angle_degrees",
            "anchor_targets_used",
            "per_block_angle_override_reason",
            "center_shift_limit_px",
            "center_refinement_score",
            "edge_alignment_score",
            "target_fill_score",
            "stripe_boundary_profile_fit",
            "stripe_periodicity_score",
            "stripe_peaks_found",
            "boundary_extent_score",
            "median_block_side_px",
            "def _detect_module4_square_blocks_pass(",
            "def _detection_set_quality(",
            '"name": "strict"',
            '"name": "relaxed"',
            '"name": "edge_supported"',
            "detection_passes_tried",
            "rejection_counts_per_pass",
            "all_detection_passes_failed",
            "def _select_module4_display_window(",
            "window_quality_score",
            "visualization only",
        ):
            if analysis_helper not in module4_service_text:
                messages.append((
                    "FAIL",
                    f"Module 4 selected-slice analysis helper is missing: {analysis_helper}",
                ))
        for detector_requirement in (
            '"top": 270.0',
            '"right": 0.0',
            '"bottom": 90.0',
            '"left": 180.0',
            "max_dimension: int = 384",
            "radial >= 0.82",
            "radial <= 1.08",
        ):
            if detector_requirement not in module4_service_text:
                messages.append((
                    "FAIL",
                    f"Module 4 cardinal perimeter detector is missing: {detector_requirement}",
                ))
        for forbidden_selector in (
            "def locate_module4_candidates(",
            "def choose_best_module4_slice(",
            '"default_required_lp_cm":',
            '"protocol_profiles":',
        ):
            if forbidden_selector in module4_service_text:
                messages.append((
                    "FAIL",
                    f"Module 4 service still contains a competing selector: {forbidden_selector}",
                ))

    module4_script = root / "templates" / "partials" / "script_parts" / "dicom_module4.html"
    if False and module4_script.exists():
        module4_script_text = read(module4_script)
        for expected_result_section in (
            "Needs Review",
            "Preliminary Raw-HU and Profile Data",
            "<th>Target</th><th>Role</th><th>ROI Source</th><th>Data</th>",
            "<th>Periodicity</th><th>Profile Quality</th>",
            '<span class="metric-label">Official ACR result</span><strong class="metric-value">Physicist review required</strong>',
            "module4-primary-summary",
            "Preliminary automated votes are for review only.",
            "Default insert radius ratio:",
            "Calibrated insert radius ratio:",
            "Pre-calibration centers:",
            "Post-calibration centers:",
            "Pre-correction radius ratio:",
            "Post-correction radius ratio:",
            "Phi correction:",
            "Ring fit status:",
            "Calibrated ring center:",
            "Median center error before / after:",
            "Full fit status:",
            "ROI side ratio before / after:",
            "ROI angle offset before / after:",
            "Focused shared ROI angle before / after:",
            "Angle fit status:",
            "Targets used for angle fit:",
            "Center freeze status:",
            "Shared ROI side before / after:",
            "Maximum center shift after polish:",
            "Local square method:",
            "All targets retained:",
            "6 lp/cm target drawn:",
            "Micro-refinements applied:",
            "Maximum micro center shift:",
            "Maximum micro side change:",
            "Maximum micro angle change:",
            'const order = ["B8", "B7", "B6", "B5", "B4", "B3", "B2", "B1"]',
            "module4-primary-card-grid",
            "module4-review-meter",
            "Module 4 class review",
            "module4-class-review-table",
            "function getModule4ClassLabel(target)",
            "function getModule4TargetLabel(target)",
            "function getModule4FullLabel(target)",
            "Not scored · Mapping needed",
            "Mapping configuration is disabled; expected lp/cm mapping is not configured.",
            "How to read this",
            "Primary Target Comparison",
            "Peak-Valley HU",
            "Primary Profile Quality",
            "Primary signal ladder",
            "primarySignalIsMonotonic",
            "module4-explain-box",
            "module4-term-grid",
            "module4-comparison-section",
            "module4-bar-chart",
            "module4-trend-summary",
            "module4-visibility-ladder",
            "module4-ladder",
            "module4-chart-note",
            "module4-muted-note",
            "const graphData = result.module4_graph_data || {};",
            "block.module4_preliminary_analysis || {}",
            "Preliminary automated test votes",
            "Preliminary resolution review",
            "Physicist review is required.",
            "module4-vote-table",
            "module4-vote-badge",
            "module4-resolution-grid",
            "const voteSummary = result.module4_vote_summary || {};",
            "const preliminaryResolution = result.module4_preliminary_resolution || {};",
            "const lpCmConfig = result.module4_expected_lp_cm_config || {};",
            "Module 4 analysis configuration",
            "Mapping review",
            "module4-mapping-review-section",
            "Expected lp/cm mapping is not enabled.",
            "The lp/cm mapping configuration is enabled.",
            "Enable a confirmed mapping configuration before calculating preliminary resolved lp/cm.",
            "module4-mapping-table",
            "Current expected lp/cm",
            "Suggestion confidence",
            "const analysisConfig = result.module4_analysis_config || {};",
            "const mappingReview = result.mapping_review || result.suggested_mapping || {};",
            "const thresholdDiagnostics = result.module4_threshold_tuning_diagnostics || {};",
            "const adaptiveThresholds = result.module4_adaptive_thresholds || {};",
            "Threshold details",
            "Shows automatically calculated FFT/profile/STD threshold evidence for detailed review.",
            "Threshold details are for preliminary review only. Do not use them as official ACR scoring.",
            "module4-threshold-table",
            "Primary and all-target summary statistics",
            "Automatic target-specific thresholds",
            "const preliminaryScoring = result.module4_auto_preliminary_scoring",
            "const methodScoring = result.module4_auto_method_scoring",
            "Automatic Module 4 scoring result",
            "Automatic preliminary result",
            "Resolved resolution",
            "First unresolved",
            "Manual threshold used: No",
            "Resolution ladder",
            "Profile / peaks",
            "FFT support",
            "Contrast / STD",
            "Three-method scoring summary",
            "Why scoring stopped here",
            "Compact target scoring table",
            "Official ACR result: Physicist review required",
            "Advanced details and raw metrics",
            "Module 4 renderer: automatic scoring UI",
            "M4_AUTO_SCORE_VISIBLE_UI_2026_08_03",
            "Target diagnostic strength",
            "Profile score evidence",
            "Profile SNR evidence",
            "FFT SNR evidence",
            "Contrast evidence",
            "Target review cards",
            "Automatic analysis explanation",
            "function buildModule4DashboardRows(result)",
            "function renderModule4DashboardBarGraph(title, subtitle, rows, options)",
            "m4-dashboard-grid",
            "m4-graph-card",
            "m4-bar-fill",
            "m4-threshold-marker",
            "m4-target-card-grid",
            "m4-target-review-card",
            'data-action="copy_debug_info"',
            "Each Module 4 target receives thresholds calculated from its own signal data and target-family context.",
            "Target-specific thresholds:",
            "Threshold source",
            "module4-adaptive-margin-table",
            "Actual values and margins compare each target against that target's adaptive threshold for this run.",
            "Target-specific adaptive thresholds are preliminary review metrics.",
            "Target-specific thresholds are diagnostic only until analytical mapping and scoring configuration are enabled.",
            "const targetThresholdReviews = Array.isArray(result.module4_target_threshold_review)",
            "const targetThresholdValidation = result.module4_target_threshold_validation || {};",
            "const thresholdQualityReview = result.module4_threshold_quality_review || {};",
            "result.module4_fft_threshold_debug",
            "module4_target_threshold_validation: targetThresholdValidation",
            "module4_threshold_quality_review: thresholdQualityReview",
            "module4_fft_threshold_debug: fftThresholdDebug",
            "module4_threshold_source_reasons: result.module4_threshold_source_reasons || []",
            "Threshold review:",
            "FFT thresholds use outlier-resistant family context so very strong targets do not force weaker targets to max guardrails.",
            "FFT outlier adjusted",
            "Max guardrail hit",
            "Strength label review",
            "function module4TargetLabel(target)",
            "4 through 7 lp/cm",
            "renderAcrModule4LocationResult",
            'formData.append("performance_mode", "fast")',
            "Running Module 4 geometry/local ROI analysis",
            "new AbortController()",
            "exceeded 90 seconds",
            "const finalRoi = block.final_roi || block;",
            "finalRoi.final_side_px",
            "finalRoi.final_angle_degrees",
        ):
            if expected_result_section not in module4_script_text:
                messages.append((
                    "FAIL",
                    f"Module 4 result renderer is missing: {expected_result_section}",
                ))
        dashboard_banner_position = module4_script_text.find(
            "Automatic Module 4 scoring result"
        )
        roi_details_position = module4_script_text.find(
            "Module 4 ROI Data Review"
        )
        if (
            dashboard_banner_position < 0 or roi_details_position < 0
            or dashboard_banner_position >= roi_details_position
        ):
            messages.append((
                "FAIL",
                "Automatic Module 4 scoring result must render before ROI data review details.",
            ))
        score_card_position = module4_script_text.find(
            "Automatic Module 4 scoring result"
        )
        additional_details_position = module4_script_text.find(
            "Advanced details and raw metrics"
        )
        mapping_config_position = module4_script_text.find(
            "Module 4 analysis configuration"
        )
        if (
            score_card_position < 0 or additional_details_position < 0
            or mapping_config_position < 0
            or score_card_position >= additional_details_position
            or additional_details_position >= mapping_config_position
        ):
            messages.append((
                "FAIL",
                "Module 4 compact score must precede mapping/configuration placeholders, which must be collapsed under advanced details.",
            ))
        visible_order = [
            module4_script_text.find("Automatic Module 4 scoring result"),
            module4_script_text.find("Resolution ladder"),
            module4_script_text.find("Three-method scoring summary"),
            module4_script_text.find("Why scoring stopped here"),
            module4_script_text.find("Compact target scoring table"),
            module4_script_text.find("Advanced details and raw metrics"),
        ]
        if any(position < 0 for position in visible_order) or visible_order != sorted(visible_order):
            messages.append((
                "FAIL",
                "Module 4 visible result order must be score, ladder, three-method summary, stop reason, compact table, then advanced details.",
            ))
        score_card_text = module4_script_text[
            score_card_position:additional_details_position
        ] if score_card_position >= 0 and additional_details_position >= 0 else ""
        renderer_position = module4_script_text.find(
            "window.renderAcrModule4LocationResult = function (data)"
        )
        host_render_position = module4_script_text.find(
            "host.innerHTML = `", renderer_position
        )
        first_visible_section_position = module4_script_text.find(
            '<section class="analysis-section result-section', host_render_position
        )
        second_visible_section_position = module4_script_text.find(
            '<section class="analysis-section result-section',
            first_visible_section_position + 1,
        )
        if (
            renderer_position < 0 or host_render_position < 0
            or first_visible_section_position < 0
            or second_visible_section_position < 0
            or not (
                first_visible_section_position
                <= score_card_position
                < second_visible_section_position
            )
            or "Module 4 automatic analysis" in module4_script_text[
                host_render_position:score_card_position
            ]
            or "Module 4 ROI Data Review" in module4_script_text[
                host_render_position:score_card_position
            ]
        ):
            messages.append((
                "FAIL",
                "Module 4 active renderer must begin with the automatic scoring result, not the legacy analysis or ROI review dashboard.",
            ))
        for forbidden_score_card_text in (
            "Default review threshold",
            "Mapping needed",
            "Not available resolved lp/cm",
            "Not configured",
            "Not available",
            "Runtime diagnostics",
            "Development",
            "Debug Details",
            "Not scored",
            "Not implemented",
        ):
            if forbidden_score_card_text in score_card_text:
                messages.append((
                    "FAIL",
                    "Module 4 main automatic score still shows an old blocker: "
                    f"{forbidden_score_card_text}",
                ))
        for forbidden_visible_section in (
            '<h4 class="result-section-heading">Candidate Slice Ranking</h4>',
            "<summary>Eliminated Slices",
            "Draft ROI Scores",
            "<th>ROI Score</th>",
            "<th>Draft Status</th>",
            "Development graphs",
            "Official Pass",
            "Official Fail",
            "Meets ACR",
            "Final lp/cm",
            ">Dev test dashboard<",
            ">MODULE 4 DASHBOARD ACTIVE<",
            "DASHBOARD_RENDER_ASSERTIONS_PASS",
            ">self-test<",
            ">render assertion<",
        ):
            if forbidden_visible_section in module4_script_text:
                messages.append((
                    "FAIL",
                    f"Module 4 normal results still show debug-only slice data: {forbidden_visible_section}",
                ))

    if False and module4_script.exists():
        module4_script_text = read(module4_script)
        for expected_location_ui in (
            "Module 4 · High-Contrast Resolution",
            "Visible through",
            "Next unresolved",
            "Standard Deviation evidence",
            "ROI location details",
            "Technical details",
            "SD ratio",
            "1 of 3 evidence methods available",
            "m4-resolution-ladder",
            "m4-compact-evidence",
            "m4-raw-json",
            "window.renderAcrModule4LocationResult = function (data)",
            'formData.append("performance_mode", "fast")',
        ):
            if expected_location_ui not in module4_script_text:
                messages.append((
                    "FAIL",
                    f"Module 4 location-only renderer is missing: {expected_location_ui}",
                ))
        for forbidden_scoring_ui in (
            "Module 4 target location review",
            "Scoring Disabled",
            "Location notes",
            "B1–B8 ROI location table",
            "standardDeviationCards",
            "m4-std-target-grid",
            "Automatic Module 4 scoring result",
            "Automatic preliminary result",
            "Three-method scoring summary",
            ">PASS<",
            ">FAIL<",
            "Overall PASS",
            "Overall FAIL",
            "Resolved resolution",
            "First unresolved",
            "Method vote",
            "Mapping needed",
            "Not scored",
            "Target-specific thresholds",
            "Preliminary automated test votes",
            "module4_auto_preliminary_scoring",
            "module4_auto_method_scoring",
            "module4_method_graphs",
        ):
            if forbidden_scoring_ui in module4_script_text:
                messages.append((
                    "FAIL",
                    f"Module 4 renderer still contains scoring UI: {forbidden_scoring_ui}",
                ))

    if module4_script.exists():
        module4_script_text = read(module4_script)
        for expected_module4_ui in (
            "ACR CT Module 4",
            "High-Contrast Resolution",
            "Automatic frequency ROI location and Standard Deviation evidence",
            "SD Evidence Available",
            "Selected Slice",
            "Targets Located",
            "Visible Through",
            "Next Unresolved",
            "Frequency ROI Overlay",
            "How the SD result is calculated",
            "SD / Noise Ratio by Frequency",
            "Standard Deviation Evidence",
            "Excess SD",
            "Noise Source",
            "Noise Reference",
            "Resolution transition",
            "ROI Location Details",
            "Technical Details",
            "Threshold Configuration",
            "Noise Calculation",
            "Target Calculations",
            "Quality Flags",
            "Raw JSON",
            "analysis-result-shell",
            "analysis-summary-header",
            "analysis-metric-grid",
            "analysis-metric-card metric-card",
            "analysis-image-stage image-stage",
            "analysis-table-card",
            "details-box",
            "window.renderAcrModule4LocationResult = function (data)",
            'formData.append("performance_mode", "fast")',
        ):
            if expected_module4_ui not in module4_script_text:
                messages.append((
                    "FAIL",
                    f"Module 4 shared result structure is missing: {expected_module4_ui}",
                ))
        for rejected_module4_ui in (
            "Standard Deviation resolution ladder",
            "Highest continuously visible by SD method",
            "1 of 3 evidence methods available",
            "m4-resolution-ladder",
            "m4-resolution-node",
            "m4-primary-result",
            "m4-compact-dashboard",
            "Module 4 target location review",
            "Scoring Disabled",
            "Location notes",
            "Automatic Module 4 scoring result",
            "Three-method scoring summary",
            "Method vote",
            "Overall PASS",
            "Overall FAIL",
            "Mapping needed",
        ):
            if rejected_module4_ui in module4_script_text:
                messages.append((
                    "FAIL",
                    f"Module 4 renderer still contains rejected UI: {rejected_module4_ui}",
                ))
        visible_metrics_start = module4_script_text.find(
            '<div class="analysis-metric-grid">'
        )
        visible_metrics_end = module4_script_text.find(
            '</div>\n            <div class="analysis-review-note"',
            visible_metrics_start,
        )
        visible_metric_text = module4_script_text[
            visible_metrics_start:visible_metrics_end
        ] if visible_metrics_start >= 0 and visible_metrics_end >= 0 else ""
        visible_metric_count = visible_metric_text.count(
            'class="analysis-metric-card metric-card"'
        )
        if visible_metric_count != 4:
            messages.append((
                "FAIL",
                f"Module 4 must show exactly four summary metric cards; found {visible_metric_count}.",
            ))

    tool_selection_path = root / "templates" / "partials" / "tool_selection.html"
    if tool_selection_path.exists():
        tool_selection_text = read(tool_selection_path)
        if "histogram_card.html" in tool_selection_text or re.search(
            r"\bdata-open-tool\s*=\s*['\"]histogram['\"]",
            tool_selection_text,
            re.IGNORECASE,
        ):
            messages.append(("FAIL", "Histogram card is still active in the launcher."))
        if "module4_high_contrast_card.html" not in tool_selection_text:
            messages.append(("FAIL", "Module 4 launcher card include is missing."))

    tool_card_onclick = count_pattern(html_files, r'class\s*=\s*["\'][^"\']*\btool-card\b[^"\']*["\'][^>]*\bonclick\s*=')
    tab_onclick = count_pattern(html_files, r'id\s*=\s*["\']tab-[^"\']+["\'][^>]*\bonclick\s*=')
    raw_fetch = count_pattern(script_files, r"fetch\(\s*['\"]/(?:process|dicom-[^'\"]+)['\"]")
    route_url = count_pattern(script_files, r"UIRegistry\.routeUrl\(")
    data_actions = count_pattern(html_files, r"\bdata-action\s*=")
    data_open_tool = count_pattern(html_files, r"\bdata-open-tool\s*=")
    data_open_panel = count_pattern(html_files, r"\bdata-open-panel\s*=")
    mutation_observer = count_pattern(script_files, r"\bMutationObserver\b")
    stop_propagation = count_pattern(script_files, r"\bstopPropagation\s*\(")
    programmatic_click = count_pattern(script_files, r"\.click\s*\(")

    if tool_card_onclick:
        messages.append(("WARN", f"Tool-card inline onclick count is {tool_card_onclick}."))
    if tab_onclick:
        messages.append(("WARN", f"DICOM tab inline onclick count is {tab_onclick}."))
    if raw_fetch:
        messages.append(("WARN", f"Raw hard-coded fetch route count is {raw_fetch}."))
    if mutation_observer:
        messages.append(("WARN", f"MutationObserver count is {mutation_observer}."))
    if programmatic_click:
        messages.append(("WARN", f"Programmatic .click() count is {programmatic_click}."))

    messages.extend(check_js_syntax(root, script_files))

    print()
    print("UI ARCHITECTURE HEALTH CHECK")
    print("=" * 32)
    print()
    print("Files scanned:", len(html_files))
    print("Script files scanned:", len(script_files))
    print()
    print("Architecture counts:")
    print("  data-action:", data_actions)
    print("  data-open-tool:", data_open_tool)
    print("  data-open-panel:", data_open_panel)
    print("  UIRegistry.routeUrl calls:", route_url)
    print("  raw fetch routes:", raw_fetch)
    print("  tool-card inline onclick:", tool_card_onclick)
    print("  tab inline onclick:", tab_onclick)
    print("  MutationObserver:", mutation_observer)
    print("  stopPropagation:", stop_propagation)
    print("  programmatic .click():", programmatic_click)
    print()
    print("DICOM tabs:", sorted(tab_keys))
    print("DICOM panels:", sorted(panel_keys))
    print()

    fail_count = sum(1 for level, _ in messages if level == "FAIL")
    warn_count = sum(1 for level, _ in messages if level == "WARN")

    if messages:
        print("Messages:")
        for level, message in messages:
            print(f"  [{level}] {message}")
    else:
        print("Messages:")
        print("  [PASS] No architecture warnings found.")

    print()
    print("Summary:")
    print("  FAIL:", fail_count)
    print("  WARN:", warn_count)

    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
