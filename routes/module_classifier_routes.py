from __future__ import annotations

from collections import OrderedDict

from flask import Blueprint, jsonify, request

from services.acr_module3_full_analysis import (
    create_integrated_module3_analysis,
)
from services.acr_module_classifier import (
    create_acr_module_classification,
)


module_classifier_bp = Blueprint(
    "module_classifier",
    __name__,
)


_CLASSIFICATION_CACHE = OrderedDict()
_MAX_CACHE_ITEMS = 8


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
