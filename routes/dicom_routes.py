from flask import Blueprint, request, jsonify

from services.dicom_helpers import (
    dicom_to_image,
    analyze_dicom_roi,
    create_dicom_stack_preview,
    create_dicom_window_preview,
    create_acr_module3_analysis
)

from services.image_helpers import image_to_base64


dicom_bp = Blueprint("dicom", __name__)


@dicom_bp.route("/dicom-preview", methods=["POST"])
def dicom_preview():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        uploaded_file = request.files["image"]

        img, info = dicom_to_image(uploaded_file)

        return jsonify({
            "image": image_to_base64(img.convert("RGB")),
            "info": info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dicom_bp.route("/dicom-stack-preview", methods=["POST"])
def dicom_stack_preview():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        uploaded_file = request.files["image"]

        result = create_dicom_stack_preview(uploaded_file)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dicom_bp.route("/dicom-window-preview", methods=["POST"])
def dicom_window_preview():
    try:
        stack_id = request.form.get("stack_id", "")
        uploaded_file = request.files.get("image", None)

        if not stack_id and uploaded_file is None:
            return jsonify({"error": "No DICOM stack ID or image provided"}), 400

        slice_index = int(request.form.get("slice_index", 0))
        window_width = float(request.form.get("window_width", 400))
        window_level = float(request.form.get("window_level", 40))

        result = create_dicom_window_preview(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            slice_index=slice_index,
            window_width=window_width,
            window_level=window_level
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dicom_bp.route("/dicom-roi-analysis", methods=["POST"])
def dicom_roi_analysis():
    try:
        stack_id = request.form.get("stack_id", "")
        uploaded_file = request.files.get("image", None)

        if not stack_id and uploaded_file is None:
            return jsonify({"error": "No DICOM stack ID or image provided"}), 400

        ymin = int(request.form.get("ymin", 0))
        ymax = int(request.form.get("ymax", 0))
        xmin = int(request.form.get("xmin", 0))
        xmax = int(request.form.get("xmax", 0))
        slice_index = int(request.form.get("slice_index", 0))

        result = analyze_dicom_roi(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            ymin=ymin,
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
            slice_index=slice_index
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dicom_bp.route("/dicom-acr-module3", methods=["POST"])
def dicom_acr_module3():
    try:
        stack_id = request.form.get("stack_id", "")
        uploaded_file = request.files.get("image", None)

        if not stack_id and uploaded_file is None:
            return jsonify({"error": "No DICOM stack ID or image provided"}), 400

        slice_index = int(request.form.get("slice_index", 0))

        window_width = float(request.form.get("window_width", 100))
        window_level = float(request.form.get("window_level", 0))

        result = create_acr_module3_analysis(
            stack_id=stack_id,
            uploaded_file=uploaded_file,
            slice_index=slice_index,
            window_width=window_width,
            window_level=window_level
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500