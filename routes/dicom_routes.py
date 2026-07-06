from flask import Blueprint, request, jsonify

from services.dicom_helpers import (
    dicom_to_image,
    analyze_dicom_roi,
    create_dicom_stack_preview,
    create_dicom_window_preview,
    create_acr_module3_analysis
)

from services.image_helpers import image_to_base64


dicom_bp = Blueprint("dicom_bp", __name__)

def _get_uploaded_file():
    return (
        request.files.get("image")
        or request.files.get("file")
        or request.files.get("dicom")
        or request.files.get("dicomFile")
    )


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
def dicom_acr_module3_route():
    try:
        file_storage = _get_uploaded_file()
        stack_id = request.form.get("stack_id") or request.form.get("stackId")

        if not stack_id and file_storage is None:
            return jsonify({
                "success": False,
                "error": "No cached stack or DICOM file was provided."
            }), 400

        try:
            slice_index = int(float(
                request.form.get("slice_index")
                or request.form.get("sliceIndex")
                or 0
            ))
        except Exception:
            slice_index = 0

        try:
            window_width = float(
                request.form.get("window_width")
                or request.form.get("windowWidth")
                or 100
            )
        except Exception:
            window_width = 100

        try:
            window_level = float(
                request.form.get("window_level")
                or request.form.get("windowLevel")
                or 0
            )
        except Exception:
            window_level = 0

        auto_scan_value = request.form.get("auto_scan", "0")
        auto_scan = str(auto_scan_value).lower().strip() in ["1", "true", "yes", "on"]

        result = create_acr_module3_analysis(
            slice_index=slice_index,
            stack_id=stack_id,
            file_storage=file_storage,
            window_width=window_width,
            window_level=window_level,
            auto_scan=auto_scan
        )

        return jsonify({
            "success": True,
            **result
        })

    except ValueError as exc:
        message = str(exc)

        return jsonify({
            "success": False,
            "error": message,
            "message": message
        }), 400

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400