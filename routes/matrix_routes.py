from flask import Blueprint, request, jsonify

from services.matrix_helpers import apply_matrix_filter


matrix_bp = Blueprint("matrix", __name__)


@matrix_bp.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        kernel_text = request.form.get("kernel")

        if not kernel_text:
            return jsonify({"error": "No matrix provided"}), 400

        uploaded_file = request.files["image"]

        result = apply_matrix_filter(uploaded_file, kernel_text)

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500