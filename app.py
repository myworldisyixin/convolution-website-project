from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from scipy.ndimage import convolve
import numpy as np
import io
import json
import base64

app = Flask(__name__)
CORS(app)

MAX_IMAGE_SIZE = 1200
MAX_MATRIX_SIZE = 25


def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return "data:image/png;base64," + encoded


def normalize_kernel(kernel):
    total = np.sum(kernel)

    # If sum is not zero, normalize so brightness does not blow out.
    if abs(total) > 0.000001:
        return kernel / total

    # If sum is zero, do NOT normalize.
    # Zero-sum kernels are usually edge detectors.
    return kernel


def predict_effect(original_kernel, normalized_kernel):
    rows, cols = original_kernel.shape
    center_value = original_kernel[rows // 2, cols // 2]
    total_sum = float(np.sum(original_kernel))
    positive_count = int(np.sum(original_kernel > 0))
    negative_count = int(np.sum(original_kernel < 0))
    value_count = rows * cols

    all_positive_or_zero = negative_count == 0
    has_positive_and_negative = positive_count > 0 and negative_count > 0
    is_uniform = np.std(original_kernel) < 0.000001

    if abs(total_sum) < 0.000001 and has_positive_and_negative:
        return {
            "name": "Edge detection / outline",
            "description": "This matrix has positive and negative values that cancel out. It will mostly highlight edges and outlines."
        }

    if all_positive_or_zero and is_uniform:
        return {
            "name": "Blur / smoothing",
            "description": "All values are the same and positive. After normalization, this averages nearby pixels, so the image becomes smoother."
        }

    if all_positive_or_zero:
        return {
            "name": "Weighted blur / smoothing",
            "description": "All values are positive. This will average nearby pixels. Larger numbers give those nearby pixels more influence."
        }

    if has_positive_and_negative and center_value > 0:
        return {
            "name": "Sharpen / detail enhancement",
            "description": "This matrix mixes a strong positive center with negative surrounding values. It will increase edges and detail."
        }

    if has_positive_and_negative:
        return {
            "name": "High contrast / edge effect",
            "description": "This matrix contains both positive and negative values, so it may create strong contrast or edge-like effects."
        }

    return {
        "name": "Custom effect",
        "description": "This matrix creates a custom transformation. Check the output min/max to see whether it is too strong."
    }


def process_image(gray_array, kernel):
    raw_result = convolve(
        gray_array.astype(np.float32),
        kernel,
        mode="reflect"
    )

    raw_min = float(np.min(raw_result))
    raw_max = float(np.max(raw_result))

    clipped_low = float(np.mean(raw_result < 0) * 100)
    clipped_high = float(np.mean(raw_result > 255) * 100)

    result = np.clip(raw_result, 0, 255)

    return result.astype(np.uint8), raw_min, raw_max, clipped_low, clipped_high


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        kernel_text = request.form.get("kernel")

        if not kernel_text:
            return jsonify({"error": "No matrix provided"}), 400

        original_kernel = np.array(json.loads(kernel_text), dtype=np.float32)

        if original_kernel.ndim != 2:
            return jsonify({"error": "Matrix must be 2D"}), 400

        rows, cols = original_kernel.shape

        if rows < 1 or cols < 1:
            return jsonify({"error": "Matrix cannot be empty"}), 400

        if rows > MAX_MATRIX_SIZE or cols > MAX_MATRIX_SIZE:
            return jsonify({
                "error": f"Matrix is too large. Please use {MAX_MATRIX_SIZE} x {MAX_MATRIX_SIZE} or smaller."
            }), 400

        kernel = normalize_kernel(original_kernel)

        img = Image.open(request.files["image"]).convert("L")

        original_width, original_height = img.size
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        processed_width, processed_height = img.size

        max_rows_allowed = min(MAX_MATRIX_SIZE, processed_height)
        max_cols_allowed = min(MAX_MATRIX_SIZE, processed_width)

        gray_array = np.array(img)

        image_min = int(np.min(gray_array))
        image_max = int(np.max(gray_array))
        image_mean = float(np.mean(gray_array))

        result_array, raw_min, raw_max, clipped_low, clipped_high = process_image(
            gray_array,
            kernel
        )

        result_img = Image.fromarray(result_array)

        original_sum = float(np.sum(original_kernel))
        normalized_sum = float(np.sum(kernel))

        safe_equal_value = 1 / (rows * cols)

        positive_sum = float(np.sum(kernel[kernel > 0]))
        negative_sum = float(np.sum(kernel[kernel < 0]))

        theoretical_min = 255 * negative_sum
        theoretical_max = 255 * positive_sum

        prediction = predict_effect(original_kernel, kernel)

        if clipped_high > 10:
            strength_message = "This matrix may make large areas too white."
        elif clipped_low > 10:
            strength_message = "This matrix may make large areas too black."
        elif abs(raw_max - raw_min) < 15:
            strength_message = "This matrix may create only a very subtle change."
        else:
            strength_message = "This matrix should produce a visible effect without extreme clipping."

        return jsonify({
            "image": image_to_base64(result_img),
            "stats": {
                "originalImageSize": f"{original_width} x {original_height}",
                "processedImageSize": f"{processed_width} x {processed_height}",

                "imageMin": image_min,
                "imageMax": image_max,
                "imageMean": round(image_mean, 2),

                "maxMatrixAllowed": f"{max_rows_allowed} x {max_cols_allowed}",
                "matrixSize": f"{rows} x {cols}",
                "numberOfValues": rows * cols,

                "originalMatrixSum": round(original_sum, 4),
                "normalizedMatrixSum": round(normalized_sum, 4),
                "safeEqualValue": round(safe_equal_value, 6),

                "rawOutputMin": round(raw_min, 2),
                "rawOutputMax": round(raw_max, 2),

                "theoreticalMin": round(theoretical_min, 2),
                "theoreticalMax": round(theoretical_max, 2),

                "clippedLowPercent": round(clipped_low, 2),
                "clippedHighPercent": round(clipped_high, 2),

                "predictionName": prediction["name"],
                "predictionDescription": prediction["description"],
                "strengthMessage": strength_message
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    