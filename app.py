from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageDraw
from scipy.ndimage import convolve
from skimage import filters
import numpy as np
import io
import json
import base64
import pydicom

app = Flask(__name__)
CORS(app)

MAX_IMAGE_SIZE = 1200
MAX_MATRIX_SIZE = 25


# -----------------------------
# BASIC IMAGE HELPERS
# -----------------------------
def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return "data:image/png;base64," + encoded


def array_to_base64_image(array):
    array = np.asarray(array, dtype=np.float32)

    arr_min = float(np.min(array))
    arr_max = float(np.max(array))

    if arr_max - arr_min < 0.000001:
        display = np.zeros_like(array)
    else:
        display = (array - arr_min) / (arr_max - arr_min) * 255

    img = Image.fromarray(display.astype(np.uint8)).convert("L")
    return image_to_base64(img)


def normalize_for_display(array):
    array = np.asarray(array, dtype=np.float32)

    arr_min = float(np.min(array))
    arr_max = float(np.max(array))

    if arr_max - arr_min < 0.000001:
        return np.zeros_like(array, dtype=np.float32)

    return (array - arr_min) / (arr_max - arr_min)


# -----------------------------
# MATRIX FILTER FUNCTIONS
# -----------------------------
def normalize_kernel(kernel):
    total = np.sum(kernel)

    if abs(total) > 0.000001:
        return kernel / total

    return kernel


def predict_effect(original_kernel, normalized_kernel):
    rows, cols = original_kernel.shape
    center_value = original_kernel[rows // 2, cols // 2]
    total_sum = float(np.sum(original_kernel))
    positive_count = int(np.sum(original_kernel > 0))
    negative_count = int(np.sum(original_kernel < 0))

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


# -----------------------------
# FILE LOADING FUNCTIONS
# -----------------------------
def dicom_to_image(uploaded_file):
    uploaded_file.stream.seek(0)
    dicom_data = pydicom.dcmread(uploaded_file, force=True)

    if not hasattr(dicom_data, "pixel_array"):
        raise ValueError("This DICOM file does not contain pixel data.")

    pixel_array = dicom_data.pixel_array.astype(np.float32)

    slope = float(getattr(dicom_data, "RescaleSlope", 1))
    intercept = float(getattr(dicom_data, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    photometric = str(getattr(dicom_data, "PhotometricInterpretation", "")).upper()

    display_array = normalize_for_display(pixel_array) * 255

    if photometric == "MONOCHROME1":
        display_array = 255 - display_array

    img = Image.fromarray(display_array.astype(np.uint8)).convert("L")

    dicom_info = {
        "isDicom": True,
        "patientID": str(getattr(dicom_data, "PatientID", "Unknown")),
        "modality": str(getattr(dicom_data, "Modality", "Unknown")),
        "studyDescription": str(getattr(dicom_data, "StudyDescription", "Unknown")),
        "seriesDescription": str(getattr(dicom_data, "SeriesDescription", "Unknown")),
        "rows": int(getattr(dicom_data, "Rows", 0)),
        "columns": int(getattr(dicom_data, "Columns", 0)),
        "rescaleSlope": slope,
        "rescaleIntercept": intercept,
        "photometricInterpretation": photometric or "Unknown",
        "originalPixelMin": round(float(np.min(pixel_array)), 2),
        "originalPixelMax": round(float(np.max(pixel_array)), 2)
    }

    return img, dicom_info


def normal_image_to_image(uploaded_file):
    uploaded_file.stream.seek(0)
    img = Image.open(uploaded_file).convert("L")

    image_info = {
        "isDicom": False,
        "patientID": "Not DICOM",
        "modality": "Normal image",
        "studyDescription": "Not DICOM",
        "seriesDescription": "Not DICOM",
        "rows": img.height,
        "columns": img.width,
        "rescaleSlope": "N/A",
        "rescaleIntercept": "N/A",
        "photometricInterpretation": "N/A",
        "originalPixelMin": "N/A",
        "originalPixelMax": "N/A"
    }

    return img, image_info


def load_uploaded_file_as_grayscale(uploaded_file):
    try:
        return dicom_to_image(uploaded_file)
    except Exception:
        pass

    try:
        return normal_image_to_image(uploaded_file)
    except Exception as e:
        raise ValueError(
            "File could not be read as DICOM or as a normal image. "
            f"Original error: {str(e)}"
        )


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


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

        uploaded_file = request.files["image"]
        img, file_info = load_uploaded_file_as_grayscale(uploaded_file)

        original_width, original_height = img.size
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        processed_width, processed_height = img.size

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
                "strengthMessage": strength_message,
                "isDicom": file_info["isDicom"],
                "patientID": file_info["patientID"],
                "modality": file_info["modality"],
                "studyDescription": file_info["studyDescription"],
                "seriesDescription": file_info["seriesDescription"],
                "dicomRows": file_info["rows"],
                "dicomColumns": file_info["columns"],
                "rescaleSlope": file_info["rescaleSlope"],
                "rescaleIntercept": file_info["rescaleIntercept"],
                "photometricInterpretation": file_info["photometricInterpretation"],
                "originalPixelMin": file_info["originalPixelMin"],
                "originalPixelMax": file_info["originalPixelMax"]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dicom-roi-analysis", methods=["POST"])
def dicom_roi_analysis():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        uploaded_file = request.files["image"]

        ymin = int(request.form.get("ymin", 0))
        ymax = int(request.form.get("ymax", 0))
        xmin = int(request.form.get("xmin", 0))
        xmax = int(request.form.get("xmax", 0))

        uploaded_file.stream.seek(0)
        ds = pydicom.dcmread(uploaded_file, force=True)

        if not hasattr(ds, "pixel_array"):
            return jsonify({"error": "This DICOM file does not contain pixel data."}), 400

        raw_pixels = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        raw_pixels = raw_pixels * slope + intercept

        if raw_pixels.ndim != 2:
            return jsonify({"error": "This demo currently supports single-slice 2D DICOM images only."}), 400

        height, width = raw_pixels.shape

        if ymin < 0 or ymax > height or xmin < 0 or xmax > width:
            return jsonify({
                "error": f"ROI is outside the image. Image size is {height} rows x {width} columns."
            }), 400

        if ymin >= ymax or xmin >= xmax:
            return jsonify({"error": "Invalid ROI coordinates."}), 400

        norm_pixels = normalize_for_display(raw_pixels)

        smooth_img = filters.gaussian(norm_pixels, sigma=2.0)
        sharp_img = filters.unsharp_mask(norm_pixels, radius=2.0, amount=1.5)
        edge_img = filters.sobel(norm_pixels)

        roi_pixels = raw_pixels[ymin:ymax, xmin:xmax]

        roi_mean = float(np.mean(roi_pixels))
        roi_std = float(np.std(roi_pixels))
        roi_min = float(np.min(roi_pixels))
        roi_max = float(np.max(roi_pixels))

        original_img = Image.fromarray((norm_pixels * 255).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(original_img)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        return jsonify({
            "images": {
                "original": image_to_base64(original_img),
                "smooth": array_to_base64_image(smooth_img),
                "sharp": array_to_base64_image(sharp_img),
                "edge": array_to_base64_image(edge_img)
            },
            "roi": {
                "ymin": ymin,
                "ymax": ymax,
                "xmin": xmin,
                "xmax": xmax,
                "mean": round(roi_mean, 2),
                "std": round(roi_std, 2),
                "min": round(roi_min, 2),
                "max": round(roi_max, 2)
            },
            "imageInfo": {
                "rows": int(height),
                "columns": int(width),
                "modality": str(getattr(ds, "Modality", "Unknown")),
                "studyDescription": str(getattr(ds, "StudyDescription", "Unknown")),
                "seriesDescription": str(getattr(ds, "SeriesDescription", "Unknown")),
                "rescaleSlope": slope,
                "rescaleIntercept": intercept
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
