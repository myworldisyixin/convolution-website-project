import json
import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from services.dicom_helpers import dicom_to_image
from services.image_helpers import image_to_base64


MAX_IMAGE_SIZE = 1200
MAX_MATRIX_SIZE = 25


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


def apply_matrix_filter(uploaded_file, kernel_text):
    original_kernel = np.array(json.loads(kernel_text), dtype=np.float32)

    if original_kernel.ndim != 2:
        raise ValueError("Matrix must be 2D.")

    rows, cols = original_kernel.shape

    if rows < 1 or cols < 1:
        raise ValueError("Matrix cannot be empty.")

    if rows > MAX_MATRIX_SIZE or cols > MAX_MATRIX_SIZE:
        raise ValueError(
            f"Matrix is too large. Please use {MAX_MATRIX_SIZE} x {MAX_MATRIX_SIZE} or smaller."
        )

    kernel = normalize_kernel(original_kernel)

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

    return {
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
    }