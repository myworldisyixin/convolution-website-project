import io
import re
import zipfile

import numpy as np
import pydicom

from services.dicom_config import ROBUST_VERSION


def _read_uploaded_bytes(uploaded_file):
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    try:
        uploaded_file.stream.seek(0)
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    data = uploaded_file.read()

    try:
        uploaded_file.stream.seek(0)
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    if not data:
        raise ValueError("Uploaded file is empty.")

    return data


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _natural_sort_text(text):
    parts = re.split(r"(\d+)", str(text))
    out = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out


def _get_pixel_spacing(ds):
    try:
        spacing = getattr(ds, "PixelSpacing", None)

        if spacing is not None and len(spacing) >= 2:
            row_spacing = float(spacing[0])
            col_spacing = float(spacing[1])

            if row_spacing > 0 and col_spacing > 0:
                return row_spacing, col_spacing

    except Exception:
        pass

    return None, None


def _get_image_position_z(ds):
    try:
        image_position = getattr(ds, "ImagePositionPatient", None)

        if image_position is not None and len(image_position) >= 3:
            return float(image_position[2])

    except Exception:
        pass

    return None


def _set_default_transfer_syntax_if_missing(ds):
    try:
        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()

        if not getattr(ds.file_meta, "TransferSyntaxUID", None):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    except Exception:
        pass


def _manual_uncompressed_pixel_array(ds):
    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    bits_allocated = int(getattr(ds, "BitsAllocated", 16) or 16)
    pixel_representation = int(getattr(ds, "PixelRepresentation", 0) or 0)

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid Rows/Columns: Rows={rows}, Columns={cols}")

    if bits_allocated == 16:
        dtype = np.int16 if pixel_representation == 1 else np.uint16
    elif bits_allocated == 8:
        dtype = np.int8 if pixel_representation == 1 else np.uint8
    else:
        raise ValueError(f"Unsupported BitsAllocated={bits_allocated}")

    raw = np.frombuffer(ds.PixelData, dtype=dtype)
    expected_single = rows * cols * samples
    expected_total = expected_single * frames

    if raw.size < expected_single:
        raise ValueError(
            f"PixelData too small. Raw values={raw.size}, expected at least {expected_single}."
        )

    if frames > 1 and raw.size >= expected_total:
        if samples > 1:
            return raw[:expected_total].reshape(frames, rows, cols, samples)
        return raw[:expected_total].reshape(frames, rows, cols)

    if samples > 1:
        return raw[:expected_single].reshape(rows, cols, samples)

    return raw[:expected_single].reshape(rows, cols)


def _color_to_gray(arr):
    a = np.asarray(arr)

    if a.ndim != 3:
        raise ValueError(f"Color image has unsupported shape: {a.shape}")

    if a.shape[-1] >= 3:
        r = a[..., 0].astype(np.float32)
        g = a[..., 1].astype(np.float32)
        b = a[..., 2].astype(np.float32)
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

    if a.shape[0] >= 3:
        r = a[0, ...].astype(np.float32)
        g = a[1, ...].astype(np.float32)
        b = a[2, ...].astype(np.float32)
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

    raise ValueError(f"Color image has unsupported shape: {a.shape}")


def _usable_2d_pixels(pixels):
    if pixels is None:
        return False

    if pixels.ndim != 2:
        return False

    if pixels.shape[0] < 16 or pixels.shape[1] < 16:
        return False

    finite = pixels[np.isfinite(pixels)]
    return finite.size >= 100


def _make_info(ds, pixels, source_name, frame_index=None, is_color=False, photometric_override=None):
    row_spacing, col_spacing = _get_pixel_spacing(ds)
    photometric = photometric_override or str(getattr(ds, "PhotometricInterpretation", "") or "").upper()

    try:
        transfer_syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
    except Exception:
        transfer_syntax = ""

    instance_number = _safe_int(getattr(ds, "InstanceNumber", None), None)

    if frame_index is not None:
        if instance_number is not None:
            instance_number = int(instance_number) + int(frame_index)
        else:
            instance_number = int(frame_index) + 1

    info = {
        "isDicom": True,
        "sourceName": source_name,
        "patientID": str(getattr(ds, "PatientID", "Unknown")),
        "patientName": str(getattr(ds, "PatientName", "")),
        "modality": str(getattr(ds, "Modality", "Unknown")),
        "studyDescription": str(getattr(ds, "StudyDescription", "Unknown")),
        "seriesDescription": str(getattr(ds, "SeriesDescription", "Unknown")),
        "rows": int(pixels.shape[-2]),
        "columns": int(pixels.shape[-1]),
        "rescaleSlope": _safe_float(getattr(ds, "RescaleSlope", 1), 1.0),
        "rescaleIntercept": _safe_float(getattr(ds, "RescaleIntercept", 0), 0.0),
        "photometricInterpretation": photometric or "MONOCHROME2",
        "photometric": photometric or "MONOCHROME2",
        "originalPixelMin": round(float(np.nanmin(pixels)), 2),
        "originalPixelMax": round(float(np.nanmax(pixels)), 2),
        "instanceNumber": instance_number,
        "sliceLocation": _safe_float(getattr(ds, "SliceLocation", None), None),
        "imagePositionZ": _get_image_position_z(ds),
        "pixelSpacingRow": row_spacing,
        "pixelSpacingCol": col_spacing,
        "samplesPerPixel": _safe_int(getattr(ds, "SamplesPerPixel", 1), 1),
        "isColorDicom": bool(is_color),
        "colorNote": (
            "Color DICOM converted to grayscale for display. ACR HU analysis requires original grayscale CT DICOM."
            if is_color
            else ""
        ),
        "transferSyntaxUID": transfer_syntax,
        "readerVersion": ROBUST_VERSION,
    }

    if frame_index is not None:
        info["frameIndex"] = int(frame_index)

    return info


def _extract_slices_from_dicom_bytes(file_bytes, source_name="uploaded_file"):
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True, stop_before_pixels=False)
    except Exception as exc:
        raise ValueError(f"pydicom could not read file: {exc}")

    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM has no PixelData tag.")

    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid Rows/Columns: Rows={rows}, Columns={cols}")

    _set_default_transfer_syntax_if_missing(ds)

    try:
        pixel_array = ds.pixel_array
    except Exception as exc:
        try:
            pixel_array = _manual_uncompressed_pixel_array(ds)
        except Exception as manual_exc:
            raise ValueError(
                "Could not decode PixelData. This may be compressed DICOM needing pylibjpeg/gdcm codecs. "
                f"pydicom error: {exc}; manual fallback error: {manual_exc}"
            )

    arr = np.asarray(pixel_array)

    if arr.size == 0:
        raise ValueError("pixel_array is empty.")

    samples_per_pixel = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
    is_color = samples_per_pixel > 1 or "RGB" in photometric or "YBR" in photometric

    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)

    slices = []

    def append_slice(pixels, label, frame_index=None, color=False):
        pixels = np.asarray(pixels).astype(np.float32)

        if not _usable_2d_pixels(pixels):
            raise ValueError(f"Decoded pixels are not usable 2D image pixels. Shape={pixels.shape}")

        info = _make_info(
            ds=ds,
            pixels=pixels,
            source_name=source_name,
            frame_index=frame_index,
            is_color=color,
            photometric_override="MONOCHROME2" if color else photometric or "MONOCHROME2",
        )

        slices.append({
            "pixels": pixels,
            "info": info,
            "photometric": "MONOCHROME2" if color else photometric or "MONOCHROME2",
            "label": label,
            "sourceName": source_name,
        })

    if is_color:
        if arr.ndim == 3:
            gray = _color_to_gray(arr)
            append_slice(gray, source_name, color=True)
            return slices

        if arr.ndim == 4 and arr.shape[-1] >= 3:
            number_of_frames = int(getattr(ds, "NumberOfFrames", arr.shape[0]) or arr.shape[0])

            for frame_index in range(min(number_of_frames, arr.shape[0])):
                gray = _color_to_gray(arr[frame_index])
                append_slice(
                    gray,
                    f"{source_name} frame {frame_index + 1}",
                    frame_index=frame_index,
                    color=True
                )

            return slices

        raise ValueError(f"Unsupported color DICOM pixel array shape: {arr.shape}")

    if arr.ndim == 2:
        pixels = arr.astype(np.float32) * slope + intercept
        append_slice(pixels, source_name, color=False)
        return slices

    if arr.ndim == 3:
        number_of_frames = int(getattr(ds, "NumberOfFrames", 0) or 0)

        if number_of_frames <= 1:
            raise ValueError(f"3D grayscale pixel array but NumberOfFrames missing/invalid. Shape={arr.shape}")

        if arr.shape[0] != number_of_frames or arr.shape[1] != rows or arr.shape[2] != cols:
            raise ValueError(
                f"3D grayscale shape mismatch. Shape={arr.shape}, NumberOfFrames={number_of_frames}, Rows={rows}, Cols={cols}"
            )

        for frame_index in range(number_of_frames):
            pixels = arr[frame_index].astype(np.float32) * slope + intercept
            append_slice(
                pixels,
                f"{source_name} frame {frame_index + 1}",
                frame_index=frame_index,
                color=False
            )

        return slices

    raise ValueError(f"Unsupported pixel array shape: {arr.shape}, SamplesPerPixel={samples_per_pixel}")


def read_dicom_stack(uploaded_file):
    original_filename = getattr(uploaded_file, "filename", "") or "uploaded_file"
    file_bytes = _read_uploaded_bytes(uploaded_file)

    slices = []
    scanned_files = []
    read_errors = []

    def try_add_dicom(data, source_name):
        if not data:
            return

        scanned_files.append(source_name)

        try:
            found = _extract_slices_from_dicom_bytes(data, source_name=source_name)

            if found:
                slices.extend(found)

        except Exception as exc:
            if len(read_errors) < 30:
                read_errors.append(f"{source_name}: {type(exc).__name__}: {exc}")

    def scan_zip_bytes(zip_bytes, zip_name, depth=0):
        if depth > 5:
            read_errors.append(f"{zip_name}: nested ZIP depth > 5 skipped")
            return

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue

                    member_name = member.filename or "unknown_file"
                    lower_name = member_name.lower()

                    if (
                        lower_name.startswith("__macosx/")
                        or lower_name.endswith(".ds_store")
                        or lower_name.endswith("thumbs.db")
                    ):
                        continue

                    try:
                        member_bytes = zf.read(member)
                    except Exception as exc:
                        if len(read_errors) < 30:
                            read_errors.append(f"{member_name}: ZIP read error: {exc}")
                        continue

                    if not member_bytes:
                        continue

                    if zipfile.is_zipfile(io.BytesIO(member_bytes)):
                        scan_zip_bytes(member_bytes, member_name, depth + 1)
                    else:
                        try_add_dicom(member_bytes, member_name)

        except Exception as exc:
            if len(read_errors) < 30:
                read_errors.append(f"{zip_name}: ZIP open error: {exc}")

    if zipfile.is_zipfile(io.BytesIO(file_bytes)):
        scan_zip_bytes(file_bytes, original_filename, depth=0)
    else:
        try_add_dicom(file_bytes, original_filename)

    if not slices:
        preview = scanned_files[:20]
        error_preview = read_errors[:10]

        raise ValueError(
            f"[version={ROBUST_VERSION}] No readable DICOM image files were found. "
            "The ZIP/file was opened, but none decoded into usable image pixels. "
            "Scanned examples: "
            + ", ".join(preview)
            + " | Read errors: "
            + " || ".join(error_preview)
        )

    def sort_key(item):
        info = item.get("info", {})
        image_position_z = info.get("imagePositionZ")
        instance = info.get("instanceNumber")
        z_position = info.get("sliceLocation")
        source = item.get("sourceName", item.get("label", ""))

        try:
            if image_position_z is not None:
                return (0, float(image_position_z), _natural_sort_text(source))
        except Exception:
            pass

        try:
            if instance is not None:
                return (1, float(instance), _natural_sort_text(source))
        except Exception:
            pass

        try:
            if z_position is not None:
                return (2, float(z_position), _natural_sort_text(source))
        except Exception:
            pass

        return (3, 0, _natural_sort_text(source))

    slices.sort(key=sort_key)

    total = len(slices)

    for index, item in enumerate(slices):
        item["sliceIndex"] = index
        item["label"] = f"Slice {index + 1} of {total}"
        item["info"]["sliceIndex"] = index
        item["info"]["sliceLabel"] = item["label"]
        item["info"]["readerVersion"] = ROBUST_VERSION

    return slices
