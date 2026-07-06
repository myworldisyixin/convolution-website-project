import time
import uuid

from services.dicom_config import CACHE_MAX_AGE_SECONDS, CACHE_MAX_ITEMS


DICOM_STACK_CACHE = {}


def cleanup_stack_cache():
    now = time.time()

    expired = []
    for key, value in DICOM_STACK_CACHE.items():
        if now - value.get("created_at", now) > CACHE_MAX_AGE_SECONDS:
            expired.append(key)

    for key in expired:
        DICOM_STACK_CACHE.pop(key, None)

    while len(DICOM_STACK_CACHE) > CACHE_MAX_ITEMS:
        oldest = min(
            DICOM_STACK_CACHE.keys(),
            key=lambda k: DICOM_STACK_CACHE[k].get("created_at", now)
        )
        DICOM_STACK_CACHE.pop(oldest, None)


def store_dicom_stack(slices, filename=""):
    cleanup_stack_cache()

    stack_id = str(uuid.uuid4())
    DICOM_STACK_CACHE[stack_id] = {
        "created_at": time.time(),
        "filename": filename,
        "slices": slices,
    }
    return stack_id


def get_cached_dicom_stack(stack_id):
    cleanup_stack_cache()

    if not stack_id:
        raise ValueError("Missing DICOM stack ID.")

    cached = DICOM_STACK_CACHE.get(stack_id)

    if not cached:
        raise ValueError(
            "DICOM stack cache expired or was not found. Please upload the DICOM file again."
        )

    cached["created_at"] = time.time()
    return cached["slices"]
