import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _pad_mode(mode):
    return {
        "reflect": "reflect",
        "mirror": "symmetric",
        "nearest": "edge",
        "wrap": "wrap",
        "constant": "constant",
    }.get(str(mode).lower(), "reflect")


def _convolve_2d(image, kernel, mode="reflect", cval=0.0):
    image = np.asarray(image)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError(
            f"Expected 2D image and 2D kernel, got {image.shape} and {kernel.shape}."
        )

    kh, kw = kernel.shape
    top = kh // 2
    bottom = kh - 1 - top
    left = kw // 2
    right = kw - 1 - left

    np_mode = _pad_mode(mode)
    pads = ((top, bottom), (left, right))

    if np_mode == "constant":
        padded = np.pad(image, pads, mode=np_mode, constant_values=float(cval))
    else:
        padded = np.pad(image, pads, mode=np_mode)

    windows = sliding_window_view(padded, (kh, kw))
    flipped = kernel[::-1, ::-1]
    return np.einsum("ijkl,kl->ij", windows, flipped, optimize=True)


def convolve(input, weights, output=None, mode="reflect", cval=0.0, origin=0):
    if origin not in (0, (0, 0), [0, 0]):
        raise ValueError("This helper supports origin=0 only.")

    image = np.asarray(input)
    kernel = np.asarray(weights)

    if image.ndim == 2:
        result = _convolve_2d(image, kernel, mode, cval)
    elif image.ndim == 3 and kernel.ndim == 2:
        result = np.stack(
            [_convolve_2d(image[..., c], kernel, mode, cval)
             for c in range(image.shape[-1])],
            axis=-1,
        )
    else:
        raise ValueError(
            f"Supported: 2D image or channel-last 3D image with 2D kernel. "
            f"Got image={image.shape}, kernel={kernel.shape}."
        )

    if output is not None:
        if isinstance(output, np.ndarray):
            output[...] = result.astype(output.dtype, copy=False)
            return output
        return result.astype(output, copy=False)

    return result
