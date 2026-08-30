from __future__ import annotations

from PIL import Image, ImageOps


def _pixels(gray: Image.Image) -> list[int]:
    if hasattr(gray, "get_flattened_data"):
        return list(gray.get_flattened_data())
    return list(gray.getdata())


def dhash(im: Image.Image, hash_size: int = 12) -> int:
    gray = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = _pixels(gray)
    bits = 0
    row_w = hash_size + 1
    for y in range(hash_size):
        row = pixels[y * row_w : (y + 1) * row_w]
        for x in range(hash_size):
            bits = (bits << 1) | (1 if row[x] > row[x + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def fill_crop(im: Image.Image, tw: int, th: int) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    scale = max(tw / im.width, th / im.height)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return im.crop((left, top, left + tw, top + th))


def sample_size(query_w: int, query_h: int, long_edge: int = 320) -> tuple[int, int]:
    aspect = query_w / max(query_h, 1)
    if aspect >= 1:
        tw = long_edge
        th = max(1, int(round(long_edge / aspect)))
    else:
        th = long_edge
        tw = max(1, int(round(long_edge * aspect)))
    return tw, th


def center_box(w: int, h: int) -> tuple[int, int, int, int]:
    """Crop that skips typical desktop widgets in the corners."""
    left = int(w * 0.28)
    right = int(w * 0.72)
    top = int(h * 0.12)
    bottom = int(h * 0.78)
    if right - left < 32 or bottom - top < 32:
        return (0, 0, w, h)
    return (left, top, right, bottom)


def query_hashes(query: Image.Image) -> tuple[int, int, int, int]:
    tw, th = sample_size(query.width, query.height)
    fitted = fill_crop(query, tw, th)
    box = center_box(tw, th)
    return dhash(fitted), dhash(fitted.crop(box)), tw, th
