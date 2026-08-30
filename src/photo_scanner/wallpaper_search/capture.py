from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from photo_scanner.paths import CACHE_DIR, ensure_data_dirs

SWIFT_SRC = Path(__file__).resolve().parent / "wallpaper_capture.swift"


def mapped_originals() -> list[Path]:
    proc = subprocess.run(["pgrep", "-n", "-f", "WallpaperImageExtension"], capture_output=True, text=True)
    pid = proc.stdout.strip().split("\n")[0] if proc.returncode == 0 else ""
    if not pid:
        return []
    listed = subprocess.run(["lsof", "-p", pid], capture_output=True, text=True)
    if listed.returncode != 0:
        return []
    paths: list[Path] = []
    seen: set[str] = set()
    for line in listed.stdout.splitlines():
        marker = "photoslibrary/originals/"
        if marker not in line:
            continue
        path = Path(line[line.find("/") :].strip())
        if path.suffix.lower() not in {".jpeg", ".jpg", ".png", ".heic", ".heif"}:
            continue
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _compile_capture(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ran = subprocess.run(
        ["swiftc", "-parse-as-library", str(src), "-o", str(dest)],
        capture_output=True,
        text=True,
    )
    if ran.returncode != 0:
        raise RuntimeError(f"swiftc failed:\n{ran.stderr or ran.stdout}")


def capture_binary() -> Path:
    ensure_data_dirs()
    digest = hashlib.sha256(SWIFT_SRC.read_bytes()).hexdigest()[:16]
    dest = CACHE_DIR / f"wallpaper-capture-{digest}"
    if dest.is_file() and dest.stat().st_mtime >= SWIFT_SRC.stat().st_mtime:
        return dest
    _compile_capture(SWIFT_SRC, dest)
    return dest


def capture_wallpaper() -> Image.Image:
    binary = capture_binary()
    fd, name = tempfile.mkstemp(suffix=".png", prefix="wallpaper-")
    os.close(fd)
    path = Path(name)
    try:
        ran = subprocess.run([str(binary), str(path)], capture_output=True, text=True)
        if ran.returncode != 0 or not path.is_file() or path.stat().st_size == 0:
            err = (ran.stderr or ran.stdout or "capture failed").strip()
            raise RuntimeError(
                f"{err}\nGrant Screen Recording to your terminal (or Python), then retry. "
                "Or pass a screenshot: image-search shot.png"
            )
        return Image.open(path).convert("RGB")
    finally:
        path.unlink(missing_ok=True)
