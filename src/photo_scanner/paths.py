"""Repo-root locations for data that stays outside src/."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
CACHE_DIR = REPO_ROOT / ".cache"
OUTPUT_DIR = REPO_ROOT / "output"
BAD_PHOTOS_DIR = REPO_ROOT / "BadPhotos"
CONFIG_PATH = REPO_ROOT / "config.json"
TOOLS_DIR = SRC_DIR / "tools"
STATIC_DIR = SRC_DIR / "review_gui" / "static"
THUMB_DIR = CACHE_DIR / "review_thumbs"
SESSION_FILE = OUTPUT_DIR / "review_session.json"
MODEL_PATH = CACHE_DIR / "aesthetic_model.pkl"
LAUNCHER_STATE = CACHE_DIR / "launcher.json"
USERNAME_CAPTIONS_PATH = CACHE_DIR / "username_captions.txt"


def ensure_data_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
