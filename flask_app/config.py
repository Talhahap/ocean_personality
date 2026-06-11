"""
Konfigurasi Flask app.
Override via environment variables (lihat .env.example).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# load .env dari root project (parent dari flask_app/)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')


class Config:
    # ─── Flask ───────────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-in-production')
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 24  # 1 day

    # ─── Upload ──────────────────────────────────────────────────────────────
    UPLOAD_FOLDER     = str(Path(__file__).parent / 'uploads')
    ALLOWED_EXT       = {'mp4', 'webm', 'mov', 'avi', 'mkv'}
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

    # ─── Model ───────────────────────────────────────────────────────────────
    # Path ke model .h5 hasil training.
    # User bisa override via env: MODEL_PATH=weights/faces/adam/face_full_model.h5
    MODEL_PATH        = os.getenv(
        'MODEL_PATH',
        str(ROOT / 'weights' / 'faces' / 'adam' / 'face_full_model.h5')
    )
    ARCFACE_WEIGHTS   = str(ROOT / 'arcface_weights.h5')
    NUM_FRAMES        = 10
    IMG_SIZE          = (112, 112)

    # ─── MySQL ───────────────────────────────────────────────────────────────
    MYSQL_HOST     = os.getenv('MYSQL_HOST',     'localhost')
    MYSQL_PORT     = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER     = os.getenv('MYSQL_USER',     'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DB       = os.getenv('MYSQL_DB',       'ocean_personality')
