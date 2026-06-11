"""
inference.py
------------
Pipeline inference untuk Flask:
    video file → 10 frame wajah (MTCNN) → model ArcFace+LSTM → 5 skor OCEAN.

Model di-load sekali saat aplikasi start (lazy / singleton).
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np


# Lazy imports — biar Flask bisa start dulu tanpa TF kalau diperlukan untuk debug.
_model      = None
_detector   = None
_model_lock = None


OCEAN_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']


def _load_model(model_path, arcface_weights_path):
    """Lazy-load Keras model. Di-import di sini biar Flask start gak lama."""
    global _model
    import tensorflow as tf  # noqa
    from tensorflow import keras

    # Coba load sebagai full model dulu. Kalau gagal (cuma weights), rebuild arsitektur.
    try:
        # pyrefly: ignore [missing-import]
        import tensorflow.keras.backend as K
        _model = keras.models.load_model(model_path, compile=False, custom_objects={'K': K})
        print(f'[inference] Loaded full model: {model_path}')
    except (IOError, ValueError, OSError) as e:
        print(f'[inference] load_model gagal ({e}); coba rebuild + load_weights...')
        import sys
        ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(ROOT))
        from models import create_arcface_model, create_arcface_personality_model

        arcface = create_arcface_model(load_weights=False)
        # Load ArcFace weights kalau ada (biar struktur konsisten)
        if os.path.exists(arcface_weights_path):
            try:
                arcface.load_weights(arcface_weights_path)
            except Exception as ex:
                print(f'[inference] WARNING: ArcFace weights load gagal: {ex}')

        _model = create_arcface_personality_model(arcface, optimizer=None)
        _model.load_weights(model_path)
        print(f'[inference] Rebuilt model and loaded weights from: {model_path}')

    return _model


def _get_detector():
    """Lazy-load MTCNN detector."""
    global _detector
    if _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector


def init_model(model_path, arcface_weights_path):
    """Panggil saat app start supaya inference call pertama gak lambat."""
    _load_model(model_path, arcface_weights_path)
    _get_detector()


def extract_faces_from_video(video_path, num_frames=10, image_size=(112, 112)):
    """
    Ekstraksi `num_frames` wajah dari video, evenly spaced.
    Fallback: pakai wajah terakhir yang valid; kalau belum ada → gambar putih.

    Return: np.ndarray shape (num_frames, H, W, 3) BGR uint8 (range [0,255]).
    """
    detector = _get_detector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Tidak bisa membuka video: {video_path}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        raise ValueError(f'Video kosong / tidak terbaca: {video_path}')

    frame_idxs = np.linspace(0, total - 1, num_frames, dtype=int)

    white = (np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255)
    last_valid = white.copy()
    has_valid = False

    faces = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            faces.append(last_valid.copy())
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detector.detect_faces(rgb)
        if boxes:
            x, y, w, h = boxes[0]['box']
            x, y = max(0, x), max(0, y)
            crop = frame[y:y+h, x:x+w]
            if crop.size > 0:
                face = cv2.resize(crop, image_size, interpolation=cv2.INTER_AREA)
                faces.append(face)
                last_valid = face
                has_valid = True
                continue
        faces.append(last_valid.copy())

    cap.release()

    if not has_valid:
        # tidak ada wajah terdeteksi sama sekali — pakai 10x gambar putih
        faces = [white.copy() for _ in range(num_frames)]

    return np.stack(faces, axis=0)        # (T, H, W, 3) BGR uint8


def predict_ocean(video_path, num_frames=10, image_size=(112, 112)):
    """
    Full pipeline: video → 10 wajah → model → dict skor OCEAN (dalam PERSEN).
    """
    global _model
    if _model is None:
        raise RuntimeError('Model belum di-load. Panggil init_model() dulu.')

    t0 = time.time()

    # 1. Extract faces
    faces_bgr = extract_faces_from_video(video_path, num_frames, image_size)

    # 2. BGR → RGB
    faces_rgb = faces_bgr[..., ::-1].astype(np.float32)

    # 3. Predict
    batch = np.expand_dims(faces_rgb, axis=0)
    pred = _model.predict(batch, verbose=0)[0]
    pred = np.clip(pred, 0.0, 1.0)

    # Percentage untuk output yang lebih intuitif (0-100%).
    pred_percent = pred * 100

    scores = {OCEAN_TRAITS[i]: float(pred_percent[i]) for i in range(5)}
    elapsed = time.time() - t0

    return {
        'scores': scores,
        'processing_time': round(elapsed, 3),
        'num_faces_detected': int(num_frames),
    }