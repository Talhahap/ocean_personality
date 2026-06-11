"""
utils.py
--------
Utility functions untuk pipeline OCEAN personality prediction:
- Preprocessing video -> face extraction (MTCNN + OpenCV)
- Load/save tf.data.Dataset
- Load anotasi OCEAN
- Plot training history
- Hitung accuracy per trait
"""

import os
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import tensorflow as tf
from mtcnn import MTCNN
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OCEAN_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
OCEAN_SHORT  = ['O', 'C', 'E', 'A', 'N']
IMG_SIZE     = (112, 112)        # ArcFace native input
NUM_FRAMES   = 10                # 10 frame per video


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def loss_val_graph(history, save_path=None):
    """Plot training/validation loss & MAE dari history Keras."""
    mae      = history.history.get('mae',     [])
    val_mae  = history.history.get('val_mae', [])
    loss     = history.history.get('loss',     [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mae,     label='Training MAE')
    plt.plot(val_mae, label='Validation MAE')
    plt.legend(loc='upper right')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')

    plt.subplot(1, 2, 2)
    plt.plot(loss,     label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Accuracy per OCEAN trait
# ---------------------------------------------------------------------------
def per_trait_accuracy(y_true, y_pred):
    """
    Hitung accuracy per trait OCEAN.
    Konvensi ChaLearn: accuracy = 1 - MAE (skor di range [0, 1]).
    Mengembalikan dict {trait: accuracy_persen}.
    """
    y_true = np.asarray(y_true, np.float32)
    y_pred = np.asarray(y_pred, np.float32)
    mae_per_trait = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    return {
        OCEAN_TRAITS[i]: float((1.0 - mae_per_trait[i]) * 100.0)
        for i in range(5)
    }


def mean_acc(y_true, y_pred):
    """Mean accuracy across 5 traits."""
    accs = per_trait_accuracy(y_true, y_pred)
    return float(np.mean(list(accs.values())))


# ---------------------------------------------------------------------------
# Face extraction
# ---------------------------------------------------------------------------
def extract_face_from_frame(frame_bgr, detector, image_size=IMG_SIZE):
    """
    Deteksi wajah pada satu frame BGR (OpenCV) menggunakan MTCNN.
    Return: ndarray (H, W, 3) BGR atau None kalau gagal.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = detector.detect_faces(frame_rgb)
    if not boxes:
        return None
    x1, y1, w, h = boxes[0]['box']
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + w, y1 + h
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, image_size, interpolation=cv2.INTER_AREA)
    return face


def extract_face_from_video(video_dir, save_dir, num_images=NUM_FRAMES, image_size=IMG_SIZE):
    """
    Iterasi semua .mp4 di video_dir, ekstrak `num_images` wajah merata per video
    pakai MTCNN, simpan ke save_dir/<video_stem>/face_*.jpg.

    Fallback chain:
      1. Pakai wajah valid dari frame sebelumnya kalau deteksi gagal.
      2. Kalau dari awal belum pernah ada wajah valid, pakai gambar putih.
    """
    detector = MTCNN()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted([f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')])
    for video_name in tqdm(videos, desc='Ekstraksi -> ' + save_dir.name, leave=False, ncols=80, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
        stem = Path(video_name).stem
        out_path = save_dir / stem
        out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(Path(video_dir) / video_name))
        if not cap.isOpened():
            print('[WARN] Tidak bisa buka ' + video_name)
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            continue

        frame_idxs = np.linspace(0, total - 1, num_images, dtype=int)

        white = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
        last_valid = white.copy()
        has_valid = False

        faces = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                faces.append(last_valid.copy())
                continue

            face = extract_face_from_frame(frame, detector, image_size)
            if face is not None:
                faces.append(face)
                last_valid = face
                has_valid = True
            else:
                faces.append(last_valid.copy())

        cap.release()

        if not has_valid:
            faces = [white.copy() for _ in range(num_images)]

        for i, f in enumerate(faces[:num_images]):
            cv2.imwrite(str(out_path / ('face_' + str(i) + '.jpg')), f)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
def load_annotations(annotation_dir='./annotations'):
    """Load ChaLearn OCEAN annotation .pkl files."""
    paths = {
        'train': os.path.join(annotation_dir, 'annotation_training.pkl'),
        'val':   os.path.join(annotation_dir, 'annotation_validation.pkl'),
        'test':  os.path.join(annotation_dir, 'annotation_test.pkl'),
    }
    out = {}
    for k, p in paths.items():
        with open(p, 'rb') as f:
            out[k] = pickle.load(f, encoding='latin1')
    return out['train'], out['val'], out['test']


def read_ocean_data(video_stem, annotation):
    """
    Ambil 5 skor OCEAN untuk satu video.
    `video_stem` boleh tanpa .mp4; akan ditambahkan secara otomatis.
    Return: np.ndarray (5,) float32 atau None jika tidak ada di anotasi.
    """
    if video_stem.endswith('.mp4'):
        key = video_stem
    else:
        key = os.path.basename(video_stem) + '.mp4'
    try:
        scores = [annotation[t][key] for t in OCEAN_TRAITS]
        return np.asarray(scores, np.float32)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Dataset builder: faces folder -> tf.data.Dataset
# ---------------------------------------------------------------------------
def build_dataset_from_faces(face_dir, annotation, num_images=NUM_FRAMES,
                             image_size=IMG_SIZE, batch_size=8):
    """
    Bangun tf.data.Dataset dari folder hasil ekstraksi wajah.
    Struktur folder yang diharapkan:
        face_dir/<video_stem>/face_0.jpg ... face_9.jpg
    Output tensor: (num_images, H, W, 3) float32, label (5,) float32.
    """
    X, y = [], []
    folders = sorted(os.listdir(face_dir))
    for folder in tqdm(folders, desc='Loading ' + Path(face_dir).name, leave=False, ncols=80, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
        p = os.path.join(face_dir, folder)
        if not os.path.isdir(p):
            continue
        scores = read_ocean_data(folder, annotation)
        if scores is None:
            continue

        files = sorted([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.png'))])
        if len(files) < num_images:
            continue

        seq = []
        for f in files[:num_images]:
            img = cv2.imread(os.path.join(p, f))
            if img is None:
                img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != image_size:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            seq.append(img.astype(np.float32))
        X.append(np.array(seq, dtype=np.float32))
        y.append(scores)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print('  -> X shape=' + str(X.shape) + ', y shape=' + str(y.shape))

    def gen():
        for a, b in zip(X, y):
            yield a, b

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(num_images, image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(5,), dtype=tf.float32),
        ),
    )
    return ds.batch(batch_size)


# ---------------------------------------------------------------------------
# Audio (kept for completeness, opsional)
# ---------------------------------------------------------------------------
def extract_audio(video_dir, save_dir):
    """Extract .wav audio dari setiap video pakai ffmpeg (opsional)."""
    os.makedirs(save_dir, exist_ok=True)
    for video_name in tqdm(os.listdir(video_dir)):
        stem = Path(video_name).stem
        cmd = ('ffmpeg -y -i "' + str(Path(video_dir) / video_name)
               + '" -ab 320k -ac 2 -ar 44100 -vn "'
               + str(Path(save_dir) / (stem + '.wav')) + '"')
        subprocess.call(cmd, shell=True)


# ---------------------------------------------------------------------------
# Sanity checker
# ---------------------------------------------------------------------------
def check_number_of_images(_dir, n=NUM_FRAMES):
    """Return list of (folder_name, count) yang punya < n gambar."""
    bad = []
    for f in os.listdir(_dir):
        p = os.path.join(_dir, f)
        if os.path.isdir(p):
            k = os.listdir(p)
            if len(k) < n:
                bad.append((f, len(k)))
    return bad
