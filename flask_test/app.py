import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template, jsonify
from mtcnn import MTCNN
from werkzeug.utils import secure_filename
import sys

# ── Konfigurasi ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXT    = {'mp4', 'avi', 'mov', 'mkv'}
IMAGE_SIZE     = (112, 112)
NUM_FRAMES     = 10
TRAITS         = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
TRAIT_DESC     = {
    'Openness'         : 'Kreativitas, rasa ingin tahu, dan keterbukaan terhadap pengalaman baru.',
    'Conscientiousness': 'Disiplin, keteraturan, dan rasa tanggung jawab yang tinggi.',
    'Extraversion'     : 'Energi sosial, antusiasme, dan kenyamanan dalam berinteraksi.',
    'Agreeableness'    : 'Empati, kooperatif, dan kebaikan hati.',
    'Neuroticism'      : 'Kecenderungan mengalami emosi negatif dan ketidakstabilan emosi.',
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from models import create_arcface_model

print('Memuat ArcFace model...')
arcface = create_arcface_model(load_weights=True)
arcface.trainable = False

# Cari full model terbaru
def find_latest_full_model():
    faces_dir = os.path.join(BASE_DIR, 'weights', 'faces')
    best = None
    best_time = 0
    for root, dirs, files in os.walk(faces_dir):
        for f in files:
            if f == 'face_full_model.h5':
                fp = os.path.join(root, f)
                t  = os.path.getmtime(fp)
                if t > best_time:
                    best_time = t
                    best = fp
    return best

# Build model arsitektur (sama dengan training)
def build_personality_model():
    from tensorflow.keras import regularizers
    inputs = keras.layers.Input(shape=(NUM_FRAMES, *IMAGE_SIZE, 3), name='Input')
    x = keras.layers.TimeDistributed(
        keras.layers.Rescaling(scale=1./255.0), name='Rescaling')(inputs)
    x = keras.layers.TimeDistributed(arcface, name='ArcFace')(x)
    x = keras.layers.LSTM(128, return_sequences=True, name='LSTM_1')(x)
    x = keras.layers.LSTM(64, name='LSTM_2')(x)
    x = keras.layers.Dropout(0.2, name='Dropout_LSTM')(x)
    x = keras.layers.Dense(1024, activation='relu',
                           kernel_regularizer=regularizers.l2(0.01), name='Dense_1')(x)
    x = keras.layers.Dropout(0.3, name='Dropout_1')(x)
    x = keras.layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l2(0.01), name='Dense_2')(x)
    x = keras.layers.Dropout(0.3, name='Dropout_2')(x)
    x = keras.layers.Dense(256, activation='relu',
                           kernel_regularizer=regularizers.l2(0.01), name='Dense_3')(x)
    x = keras.layers.Dropout(0.5, name='Dropout_3')(x)
    output = keras.layers.Dense(5, activation='sigmoid', name='OCEAN_Output')(x)
    return keras.models.Model(inputs=inputs, outputs=output)

# Coba load full model, fallback ke rebuild + load weights
model = None
full_model_path = find_latest_full_model()
if full_model_path:
    print(f'Mencoba load full model: {full_model_path}')
    try:
        model = keras.models.load_model(full_model_path, compile=False)
        print('Full model berhasil di-load!')
    except Exception as e:
        print(f'Full model gagal: {e}')

if model is None:
    # Fallback: build ulang & load weights
    print('Fallback: rebuild model + load weights...')
    model = build_personality_model()
    # Cari face.h5 terbaru
    faces_dir = os.path.join(BASE_DIR, 'weights', 'faces')
    best_w, best_time = None, 0
    for root, _, files in os.walk(faces_dir):
        for f in files:
            if f == 'face.h5':
                fp = os.path.join(root, f)
                t  = os.path.getmtime(fp)
                if t > best_time:
                    best_time = t
                    best_w = fp
    if best_w:
        model.load_weights(best_w)
        print(f'Weights di-load dari: {best_w}')

model.compile(loss='mse', metrics=['mae'])
print('Model siap digunakan!')

# ── Face extraction ────────────────────────────────────────────────────────────
detector = MTCNN()

def extract_faces_from_video(video_path, num_images=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // num_images, 1)
    faces = []
    last_valid = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)

    for i in range(num_images):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            faces.append(last_valid)
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detector.detect_faces(frame_rgb)
        if boxes:
            x, y, w, h = boxes[0]['box']
            x, y = max(0, x), max(0, y)
            crop = frame_rgb[y:y+h, x:x+w]
            if crop.size > 0:
                face = cv2.resize(crop, IMAGE_SIZE).astype(np.float32)
                faces.append(face)
                last_valid = face
                continue
        faces.append(last_valid)

    cap.release()
    while len(faces) < num_images:
        faces.append(last_valid)
    return np.array(faces[:num_images], dtype=np.float32)

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html', traits=TRAITS, trait_desc=TRAIT_DESC)

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'Tidak ada file video'}), 400
    file = request.files['video']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak didukung. Gunakan MP4/AVI/MOV/MKV'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        faces = extract_faces_from_video(filepath)
        faces_batch = np.expand_dims(faces, axis=0)  # (1, 10, 112, 112, 3)
        predictions = model.predict(faces_batch, verbose=0)[0]

        results = []
        for trait, score in zip(TRAITS, predictions):
            pct = float(score) * 100
            results.append({
                'trait'      : trait,
                'score'      : round(float(score), 4),
                'percent'    : round(pct, 2),
                'description': TRAIT_DESC[trait],
                'level'      : 'Tinggi' if pct >= 60 else ('Sedang' if pct >= 40 else 'Rendah'),
            })

        # Deteksi stuck: semua nilai hampir sama
        scores = [r['score'] for r in results]
        is_stuck = (max(scores) - min(scores)) < 0.05

        os.remove(filepath)
        return jsonify({
            'success'   : True,
            'results'   : results,
            'is_stuck'  : is_stuck,
            'stuck_msg' : 'Model mungkin STUCK — semua nilai terlalu mirip!' if is_stuck else '',
            'model_path': full_model_path or 'weights rebuilt',
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
