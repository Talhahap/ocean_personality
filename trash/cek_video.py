import numpy as np
import cv2
import tensorflow as tf
import os
import traceback

# ================= KONFIGURASI =================
# PENTING: Gunakan path ke folder weights, bukan file langsung
MODEL_PATH = r'C:\Users\ALFIAN\TA_CODING\ZIP_FILE\arcface_oceanmodel\ocean-project-deepface\weights\1030_122026'

# Sesuaikan ukuran gambar dengan trainingmu (112 atau 160?)
IMG_SIZE = (112, 112) 

# Jumlah frame yang diambil per video
NUM_FRAMES = 10 
# ===============================================

def load_trained_model(model_path):
    """
    Load model dari folder weights dengan format .t5
    
    Args:
        model_path: Path ke folder yang berisi face.t5 files
    """
    # Cek apakah file-file checkpoint ada
    checkpoint_file = os.path.join(model_path, 'face.t5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Folder tidak ditemukan: {model_path}")
    
    files_required = ['face.t5.index', 'face.t5.data-00000-of-00001', 'checkpoint']
    for f in files_required:
        full_path = os.path.join(model_path, f)
        if not os.path.exists(full_path):
            print(f"⚠️ Warning: File {f} tidak ditemukan di {model_path}")
    
    print(f"Loading checkpoint dari: {checkpoint_file}")
    
    # Build architecture dan load weights
    model = build_full_model()
    model.load_weights(checkpoint_file)
    print("✅ Model berhasil di-load!")
    return model

def build_full_model():
    """
    Build full model architecture sesuai dengan training weights
    """
    import tensorflow.keras as keras
    
    # Build full model dengan LSTM
    inputs = keras.layers.Input(shape=(10, 112, 112, 3), name='Input')
    
    # Normalisasi
    x = keras.layers.Rescaling(scale=1./255.0)(inputs)
    
    # Flatten frame untuk LSTM
    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
    
    # LSTM layers - SESUAIKAN DENGAN CHECKPOINT
    x = keras.layers.LSTM(units=256, return_sequences=True, name='LSTM_1')(x)
    x = keras.layers.LSTM(units=512, name='LSTM_2')(x)  
    
    # Dense layers
    x = keras.layers.Dense(256, activation='relu', name='Dense_1')(x)
    x = keras.layers.Dropout(0.5, name='Dropout_1')(x)
    x = keras.layers.Dense(128, activation='relu', name='Dense_2')(x)
    x = keras.layers.Dropout(0.3, name='Dropout_2')(x)
    
    # Output
    outputs = keras.layers.Dense(5, activation='sigmoid', name='Output')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='OCEAN_Face_Model')
    return model

def process_video(video_path):
    """
    Fungsi ini membuka video, mengambil 10 frame rata, 
    resize, dan normalisasi.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"⚠️ Video {video_path} tidak bisa dibaca atau kosong")
        return None
    
    # Ambil frame dengan interval yang merata
    if total_frames > NUM_FRAMES:
        skip = total_frames // NUM_FRAMES
    else:
        skip = 1

    count = 0
    for i in range(NUM_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, count * skip)
        ret, frame = cap.read()
        if ret:
            # 1. Resize
            frame = cv2.resize(frame, IMG_SIZE)
            # 2. Ganti BGR ke RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 3. Normalisasi 0-1
            frame = frame.astype('float32') / 255.0
            frames.append(frame)
        else:
            # Padding dengan frame hitam jika tidak cukup
            frames.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype='float32'))
        count += 1
    
    cap.release()
    
    # Ubah jadi numpy array dan tambah dimensi Batch
    # Output shape: (1, 10, 112, 112, 3)
    data = np.array(frames, dtype='float32')
    data = np.expand_dims(data, axis=0) 
    return data

# --- MULAI PENGECEKAN ---
print("="*50)
print("SANITY CHECK - MODEL PREDICTION")
print("="*50)

print("\n📂 Sedang meload model...")
try:
    model = load_trained_model(MODEL_PATH)
    print("✅ Model berhasil di-load!")
    print(f"\nModel Summary:")
    model.summary()
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Path video untuk testing
video1_path = r'C:\Users\ALFIAN\TA_CODING\ZIP_FILE\arcface_oceanmodel\ocean-project-deepface\dataset\videos\test\_g3M_MoFIvA.003.mp4'
video2_path = r'C:\Users\ALFIAN\TA_CODING\ZIP_FILE\arcface_oceanmodel\ocean-project-deepface\dataset\videos\test\m4-vvEeWP8s.000.mp4'

print(f"\n🎬 Memproses video 1: {video1_path}...")
input1 = process_video(video1_path)
if input1 is not None:
    pred1 = model.predict(input1, verbose=0)
else:
    print("❌ Gagal memproses video 1")
    exit()

print(f"🎬 Memproses video 2: {video2_path}...")
input2 = process_video(video2_path)
if input2 is not None:
    pred2 = model.predict(input2, verbose=0)
else:
    print("❌ Gagal memproses video 2")
    exit()

# HASIL
print("\n" + "="*50)
print("     HASIL SANITY CHECK     ")
print("="*50)
print(f"Prediksi Video A: {pred1[0]}")
print(f"Prediksi Video B: {pred2[0]}")

# Hitung selisih rata-rata
diff = np.mean(np.abs(pred1 - pred2))
print(f"\nSelisih Rata-rata: {diff:.6f}")
print("-" * 50)

if diff < 0.0001:
    print("🚨 KESIMPULAN: MODEL LAZY / RUSAK")
    print("   Model mengeluarkan prediksi yang SAMA untuk video berbeda")
    print("   Kemungkinan: Model Collapse di training")
else:
    print("✅ KESIMPULAN: MODEL SEHAT")
    print("   Model mengeluarkan prediksi yang BERBEDA untuk video berbeda")
    print("   Jika Web masih error, check input preprocessing-nya")
print("="*50)