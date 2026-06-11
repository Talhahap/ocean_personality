# REVIEW: Perbedaan Training Model antara train.py dan File Notebook 02

## 📋 Ringkasan Eksekutif

Terdapat **perbedaan signifikan** antara `train.py` dan empat notebook (02_Face_*) dalam hal:
1. **Optimizer yang digunakan** (Adam, SGD dengan momentum, RMSprop, atau SGD default)
2. **Arsitektur Dense layer** (jumlah unit dan regularisasi)
3. **Hyperparameter** (learning rate, dropout, regularisasi L2)
4. **Batch size** (implisit vs eksplisit)

Perbedaan ini **memastikan hasil training berbeda** di setiap varian karena perubahan landscape optimasi.

---

## 1️⃣ ARKITEKTUR MODEL

### ✅ Bagian yang SAMA di semua varian:

```
Input (batch, 10, 112, 112, 3)
    ↓
Rescaling (÷255) → [0, 1]
    ↓
TimeDistributed(ArcFace frozen) → (batch, 10, 512)
    ↓
LSTM(128, return_sequences=True) + recurrent_dropout=0.2
    ↓
LSTM(64) + recurrent_dropout=0.2
    ↓
Dropout(0.2)
    ↓
Dense Head (BERBEDA di setiap varian)
    ↓
Dense(5, sigmoid) → 5 trait OCEAN [0,1]
```

### ❌ Bagian yang BERBEDA - Dense Head:

#### **train.py + 02_Face_nooptimizer.ipynb**
```python
Dense(256, relu, L2=1e-4)
    ↓ Dropout(0.3)
Dense(128, relu, L2=1e-4)
    ↓ Dropout(0.5)
Dense(5, sigmoid)
```
**Total trainable params: ~300K**

#### **02_Face_Optimizer_adam.ipynb**
```python
Dense(1024, relu, L2=0.01)  ← JAUH lebih besar + regularisasi lebih kuat
    ↓ Dropout(0.3)
Dense(512, relu, L2=0.01)   ← JAUH lebih besar
    ↓ Dropout(0.3)
Dense(256, relu, L2=0.01)   ← Ada tambahan layer
    ↓ Dropout(0.5)
Dense(5, sigmoid)
```
**Total trainable params: ~2M (7x lebih banyak dari train.py!)**

#### **02_Face_Optimizer_RMS.ipynb + 02_Face_Optimizer_SGD.ipynb**
```python
Sama dengan train.py:
Dense(256, relu, L2=1e-4)
    ↓ Dropout(0.3)
Dense(128, relu, L2=1e-4)
    ↓ Dropout(0.5)
Dense(5, sigmoid)
```
**Total trainable params: ~300K**

---

## 2️⃣ OPTIMIZER DAN LEARNING RATE

| File | Optimizer | Learning Rate | Momentum | Notes |
|------|-----------|---------------|----------|-------|
| **train.py** | `Adam()` | default: 0.001 | N/A | Bisa di-override via argparse |
| **02_Face_nooptimizer.ipynb** | SGD (default) | default: 0.01 | 0 | **Baseline murni, tanpa optimizer eksplisit** |
| **02_Face_Optimizer_adam.ipynb** | `Adam()` | default: 0.001 | N/A | Sama seperti train.py default |
| **02_Face_Optimizer_RMS.ipynb** | `RMSprop()` | default: 0.001 | N/A | Adaptive learning rate |
| **02_Face_Optimizer_SGD.ipynb** | `SGD(momentum=0.9)` | default: 0.01 | 0.9 | Dengan momentum untuk akselerasi konvergensi |

### Implikasi:
- **SGD default** (nooptimizer): Paling sederhana, paling lambat konvergen
- **SGD dengan momentum**: Lebih cepat konvergen, lebih stabil
- **Adam**: Adaptive learning rate per parameter, biasanya tercepat
- **RMSprop**: Adaptive, tengah-tengah antara SGD dan Adam

---

## 3️⃣ REGULARISASI L2

| File | L2 Dense Layer | Impact |
|------|---|---|
| train.py | `1e-4` | Ringan: 0.0001 |
| 02_Face_nooptimizer.ipynb | `1e-4` | Ringan: 0.0001 |
| **02_Face_Optimizer_adam.ipynb** | **`0.01`** | **100x lebih KUAT** |
| 02_Face_Optimizer_RMS.ipynb | `1e-4` | Ringan: 0.0001 |
| 02_Face_Optimizer_SGD.ipynb | `1e-4` | Ringan: 0.0001 |

### Implikasi:
- Regularisasi yang lebih kuat **mengurangi overfitting** tapi **meningkatkan underfitting**
- Notebook Adam punya penalty berat untuk weight besar → model lebih "konservatif"
- Ini bisa membuat validation MAE lebih tinggi di notebook Adam dibanding yang lain

---

## 4️⃣ HYPERPARAMETER LAINNYA

### Batch Size:
- **train.py**: Default argparse = 8 (bisa di-override)
- **Semua notebook**: Tidak eksplisit di `model.fit()` → Keras default = 32

```python
# train.py (explicit batch_size)
model.fit(train_ds, ..., batch_size=args.batch_size)  # default 8

# Notebook (implisit)
model.fit(train_ds, ..., epochs=100)  # Keras gunakan batch_size=32
```

**PENTING**: Batch size 8 vs 32 → gradient descent lebih berbeda per epoch!

### EarlyStopping:
Semua menggunakan `patience=10` identik ✓

### Epochs:
- **train.py**: default 100 (bisa di-override)
- **Semua notebook**: 100 eksplisit

---

## 5️⃣ DATASET LOADING

### train.py:
```python
train_ds = (
    tf.data.Dataset.load(args.train_ds)
    .cache()
    .shuffle(buffer_size=1000, seed=42)
    .prefetch(tf.data.AUTOTUNE)
)
```

### Notebook:
```python
train_ds = tf.data.Dataset.load(r'data/ds/train_ds') \
    .cache().shuffle(buffer_size=1000, seed=42)\
    .prefetch(buffer_size=tf.data.AUTOTUNE)
```

**SAMA** ✓ (seed=42 memastikan shuffle reproducible)

### Validation Dataset:

**train.py:**
```python
val_ds = (
    tf.data.Dataset.load(args.val_ds)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
```

**Notebook:**
```python
valid_ds = tf.data.Dataset.load(r'data/ds/val_ds') \
    .cache().shuffle(buffer_size=1000, seed=42)\  # ← SHUFFLE PADA VALIDATION!
    .prefetch(buffer_size=tf.data.AUTOTUNE)
```

⚠️ **PERBEDAAN KRITIS**: Notebook men-shuffle validation data, train.py **tidak**!

Shuffle pada val_ds akan membuat validation metrics kurang stabil. Ini adalah **bug** di notebook.

---

## 6️⃣ PERBANDINGAN DETAILED - PARAMETER LAYER

### train.py:
```python
# Dense layer
Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='Dense_1')(x)
Dropout(0.3, name='Dropout_1')(x)

Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='Dense_2')(x)
Dropout(0.5, name='Dropout_2')(x)

# Output
Dense(5, activation='sigmoid', name='OCEAN_Output')(x)
```

### 02_Face_Optimizer_adam.ipynb:
```python
# Dense layer - JAUH LEBIH BESAR
Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='Dense_1')(x)
Dropout(0.3, name='Dropout_1')(x)

Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='Dense_2')(x)
Dropout(0.3, name='Dropout_2')(x)

Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='Dense_3')(x)  # Layer tambahan!
Dropout(0.5, name='Dropout_3')(x)

# Output
Dense(5, activation='sigmoid', name='OCEAN_Output')(x)
```

---

## 7️⃣ RINGKASAN PERBEDAAN KUNCI

| Aspek | train.py | Nooptimizer | Adam | RMSprop | SGD+Mom |
|-------|----------|-------------|------|---------|---------|
| **Dense units** | 256→128 | 256→128 | **1024→512→256** | 256→128 | 256→128 |
| **L2 Regularisasi** | 1e-4 | 1e-4 | **0.01** | 1e-4 | 1e-4 |
| **Total params** | ~300K | ~300K | **~2M** | ~300K | ~300K |
| **Optimizer** | Adam | SGD (default) | Adam | RMSprop | SGD(mom=0.9) |
| **Learning Rate** | 0.001 | 0.01 | 0.001 | 0.001 | 0.01 |
| **Batch Size** | 8 | 32 | 32 | 32 | 32 |
| **Val Shuffle** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Epochs** | 100 | 100 | 100 | 100 | 100 |

---

## 8️⃣ MENGAPA HASIL BISA BERBEDA?

### 1. **Landscape Optimasi Berbeda**
- Model Adam dengan 2M params → ruang parameter 7x lebih besar
- Optimizer yang berbeda → trajectory konvergensi berbeda
- Learning rate berbeda → step size berbeda per iterasi

### 2. **Regularisasi yang Lebih Kuat**
- Notebook Adam punya L2=0.01 (100x lebih kuat)
- Ini mendorong weights lebih kecil → underfitting?
- Val MAE bisa lebih tinggi meski training loss lebih rendah

### 3. **Batch Size Impact**
- Batch 8 vs 32 → gradient estimates sangat berbeda
- Batch yang lebih kecil = gradient lebih berisik = training lebih "random"
- Bisa berakibat convergence rate berbeda, bahkan local minima berbeda

### 4. **Optimizer Properties**
- **SGD default**: Sederhana, tapi bisa stuck di flat region
- **SGD + momentum**: Lebih cepat through plateaus
- **Adam**: Adaptive per parameter, biasanya convergence lebih smooth
- **RMSprop**: Tengah-tengah, adaptif tapi lebih stable dari Adam

### 5. **Random Initialization**
- Meskipun seed=42 di data shuffle, **weight initialization berbeda**
- Setiap kali run = random weights baru → training path berbeda
- Ini normal & expected

### 6. **EarlyStopping Trigger**
- Patience=10 identik, tapi monitor metric = validation MAE
- Model dengan architecture berbeda → convergence pattern berbeda
- Bisa stop di epoch 20 atau 95 tergantung model complexity

---

## 9️⃣ PREDIKSI URUTAN PERFORMA (Val MAE)

Berdasarkan prinsip machine learning:

1. **02_Face_Optimizer_adam.ipynb**: Mungkin **UNDERFITTING**
   - 7x lebih banyak params + regularisasi 100x lebih kuat
   - Efektif mengurangi kapasitas
   - Prediksi: Val MAE tertinggi

2. **02_Face_Optimizer_RMS.ipynb**: Tengah-tengah
   - Ukuran model standard, optimizer adaptive
   - Biasanya cukup stabil

3. **train.py + 02_Face_Optimizer_SGD.ipynb**: Mirip, tapi SGD+mom sedikit lebih baik
   - Model kecil, optimizer sederhana
   - Batch size 8 bisa jadi lebih noisy (kurang stabil)

4. **02_Face_nooptimizer.ipynb**: Mungkin **OVERFITTING** tapi fast training
   - SGD pure tanpa adaptasi
   - Learning rate 0.01 yang lebih tinggi
   - Convergence cepat tapi bisa tidak optimal

**CATATAN**: Prediksi ini bisa salah jika dataset atau initialization berbeda!

---

## 🔟 REKOMENDASI

### Untuk Konsistensi:
1. **Standardisasi batch size** di semua notebook ke 8 (match train.py)
   ```python
   model.fit(train_ds, batch_size=8, ...)
   ```

2. **Buang shuffle di validation data**
   ```python
   # Jangan:
   valid_ds = ... .shuffle(...) 
   # Harus:
   valid_ds = ...
   ```

3. **Standardisasi architecture**
   - Gunakan 256→128 untuk semua
   - L2=1e-4 untuk semua
   - Notebook Adam harus di-update

4. **Dokumentasikan peruntukan**
   - train.py = production script
   - 02_Face_*.ipynb = experiment/research
   - Atau buat satu notebook master, notebook lain hanya load weights

### Untuk Experiment Terkontrol:
```python
CONFIGS = {
    'baseline': {'optimizer': 'sgd', 'dense_units': [256, 128], 'l2': 1e-4},
    'adam': {'optimizer': 'adam', 'dense_units': [256, 128], 'l2': 1e-4},
    'rmsprop': {'optimizer': 'rmsprop', 'dense_units': [256, 128], 'l2': 1e-4},
}
```

---

## 📊 KESIMPULAN

| Aspek | Kondisi |
|-------|---------|
| **Kode terpusat?** | ❌ Tersebar di 5 files |
| **Reproducible?** | ⚠️ Mirip tapi banyak subtle differences |
| **Maintainable?** | ❌ Duplikasi kode tinggi |
| **Hasil comparable?** | ❌ Terlalu banyak variabel |

**Rekomendasi utama**: Refactor ke satu source of truth (train.py), notebook hanya untuk visualization/evaluation.

