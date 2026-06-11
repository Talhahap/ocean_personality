# OCEAN Personality Prediction — ArcFace + Flask

Sistem prediksi kepribadian **Big Five (OCEAN)** dari video wajah, menggunakan
**ArcFace** (transfer learning) sebagai backbone face encoder + **LSTM** untuk
temporal modelling, dan di-deploy via **Flask + MySQL InnoDB**.

Dataset: **ChaLearn First Impressions V2 (CVPR'17)**.

---

## Pipeline

```
            ChaLearn V2 videos
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ 00_DataPreparation.ipynb        │   MTCNN → 10 wajah / video, 112×112
   │  → data/faces/{train,val,test}/ │   tf.data.Dataset → data/ds/
   └─────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ 02_Face_Optimizer_*.ipynb       │   ArcFace (frozen) + TimeDistributed
   │  · Adam   · SGD                 │   + LSTM + Dense → 5 skor OCEAN
   │  · RMS    · NoOpt               │   weights/faces/*/face*.h5
   └─────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ 03_OptimizerComparison.ipynb    │   Bandingkan 4 optimizer
   └─────────────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ flask_app/                      │   Login / Register / Upload-Record video
   │  · Flask + Jinja2               │   → prediksi OCEAN → simpan ke MySQL
   │  · MySQL (InnoDB)               │
   └─────────────────────────────────┘
```

---

## Struktur folder

```
joki 15 mei/
├── README.md                          ← lu lagi baca ini
├── .env.example                       ← copy ke .env, isi kredensial MySQL
├── .gitignore
├── requirements.txt
│
├── arcface_weights.h5                 ← pretrained ArcFace (131 MB)
├── models.py                          ← arsitektur ArcFace + OCEAN regressor
├── utils.py                           ← preprocessing, eval, dataset builder
├── train.py                           ← training script standalone (CLI)
│
├── 00_DataPreparation.ipynb           ← preprocessing video → tf.data.Dataset
├── 02_Face_Optimizer_adam.ipynb       ← training Adam
├── 02_Face_Optimizer_SGD.ipynb        ← training SGD (momentum 0.9)
├── 02_Face_Optimizer_RMS.ipynb        ← training RMSprop
├── 02_Face_nooptimizer.ipynb          ← baseline SGD default
├── 03_OptimizerComparison.ipynb       ← compare 4 optimizer
│
├── annotations/                       ← .pkl skor OCEAN
│   ├── annotation_training.pkl
│   ├── annotation_validation.pkl
│   └── annotation_test.pkl
├── transcriptions/                    ← .pkl transcript audio (opsional)
├── CVPR 2017 dataset/                 ← 1000 video (600 / 200 / 200)
│   ├── train 600/
│   ├── val 200/
│   └── test 200/
│
└── flask_app/
    ├── app.py                         ← main Flask application
    ├── config.py
    ├── database.py                    ← MySQL pool + helpers
    ├── inference.py                   ← video → faces → OCEAN
    ├── schema.sql                     ← DDL InnoDB
    ├── static/
    │   ├── css/style.css
    │   └── js/main.js                 ← tab + webcam recording
    ├── templates/
    │   ├── base.html
    │   ├── login.html
    │   ├── register.html
    │   ├── dashboard.html
    │   └── result.html
    └── uploads/                       ← (gitignored) video user
```

---

## Setup (one-time)

### 1. Buat virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Catatan TensorFlow.** Versi yang dipin (`2.13.0`) terbukti jalan bareng
> `mtcnn 0.1.1` di Windows + Python 3.10. Kalau pakai GPU, install
> `tensorflow-gpu` matching CUDA lu.

### 3. ArcFace pretrained weights

File `arcface_weights.h5` (≈131 MB) sudah di-include di root project.
Kalau hilang, download ulang:

```bash
curl -L -o arcface_weights.h5 \
  https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5
```

### 4. Setup MySQL (InnoDB)

Pastikan MySQL Server jalan, lalu copy `.env.example` ke `.env` dan
sesuaikan kredensialnya:

```bash
cp .env.example .env
# edit .env → set MYSQL_USER & MYSQL_PASSWORD
```

Bikin database & tabel:

```bash
# opsi A: pakai schema.sql langsung
mysql -u root -p < flask_app/schema.sql

# opsi B: pakai bootstrap di Flask
cd flask_app
python app.py --init-db
```

---

## Training pipeline

### Step 1 — Preprocessing (sekali aja, lama)

Notebook: **`00_DataPreparation.ipynb`**

Output:
- `data/faces/{train,val,test}/<video_id>/face_*.jpg` — 10 wajah per video, 112×112
- `data/ds/{train,val,test}_ds/` — `tf.data.Dataset` siap konsumsi

Atau jalanin via Python:

```python
from utils import extract_face_from_video, build_dataset_from_faces, load_annotations
import tensorflow as tf

ann_train, ann_val, ann_test = load_annotations('annotations')

extract_face_from_video('CVPR 2017 dataset/train 600', 'data/faces/train')
ds = build_dataset_from_faces('data/faces/train', ann_train, batch_size=8)
tf.data.Dataset.save(ds, 'data/ds/train_ds')
```

> **Tips.** MTCNN di CPU ≈ 4–5 detik/video → 1000 video ≈ 1.5 jam.
> Run sekali, kemudian re-use hasilnya untuk semua eksperimen.

### Step 2 — Training (pilih satu optimizer)

**Via notebook** — buka salah satu:
- `02_Face_Optimizer_adam.ipynb`
- `02_Face_Optimizer_SGD.ipynb`
- `02_Face_Optimizer_RMS.ipynb`
- `02_Face_nooptimizer.ipynb`

**Via CLI** (lebih cepat reproduce):

```bash
python train.py --optimizer adam    --epochs 100 --batch_size 8
python train.py --optimizer sgd
python train.py --optimizer rmsprop
python train.py --optimizer noopt
```

Output:
- `weights/faces/<opt>/face.h5`               — best weights (val_mae)
- `weights/faces/<opt>/face_full_model.h5`    — full model untuk Flask
- `weights/faces/<opt>/history.csv`           — log per-epoch
- `weights/faces/<opt>/training_curve.png`    — plot loss & MAE

Config sesuai request client:
- Loss MSE / metric MAE → accuracy = 1 − MAE (ChaLearn convention)
- Optimizer: pilihan 4 (Adam / SGD / RMSprop / NoOpt)
- Callback: `EarlyStopping(patience=10)` + `ModelCheckpoint(monitor='val_mae')`
- Batch size 8, max 100 epoch
- ArcFace backbone **frozen**, head LSTM + Dense ter-train

### Step 3 — Perbandingan optimizer

Notebook: **`03_OptimizerComparison.ipynb`**

Update path `OPT_WEIGHTS` di cell 2 dengan lokasi best weights masing-masing
optimizer. Output:
- Tabel ringkas accuracy train/val/test
- Bar chart per-trait OCEAN per optimizer → `reports/optimizer_comparison.png`
- Dump JSON → `reports/optimizer_comparison.json`

---

## Deployment — Flask web app

### 1. Pastikan model siap

Set `MODEL_PATH` di `.env` ke file `.h5` hasil training yang lu mau pakai:

```bash
MODEL_PATH=weights/faces/adam/face_full_model.h5
```

### 2. Run

```bash
cd flask_app
python app.py --debug
```

Buka **http://localhost:5000** → register → login → dashboard.

Flag CLI:
```bash
python app.py --init-db          # bootstrap MySQL schema lalu exit
python app.py --no-model         # skip load model (untuk dev UI saja)
python app.py --host 0.0.0.0 --port 8080
```

### 3. Fitur

- **Register / Login** — password di-hash via Werkzeug PBKDF2
- **Dashboard** — 2 tab:
  - **Upload Video** — drag mp4/webm/mov/avi/mkv (max 200 MB)
  - **Rekam Webcam** — pakai MediaRecorder API, output `.webm`
- **Prediksi** — async fetch ke `/predict`, return JSON 5 skor OCEAN
- **History** — 20 prediksi terbaru per user, ada link ke detail
- **Detail page** — video playback + bar chart 5 trait

---

## Catatan teknis

### Loss function

Notebook saat ini pakai **MSE + MAE metric** (standar regresi OCEAN, mengikuti
ChaLearn).

Request client juga menyebut **"Additive Angular Margin (ArcFace loss)"**.
ArcFace loss adalah loss klasifikasi (margin di angular space) — tidak
diaplikasi langsung di task **regresi** 5 skor kontinu OCEAN, melainkan di
fase **pre-training backbone ArcFace** (face recognition di MS1M). Backbone
yang sudah di-train dengan ArcFace loss kita pakai sebagai **frozen feature
extractor**, lalu head regression OCEAN di-fine-tune dengan MSE/MAE.

Pendekatan ini adalah praktik standar transfer learning ArcFace untuk task
hilir regresi (lihat literatur ChaLearn 2017). Jika dosen pembimbing minta
ArcFace loss di-train ulang, itu lingkup yang sangat berbeda dan butuh
dataset face-ID berlabel kelas (bukan OCEAN).

### Per-trait accuracy

`utils.per_trait_accuracy()` menghitung `1 − MAE` per dimensi OCEAN,
dikalikan 100. Convention dari **ChaLearn First Impressions 2017**.

### Fallback wajah

Kalau MTCNN gagal deteksi:
1. Pakai wajah valid terakhir.
2. Kalau dari awal belum pernah ada wajah valid → gambar putih (sesuai request).

---

## Troubleshooting

| Gejala | Solusi |
|---|---|
| `mtcnn` install error | `pip install mtcnn==0.1.1` (pin ke versi lama, baru kompat dengan TF 2.13) |
| `mysql.connector.errors.InterfaceError: 2003` | Cek MySQL service jalan. `mysql -u root -p` dulu manual. |
| `MODEL_PATH not found` warning di Flask | Train dulu, atau set `MODEL_PATH=...` di `.env` |
| `403` saat akses `/uploads/<file>` | Itu fitur (ownership check). Video cuma bisa diakses pemiliknya. |
| Webcam tidak bisa diakses | Browser butuh **HTTPS** atau **localhost**. Pakai `localhost:5000`, bukan IP LAN. |
| OOM saat training | Turunkan `--batch_size` ke 4 atau 2 |

---

## Lisensi & kredit

- **ArcFace implementation**: `serengil/deepface`
- **Dataset**: ChaLearn First Impressions V2 (CVPR 2017)
- **Annotation source**: Amazon Mechanical Turk (AMT)

Kode dalam repo ini untuk keperluan tugas akhir; harap kreditkan sumber-sumber
di atas dalam laporan / publikasi.
