"""
app.py — OCEAN Personality Prediction Web App
=============================================
Flask + MySQL InnoDB + ArcFace model.

Fitur:
- Login / Register (password hashed dengan Werkzeug PBKDF2).
- Upload video atau rekam via webcam (MediaRecorder API → POST .webm).
- Inference: ekstraksi 10 wajah (MTCNN) → model → 5 skor OCEAN.
- Simpan riwayat prediksi ke MySQL.
- History page.

Run:
    cd flask_app
    python app.py
"""
import os
import uuid
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, jsonify, send_from_directory, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from database import init_pool, close_db, query, execute, init_schema
import inference


# ─── App factory ─────────────────────────────────────────────────────────────
def create_app(load_model_at_start=True):
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # DB
    init_pool(app)
    app.teardown_appcontext(close_db)

    # Model
    if load_model_at_start:
        if os.path.exists(app.config['MODEL_PATH']):
            print(f'[app] Loading model from: {app.config["MODEL_PATH"]}')
            try:
                inference.init_model(
                    app.config['MODEL_PATH'],
                    app.config['ARCFACE_WEIGHTS'],
                )
                print('[app] Model loaded.')
            except Exception as e:
                print(f'[app] WARNING: model gagal di-load: {e}')
                print('[app] Endpoint /predict akan return 503 sampai model tersedia.')
        else:
            print(f'[app] WARNING: MODEL_PATH tidak ditemukan: {app.config["MODEL_PATH"]}')
            print('[app] Endpoint /predict akan return 503 sampai model tersedia.')

    register_routes(app)
    return app


# ─── Auth helpers ────────────────────────────────────────────────────────────
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            flash('Lu harus login dulu, bro.', 'warning')
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    if 'user_id' not in session:
        return None
    return query('SELECT id, username, email, full_name FROM users WHERE id=%s',
                 (session['user_id'],), fetchone=True)


# ─── File helpers ────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXT


# ─── Routes ──────────────────────────────────────────────────────────────────
def register_routes(app):

    @app.route('/')
    def index():
        if 'user_id' in session:
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))

    # ─── Auth ────────────────────────────────────────────────────────────────
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username  = request.form.get('username', '').strip()
            email     = request.form.get('email', '').strip().lower()
            password  = request.form.get('password', '')
            full_name = request.form.get('full_name', '').strip() or None

            if not (username and email and password):
                flash('Username, email, dan password wajib diisi.', 'danger')
                return redirect(url_for('register'))
            if len(password) < 6:
                flash('Password minimal 6 karakter.', 'danger')
                return redirect(url_for('register'))

            # cek duplikat
            dupe = query(
                'SELECT id FROM users WHERE username=%s OR email=%s LIMIT 1',
                (username, email), fetchone=True
            )
            if dupe:
                flash('Username atau email udah dipakai.', 'danger')
                return redirect(url_for('register'))

            pw_hash = generate_password_hash(password)
            execute(
                'INSERT INTO users (username, email, password_hash, full_name) '
                'VALUES (%s, %s, %s, %s)',
                (username, email, pw_hash, full_name),
            )
            flash('Akun berhasil dibuat. Silakan login.', 'success')
            return redirect(url_for('login'))

        return render_template('register.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            identifier = request.form.get('username', '').strip()
            password   = request.form.get('password', '')

            user = query(
                'SELECT id, username, password_hash FROM users '
                'WHERE username=%s OR email=%s LIMIT 1',
                (identifier, identifier.lower()), fetchone=True
            )
            if not user or not check_password_hash(user['password_hash'], password):
                flash('Username atau password salah.', 'danger')
                return redirect(url_for('login'))

            session.clear()
            session['user_id']  = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('dashboard'))

        return render_template('login.html')

    @app.route('/logout')
    def logout():
        session.clear()
        flash('Lu udah logout.', 'info')
        return redirect(url_for('login'))

    # ─── Dashboard ───────────────────────────────────────────────────────────
    @app.route('/dashboard')
    @login_required
    def dashboard():
        user = current_user()
        history = query(
            'SELECT id, subject_name, source, openness, conscientiousness, '
            'extraversion, agreeableness, neuroticism, created_at '
            'FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 20',
            (session['user_id'],)
        )
        return render_template('dashboard.html', user=user, history=history)

    # ─── Prediction ──────────────────────────────────────────────────────────
    @app.route('/predict', methods=['POST'])
    @login_required
    def predict():
        if inference._model is None:
            return jsonify({'error': 'Model belum siap di server. Cek MODEL_PATH.'}), 503

        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded.'}), 400

        f = request.files['video']
        if f.filename == '':
            return jsonify({'error': 'Empty filename.'}), 400

        # Werkzeug secure_filename + UUID prefix biar tidak bentrok
        ext = (f.filename.rsplit('.', 1)[-1] or 'mp4').lower()
        if ext not in Config.ALLOWED_EXT:
            return jsonify({'error': f'Format tidak didukung: .{ext}'}), 400

        unique_name = f'{uuid.uuid4().hex}_{secure_filename(f.filename)}'
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        f.save(save_path)

        source       = request.form.get('source', 'upload')
        subject_name = request.form.get('subject_name', '').strip() or None

        try:
            result = inference.predict_ocean(
                video_path=save_path,
                num_frames=app.config['NUM_FRAMES'],
                image_size=app.config['IMG_SIZE'],
            )
        except Exception as e:
            app.logger.exception('predict_ocean failed')
            # bersihin file kalau gagal
            try: os.remove(save_path)
            except OSError: pass
            return jsonify({'error': f'Inference gagal: {e}'}), 500

        s = result['scores']
        pid = execute(
            'INSERT INTO predictions '
            '(user_id, subject_name, video_filename, source, '
            ' openness, conscientiousness, extraversion, agreeableness, neuroticism, '
            ' processing_time) '
            'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
            (
                session['user_id'], subject_name, unique_name, source,
                s['openness'], s['conscientiousness'], s['extraversion'],
                s['agreeableness'], s['neuroticism'],
                result['processing_time'],
            ),
        )

        return jsonify({
            'prediction_id'   : pid,
            'subject_name'    : subject_name,
            'scores'          : s,
            'processing_time' : result['processing_time'],
            'video_url'       : url_for('uploaded_file', filename=unique_name),
            'result_url'      : url_for('view_result', pid=pid),
        })

    @app.route('/result/<int:pid>')
    @login_required
    def view_result(pid):
        row = query(
            'SELECT * FROM predictions WHERE id=%s AND user_id=%s',
            (pid, session['user_id']), fetchone=True
        )
        if not row:
            abort(404)
        return render_template('result.html', pred=row, user=current_user())

    @app.route('/uploads/<path:filename>')
    @login_required
    def uploaded_file(filename):
        # Cuma user yang punya video boleh akses → cek ownership
        row = query(
            'SELECT user_id FROM predictions WHERE video_filename=%s LIMIT 1',
            (filename,), fetchone=True
        )
        if not row or row['user_id'] != session['user_id']:
            abort(403)
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    # ─── Health ──────────────────────────────────────────────────────────────
    @app.route('/healthz')
    def healthz():
        return jsonify({
            'status'      : 'ok',
            'model_loaded': inference._model is not None,
        })

    # ─── Error handlers ──────────────────────────────────────────────────────
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File terlalu besar (max 200 MB).'}), 413


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-db',    action='store_true',
                        help='Init MySQL schema lalu exit.')
    parser.add_argument('--no-model',   action='store_true',
                        help='Skip load model (dev mode, UI only).')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.init_db:
        # bootstrap schema saja
        app = Flask(__name__)
        app.config.from_object(Config)
        init_schema(app)
        print('Schema initialized. Exit.')
        raise SystemExit(0)

    app = create_app(load_model_at_start=not args.no_model)
    app.run(host=args.host, port=args.port, debug=args.debug)
