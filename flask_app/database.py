"""
database.py
-----------
Thin wrapper untuk koneksi MySQL InnoDB.
Pakai `mysql-connector-python` (pure Python, gak butuh build deps).
"""
import mysql.connector
from mysql.connector import pooling, Error
from flask import g, current_app


_POOL = None


def init_pool(app):
    """Inisialisasi connection pool sekali saat app start."""
    global _POOL
    if _POOL is not None:
        return
    _POOL = pooling.MySQLConnectionPool(
        pool_name='ocean_pool',
        pool_size=5,
        host=app.config['MYSQL_HOST'],
        port=app.config['MYSQL_PORT'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        database=app.config['MYSQL_DB'],
        autocommit=False,
        charset='utf8mb4',
    )
    app.logger.info('MySQL pool initialized.')


def get_db():
    """Ambil koneksi dari pool, simpan di flask.g supaya 1 koneksi per request."""
    if 'db' not in g:
        if _POOL is None:
            init_pool(current_app)
        g.db = _POOL.get_connection()
    return g.db


def close_db(exc=None):
    """Tutup koneksi (kembalikan ke pool) saat request selesai."""
    db = g.pop('db', None)
    if db is not None:
        try:
            db.close()
        except Error:
            pass


# ─── Query helpers ───────────────────────────────────────────────────────────
def query(sql, params=None, fetchone=False):
    """SELECT helper."""
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, params or ())
    result = cur.fetchone() if fetchone else cur.fetchall()
    cur.close()
    return result


def execute(sql, params=None):
    """INSERT/UPDATE/DELETE helper. Returns lastrowid."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    last_id = cur.lastrowid
    conn.commit()
    cur.close()
    return last_id


# ─── Schema bootstrap ────────────────────────────────────────────────────────
def init_schema(app):
    """
    Buat database & tabel kalau belum ada.
    Jalanin sekali setelah deploy.
    """
    import os
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r', encoding='utf-8') as f:
        ddl = f.read()

    # Connect tanpa specify database supaya bisa CREATE DATABASE
    conn = mysql.connector.connect(
        host=app.config['MYSQL_HOST'],
        port=app.config['MYSQL_PORT'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
    )
    cur = conn.cursor()
    for stmt in ddl.split(';'):
        s = stmt.strip()
        if s:
            cur.execute(s)
    conn.commit()
    cur.close()
    conn.close()
    app.logger.info('Schema initialized.')
