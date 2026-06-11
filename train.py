"""
train.py
--------
Standalone training script (kalau gak mau pakai notebook).
Train ArcFace + LSTM regressor untuk prediksi OCEAN.

Usage:
    python train.py --optimizer adam --epochs 100 --batch_size 8
    python train.py --optimizer sgd  --epochs 100
    python train.py --optimizer rmsprop
    python train.py --optimizer noopt        # SGD default (baseline)
"""
import argparse
import datetime
import os
import warnings

import tensorflow as tf
from tensorflow import keras

from models import create_arcface_model, create_arcface_personality_model
from utils import loss_val_graph

warnings.filterwarnings('ignore')


def get_optimizer(name):
    name = name.lower()
    if name == 'adam':
        return keras.optimizers.Adam()
    if name == 'sgd':
        return keras.optimizers.SGD(momentum=0.9)
    if name == 'rmsprop':
        return keras.optimizers.RMSprop()
    if name == 'noopt':
        # baseline: SGD default (lr=0.01, no momentum)
        return keras.optimizers.SGD()
    raise ValueError(f'Unknown optimizer: {name}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', default='adam',
                   choices=['adam', 'sgd', 'rmsprop', 'noopt'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--train_ds', default='data/ds/train_ds')
    p.add_argument('--val_ds',   default='data/ds/val_ds')
    p.add_argument('--out_dir',  default=None,
                   help='Output dir for weights. Default: weights/faces/<optimizer>')
    args = p.parse_args()

    out_dir = args.out_dir or f'weights/faces/{args.optimizer}'
    os.makedirs(out_dir, exist_ok=True)
    weight_path = os.path.join(out_dir, 'face.h5')

    print(f'TensorFlow: {tf.__version__}')
    print(f'Optimizer : {args.optimizer}')
    print(f'Weights → : {weight_path}')

    # Load datasets
    train_ds = (
        tf.data.Dataset.load(args.train_ds)
        .cache()
        .shuffle(buffer_size=1000, seed=42)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.load(args.val_ds)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build model
    arcface = create_arcface_model(load_weights=True)
    arcface.trainable = False
    model = create_arcface_personality_model(
        arcface, optimizer=get_optimizer(args.optimizer)
    )
    model.summary()

    # Callbacks (sesuai request client: EarlyStopping patience=10, ModelCheckpoint)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_mae', patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=weight_path,
            monitor='val_mae', save_best_only=True,
            save_weights_only=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(os.path.join(out_dir, 'history.csv')),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks, verbose=1,
    )

    # Save full model untuk deployment Flask
    full_path = os.path.join(out_dir, 'face_full_model.h5')
    model.save(full_path)
    print(f'\n[OK] Full model saved → {full_path}')

    # Plot
    plot_path = os.path.join(out_dir, 'training_curve.png')
    loss_val_graph(history, save_path=plot_path)
    print(f'[OK] Plot saved → {plot_path}')


if __name__ == '__main__':
    main()
