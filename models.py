import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def create_arcface_model(load_weights=True):
    """
    Membangun model ArcFace (ResNet34 backbone) untuk face recognition.
    
    Input  : (112, 112, 3)  - gambar wajah BGR/RGB
    Output : (512,)         - embedding wajah yang sudah di-L2 normalize
    
    Args:
        load_weights (bool): Jika True, load pre-trained weights dari arcface_weights.h5
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        ZeroPadding2D, Input, Conv2D, BatchNormalization,
        PReLU, Add, Dropout, Flatten, Dense,
    )

    # ── ResNet34 building blocks ───────────────────────────────────────────────
    def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
        bn_axis = 3
        if conv_shortcut:
            shortcut = Conv2D(filters, 1, strides=stride, use_bias=False,
                              kernel_initializer="glorot_normal",
                              name=name + "_0_conv")(x)
            shortcut = BatchNormalization(axis=bn_axis, epsilon=2e-5,
                                          momentum=0.9, name=name + "_0_bn")(shortcut)
        else:
            shortcut = x

        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn")(x)
        x = ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
        x = Conv2D(filters, 3, strides=1, kernel_initializer="glorot_normal",
                   use_bias=False, name=name + "_1_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn")(x)
        x = PReLU(shared_axes=[1, 2], name=name + "_1_prelu")(x)

        x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
        x = Conv2D(filters, kernel_size, strides=stride, kernel_initializer="glorot_normal",
                   use_bias=False, name=name + "_2_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn")(x)
        x = Add(name=name + "_add")([shortcut, x])
        return x

    def stack1(x, filters, blocks, stride1=2, name=None):
        x = block1(x, filters, stride=stride1, name=name + "_block1")
        for i in range(2, blocks + 1):
            x = block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
        return x

    def stack_fn(x):
        x = stack1(x, 64,  3, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    # ── Bangun arsitektur ──────────────────────────────────────────────────────
    img_input = Input(shape=(112, 112, 3))
    x = ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
    x = Conv2D(64, 3, strides=1, use_bias=False,
               kernel_initializer="glorot_normal", name="conv1_conv")(x)
    x = BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name="conv1_bn")(x)
    x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
    x = stack_fn(x)

    # ArcFace head
    x = BatchNormalization(momentum=0.9, epsilon=2e-5)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(x)
    embedding = BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(x)
    embedding = keras.layers.Lambda(
        lambda v: K.l2_normalize(v, axis=1), name='norm_layer'
    )(embedding)

    model = Model(inputs=img_input, outputs=embedding, name='ArcFace')

    # ── Load pre-trained weights ───────────────────────────────────────────────
    if load_weights:
        # Path relatif terhadap lokasi file models.py ini
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        weight_file = os.path.join(base_dir, 'arcface_weights.h5')
        try:
            model.load_weights(weight_file)
            print(f"[OK] ArcFace weights berhasil dimuat dari: {weight_file}")
        except Exception as e:
            print(f"[WARNING] Gagal memuat weights dari: {weight_file}")
            print(f"  Error : {e}")
            print("  Model akan diinisialisasi dengan random weights.")

    return model


def create_arcface_personality_model(arcface_model, optimizer=None, learning_rate=1e-3):
    """
    Membangun model prediksi kepribadian OCEAN di atas ArcFace yang sudah dibekukan.
    
    Arsitektur:
        Input (10, 112, 112, 3)
        → Rescaling (÷255)
        → TimeDistributed(ArcFace frozen)  → (10, 512)
        → LSTM(128, return_sequences=True)
        → LSTM(64)
        → Dropout(0.2)
        → Dense(256, relu)  → Dropout(0.3)
        → Dense(128, relu)  → Dropout(0.5)
        → Dense(5, sigmoid)   ← 5 trait OCEAN [0, 1]

    Args:
        arcface_model  : Model ArcFace yang sudah dibuild via create_arcface_model().
        optimizer      : Keras optimizer instance atau None.
                         Jika None, model tidak di-compile (compile manual di notebook).
        learning_rate  : Hanya dipakai jika optimizer=None dan compile otomatis.

    Returns:
        Keras Model yang sudah di-compile (jika optimizer diberikan).
    """
    from tensorflow.keras import regularizers

    arcface_model.trainable = False   # Bekukan bobot ArcFace

    inputs = keras.layers.Input(shape=(10, 112, 112, 3), name='Input_Sequence')

    # Rescale [0,255] → [0,1]
    x = keras.layers.TimeDistributed(
        keras.layers.Rescaling(scale=1./255.0), name='Rescaling'
    )(inputs)

    # Ekstrak embedding per-frame dengan ArcFace  → (batch, 10, 512)
    x = keras.layers.TimeDistributed(arcface_model, name='ArcFace_Embeddings')(x)

    # Temporal modelling
    x = keras.layers.LSTM(128, return_sequences=True,
                          recurrent_dropout=0.2, name='LSTM_1')(x)
    x = keras.layers.LSTM(64, recurrent_dropout=0.2, name='LSTM_2')(x)
    x = keras.layers.Dropout(0.2, name='Dropout_LSTM')(x)

    # Dense head
    x = keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=regularizers.l2(1e-4), name='Dense_1')(x)
    x = keras.layers.Dropout(0.3, name='Dropout_1')(x)
    x = keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=regularizers.l2(1e-4), name='Dense_2')(x)
    x = keras.layers.Dropout(0.5, name='Dropout_2')(x)

    # Output: 5 trait OCEAN, nilai antara 0–1
    x = keras.layers.Dense(5, activation='sigmoid', name='OCEAN_Output')(x)

    model = keras.models.Model(inputs=inputs, outputs=x,
                               name='ArcFace_OCEAN_Predictor')

    if optimizer is not None:
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    return model
