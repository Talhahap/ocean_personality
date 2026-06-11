import json

with open('02_Face_train_ADAM.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Markdown cell
md_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['### Evaluasi Model']
}

# 2. Evaluasi Train code cell
train_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# \u2500\u2500 EVALUASI TRAIN \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
        "train_ds_eval = tf.data.Dataset.load(r'data/ds/train_ds').cache().prefetch(buffer_size=tf.data.AUTOTUNE)\n",
        '\n',
        'loss, mae = model.evaluate(train_ds_eval)\n',
        "print(f'Global Accuracy (Train): {(1 - mae) * 100:.2f}%')\n",
        '\n',
        'y_true_train = np.concatenate([y for x, y in train_ds_eval])\n',
        'y_pred_train = model.predict(train_ds_eval)\n',
        '\n',
        "mae_train = mean_absolute_error(y_true_train, y_pred_train, multioutput='raw_values')\n",
        "print('Accuracy per Trait (O, C, E, A, N):', (1 - mae_train) * 100)\n",
        "print(f'Mean Accuracy (Train): {(1 - np.mean(mae_train)) * 100:.2f}%')"
    ]
}

# 3. Evaluasi Validation code cell
val_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# \u2500\u2500 EVALUASI VALIDATION \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
        "val_ds_eval = tf.data.Dataset.load(r'data/ds/val_ds').cache().prefetch(buffer_size=tf.data.AUTOTUNE)\n",
        '\n',
        'y_true_val = np.concatenate([y for x, y in val_ds_eval])\n',
        'y_pred_val = model.predict(val_ds_eval)\n',
        '\n',
        "mae_val = mean_absolute_error(y_true_val, y_pred_val, multioutput='raw_values')\n",
        "print('Accuracy per Trait (O, C, E, A, N):', (1 - mae_val) * 100)\n",
        "print(f'Mean Accuracy (Val): {(1 - np.mean(mae_val)) * 100:.2f}%')"
    ]
}

# 4. Evaluasi Test code cell
test_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# \u2500\u2500 EVALUASI TEST \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
        "test_ds_eval = tf.data.Dataset.load(r'data/ds/test_ds').cache().prefetch(buffer_size=tf.data.AUTOTUNE)\n",
        '\n',
        'y_true_test = np.concatenate([y for x, y in test_ds_eval])\n',
        'y_pred_test = model.predict(test_ds_eval)\n',
        '\n',
        "mae_test = mean_absolute_error(y_true_test, y_pred_test, multioutput='raw_values')\n",
        "print('Accuracy per Trait (O, C, E, A, N):', (1 - mae_test) * 100)\n",
        "print(f'Mean Accuracy (Test): {(1 - np.mean(mae_test)) * 100:.2f}%')"
    ]
}

nb['cells'].append(md_cell)
nb['cells'].append(train_cell)
nb['cells'].append(val_cell)
nb['cells'].append(test_cell)

with open('02_Face_train_ADAM.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Done. Total cells:', len(nb['cells']))
