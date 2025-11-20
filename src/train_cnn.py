# train_cnn.py
# Simple CNN training script for grayscale phantom images
# Usage: python train_cnn.py
# Requirements: tensorflow, pandas, scikit-learn, matplotlib, opencv-python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Activation,
    BatchNormalization, Dropout, Flatten, Dense
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
DATA_DIR = os.path.join("data", "synthetic")
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
OUT_DIR = "results"
IMG_SIZE = (224, 224)         # height, width
BATCH_SIZE = 16               # reduce to 8 or 4 if needed
SEED = 42
EPOCHS = 30
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


# -------- LABEL PROCESSING ----------
def map_label_to_binary(lbl):
    s = str(lbl)
    return "tumor" if s.startswith("tumor") else "no_tumor"


def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    df['image_path'] = df['image'].apply(
        lambda p: os.path.join(DATA_DIR, p) if not os.path.isabs(p) else p
    )
    df['label_bin'] = df['label'].apply(map_label_to_binary)

    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    return df


# -------- DATA GENERATORS -----------
def build_generators(train_df, val_df):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=12,
        width_shift_range=0.06,
        height_shift_range=0.06,
        zoom_range=0.08,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='reflect'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='label_bin',
        target_size=IMG_SIZE,
        color_mode='grayscale',
        class_mode='binary',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='label_bin',
        target_size=IMG_SIZE,
        color_mode='grayscale',
        class_mode='binary',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_gen, val_gen


# -------- SIMPLE CNN -----------
def build_simple_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dropout_rate=0.35):
    inp = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    out = Dense(1, activation='sigmoid')(x)

    return Model(inputs=inp, outputs=out)


# -------- TRAINING PLOTS -----------
def plot_history(history, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()


# -------- EVALUATION -----------
def evaluate_and_save(model, test_df, out_dir):
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        test_df,
        x_col='image_path',
        y_col='label_bin',
        target_size=IMG_SIZE,
        color_mode='grayscale',
        class_mode='binary',
        batch_size=1,
        shuffle=False
    )

    preds = model.predict(test_gen, verbose=1)
    y_pred = (preds.ravel() >= 0.5).astype(int)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=labels)
    cm = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, preds)
    except:
        auc = None

    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)
    if auc is not None:
        print("ROC AUC:", auc)

    with open(os.path.join(out_dir, 'test_report.txt'), 'w') as f:
        f.write(report + "\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n")
        if auc is not None:
            f.write("ROC AUC: %.4f\n" % auc)


# -------------- MAIN --------------
def main():
    print("\nLoading CSV:", CSV_PATH)
    df = prepare_dataframe(CSV_PATH)
    print("Total samples:", len(df))

    # splits
    df_trainval, df_test = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df['label_bin'], random_state=RANDOM_STATE
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        stratify=df_trainval['label_bin'],
        random_state=RANDOM_STATE
    )

    print("Train/Val/Test:", len(df_train), len(df_val), len(df_test))

    train_gen, val_gen = build_generators(df_train, df_val)

    model = build_simple_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    chk_path = os.path.join(OUT_DIR, 'best_model.keras')
    checkpoint = ModelCheckpoint(chk_path, save_best_only=True, monitor='val_loss')

    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early, reduce]
    )

    plot_history(history, OUT_DIR)

    best = load_model(chk_path)
    evaluate_and_save(best, df_test, OUT_DIR)

    final_path = os.path.join(OUT_DIR, 'final_model.keras')
    best.save(final_path)
    print("\nDONE. Results saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
