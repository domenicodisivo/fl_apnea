import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os
from config import get_models_path


def prepare_windows(bcg_data, amplitudes, window_size=30000, step_size=1000):
    n_windows = len(amplitudes)
    X = np.zeros((n_windows, window_size, 12))

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        X[i] = bcg_data[start_idx:end_idx]

    return X


def prepare_labels(amplitudes):
    y = np.zeros(len(amplitudes))
    y[amplitudes >= 0.35] = 2
    y[(amplitudes >= 0.15) & (amplitudes < 0.35)] = 1
    return y


def create_cnn_model(input_shape=(30000, 12), num_classes=3):
    model = models.Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print("Caricamento dati...")
    data = np.load('processed_data.npz')
    bcg_data = data['bcg']
    amplitudes = data['amplitudes']

    print("Preparazione dati...")
    X = prepare_windows(bcg_data, amplitudes)
    y = prepare_labels(amplitudes)

    print("Normalizzazione...")
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    X = (X - mean) / (std + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Creazione modello...")
    model = create_cnn_model()

    model_path = os.path.join(get_models_path(), "cnn_bcg_apnea_model_best.h5")

    print("Addestramento...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', verbose=1)
        ]
    )

    elapsed = time.time() - start_time
    print(f"\n⏱ Addestramento completato in {elapsed:.2f} secondi")

    print("Salvataggio grafico andamento loss/accuracy...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoca')
    plt.ylabel('Accuratezza')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()

    print(f"\n✅ Modello salvato in: {model_path}")
