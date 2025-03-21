import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_model():
    model = models.Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=(30000, 12)),
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
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

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

if __name__ == "__main__":
    print("Caricamento dati processati...")
    data = np.load('processed_data.npz')
    bcg_data = data['bcg']
    amplitudes = data['amplitudes']
    
    print("Preparazione finestre BCG...")
    X = prepare_windows(bcg_data, amplitudes)
    y = prepare_labels(amplitudes)
    
    print("Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Normalizzazione dati...")
    for i in range(12):
        mean = np.mean(X_train[:,:,i])
        std = np.std(X_train[:,:,i])
        X_train[:,:,i] = (X_train[:,:,i] - mean) / std
        X_test[:,:,i] = (X_test[:,:,i] - mean) / std
    
    print("Creazione e addestramento modello...")
    model = create_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50, batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    print("\nValutazione modello...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['No Apnea', 'Media Prob.', 'Alta Prob.']))
    
    print("\nPlot training history...")
    plot_training_history(history)
    
    print("\nSalvataggio modello...")
    model.save('bcg_apnea_model.h5')