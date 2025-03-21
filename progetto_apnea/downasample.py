import pandas as pd
import numpy as np
from scipy.signal import resample, find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from config import get_raw_data_path
import os

# Parametri globali
original_rate = 1000
target_rate = 100
window_size = 30  # secondi
step_size = 1  # secondi
chunk_size = 50000  # Numero di righe per batch durante la lettura

def load_and_downsample(csv_file, original_rate=1000, target_rate=100, chunk_size=50000):
    downsample_factor = original_rate // target_rate
    all_bcg = []
    all_ecg = []

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        bcg_signals = chunk.iloc[:, 1:13].astype(np.float32).values  # Convertire in float32
        ecg_signal = chunk.iloc[:, -1].astype(np.float32).values

        new_n_samples = len(ecg_signal) // downsample_factor
        
        bcg_downsampled = np.zeros((new_n_samples, 12), dtype=np.float32)
        for i in range(12):
            bcg_downsampled[:, i] = resample(bcg_signals[:, i], new_n_samples)
        
        ecg_downsampled = resample(ecg_signal, new_n_samples)
        
        all_bcg.append(bcg_downsampled)
        all_ecg.append(ecg_downsampled)
    
    return np.vstack(all_bcg), np.concatenate(all_ecg), target_rate

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def discretize_probability(prob):
    if prob < 0.4:
        return 0  # No Apnea
    elif prob < 0.7:
        return 1  # Media Probabilità
    else:
        return 2  # Alta Probabilità

def create_windowed_dataset(bcg_signals, probabilities, window_size=30, step_size=1, sampling_rate=100):
    bcg_signals = bcg_signals.T
    
    window_samples = window_size * sampling_rate
    step_samples = step_size * sampling_rate
    n_samples = bcg_signals.shape[1]
    
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    X = np.zeros((n_windows, 12, window_samples), dtype=np.float32)
    y = np.zeros(n_windows, dtype=int)
    
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        
        X[i] = bcg_signals[:, start_idx:end_idx]
        
        prob_window = probabilities[start_idx:end_idx]
        max_prob = np.max(prob_window)
        y[i] = discretize_probability(max_prob)
    
    return X, y

def plot_apnea_analysis(probabilities, y, step_size=1):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    time = np.arange(len(probabilities)) / 100
    ax1.plot(time, probabilities, 'r-', label='Probabilità Apnea')
    ax1.set_ylabel('Probabilità')
    ax1.set_title('Probabilità di Apnea nel Tempo')
    ax1.grid(True)
    
    window_times = np.arange(len(y)) * step_size
    colors = ['black', 'green', 'purple']
    labels = ['No Apnea', 'Media Prob.', 'Alta Prob.']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = y == i
        if np.any(mask):
            ax2.scatter(window_times[mask], [i]*np.sum(mask), c=color, label=label, alpha=0.6)
    
    ax2.set_xlabel('Tempo (secondi)')
    ax2.set_ylabel('Classe di Apnea')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(labels)
    ax2.grid(True)
    
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def compute_apnea_probability(ecg_signal, sampling_rate=100):
    peaks, _ = find_peaks(ecg_signal, distance=60, height=np.mean(ecg_signal))
    rr_intervals_seconds = np.diff(peaks) / sampling_rate
    
    rr_mean = np.mean(rr_intervals_seconds)
    rr_std = np.std(rr_intervals_seconds)
    rr_z_scores = (rr_intervals_seconds - rr_mean) / rr_std
    
    rr_apnea_probability = sigmoid(rr_z_scores)
    
    peak_times = peaks[:-1] / sampling_rate
    interp_func = interp1d(peak_times, rr_apnea_probability, kind='linear', fill_value="extrapolate")
    
    time = np.arange(len(ecg_signal)) / sampling_rate
    probabilities = interp_func(time)
    
    return probabilities, time

if __name__ == "__main__":
    csv_file = os.path.join(get_raw_data_path(), "ecg_bcg1_indexed_clean5.csv")
    
    print("Caricamento e sottocampionamento...")
    bcg_downsampled, ecg_downsampled, new_rate = load_and_downsample(csv_file, original_rate, target_rate, chunk_size)
    
    print("Calcolo probabilità di apnea...")
    probabilities, _ = compute_apnea_probability(ecg_downsampled, new_rate)
    
    print("Creazione dataset con finestre...")
    X, y = create_windowed_dataset(bcg_downsampled, probabilities, window_size, step_size, new_rate)
    
    print("Creazione grafico di analisi...")
    plot_apnea_analysis(probabilities, y, step_size)
    
    print(f"Dataset creato con successo! {X.shape[0]} finestre generate con dimensione {X.shape} e frequenza di campionamento {new_rate} Hz")
