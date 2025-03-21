import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from config import get_raw_data_path
import os

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return y

def load_data(csv_file, chunk_size=50000):
    all_bcg = []
    all_ecg = []

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        bcg_signals = chunk.iloc[:, 1:13].astype(np.float32).values
        ecg_signal = chunk.iloc[:, -1].astype(np.float32).values
        all_bcg.append(bcg_signals)
        all_ecg.append(ecg_signal)
    
    return np.vstack(all_bcg), np.concatenate(all_ecg)

def compute_apnea_probability(ecg_signal, sampling_rate=1000):
    peaks, _ = find_peaks(ecg_signal, distance=600, height=np.mean(ecg_signal))
    rr_intervals_seconds = np.diff(peaks) / sampling_rate
    
    rr_mean = np.mean(rr_intervals_seconds)
    rr_std = np.std(rr_intervals_seconds)
    rr_z_scores = (rr_intervals_seconds - rr_mean) / rr_std
    
    rr_apnea_probability = 1 / (1 + np.exp(-rr_z_scores))
    
    peak_times = peaks[:-1] / sampling_rate
    interp_func = interp1d(peak_times, rr_apnea_probability, kind='linear', fill_value="extrapolate")
    
    time = np.arange(len(ecg_signal)) / sampling_rate
    probabilities = interp_func(time)
    
    return probabilities, time

def compute_amplitudes(probabilities, window_size=30, step_size=1, sampling_rate=1000):
    filtered_probabilities = apply_lowpass_filter(probabilities, cutoff=0.1, fs=sampling_rate)
    
    window_samples = window_size * sampling_rate
    step_samples = step_size * sampling_rate
    n_windows = (len(filtered_probabilities) - window_samples) // step_samples + 1
    
    amplitudes = np.zeros(n_windows)
    
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        prob_window = filtered_probabilities[start_idx:end_idx]
        amplitudes[i] = np.max(prob_window) - np.min(prob_window)
    
    return amplitudes, filtered_probabilities

def plot_analysis(probabilities, filtered_probabilities, amplitudes):
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3,1,1)
    time = np.arange(len(probabilities))/1000
    plt.plot(time, probabilities)
    plt.title('Probabilità Originali')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Probabilità')
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.plot(time, filtered_probabilities)
    plt.title('Probabilità Filtrate (Passa-basso 0.1Hz)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Probabilità')
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.hist(amplitudes, bins=50, color='blue', alpha=0.7)
    plt.axvline(x=0.15, color='r', linestyle='--', label='Soglia No Apnea (0.15)')
    plt.axvline(x=0.35, color='g', linestyle='--', label='Soglia Media Prob. (0.35)')
    plt.xlabel('Variazione della Probabilità nella Finestra (max-min)')
    plt.ylabel('Numero di Finestre')
    plt.title('Distribuzione della Variazione di Probabilità')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_file = os.path.join(get_raw_data_path(), "ecg_bcg1_indexed_clean5.csv")
    
    print("Caricamento dati...")
    bcg_data, ecg_data = load_data(csv_file)
    
    print("Calcolo probabilità di apnea...")
    probabilities, _ = compute_apnea_probability(ecg_data)
    
    print("Calcolo delle ampiezze con filtro passa-basso...")
    amplitudes, filtered_probabilities = compute_amplitudes(probabilities)
    
    print("Creazione grafici di analisi...")
    plot_analysis(probabilities, filtered_probabilities, amplitudes)
    
    print("Salvataggio dati processati...")
    np.savez('processed_data.npz',
             bcg=bcg_data,
             ecg=ecg_data,
             amplitudes=amplitudes,
             filtered_probabilities=filtered_probabilities)
    
    print("\nStatistiche delle ampiezze:")
    print(f"- Numero finestre: {len(amplitudes)}")
    print(f"- Ampiezza media: {np.mean(amplitudes):.3f}")
    print(f"- Ampiezza mediana: {np.median(amplitudes):.3f}")
    print(f"- Deviazione standard: {np.std(amplitudes):.3f}")
    print(f"- Distribuzione classi:")
    print(f"  No Apnea (<0.15): {np.sum(amplitudes < 0.15)} finestre")
    print(f"  Media Prob. (0.15-0.35): {np.sum((amplitudes >= 0.15) & (amplitudes < 0.35))} finestre")
    print(f"  Alta Prob. (>0.35): {np.sum(amplitudes >= 0.35)} finestre")