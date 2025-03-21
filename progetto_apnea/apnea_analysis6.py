import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def load_data(file_path):
    df = pd.read_csv(file_path)
    bcg_signals = df.iloc[:, 1:13]
    ecg_signal = df.iloc[:, -1]
    return bcg_signals, ecg_signal

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_apnea_probability(ecg_signal, sampling_rate=1000):
    peaks, _ = find_peaks(ecg_signal, distance=600, height=np.mean(ecg_signal))
    rr_intervals_seconds = np.diff(peaks) / sampling_rate
    
    rr_mean = np.mean(rr_intervals_seconds)
    rr_std = np.std(rr_intervals_seconds)
    rr_z_scores = (rr_intervals_seconds - rr_mean) / rr_std
    
    rr_apnea_probability = sigmoid(rr_z_scores)
    
    peak_times = peaks[:-1] / sampling_rate
    interp_func = interp1d(peak_times, rr_apnea_probability, kind='linear', 
                          fill_value="extrapolate")
    
    time = np.arange(len(ecg_signal)) / sampling_rate
    probabilities = interp_func(time)
    
    return probabilities, time

def create_event_masks(signal_length, sampling_rate=1000):
    time = np.arange(signal_length) / sampling_rate
    
    breath_holding_mask = np.zeros(signal_length, dtype=bool)
    movement_mask = np.zeros(signal_length, dtype=bool)
    
    breath_holding_periods = [
        (60, 90),     # First breath-holding during inhalation
        (120, 150),   # Second breath-holding during inhalation
        (180, 210),   # Third breath-holding during exhalation
        (240, 270),   # Fourth breath-holding during exhalation
        (300, 330),   # Fifth breath-holding during exhalation
        (480, 510),   # Sixth breath-holding during inhalation
        (540, 570),   # Seventh breath-holding during inhalation
        (600, 630),   # Eighth breath-holding during exhalation
        (660, 690),   # Ninth breath-holding during exhalation
    ]
    
    movement_periods = [
        (420, 480),   # turning on the side
    ]
    
    for start, end in breath_holding_periods:
        breath_holding_mask[(time >= start) & (time <= end)] = True
    
    for start, end in movement_periods:
        movement_mask[(time >= start) & (time <= end)] = True
    
    return breath_holding_mask, movement_mask, time

def plot_analysis(bcg_signals, probabilities, breath_mask, movement_mask, time):
    plt.figure(figsize=(15, 10))
    
    # Plot BCG signal (first channel)
    bcg_signal = bcg_signals.iloc[:, 0]
    plt.plot(time, bcg_signal / np.max(np.abs(bcg_signal)) * 0.5, 'k-', 
             alpha=0.3, label='BCG Signal (normalized)')
    
    # Plot probability
    plt.plot(time, probabilities, 'r-', label='Apnea Probability', linewidth=2)
    
    # Highlight breath-holding periods
    plt.fill_between(time, 0, 1,
                    where=breath_mask, color='yellow', alpha=0.3,
                    label='Breath Holding')
    
    # Highlight movement periods
    plt.fill_between(time, 0, 1,
                    where=movement_mask, color='blue', alpha=0.3,
                    label='Movement')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Apnea Probability')
    plt.title('Apnea Probability Analysis with Breath Holding and Movement Periods')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Replace with your CSV file path
    file_path = "ecg_bcg1_indexed_clean3.csv"
    
    # Load data
    bcg_signals, ecg_signal = load_data(file_path)
    
    # Compute apnea probability
    probabilities, time = compute_apnea_probability(ecg_signal)
    
    # Create event masks
    breath_mask, movement_mask, _ = create_event_masks(len(ecg_signal))
    
    # Plot analysis
    plot_analysis(bcg_signals, probabilities, breath_mask, movement_mask, time)

if __name__ == "__main__":
    main()