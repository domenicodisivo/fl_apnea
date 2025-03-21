import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from config import get_raw_data_path, get_figures_path
import os

def carica_dati(percorso_file):
    dati = pd.read_csv(percorso_file)
    segnali_bcg = dati.iloc[:, 1:13]
    segnale_ecg = dati.iloc[:, -1]
    return segnali_bcg, segnale_ecg

def funzione_sigmoide(x):
    return 1 / (1 + np.exp(-x))

def calcola_probabilita_apnea(segnale_ecg, frequenza_campionamento=1000):
    picchi, _ = find_peaks(segnale_ecg, distance=600, height=np.mean(segnale_ecg))
    intervalli_rr_secondi = np.diff(picchi) / frequenza_campionamento
    
    media_rr = np.mean(intervalli_rr_secondi)
    deviazione_standard_rr = np.std(intervalli_rr_secondi)
    punteggi_z_rr = (intervalli_rr_secondi - media_rr) / deviazione_standard_rr
    
    probabilita_apnea_rr = funzione_sigmoide(punteggi_z_rr)
    
    tempi_picchi = picchi[:-1] / frequenza_campionamento
    funzione_interpolazione = interp1d(tempi_picchi, probabilita_apnea_rr, kind='linear', 
                          fill_value="extrapolate")
    
    tempo = np.arange(len(segnale_ecg)) / frequenza_campionamento
    probabilita = funzione_interpolazione(tempo)
    
    return probabilita, tempo

def crea_maschere_eventi(lunghezza_segnale, frequenza_campionamento=1000):
    tempo = np.arange(lunghezza_segnale) / frequenza_campionamento
    
    maschera_apnea = np.zeros(lunghezza_segnale, dtype=bool)
    maschera_movimento = np.zeros(lunghezza_segnale, dtype=bool)
    
    periodi_apnea = [
        (60, 90),     # Prima apnea durante inspirazione
        (120, 150),   # Seconda apnea durante inspirazione
        (180, 210),   # Terza apnea durante espirazione
        (240, 270),   # Quarta apnea durante espirazione
        (300, 330),   # Quinta apnea durante espirazione
        (480, 510),   # Sesta apnea durante inspirazione
        (540, 570),   # Settima apnea durante inspirazione
        (600, 630),   # Ottava apnea durante espirazione
        (660, 690),   # Nona apnea durante espirazione
    ]
    
    periodi_movimento = [
        (420, 480),   # Girarsi su un fianco
    ]
    
    for inizio, fine in periodi_apnea:
        maschera_apnea[(tempo >= inizio) & (tempo <= fine)] = True
    
    for inizio, fine in periodi_movimento:
        maschera_movimento[(tempo >= inizio) & (tempo <= fine)] = True
    
    return maschera_apnea, maschera_movimento, tempo

def visualizza_analisi(segnali_bcg, probabilita, maschera_apnea, maschera_movimento, tempo):
    plt.figure(figsize=(15, 10))
    
    # Grafico del segnale BCG (primo canale)
    segnale_bcg = segnali_bcg.iloc[:, 0]
    plt.plot(tempo, segnale_bcg / np.max(np.abs(segnale_bcg)) * 0.5, 'k-', 
             alpha=0.3, label='Segnale BCG (normalizzato)')
    
    # Grafico della probabilità
    plt.plot(tempo, probabilita, 'r-', label='Probabilità Apnea', linewidth=2)
    
    # Evidenzia periodi di apnea
    plt.fill_between(tempo, 0, 1,
                    where=maschera_apnea, color='yellow', alpha=0.3,
                    label='Trattenimento Respiro')
    
    # Evidenzia periodi di movimento
    plt.fill_between(tempo, 0, 1,
                    where=maschera_movimento, color='blue', alpha=0.3,
                    label='Movimento')
    
    plt.xlabel('Tempo (secondi)')
    plt.ylabel('Probabilità Apnea')
    plt.title('Analisi della Probabilità di Apnea con Periodi di Apnea e Movimento')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.savefig(os.path.join(get_figures_path(), 'analisi_apnea.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Usa il percorso dal config per il file CSV
    percorso_file = os.path.join(get_raw_data_path(), "ecg_bcg1_indexed_clean3.csv")
    
    # Carica i dati
    segnali_bcg, segnale_ecg = carica_dati(percorso_file)
    
    # Calcola la probabilità di apnea
    probabilita, tempo = calcola_probabilita_apnea(segnale_ecg)
    
    # Crea le maschere degli eventi
    maschera_apnea, maschera_movimento, _ = crea_maschere_eventi(len(segnale_ecg))
    
    # Visualizza l'analisi
    visualizza_analisi(segnali_bcg, probabilita, maschera_apnea, maschera_movimento, tempo)

if __name__ == "__main__":
    main()