import numpy as np
import torch
import pandas as pd
from train_cnn_teacher import ReteApneaCNN
import matplotlib.pyplot as plt
import os

def preprocessa_bcg(bcg_data, window_size=30, step_size=1, sampling_rate=100):
    """Prepara i dati BCG per la predizione"""
    n_samples = len(bcg_data)
    window_samples = window_size * sampling_rate
    step_samples = step_size * sampling_rate
    
    n_windows = (n_samples - window_samples) // step_samples + 1
    X = np.zeros((n_windows, 12, window_samples))
    
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        X[i] = bcg_data[start_idx:end_idx].T
    
    return X

def predici_apnea(model_path, bcg_file):
    # Carica modello
    model = ReteApneaCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Carica e preprocessa BCG
    print(f"Caricamento dati BCG da {bcg_file}")
    bcg_data = pd.read_csv(bcg_file).iloc[:, 1:13].values
    X = preprocessa_bcg(bcg_data)
    X = torch.FloatTensor(X)
    
    # Predizione
    print("Esecuzione predizioni...")
    with torch.no_grad():
        predictions = model(X)
        probabilities = torch.softmax(predictions, dim=1)
        classes = torch.argmax(predictions, dim=1)
    
    # Converti in etichette
    etichette_classi = {
        0: "No Apnea",
        1: "Media Probabilità",
        2: "Alta Probabilità"
    }
    
    # Crea DataFrame con risultati e timestamp
    risultati = pd.DataFrame({
        'tempo_secondi': np.arange(len(classes)),
        'classe': classes.numpy(),
        'etichetta': [etichette_classi[c.item()] for c in classes],
        'confidenza': torch.max(probabilities, dim=1)[0].numpy(),
        'prob_no_apnea': probabilities[:, 0].numpy(),
        'prob_media': probabilities[:, 1].numpy(),
        'prob_alta': probabilities[:, 2].numpy()
    })
    
    return risultati

def visualizza_predizioni(risultati):
    """Visualizza graficamente le predizioni"""
    # Get unique classes and their labels
    classi_uniche = sorted(risultati['classe'].unique())
    etichette_classi = {
        0: "No Apnea",
        1: "Media Probabilità",
        2: "Alta Probabilità"
    }
    etichette = [etichette_classi[c] for c in classi_uniche]
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Probabilità per ogni classe
    plt.subplot(3, 1, 1)
    if 'prob_no_apnea' in risultati.columns:
        plt.plot(risultati['tempo_secondi'], risultati['prob_no_apnea'], 'g-', label='No Apnea', alpha=0.7)
    if 'prob_media' in risultati.columns:
        plt.plot(risultati['tempo_secondi'], risultati['prob_media'], 'y-', label='Media', alpha=0.7)
    if 'prob_alta' in risultati.columns:
        plt.plot(risultati['tempo_secondi'], risultati['prob_alta'], 'r-', label='Alta', alpha=0.7)
    plt.ylabel('Probabilità')
    plt.title('Probabilità di Apnea nel Tempo')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Classificazione con confidenza
    plt.subplot(3, 1, 2)
    scatter = plt.scatter(risultati['tempo_secondi'], risultati['classe'], 
                         c=risultati['confidenza'], cmap='viridis', 
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='Confidenza')
    plt.ylabel('Classe di Apnea')
    plt.yticks(classi_uniche, etichette)
    plt.grid(True)
    
    # Plot 3: Distribuzione delle classi
    plt.subplot(3, 1, 3)
    class_counts = risultati['classe'].value_counts().sort_index()
    bars = plt.bar(etichette, class_counts)
    plt.ylabel('Numero di Finestre')
    plt.title('Distribuzione delle Classi di Apnea')
    
    # Aggiungi etichette con le percentuali
    total = len(risultati)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}\n({height/total*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Stampa statistiche
    print("\nStatistiche delle predizioni:")
    print(f"Totale finestre analizzate: {len(risultati)}")
    print("\nDistribuzione classi:")
    for classe, count in class_counts.items():
        print(f"{etichette_classi[classe]}: {count} ({count/total*100:.1f}%)")
    print("\nConfidenza media per classe:")
    print(risultati.groupby('classe')['confidenza'].mean())

def salva_risultati(risultati, output_file):
    """Salva i risultati in formato CSV"""
    risultati.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in {output_file}")

if __name__ == "__main__":
    try:
        # Configurazione percorsi
        base_path = "C:\\Windows\\System32\\progetto_apnea"
        model_path = os.path.join(base_path, "best_model_bcg.pth")
        bcg_file = os.path.join(base_path, "nuovo_bcg.csv")
        output_file = os.path.join(base_path, "predizioni_apnea.csv")
        
        # Verifica directory
        if not os.path.exists(base_path):
            print(f"Directory non trovata: {base_path}")
            print("Assicurati che la directory esista e che tu abbia i permessi necessari")
            exit(1)
            
        # Verifica esistenza file
        if not os.path.exists(model_path):
            print(f"ATTENZIONE: Modello non trovato in {model_path}")
            print("Assicurati di aver eseguito prima l'addestramento del modello")
            exit(1)
            
        if not os.path.exists(bcg_file):
            print(f"ATTENZIONE: File BCG non trovato in {bcg_file}")
            print("Assicurati che il file dei dati BCG sia presente")
            exit(1)
            
        # Esegui predizioni
        risultati = predici_apnea(model_path, bcg_file)
        print("\nRisultati predizione:")
        print(risultati.head())
        
        # Visualizza e salva risultati
        visualizza_predizioni(risultati)
        salva_risultati(risultati, output_file)
        
    except Exception as e:
        print(f"Errore durante la predizione: {str(e)}")