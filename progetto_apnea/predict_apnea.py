import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import get_raw_data_path, get_models_path

def predici_apnea(model_path, bcg_file):
    # Carica modello
    model = tf.keras.models.load_model(model_path)
    
    # Carica e preprocessa BCG
    print(f"Caricamento dati BCG da {bcg_file}")
    bcg_data = pd.read_csv(bcg_file)
    print(f"Dimensioni dati BCG: {bcg_data.shape}")
    
    # Prendi solo le colonne BCG (dalla 2 alla 13)
    X = bcg_data.iloc[:, 1:13].values
    
    # Ridimensiona i dati nel formato corretto (batch, time_steps, channels)
    X = X.reshape(1, X.shape[0], X.shape[1])
    
    # Normalizzazione
    for i in range(12):
        mean = np.mean(X[:,:,i])
        std = np.std(X[:,:,i])
        X[:,:,i] = (X[:,:,i] - mean) / std
    
    # Predizione
    print("Esecuzione predizioni...")
    predictions = model.predict(X)
    probabilities = tf.nn.softmax(predictions).numpy()
    classes = np.argmax(predictions, axis=1)
    
    # Crea DataFrame con risultati
    risultati = pd.DataFrame({
        'classe': classes,
        'prob_no_apnea': probabilities[0,0],
        'prob_media': probabilities[0,1],
        'prob_alta': probabilities[0,2]
    }, index=[0])
    
    risultati['etichetta'] = risultati['classe'].map({
        0: "No Apnea",
        1: "Media Probabilità",
        2: "Alta Probabilità"
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
        model_path = os.path.join(get_models_path(), "bcg_apnea_model.h5")
        bcg_file = os.path.join(get_raw_data_path(), "bcg_apnea_dataset.csv")
        output_file = os.path.join(get_models_path(), "predizioni_apnea.csv")
        
        # Verifica esistenza file
        if not os.path.exists(model_path):
            print(f"ATTENZIONE: Modello non trovato in {model_path}")
            print("Assicurati di aver eseguito prima l'addestramento del modello")
            exit(1)
            
        if not os.path.exists(bcg_file):
            print(f"ATTENZIONE: File BCG non trovato in {bcg_file}")
            print("Assicurati che il file dei dati BCG sia presente nella cartella raw")
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