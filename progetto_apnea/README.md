# Progetto Apnea - Analisi e Riconoscimento di Episodi Apnoici da Segnali BCG/ECG

Questo repository contiene il codice e i dati elaborati per l'analisi e la classificazione di episodi di apnea durante test di trattenuta del respiro, utilizzando segnali BCG ed ECG. Il progetto è suddiviso in più fasi: estrazione di probabilità di apnea, creazione dataset, addestramento di modelli deep learning, e predizione con tecniche di transfer learning.

---

## 📂 Struttura del Progetto

```
progetto_apnea/
├── data/                    # Dataset grezzi e preprocessati
├── models/                  # Modelli salvati
├── figures/                 # Grafici generati
├── train_cnn_ampiezza.py    # Addestramento CNN con etichette da ampiezze
├── predict_bcg_transfer.py  # Predizione solo da BCG con transfer learning
├── process_data.py          # Estrazione ampiezze da probabilità apnea
├── analisi_apnea_plot.py    # Visualizzazione probabilità + eventi
├── config.py                # Funzioni per gestire percorsi
└── processed_data.npz       # Dataset pronto per l'addestramento
```

---

## ⚙️ Requisiti

```bash
pip install -r requirements.txt
```

Include:
- numpy
- pandas
- scipy
- matplotlib
- tensorflow
- scikit-learn

---

## 🚀 Fasi del Progetto

### 1. Estrazione delle probabilità di apnea da ECG
Script: `analisi_apnea_plot.py`

- Estrae i picchi dal segnale ECG
- Calcola intervalli RR e probabilità di apnea (funzione sigmoide)
- Interpola la curva per ottenere un segnale continuo
- Visualizza i periodi di apnea e movimento sul grafico

### 2. Calcolo delle ampiezze su finestre mobili
Script: `process_data.py`

- Applica filtro passa-basso alla curva di probabilità
- Divide il segnale in finestre da 30 secondi, con shift di 1 secondo
- Calcola l'ampiezza (max - min) della probabilità nella finestra
- Classifica le finestre: `0 = no apnea`, `1 = medio`, `2 = alta probabilità`

### 3. Addestramento modello CNN
Script: `train_cnn_ampiezza.py`

- Carica il file `processed_data.npz`
- Crea finestre 30s x 12 canali BCG
- Normalizza i dati
- Addestra una CNN 1D con early stopping e salvataggio del miglior modello
- Salva modello in `models/cnn_bcg_apnea_model_best.h5`

### 4. Transfer Learning per predizione da solo BCG
Script: `predict_bcg_transfer.py`

- Carica nuovo CSV contenente solo i 12 canali BCG
- Genera finestre come nel training
- Applica normalizzazione
- Carica il modello addestrato e predice la classe di apnea
- Esporta i risultati in `predictions_bcg_only.csv`

---

## 📊 Output generati
- `training_history.png` - andamento loss/accuracy
- `cnn_bcg_apnea_model_best.h5` - modello CNN addestrato
- `predictions_bcg_only.csv` - risultati delle predizioni

---

## 🧠 Obiettivo finale
Creare un sistema che, una volta addestrato su segnali BCG+ECG, possa essere usato per predire episodi di apnea **anche in assenza di ECG**, utilizzando solo i segnali BCG, attraverso il transfer learning.

---

## 📧 Contatti
Per domande o collaborazioni: [domenicodisivo@...]