import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DatasetApneaTeacher(Dataset):
    def __init__(self, bcg_data, ecg_labels):
        # bcg_data: [N, 12, 3000], ecg_labels: [N]
        self.bcg = torch.FloatTensor(bcg_data)
        self.labels = torch.LongTensor(ecg_labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.bcg[idx], self.labels[idx]

class ReteApneaCNN(nn.Module):
    def __init__(self):
        super(ReteApneaCNN, self).__init__()
        # Input: [batch, 12 canali BCG, 3000 campioni]
        self.conv1 = nn.Conv1d(12, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 375, 128)
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def addestra_modello_teacher(bcg_data, ecg_labels, epoche=50, batch_size=32):
    print("Preparazione dati...")
    X_train, X_test, y_train, y_test = train_test_split(bcg_data, ecg_labels, test_size=0.2)
    
    train_dataset = DatasetApneaTeacher(X_train, y_train)
    test_dataset = DatasetApneaTeacher(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("Inizializzazione modello...")
    model = ReteApneaCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    storico = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    print("Inizio addestramento...")
    for epoca in range(epoche):
        model.train()
        train_loss = 0
        corretti = 0
        totali = 0
        
        for bcg, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(bcg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            totali += labels.size(0)
            corretti += predicted.eq(labels).sum().item()
        
        acc_train = 100. * corretti / totali
        storico['train_loss'].append(train_loss / len(train_loader))
        storico['train_acc'].append(acc_train)
        
        # Validazione
        model.eval()
        val_loss = 0
        corretti = 0
        totali = 0
        
        with torch.no_grad():
            for bcg, labels in test_loader:
                outputs = model(bcg)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                totali += labels.size(0)
                corretti += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(test_loader)
        acc_val = 100. * corretti / totali
        storico['val_loss'].append(val_loss)
        storico['val_acc'].append(acc_val)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Salva il miglior modello
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), 'best_model_bcg.pth')
        
        print(f'Epoca {epoca+1}/{epoche}:')
        print(f'Loss Training: {storico["train_loss"][-1]:.3f}, Acc Training: {acc_train:.2f}%')
        print(f'Loss Val: {val_loss:.3f}, Acc Val: {acc_val:.2f}%\n')
    
    return model, storico

if __name__ == "__main__":
    try:
        # Carica dati dal dataset creato
        data = np.load("downsampled_windowed_dataset.npz")
        bcg_data = data['X']  # [N, 12, 3000]
        ecg_labels = data['y']  # [N]
        
        # Addestra modello usando BCG con labels da ECG
        model, storico = addestra_modello_teacher(bcg_data, ecg_labels)
        
        # Visualizza risultati
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(storico['train_acc'], label='Train')
        plt.plot(storico['val_acc'], label='Val')
        plt.title('Accuratezza')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(storico['train_loss'], label='Train')
        plt.plot(storico['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Errore: {str(e)}")