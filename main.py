import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch

# Détecte la RTX 3060 si CUDA est disponible, sinon reste sur le processeur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé pour l'entraînement : {device}")

# ==========================================
# 1. PARAMÈTRES GLOBAUX
# ==========================================
DATA_FOLDER = "Waveforms.csv"
TOLERANCE_TIME = 0.1
PMUS_THRESHOLD = -1.0  # Seuil durci pour trouver les "Delayed"

# Détection automatique de la carte graphique (GPU) si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé pour l'entraînement : {device}")


# ==========================================
# 2. FONCTIONS DE TRAITEMENT DU SIGNAL
# ==========================================
def extract_cycles(df):
    smoothed_flow = df['flow'].rolling(window=5, center=True).mean().fillna(0)
    is_positive = smoothed_flow > 0
    crossings = (is_positive & ~is_positive.shift(1).fillna(False))
    start_indices = df.index[crossings].tolist()

    cycles = []
    for i in range(len(start_indices) - 1):
        start_idx = start_indices[i]
        end_idx = start_indices[i + 1]
        cycle_df = df.iloc[start_idx:end_idx].copy()
        if len(cycle_df) > 50:
            cycles.append(cycle_df)
    return cycles


def label_cycle(cycle_df):
    expir_start_mask = cycle_df['flow'] < 0
    if not expir_start_mask.any():
        return None

    mech_end_time = cycle_df.loc[expir_start_mask, 'time'].iloc[0]
    pmus_active = cycle_df['pmus'] < PMUS_THRESHOLD
    if not pmus_active.any():
        return "None"

    pmus_times = cycle_df.loc[pmus_active, 'time']
    valid_pmus_times = pmus_times[pmus_times < (mech_end_time + 1.0)]

    if valid_pmus_times.empty:
        return "None"

    neural_end_time = valid_pmus_times.iloc[-1]
    time_diff = mech_end_time - neural_end_time

    if time_diff < -TOLERANCE_TIME:
        return "Premature"
    elif time_diff > TOLERANCE_TIME:
        return "Delayed"
    else:
        return "None"


# ==========================================
# 3. PRÉPARATION PYTORCH (DATASET)
# ==========================================
class AsynchronyDataset(Dataset):
    def __init__(self, cycles_df_list, labels_list, max_len=500):
        self.data = []
        self.labels = []

        # On définit 3 classes possibles
        self.label_map = {"None": 0, "Premature": 1, "Delayed": 2}

        for df, label in zip(cycles_df_list, labels_list):
            if label not in self.label_map:
                continue

            # Extraction de Flow et Paw uniquement
            sig = df[['flow', 'paw']].values

            # Padding ou Troncature
            if len(sig) > max_len:
                sig = sig[:max_len]
            else:
                pad_len = max_len - len(sig)
                sig = np.pad(sig, ((0, pad_len), (0, 0)), mode='constant')

            sig = sig.T  # Format (Variables, Temps)

            self.data.append(sig)
            self.labels.append(self.label_map[label])

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ==========================================
# 4. ARCHITECTURE DU MODÈLE DE DEEP LEARNING
# ==========================================
class CNNRespiratoire(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNRespiratoire, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # 500 / 2 / 2 / 2 = 62 (environ)
        self.fc1 = nn.Linear(64 * 62, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==========================================
# 5. BOUCLE PRINCIPALE (EXTRACTION & ENTRAÎNEMENT)
# ==========================================
if __name__ == "__main__":
    print("\n--- PHASE 1 : EXTRACTION DES SIGNAUX ---")
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

    tous_les_cycles = []
    tous_les_labels = []

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            if 'pmus' not in df.columns:
                print(f"Ignoré: {os.path.basename(file_path)} (pas de colonne 'pmus')")
                continue

            cycles = extract_cycles(df)
            for cycle in cycles:
                label = label_cycle(cycle)
                if label is not None:
                    tous_les_cycles.append(cycle)
                    tous_les_labels.append(label)
        except Exception as e:
            print(f"Erreur sur {file_path}: {e}")

    print(f"\nExtraction terminée. {len(tous_les_labels)} cycles trouvés.")

    if len(tous_les_labels) > 100:
        print("\n--- PHASE 2 : PRÉPARATION DE L'IA ---")
        # Séparation Train/Test
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            tous_les_cycles, tous_les_labels, test_size=0.2, random_state=42, stratify=tous_les_labels
        )

        # Création des Datasets et DataLoaders
        train_dataset = AsynchronyDataset(X_train_raw, y_train_raw)
        test_dataset = AsynchronyDataset(X_test_raw, y_test_raw)

        # Batch size de 64 (traite 64 respirations en même temps)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialisation du modèle
        modele = CNNRespiratoire(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()  # Gère automatiquement les probabilités
        optimizer = optim.Adam(modele.parameters(), lr=0.001)  # Algorithme d'optimisation classique

        print("\n--- PHASE 3 : ENTRAÎNEMENT DU RÉSEAU DE NEURONES ---")
        num_epochs = 10  # Nombre de passages complets sur les données

        for epoch in range(num_epochs):
            modele.train()  # Mode entraînement
            running_loss = 0.0

            for signaux, labels in train_loader:
                signaux, labels = signaux.to(device), labels.to(device)

                optimizer.zero_grad()  # 1. Remise à zéro des gradients
                outputs = modele(signaux)  # 2. Prédiction de l'IA
                loss = criterion(outputs, labels)  # 3. Calcul de l'erreur
                loss.backward()  # 4. Calcul de la correction
                optimizer.step()  # 5. Application de la correction

                running_loss += loss.item()

            print(f"Époque {epoch + 1}/{num_epochs} - Erreur (Loss): {running_loss / len(train_loader):.4f}")

        print("\n--- PHASE 4 : ÉVALUATION FINALE ---")
        modele.eval()  # Mode évaluation (désactive le Dropout)
        all_preds = []
        all_targets = []

        with torch.no_grad():  # Pas besoin de calculer les gradients pour le test
            for signaux, labels in test_loader:
                signaux, labels = signaux.to(device), labels.to(device)
                outputs = modele(signaux)

                # On prend la classe avec la plus haute probabilité
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Inversion du dictionnaire pour l'affichage
        reverse_map = {0: "None", 1: "Premature", 2: "Delayed"}
        noms_cibles = [reverse_map[i] for i in sorted(list(set(all_targets)))]

        print("\nMatrice de Confusion :")
        print(confusion_matrix(all_targets, all_preds))

        print("\nRapport de Classification :")
        print(classification_report(all_targets, all_preds, target_names=noms_cibles))

    else:
        print("Pas assez de données pour entraîner le Deep Learning.")