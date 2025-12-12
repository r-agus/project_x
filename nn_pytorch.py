"""
This module contains a Neural Network classifier using PyTorch.

This network is designed to perform a multi-classification task, from a tweet belonging to politicES dataset
it classifies the gender, profession, ideology (binary) and ideology (multi-class) of the tweet's author.

A BERT embedding layer is used to convert the tweet text into numerical format before passing it through the network.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
# from TextVectorRepresentation import (
#     vectorRepresentation_BERT,
#     vectorRepresentation_TFIDF,
#     vectorRepresentation_Word2Vec,
#     separate_x_y_vectors,
#     divide_train_val_test
# )
# from main import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ============================
#   MODEL
# ============================

class NeuralNetwork(nn.Module):
    """
    A feedforward neural network for multi-class classification.
    """
    def __init__(self):
        """
        Initializes the neural network layers.
        The network consists of:
        - An input layer that flattens the input tensor.
        - A sequence of linear layers with ReLU activations.
        - An output layer with 4 outputs corresponding to the 4 classification tasks.
        """
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(200, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )




        self.gender_head = nn.Linear(128, 2)
        self.prof_head = nn.Linear(128, 3)
        self.bin_head = nn.Linear(128, 2)
        self.multi_head = nn.Linear(128, 4)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output logits.
        """
        x = x.view(x.size(0), -1)
        h = self.shared(x)

        return {
            "gender": self.gender_head(h),
            "profession": self.prof_head(h),
            "ideology_bin": self.bin_head(h),
            "ideology_multi": self.multi_head(h)
        }


# ============================
#   LABEL MAPPING
# ============================

def map_politicES_labels(y_raw):
    """
    Maps the raw labels from the politicES dataset to numerical format. Mapping is as follows:
    - Gender: {0: Male, 1: Female}
    - Profession: {0: Politician, 1: Journalist, 2: Celebrity}
    - Ideology (binary): {0: Left, 1: Right}
    - Ideology (multi-class): {0: Left, 1: Moderate Left, 2: Moderate Right, 3: Right}
    Args:
        y_raw (array-like): Raw labels from the dataset. This array should not include the user and tweet text columns.
    Returns:
        torch.Tensor: Mapped labels in numerical format.
    """
    y_mapped = []
    for row in y_raw:
        gender = 0 if row[0] == "male" else 1

        profession_map = {"politician": 0, "journalist": 1, "celebrity": 2}
        profession = profession_map.get(row[1], -1)

        ideology_binary = 0 if row[2] == "left" else 1

        ideology_multi_map = {
            "left": 0, "moderate_left": 1,
            "moderate_right": 2, "right": 3
        }
        ideology_multi = ideology_multi_map.get(row[3], -1)

        y_mapped.append([gender, profession, ideology_binary, ideology_multi])

    return torch.tensor(y_mapped, dtype=torch.long)


# ============================
#   DEVICE + MODEL
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)

print(f"Using {device} device")
print(model)


# ============================
#   LOAD DATA
# ============================

# path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
# data = load_data(path)
# n = 3000
# data = data.sample(n=n, random_state=42)

# train_data, val_data, test_data = divide_train_val_test(data)

# X_train, y_train = separate_x_y_vectors(train_data)
# X_val, y_val = separate_x_y_vectors(val_data)
# X_test, y_test = separate_x_y_vectors(test_data)


# ============================
#   BERT VECTORIZATION
# ============================

# x_train, x_val, x_test = vectorRepresentation_BERT(X_train, X_val, X_test)
y_train = np.load('ProcessedData/y_train_30000.npy', allow_pickle=True)
y_val = np.load('ProcessedData/y_val_30000.npy', allow_pickle=True)
y_test = np.load('ProcessedData/y_test_30000.npy', allow_pickle=True)

y_train_mapped = map_politicES_labels(y_train)
y_val_mapped = map_politicES_labels(y_val)
y_test_mapped = map_politicES_labels(y_test)


# ============================
#   DATA LOADERS
# ============================

X_train = np.load("ProcessedData/x_word2vec_train_30000.npy")
X_val = np.load("ProcessedData/x_word2vec_val_30000.npy")
X_test = np.load("ProcessedData/x_word2vec_test_30000.npy")

# For BERT embeddings, the output is already dense
x_train_dense = X_train
x_val_dense   = X_val
x_test_dense  = X_test


train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train_dense, dtype=torch.float32), y_train_mapped),
    batch_size=32, shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(x_val_dense, dtype=torch.float32), y_val_mapped),
    batch_size=32, shuffle=False
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(x_test_dense, dtype=torch.float32), y_test_mapped),
    batch_size=32, shuffle=False
)


def compute_class_weights(y_train_column):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase
    """
    classes, counts = torch.unique(y_train_column, return_counts=True)
    total = counts.sum().item()
    weights = [total / c.item() for c in counts]
    # Normalizar
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


# ============================
#   TRAINING CONFIG
# ============================

weights_gender = compute_class_weights(y_train_mapped[:,0]).to(device)
criterion_gender = nn.CrossEntropyLoss(weight=weights_gender)

weights_prof = compute_class_weights(y_train_mapped[:,1]).to(device)
criterion_prof = nn.CrossEntropyLoss(weight=weights_prof)

weights_bin = compute_class_weights(y_train_mapped[:,2]).to(device)
criterion_bin = nn.CrossEntropyLoss(weight=weights_bin)

weights_multi = compute_class_weights(y_train_mapped[:,3]).to(device)
criterion_multi = nn.CrossEntropyLoss(weight=weights_multi)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# ============================
#   TRAINING LOOP WITH EARLY STOPPING
# ============================

patience = 3
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
max_epochs = 50  # número máximo de épocas
train_losses = []
val_losses = []

for epoch in range(max_epochs):
    model.train()
    total_loss = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        outputs = model(Xb)

        loss = (
            criterion_gender(outputs["gender"], yb[:,0]) +
            criterion_prof(outputs["profession"], yb[:,1]) +
            criterion_bin(outputs["ideology_bin"], yb[:,2]) +
            criterion_multi(outputs["ideology_multi"], yb[:,3])
        )

        loss = (
            criterion(outputs["gender"], yb[:,0]) +
            criterion(outputs["profession"], yb[:,1]) +
            criterion(outputs["ideology_bin"], yb[:,2]) +
            criterion(outputs["ideology_multi"], yb[:,3])
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step() 

    # Validación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(device), yv.to(device)
            out = model(Xv)
            # vloss = (
            #     criterion_gender(out["gender"], yv[:, 0]) +
            #     criterion_prof(out["profession"], yv[:, 1]) +
            #     criterion_bin(out["ideology_bin"], yv[:, 2]) +
            #     criterion_multi(out["ideology_multi"], yv[:, 3])
            # )
            vloss = (
                criterion(out["gender"], yv[:, 0]) +
                criterion(out["profession"], yv[:, 1]) +
                criterion(out["ideology_bin"], yv[:, 2]) +
                criterion(out["ideology_multi"], yv[:, 3])
            )
            val_loss += vloss.item()


    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))


# Cargar el mejor modelo
if best_model_state is not None:
    model.load_state_dict(best_model_state)
# ============================
#   TRAIN VS VALIDATION LOSS
# ============================

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
#   EVALUATION ON TEST
# ============================


def plot_confusion_matrix(y_true, y_pred, task_name, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f"Confusion Matrix - {task_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def evaluate(model, dataloader):
    model.eval()

    # Listas para acumular
    all_true = { "gender": [], "profession": [], "ideology_bin": [], "ideology_multi": [] }
    all_pred = { "gender": [], "profession": [], "ideology_bin": [], "ideology_multi": [] }
    all_proba = { "gender": [], "profession": [], "ideology_bin": [], "ideology_multi": [] }

    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            outputs = model(Xb)

            # Obtener logits y probabilidades softmax
            for key in outputs:
                logits = outputs[key]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_true[key].extend(yb[:, list(outputs.keys()).index(key)].cpu().numpy())
                all_pred[key].extend(preds.cpu().numpy())
                all_proba[key].extend(probs.cpu().numpy())


    print("\n========== FULL METRICS ==========\n")

    for task in all_true.keys():
        y_true = np.array(all_true[task])
        y_pred = np.array(all_pred[task])
        y_proba = np.array(all_proba[task])

        print(f"\n===== {task.upper()} =====")

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # Precision/Recall/F1 (macro)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro):    {recall:.4f}")
        print(f"F1-score (macro):  {f1:.4f}")

        # ROC–AUC
        num_classes = y_proba.shape[1]

        if num_classes == 2:
            # Para binaria
            auc = roc_auc_score(y_true, y_proba[:,1])
        else:
            # Para multiclase
            try:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            except:
                auc = float("nan")

        print(f"ROC-AUC:           {auc:.4f}")

        # Classification report
        print("\n" + classification_report(y_true, y_pred, zero_division=0))

        if task == "gender":
            classes = ["Male", "Female"]
        elif task == "profession":
            classes = ["Politician", "Journalist", "Celebrity"]
        elif task == "ideology_bin":
            classes = ["Left", "Right"]
        elif task == "ideology_multi":
            classes = ["Left", "Moderate Left", "Moderate Right", "Right"]

        plot_confusion_matrix(y_true, y_pred, task, class_names=classes)

evaluate(model, test_loader)
