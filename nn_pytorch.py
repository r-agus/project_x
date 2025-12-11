"""
This module contains a Neural Network classifier using PyTorch.

This network is designed to perform a multi-classification task, from a tweet belonging to politicES dataset
it classifies the gender, profession, ideology (binary) and ideology (multi-class) of the tweet's author.

A BERT embedding layer is used to convert the tweet text into numerical format before passing it through the network.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from TextVectorRepresentation import (
    vectorRepresentation_BERT,
    vectorRepresentation_TFIDF,
    vectorRepresentation_Word2Vec,
    separate_x_y_vectors,
    divide_train_val_test
)
from exploratory_analysis import load_data


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
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )



        self.gender_head = nn.Linear(256, 2)
        self.prof_head = nn.Linear(256, 3)
        self.bin_head = nn.Linear(256, 2)
        self.multi_head = nn.Linear(256, 4)

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
#   EVALUATION ON TEST
# ============================

def evaluate(model, dataloader, device):
    model.eval()
    correct_gender = 0
    correct_prof = 0
    correct_bin = 0
    correct_multi = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            outputs = model(Xb)

            pred_gender = torch.argmax(outputs["gender"], dim=1)
            pred_prof = torch.argmax(outputs["profession"], dim=1)
            pred_bin = torch.argmax(outputs["ideology_bin"], dim=1)
            pred_multi = torch.argmax(outputs["ideology_multi"], dim=1)

            correct_gender += (pred_gender == yb[:, 0]).sum().item()
            correct_prof += (pred_prof == yb[:, 1]).sum().item()
            correct_bin += (pred_bin == yb[:, 2]).sum().item()
            correct_multi += (pred_multi == yb[:, 3]).sum().item()

            total += yb.size(0)

    print("\n===== TEST EVALUATION =====")
    print(f"Accuracy Gender:        {correct_gender/total:.4f}")
    print(f"Accuracy Profession:    {correct_prof/total:.4f}")
    print(f"Accuracy Ideology bin:  {correct_bin/total:.4f}")
    print(f"Accuracy Ideology mult: {correct_multi/total:.4f}")


if __name__ == "__main__":
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

    path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
    data = load_data(path)
    n = 3000
    data = data.sample(n=n, random_state=42)

    train_data, val_data, test_data = divide_train_val_test(data)

    X_train, y_train = separate_x_y_vectors(train_data)
    X_val, y_val = separate_x_y_vectors(val_data)
    X_test, y_test = separate_x_y_vectors(test_data)


    # ============================
    #   BERT VECTORIZATION
    # ============================

    x_train, x_val, x_test = vectorRepresentation_BERT(X_train, X_val, X_test)

    y_train_mapped = map_politicES_labels(y_train.values)
    y_val_mapped = map_politicES_labels(y_val.values)
    y_test_mapped = map_politicES_labels(y_test.values)


    # ============================
    #   DATA LOADERS
    # ============================

    # x_train_dense = x_train.toarray()
    # x_val_dense   = x_val.toarray()
    # x_test_dense  = x_test.toarray()

    # For BERT embeddings, the output is already dense
    x_train_dense = x_train
    x_val_dense   = x_val
    x_test_dense  = x_test


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


    # ============================
    #   TRAINING CONFIG
    # ============================

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-6)

    # ============================
    #   TRAINING LOOP WITH EARLY STOPPING
    # ============================

    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    max_epochs = 50  # número máximo de épocas

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            outputs = model(Xb)

            loss = (
                criterion(outputs["gender"], yb[:, 0]) +
                criterion(outputs["profession"], yb[:, 1]) +
                criterion(outputs["ideology_bin"], yb[:, 2]) +
                criterion(outputs["ideology_multi"], yb[:, 3])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                out = model(Xv)
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

    # Cargar el mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    evaluate(model, test_loader, device)
