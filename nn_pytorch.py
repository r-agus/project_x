"""
This module contains a Neural Network classifier using PyTorch.

This network is designed to perform a multi-classification task, from a tweet belonging to politicES dataset
it classifies the gender, profession, ideology (binary) and ideology (multi-class) of the tweet's author.

A BERT embedding layer is used to convert the tweet text into numerical format before passing it through the network.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from TextVectorRepresentation import vectorRepresentation_BERT, vectorRepresentation_TFIDF, vectorRepresentation_Word2Vec, separate_x_y_vectors, divide_train_val_test
from main import load_data
from sklearn.model_selection import train_test_split



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
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Cabezas de salida una clasificación por cada tarea
        self.gender_head   = nn.Linear(256, 2)
        self.prof_head     = nn.Linear(256, 3)
        self.bin_head      = nn.Linear(256, 2)
        self.multi_head    = nn.Linear(256, 4)


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
        gender = 0 if row[0] == 'male' else 1
        profession_map = {'politician': 0, 'journalist': 1, 'celebrity': 2}
        profession = profession_map.get(row[1], -1)  # Default to -1 if not found
        ideology_binary = 0 if row[2] == 'left' else 1
        ideology_multi_map = {'left': 0, 'moderate_left': 1, 'moderate_right': 2, 'right': 3}
        ideology_multi = ideology_multi_map.get(row[3], -1)  # Default to -1 if not found
        y_mapped.append([gender, profession, ideology_binary, ideology_multi])

    return torch.tensor(y_mapped, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
print(f"Using {device} device")
print(model)




path = "Datasets/EvaluationData/politicES_phase_2_train_public.csv"
data = load_data(path)
data = data.head(100)  # O especifica el número de filas que necesites
train_data, val_data, test_data = divide_train_val_test(data)
X_train, y_train = separate_x_y_vectors(train_data)
X_val, y_val = separate_x_y_vectors(val_data)
X_test, y_test = separate_x_y_vectors(test_data)
x_tfidf_train, x_tfidf_val, x_tfidf_test = vectorRepresentation_BERT(X_train, X_val, X_test)


# y = map_politicES_labels(y_raw)
y_train_mapped = map_politicES_labels(y_train.values)
y_val_mapped = map_politicES_labels(y_val.values)
y_test_mapped = map_politicES_labels(y_test.values)


# Test and train shapes
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)




# Dataset and Dataloader
train_dataset = TensorDataset(torch.tensor(x_tfidf_train, dtype=torch.float32), y_train_mapped)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
# TODO: ver si se puede ajustar
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Training loop. Training para cada clase.

for epoch in range(10):
    model.train()
    total_loss = 0

    for Xb, yb in train_loader:

        Xb = Xb.to(device)
        yb = yb.to(device)

        outputs = model(Xb)

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

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")


# Evaluation on test set
model.eval()
X_test_tensor = torch.tensor(x_tfidf_test, dtype=torch.float32).to(device)

with torch.no_grad():
    outs = model(X_test_tensor)

pred_gender  = torch.argmax(outs["gender"], dim=1)
pred_prof    = torch.argmax(outs["profession"], dim=1)
pred_bin     = torch.argmax(outs["ideology_bin"], dim=1)
pred_multi   = torch.argmax(outs["ideology_multi"], dim=1)



def evaluate(model, dataloader):
    model.eval()

    correct_gender = 0
    correct_prof = 0
    correct_bin = 0
    correct_multi = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            outputs = model(Xb)

            pred_gender = torch.argmax(outputs["gender"], dim=1)
            pred_prof   = torch.argmax(outputs["profession"], dim=1)
            pred_bin    = torch.argmax(outputs["ideology_bin"], dim=1)
            pred_multi  = torch.argmax(outputs["ideology_multi"], dim=1)

            correct_gender += (pred_gender == yb[:, 0]).sum().item()
            correct_prof   += (pred_prof   == yb[:, 1]).sum().item()
            correct_bin    += (pred_bin    == yb[:, 2]).sum().item()
            correct_multi  += (pred_multi  == yb[:, 3]).sum().item()

            total += yb.size(0)

    print("\n===== EVALUACIÓN EN TEST =====")
    print(f"Accuracy Género:          {correct_gender/total:.4f}")
    print(f"Accuracy Profesión:       {correct_prof/total:.4f}")
    print(f"Accuracy Ideología bin:   {correct_bin/total:.4f}")
    print(f"Accuracy Ideología multi: {correct_multi/total:.4f}")

# Dataloader for tests
test_dataset = TensorDataset(
    torch.tensor(x_tfidf_test, dtype=torch.float32),
    y_test_mapped
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
evaluate(model, test_loader)

# ESTO SOLO LO USO PARA VER LAS SALIDAS SI SON POCASs
# with torch.no_grad():
#     outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))

# pred_gender  = torch.argmax(outputs["gender"], dim=1).cpu().numpy()
# pred_prof    = torch.argmax(outputs["profession"], dim=1).cpu().numpy()
# pred_bin     = torch.argmax(outputs["ideology_bin"], dim=1).cpu().numpy()
# pred_multi   = torch.argmax(outputs["ideology_multi"], dim=1).cpu().numpy()

# print("Predicciones listas:")
# print("Género:", pred_gender)
# print("Profesión:", pred_prof)
# print("Ideología (binaria):", pred_bin)
# print("Ideología (multi):", pred_multi)
