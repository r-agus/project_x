"""
This module contains a Neural Network classifier using PyTorch.

This network is designed to perform a multi-classification task, from a tweet belonging to politicES dataset
it classifies the gender, profession, ideology (binary) and ideology (multi-class) of the tweet's author.

A BERT embedding layer is used to convert the tweet text into numerical format before passing it through the network.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Defer heavy imports until runtime to allow Sphinx autodoc to work
tweet_embeddings = None
ytrain = None

def _load_data():
    """Load heavy data dependencies at runtime (not at import time)."""
    global tweet_embeddings, ytrain
    if tweet_embeddings is None:
        from TextVectorRepresentation import vectorRepresentation_BERT
        from main import ytrain as _yt
        ytrain = _yt
        tweet_embeddings = vectorRepresentation_BERT(ytrain)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

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
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 outputs
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output logits.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

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

if __name__ == "__main__":
    _load_data()
    X = tweet_embeddings.numpy()
    data = ytrain
    num_columns = data.shape[1]
    y_raw = data.iloc[:, 1:num_columns-1].values  # Labels are all columns except the first (user) and last (tweet text)

    y = map_politicES_labels(y_raw)

    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)


