# -*- coding: utf-8 -*-
"""
PyTorch: Custom nn Modules
--------------------------

A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""
import torch
import torch.nn as nn
from torch import Tensor, LongTensor
import scipy.io as sio
import numpy as np
from typing import Tuple, Any


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1 = self.linear1(x)
        h1_s1 = self.sigmoid(h1)
        h2 = self.linear2(h1_s1)
        h2_s2 = self.sigmoid(h2)
        return h2_s2


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
0
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    label_indices = labels - 1
    y = np.eye(num_classes)
    return np.squeeze(y[label_indices]).astype(int)


def shuffled_tensors(
        ordered_x: np.ndarray,
        ordered_y: np.ndarray) -> Tuple[Tensor, Tensor]:
    (n_spls, _) = ordered_x.shape
    indices = np.array(range(0, n_spls))
    np.random.shuffle(indices)
    shuffled_x = Tensor(ordered_x[indices, :])
    shuffled_y = LongTensor(ordered_y[indices, :])
    return shuffled_x, np.squeeze(shuffled_y)


features_path = "/Users/YWU/Projects/pytorch_test/features.mat"
labels_path = "/Users/YWU/Projects/pytorch_test/labels.mat"
features: np.array = \
    sio.loadmat(features_path)["psix"].transpose()  # shape: (3060, 36000)
labels: np.array = \
    sio.loadmat(labels_path)["imageClass"].transpose() - 1  # shape: (3060, 1)
# labels_one_hot = one_hot_embedding(labels, 102)  # shape: (3060, 102)

(n_samples, n_dimension) = features.shape
n_classes = len(np.unique(labels))

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 3060, n_dimension, 128, n_classes

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# criterion = torch.nn.MSELoss(reduction='sum')
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Create shuffled Tensors to hold inputs and outputs
    (x, y) = shuffled_tensors(features, labels)

    for b_ind in range(n_samples // N):
        batch_from: int = N * b_ind
        batch_to: int = batch_from + N
        x_batch = x[range(batch_from, batch_to)]
        y_batch = y[range(batch_from, batch_to)]

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_batch)

        # Compute and print loss
        loss = criterion(y_pred, y_batch)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
