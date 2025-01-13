import numpy as np

class Sigmoid:
    def __init__(self):
        self.activations = None  # To store activations for the backward pass
        self.trainable = False

    def forward(self, input_tensor):
        self.activations = 1 / (1 + np.exp(-input_tensor))  # Compute and store activations
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * self.activations * (1 - self.activations)  # Derivative of Sigmoid
