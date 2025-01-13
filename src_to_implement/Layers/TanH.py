import numpy as np
"""
class TanH:
    def __init__(self):
        self.activations = None  # To store activations for the backward pass

    def forward(self, input_tensor):
        
        self.activations = np.tanh(input_tensor)  # Store activations
        return self.activations

    def backward(self, error_tensor):
       
        return error_tensor * (1 - self.activations**2)  # Derivative of TanH
"""



class TanH:
    def __init__(self):
        self.trainable = False  # Add the trainable attribute
        self.activations = None  # To store activations for the backward pass

    def forward(self, input_tensor):

        self.activations = np.tanh(input_tensor)  # Store activations
        return self.activations

    def backward(self, error_tensor):

        return error_tensor * (1 - self.activations**2)  # Derivative of TanH
