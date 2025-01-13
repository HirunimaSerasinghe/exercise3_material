import numpy as np


class Dropout:
    def __init__(self, probability=0.8):

        if not (0 < probability <= 1):
            raise ValueError("Probability must be in the range (0, 1].")

        self.probability = probability
        self.testing_phase = False  # Default to training phase
        self.trainable = False  # No trainable parameters
        self.mask = None  # Mask used for dropout

    def forward(self, input_tensor):

        if not self.testing_phase:
            # Training phase: apply dropout
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability)
            return input_tensor * self.mask / self.probability
        else:
            # Testing phase: pass input tensor unchanged
            return input_tensor

    def backward(self, error_tensor):

        if not self.testing_phase:
            # Training phase: scale the gradient with the dropout mask
            return error_tensor * self.mask / self.probability
        else:
            # Testing phase: return the error tensor unchanged
            return error_tensor
