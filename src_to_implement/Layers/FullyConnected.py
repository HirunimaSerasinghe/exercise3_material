

import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.bias = None
        self.optimizer = None
        self._gradient_weights = None
        self.regularizer = None

    def initialize(self, weights_initializer, bias_initializer):
        w = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        b = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        self.weights = np.vstack((w,b))
                    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # Bias-Spalte anhängen
        batch_size = input_tensor.shape[0]
        bias_col = np.ones((batch_size, 1))
        input_with_bias = np.hstack((input_tensor, bias_col))
        output_tensor = input_with_bias @ self.weights
        return output_tensor

    def backward(self, error_tensor):
        if self.input_tensor is None:
            raise ValueError("forward() must be called before backward()")
        batch_size = self.input_tensor.shape[0]
        bias_col = np.ones((batch_size, 1))
        input_with_bias = np.hstack((self.input_tensor, bias_col))

        self._gradient_weights = input_with_bias.T @ error_tensor

        if self.optimizer and self.optimizer.regularizer:
            regularization_gradient = self.optimizer.regularizer.calculate_gradient(self.weights)
            self._gradient_weights += regularization_gradient

            # Update weights with optimizer
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)

            # Compute and return the error tensor for the previous layer
        error_tensor_prev = error_tensor @ self.weights[:-1].T
        return error_tensor_prev

    def norm(self):

        if self.regularizer is not None:
            return self.regularizer.norm(self.weights)
        return 0

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
