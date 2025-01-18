"""
import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = "train"  # Private variable to manage phase state
        self.regularizer_loss = 0.0

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        if phase not in ["train", "test"]:
            raise ValueError("Phase must be 'train' or 'test'.")
        self._phase = phase
        for layer in self.layers:
            layer.testing_phase = (phase == "test")


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            if hasattr(self.optimizer, "regularizer") and self.optimizer.regularizer:
                layer.regularizer = self.optimizer.regularizer  # Assign regularizer to the layer
        if hasattr(layer, "initialize"):
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        if not self.data_layer:
            raise ValueError("Data layer is not set. Please set a valid data layer.")
        input_tensor, label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        self.current_label_tensor = label_tensor
        return input_tensor

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def compute_regularization_loss(self):
        regularization_loss = 0
        for layer in self.layers:
            if layer.trainable and hasattr(layer, "weights") and hasattr(layer, "regularizer"):
                regularization_loss += layer.regularizer.norm(layer.weights)
        return regularization_loss

    def train(self, iterations):
        self.phase = "train"
        for _ in range(iterations):
            # Forward pass
            prediction_tensor = self.forward()

            # Compute data loss
            loss_value = self.loss_layer.forward(prediction_tensor, self.current_label_tensor)

            # Compute regularization loss
            regularization_loss = self.compute_regularization_loss()

            # Total loss = Data loss + Regularization loss
            total_loss = loss_value + regularization_loss
            self.loss.append(total_loss)

            # Backward pass
            self.backward(self.current_label_tensor)

    def test(self, input_tensor):
        self.phase = "test"
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
"""


import copy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = "train"  # Private variable to manage phase state
        self.regularizer_loss = 0.0

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        if phase not in ["train", "test"]:
            raise ValueError("Phase must be 'train' or 'test'.")
        self._phase = phase
        for layer in self.layers:
            layer.testing_phase = (phase == "test")

    def append_layer(self, layer):
        
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            if hasattr(self.optimizer, "regularizer") and self.optimizer.regularizer:
                layer.regularizer = self.optimizer.regularizer  # Assign regularizer to the layer
        if hasattr(layer, "initialize"):
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        if not self.data_layer:
            raise ValueError("Data layer is not set. Please set a valid data layer.")
        input_tensor, label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        self.current_label_tensor = label_tensor
        return input_tensor

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def compute_regularization_loss(self):
        
        regularization_loss = 0
        for layer in self.layers:
            if (
                layer.trainable and
                hasattr(layer, "weights") and
                layer.weights is not None and
                getattr(layer, "regularizer", None) is not None
            ):
                regularization_loss += layer.regularizer.norm(layer.weights)
        return regularization_loss


    def train(self, iterations):
        self.phase = "train"
        for _ in range(iterations):
            # Forward pass
            prediction_tensor = self.forward()


            # Compute data loss
            loss_value = self.loss_layer.forward(prediction_tensor, self.current_label_tensor)
            
            # Compute regularization loss
            regularization_loss = self.compute_regularization_loss()

            # Total loss = Data loss + Regularization loss
            total_loss = loss_value + regularization_loss
            self.loss.append(total_loss)

            # Backward pass
            self.backward(self.current_label_tensor)

    def test(self, input_tensor):
        self.phase = "test"
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor


