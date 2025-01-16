"""
Task:
 Implement a class NeuralNetwork in the le: NeuralNetwork.py in the same folder as
 NeuralNetworkTests.py .

 - Implement ve member variables. An optimizer object received upon construction as
 the rst argument. A list loss which will contain the loss value for each iteration after
 calling train. A list layers which will hold the architecture, a member data layer, which
 will provide input data and labels and a member loss layer referring to the special layer
 providing loss and prediction. You do not need to care for lling these members with
 actual values. They will be set within the unit tests.
 - Implement a method forward using input from the data layer and passing it through
 all layers of the network. Note that the data layer provides an input tensor and a
 label tensor upon calling next() on it. The output of this function should be the
 output of the last layer (i.e. the loss layer) of the network.
 - Implement a method backward starting from the loss layer passing it the label tensor
 for the current input and propagating it back through the network.
 - Implement the method append layer(layer). If the layer is trainable, it makes a
 deep copy of the neural networks optimizer and sets it for the layer by using its
 optimizer property. Both, trainable and non-trainable layers, are then appended to the
 list layers.
 - Note: Wewill implement optimizers that have an internal state in the upcoming exercises,
 which makes copying of the optimizer object necessary.
 - Additionally implement a convenience method train(iterations), which trains the net
work for iterations and stores the loss for each iter
ation.
 - Finally implement a convenience method test(input tensor) which propagates the in
put tensor through the network and returns the prediction of the last layer. For clas
si cation tasks we typically query the probabilistic output of the SoftMax layer.
 You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestNeuralNetwork1


"""
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
        self.phase = "train"

    @property
    def phase(self):
        return self.phase

    @phase.setter
    def phase(self, value):
        if value not in ["train", "test"]:
            raise ValueError("Phase must be 'train' or 'test'")
        self.phase = value
        for layer in self.layers:
            layer.testing_phase = (value == "test")

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        if hasattr(layer, 'initialize'):
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
        # starting from loss lyr and back propagate through all layers
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        self.phase = "train"
        for _ in range(iterations):
            prediction_tensor = self.forward()
            loss_value = self.loss_layer.forward(prediction_tensor, self.current_label_tensor)
            regularization_loss = 0
            for layer in self.layers:
                if layer.trainable and hasattr(layer, "weights"):
                    regularization_loss += self.optimizer.regularizer.norm(layer.weights)
            total_loss = loss_value + regularization_loss
            self.loss.append(total_loss)
            self.backward(self.current_label_tensor)

    def test(self, input_tensor):
        self.phase = "test"
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
"""
""">>Best up to now
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
        self._phase = "train"  # Use a private variable to avoid infinite recursion

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        if phase not in ["train", "test"]:
            raise ValueError("Phase must be 'train' or 'test'")
        self._phase = phase
        for layer in self.layers:
            layer.testing_phase = (phase == "test")

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
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
        # Start from the loss layer and backpropagate through all layers
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        self.phase = "train"
        for _ in range(iterations):
            # Forward pass
            prediction_tensor = self.forward()

            # Compute data loss
            loss_value = self.loss_layer.forward(prediction_tensor, self.current_label_tensor)

            # Compute regularization loss
            regularization_loss = 0
            for layer in self.layers:
                if layer.trainable and hasattr(layer, "norm"):
                    regularization_loss += layer.norm()

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

    """def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        if hasattr(layer, "initialize"):
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)"""

    def append_layer(self, layer):
        """Append a layer to the network, initializing weights and setting optimizer."""
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

    """def compute_regularization_loss(self):
        regularization_loss = 0
        for layer in self.layers:
            if layer.trainable and hasattr(layer, "norm"):
                regularization_loss += layer.norm()
        return regularization_loss"""

    def compute_regularization_loss(self):
        """Compute the total regularization loss across all layers."""
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
