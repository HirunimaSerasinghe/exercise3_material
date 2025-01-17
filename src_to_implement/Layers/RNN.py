import numpy as np
from Layers.FullyConnected import FullyConnected



class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable=True

        # Initialize hidden state
        self.hidden_state = np.zeros((1, hidden_size))

        # Memorize flags
        self.memorize = False

        # Trainable layers
        self.input_to_hidden = FullyConnected(input_size, hidden_size)
        self.hidden_to_hidden = FullyConnected(hidden_size, hidden_size)
        self.hidden_to_output = FullyConnected(hidden_size, output_size)

    def norm(self):
        if self.regularizer:
            return self.regularizer.norm(self.weights)
        return 0

    def forward(self, input_tensor):
        # Handle both 2D and 3D inputs
        if len(input_tensor.shape) == 2:  # Single time step input
            input_tensor = input_tensor[:, None, :]  # Add a time dimension (sequence_length = 1)

        batch_size, sequence_length, input_dim = input_tensor.shape
        hidden_states = []

        # Initialize hidden state if necessary
        if not self.memorize or self.hidden_state is None:
            self.hidden_state = np.zeros((batch_size, self.hidden_to_hidden.input_size))

        # Iterate over the sequence
        for t in range(sequence_length):
            input_t = input_tensor[:, t, :]  # Input at time step t
            self.hidden_state = np.tanh(
                self.input_to_hidden.forward(input_t) +
                self.hidden_to_hidden.forward(self.hidden_state)
            )
            hidden_states.append(self.hidden_state)

        return np.stack(hidden_states, axis=1)  # Return all hidden states

    def backward(self, error_tensor):
        """
        Perform the backward pass for the RNN using Backpropagation Through Time (BPTT).
        """
        batch_size, sequence_length, _ = error_tensor.shape
        d_hidden_next = np.zeros_like(self.hidden_state)

        for t in reversed(range(sequence_length)):
            d_output = error_tensor[:, t, :]

            # Backprop through output layer
            d_hidden = self.hidden_to_output.backward(d_output)
            d_hidden += d_hidden_next  # Add the gradient flowing back from the next time step

            # Backprop through hidden-to-hidden and input-to-hidden layers
            d_hidden_next = self.hidden_to_hidden.backward(d_hidden * (1 - self.hidden_state ** 2))
            self.input_to_hidden.backward(d_hidden * (1 - self.hidden_state ** 2))

        return d_hidden_next

    @property
    def weights(self):
        return np.concatenate([
            self.input_to_hidden.weights.flatten(),
            self.hidden_to_hidden.weights.flatten(),
            self.hidden_to_output.weights.flatten()
        ])

    @weights.setter
    def weights(self, weight_dict):
        self.input_to_hidden.weights = weight_dict['input_to_hidden']
        self.hidden_to_hidden.weights = weight_dict['hidden_to_hidden']
        self.hidden_to_output.weights = weight_dict['hidden_to_output']

    def calculate_regularization_loss(self):
        return (
                self.input_to_hidden.calculate_regularization_loss() +
                self.hidden_to_hidden.calculate_regularization_loss() +
                self.hidden_to_output.calculate_regularization_loss()
        )

    def initialize(self, weights_initializer, bias_initializer):
        self.input_to_hidden.initialize(weights_initializer, bias_initializer)
        self.hidden_to_hidden.initialize(weights_initializer, bias_initializer)
        self.hidden_to_output.initialize(weights_initializer, bias_initializer)
