import numpy as np
from Layers.FullyConnected import FullyConnected

class RNN:
    def __init__(self, input_size, hidden_size, output_size=3):  # Set output_size to 3 by default
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trainable = True
        self.hidden_state = np.zeros((1, hidden_size))
        self.memorize = False
        self.input_to_hidden = FullyConnected(input_size, hidden_size)
        self.hidden_to_hidden = FullyConnected(hidden_size, hidden_size)
        self.hidden_to_output = FullyConnected(hidden_size, output_size)  # Ensure output_size = 3

    def norm(self):
        if self.regularizer:
            return self.regularizer.norm(self.weights)
        return 0

    def forward(self, input_tensor):
        if len(input_tensor.shape) == 2: 
            input_tensor = input_tensor[:, None, :] 

        batch_size, sequence_length, input_dim = input_tensor.shape
        hidden_states = []

        if not self.memorize or self.hidden_state is None:
            self.hidden_state = np.zeros((batch_size, self.hidden_to_hidden.input_size))

        # Iterate over each time step in the sequence
        for t in range(sequence_length):
            input_t = input_tensor[:, t, :]  # Input at time step t
            self.hidden_state = np.tanh(
                self.input_to_hidden.forward(input_t) + 
                self.hidden_to_hidden.forward(self.hidden_state)
            )
            
            # Print hidden state shape after each time step
            print(f"Hidden state shape after time step {t}: {self.hidden_state.shape}")
            
            hidden_states.append(self.hidden_state)

        # Only return the last hidden state, which is used for classification
        print(f"RNN output shape (last hidden state): {self.hidden_state.shape}")
        
        # Apply the fully connected layer to map the last hidden state to the desired output shape
        output = self.hidden_to_output.forward(self.hidden_state)  # Shape: (batch_size, 3)

        return output  # Shape: (batch_size, 3)


    def backward(self, error_tensor):
        """
        Perform the backward pass for the RNN using Backpropagation Through Time (BPTT).
        Ensure error_tensor is 3D: (batch_size, sequence_length, output_size)
        """
        # If error_tensor is 2D (batch_size, output_size), add a time dimension (sequence_length = 1)
        if len(error_tensor.shape) == 2:
            error_tensor = error_tensor[:, None, :]

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
        print("Weight dictionary:", weight_dict)  # Debugging print
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
