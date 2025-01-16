"""
Task:
 Implement a class FullyConnected in the le FullyConnected.py in folder Layers , that
 inherits the base layer that we implemented earlier. This class has to provide the methods
 forward(input tensor) and backward(error tensor) as well as the property optimizer.
 
 - Write a constructor for this class, receiving the arguments (input size, output size).
 First, call its super-constructor. Set the inherited member trainable to True, as this
 layer has trainable parameters. Initialize the weights of this layer uniformly random in
 the range [01).


 - Implement a method forward(input tensor) which returns a tensor that serves as the
 input tensor for the next layer. input tensor is a matrix with input size columns
 and batch size rows. The batch size represents the number of inputs processed si
multaneously. The output size is a parameter of the layer specifying the number of
 columns of the output.
 - Addasetter and getter property optimizer which sets and returns the protected member
 optimizer for this layer. Properties o er a pythonic way of realizing getters and setters.
 Please get familiar with this concept if you are not aware of it.
 - Implement a method backward(error tensor) which returns a tensor that serves as
 the error tensor for the previous layer. Quick reminder: in the backward pass we are
 going in the other direction as in the forward pass.
 - Hint: if you discover that you need something here which is no longer available to you,
 think about storing it at the appropriate time.
 - To be able to test the gradients with respect to the weights: The member for the weights
 and biases should be named weights. For future reasons provide a property gradi
ent weights which returns the gradient with respect to the weights, after they have
 been calculated in the backward-pass. These properties are accessed by the unit tests
 and are therefore also important to pass the tests!
 - Use the method calculate update(weight tensor, gradient tensor) of your opti
mizer in your backward pass, in order to update your weights. Dont perform an
 update if the optimizer is not set.
 - You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestFullyConnected1

"""
# i have to update the weights and biases here 

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
        print(f"Setting input_tensor in forward pass: {input_tensor}")
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
        """Compute the regularization norm for this layer."""
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

"""
import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.input_tensor = None  # Initialize input_tensor attribute

    def forward(self, input_tensor):
        self.input_tensor = input_tensor  # Store input tensor
        return np.dot(input_tensor, self.weights) + self.bias

    def backward(self, error_tensor):

        # Gradient with respect to the weights
        d_weights = np.dot(self.input_tensor.T, error_tensor)
        d_bias = np.sum(error_tensor, axis=0)

        # Gradient with respect to the input tensor (needed for the previous layer)
        d_input = np.dot(error_tensor, self.weights.T)

        # Update the weights and biases if needed (for example, during training)
        self.weights -= 0.01 * d_weights  # Example learning rate of 0.01
        self.bias -= 0.01 * d_bias

        return d_input  # Return the gradient for the previous layer

    def norm(self):
        if self.regularizer:
            return self.regularizer.norm(self.weights)
        return 0

"""