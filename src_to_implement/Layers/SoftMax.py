"""
 Task:
 Implement a class SoftMax in the le: SoftMax.py in folder Layers . This class also has
 to provide the methods forward(input tensor) and backward(error tensor).
 
 - Write a constructor for this class, receiving no arguments.
 - Implement a method forward(input tensor) which returns the estimated class proba
bilities for each row representing an element of the batch.
- Implement a method backward(error tensor) which returns a tensor that serves as
 the error tensor for the previous layer.
 - Hint: again the same hint as before applies.
 Remember: Loops are slow in Python. Use NumPy functions instead!
 - You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestSoftMax



class SoftMax();
    def forward(input_tensor)
        return est_class_probability;
    def backward(error_tensor)
        return prev_layer_error_tensor;


"""
import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.output_tensor = None

    def forward(self, input_tensor):
        # Shift input for numerical stability
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        self.output_tensor = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        batch_size, num_classes = self.output_tensor.shape
        grad_input = np.zeros_like(self.output_tensor)

        for i in range(batch_size):
            y = self.output_tensor[i].reshape(-1, 1)  # Convert to column vector
            jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
            grad_input[i] = np.dot(jacobian_matrix, error_tensor[i])

        return grad_input
