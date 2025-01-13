"""
Task:
 Implement a class Flatten in the le Flatten.py in folder Layers . This class has to provide
 the methods forward(input tensor) and backward(error tensor).

 - Write a constructor for this class, receiving no arguments.
 - Implement a method forward(input tensor), which reshapes and returns the input tensor.
 - Implement a method backward(error tensor) which reshapes and returns the error tensor.
 - You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestFlatten
"""

from .Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        'self.trainable=True'

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(self.input_shape[0],-1)

    def backward(self, error_tensor):
        return  error_tensor.reshape(self.input_shape)

