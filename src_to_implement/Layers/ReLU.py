"""
 Task:
 Implement a class ReLU in the le ReLU.py in folder Layers . This class also has to
 provide the methods forward(input tensor) and backward(error tensor).

 - Write a constructor for this class, receiving no arguments. The ReLU does not have
 trainable parameters, so you dont have to change the inherited member trainable.
 - Implement a method forward(input tensor) which returns a tensor that serves as the
 input tensor for the next layer.
 - Implement a method backward(error tensor) which returns a tensor that serves as
 the error tensor for the previous layer.
 
 Hint: the same hint as before applies.
 
 You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestReLU



# member trainable of ReLU

class ReLU()

    def forward(input_tensor)
        return next_layer_input_tensor;

    def backward(error_tensor)
        return previous_layer_error_tensor();

"""
import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        
        gradient = (self.input_tensor > 0).astype(float)
        return error_tensor * gradient
