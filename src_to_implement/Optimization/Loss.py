"""
Task:
 Implement a class CrossEntropyLoss in the le: Loss.py in folder Optimization .
 When forward propagating we now additionally need the argument label tensor for
 forward(prediction tensor, label tensor) and backward(label tensor). We don t con
sider the loss function as a layer like the previous ones in our framework, thus it should not
 inherit the base layer.

 - Write a constructor for this class, receiving no arguments.
 Implement a method forward(prediction tensor, label tensor) which computes the
 Loss value according the CrossEntropy Loss formula accumulated over the batch.
 - Implement a method backward(label tensor) which returns the error tensor for the
 previous layer. The backpropagation starts here, hence no error tensor is needed.
 Instead, we need the label tensor.
 - Hint: the same hint as before applies.
 - Remember: Loops are slow in Python. Use NumPy functions instead!
 You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestCrossEntropyLoss

"""


import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.epsilon = np.finfo(np.float64).eps # A very small number (ε ≈ 2.22e-16) added to avoid taking the logarithm of zero. Without this, the code would throw errors when prediction_tensor contains zero.

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor 
        per_sample_loss = -np.sum(label_tensor * np.log(prediction_tensor + self.epsilon), axis=1) #avoid log of zero again
        return np.sum(per_sample_loss)

    def backward(self, label_tensor):
        return - label_tensor / (self.prediction_tensor + self.epsilon) 