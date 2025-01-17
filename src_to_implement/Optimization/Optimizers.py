import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        # Attribute to hold the regularizer (if any)
        self.regularizer = None
        self.learning_rate = learning_rate

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer



class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            # Apply the regularizer gradient to the existing gradients
            regularization_gradient = self.regularizer.calculate_gradient(weight_tensor)
            gradient_tensor += regularization_gradient  # Add the regularizer's gradient"""
        return weight_tensor - self.learning_rate * gradient_tensor



class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        super().__init__(learning_rate)
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            # Initialize the velocity tensor to match the shape of weights
            self.velocity = np.zeros_like(weight_tensor)

        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

            # Update the velocity using the momentum formula
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        # Update the weights by adding the velocity
        updated_weights = weight_tensor + self.velocity
        return updated_weights


class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float):
        super().__init__(learning_rate)  # Initialize the base optimizer
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        self.t += 1
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)

        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)

        updated_weights = weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_weights

