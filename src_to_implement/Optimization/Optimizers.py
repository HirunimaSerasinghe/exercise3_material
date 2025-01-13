"""
self NOTE: 
about stocastic gradient descent- same as gradient descent -(one at a time unlike batch or mini) ( decent at estimating parameters, fast processing and low computiational quality to guess the cost fn ~ how far off is our prediction from the actual value = over time those guessesgive us the learning rate) 

TASK:
 Implement the class Sgd in the le Optimizers.py in folder Optimization . DONE

 -The Sgd constructor receives the learning rate with data type foat. DONE

 -Implement the method calculate update(weight tensor, gradient tensor) that returns the updated weights
 according to the basic gradient descent update scheme. DONE

 You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestOptimizers1 TODO

"""

"""
 Task:
 Implement the classes SgdWithMomentum and Adam in the le Optimizers.py in folder
 Optimization . These classes all have to provide the method
 calculate update(weight tensor, gradient tensor).
 
- The SgdWithMomentum constructor receives the learning rate and the momen
tum rate in this order.
- The Adam constructor receives the learning rate, mu and rho, exactly in this order.
- In literature mu is often referred as 1 and rho as 2.
- Implement for both optimizers the method
 calculate update(weight tensor, gradient tensor) as it was done with the basic
 SGD Optimizer.
- You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestOptimizers2
"""

import numpy as np
"""
class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            regularization_gradient = self.regularizer.calculate_gradient(weight_tensor)
            gradient_tensor += regularization_gradient
        return self.update_weights(weight_tensor, gradient_tensor)

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.velocity
        return updated_weights

class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        
        self.t += 1
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)
        
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)
        
        updated_weights = weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_weights

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()#NOTE: datatype float
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        #TODO: check formula- explation previous wt - learning rate (guess accuracy over time) * gradient tensor ( the gradient ratio thing from that one vid) 
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor 
        return updated_weights
"""


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor):
        print(f"Weight (before update): {weight_tensor}")
        print(f"Gradient (before regularizer): {gradient_tensor}")

        if self.regularizer:

            regularization_gradient = self.regularizer.calculate_gradient(weight_tensor)
            print(f"Regularizer Gradient: {regularization_gradient}")
            gradient_tensor = gradient_tensor + regularization_gradient
        print(f"Gradient (after regularizer): {gradient_tensor}")
        return self.update_weights(weight_tensor, gradient_tensor)


    def update_weights(self, weight_tensor, gradient_tensor):
        raise NotImplementedError("This method should be implemented in subclasses.")


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None


    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)

        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        # Update the velocity
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor

        # Update the weights
        updated_weights = weight_tensor + self.velocity
        return updated_weights


class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.m = None
        self.v = None
        self.t = 0


    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)

        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        self.t += 1

        # Update moving averages of the gradient (m) and squared gradient (v)
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)

        # Bias-corrected estimates of m and v
        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)

        # Update the weights using the Adam formula
        updated_weights = weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_weights





class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate


    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights

