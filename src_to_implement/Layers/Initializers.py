
from scipy import stats
import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self,weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)



class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0,1,weights_shape)


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):

        if fan_in <= 0 or fan_out <= 0:
            raise ValueError("fan_in and fan_out must be positive integers.")

        # Calculate standard deviation for Xavier initialization
        stddev = np.sqrt(2 / (fan_in + fan_out))
            # Generate weights using a normal distribution
        return np.random.normal(0, stddev, weights_shape)


"""class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, weights_shape)
"""
class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        if fan_in is None:
            raise ValueError("fan_in cannot be None. Ensure it is passed correctly.")
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, weights_shape)


