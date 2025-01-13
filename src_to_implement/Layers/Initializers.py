"""
 Task:
 Implement four classes Constant, UniformRandom, Xavier and He in the le Initializ
ers.py in folder Layers . Each of them has to provide the method initialize(weights shape,
 fan in, fan out) which returns an initialized tensor of the desired shape.
 
- Implement all four initialization schemes. Note the following:
 The Constant class has a member that determines the constant value used for
 weight initialization. The value can be passed as a constructor argument, with a
 default of 0.1.
 
-  The support of the uniform distribution is the interval [01).
 Have a look at the exercise slides for more information on Xavier and He initializers.
 Addamethod initialize(weights initializer, bias initializer) to the class FullyCon
nected reinitializing its weights. Initialize the bias separately with the bias initializer.
 
-  Remember that the bias is usually also stored in the weights matrix.
 
 Refactor the class NeuralNetwork to receive a weights initializer and a
 bias initializer upon construction.
 Extend the method append layer(layer) in the class NeuralNetwork such that it
 initializes trainable layers with the stored initializers.
 
 You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestInitializers.

"""
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


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, weights_shape)

