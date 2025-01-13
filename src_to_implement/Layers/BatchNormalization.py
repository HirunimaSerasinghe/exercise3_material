"""
import numpy as np

class BatchNormalization:
    def __init__(self, channels):
        self.channels = channels
        self.trainable = True  # This layer has trainable parameters
        self.weights = None
        self.bias = None
        self.moving_mean = None
        self.moving_variance = None
        self.epsilon = 1e-10
        self.optimizer = None  # Placeholder for optimizer
        self.gradient_weights = None
        self.gradient_bias = None
        self.initialize(channels)



    def initialize(self, channels):

        self.weights = np.ones((channels,))
        self.bias = np.zeros((channels,))

    def forward(self, input_tensor, is_training=True):

        if is_training:
            # Compute batch mean and variance
            self.batch_mean = np.mean(input_tensor, axis=0)
            self.batch_variance = np.var(input_tensor, axis=0)

            # Normalize input
            self.normalized_input = (input_tensor - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)

            # Scale and shift
            output_tensor = self.weights * self.normalized_input + self.bias

            # Update moving mean and variance for testing phase
            if self.moving_mean is None:
                self.moving_mean = self.batch_mean
                self.moving_variance = self.batch_variance
            else:
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.batch_mean
                self.moving_variance = 0.9 * self.moving_variance + 0.1 * self.batch_variance
        else:
            # Use moving mean and variance for inference
            normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            output_tensor = self.weights * normalized_input + self.bias

        return output_tensor

    def backward(self, error_tensor):

        batch_size = error_tensor.shape[0]

        # Gradients w.r.t weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Gradient w.r.t inputs
        d_normalized_input = error_tensor * self.weights
        d_variance = np.sum(d_normalized_input * (self.normalized_input * -0.5) / (self.batch_variance + self.epsilon), axis=0)
        d_mean = np.sum(d_normalized_input * -1 / np.sqrt(self.batch_variance + self.epsilon), axis=0) + \
                 d_variance * np.sum(-2 * (self.normalized_input * np.sqrt(self.batch_variance + self.epsilon)), axis=0) / batch_size
        input_gradient = d_normalized_input / np.sqrt(self.batch_variance + self.epsilon) + \
                         d_variance * 2 * self.normalized_input / batch_size + d_mean / batch_size

        # Update weights and bias if optimizers are defined
        if self.optimizer:
            self.weights = self.optimizer.update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.update(self.bias, self.gradient_bias)

        return input_gradient

    def reformat(self, tensor):

        if len(tensor.shape) == 4:
            # Image-like to vector-like
            batch_size, channels, height, width = tensor.shape
            return tensor.reshape(batch_size, channels * height * width)
        elif len(tensor.shape) == 2:
            # Vector-like to image-like
            batch_size, vector_size = tensor.shape
            channels = self.channels
            height_width = int(np.sqrt(vector_size / channels))
            return tensor.reshape(batch_size, channels, height_width, height_width)
        else:
            raise ValueError("Invalid tensor shape for reformatting.")
"""

import numpy as np
"""
class BatchNormalization:
    def __init__(self, channels):
        self.channels = channels
        self.trainable = True  # This layer has trainable parameters
        self.weights = None
        self.bias = None
        self.moving_mean = None
        self.moving_variance = None
        self.epsilon = 1e-10
        self.optimizer = None  # Placeholder for optimizer
        self.gradient_weights = None
        self.gradient_bias = None
        self.testing_phase = False  # Add the missing attribute
        self.initialize(channels)

    def initialize(self, channels):
        
        self.weights = np.ones((channels,))
        self.bias = np.zeros((channels,))

    def forward(self, input_tensor):
        
        if not self.testing_phase:
            # Training phase
            self.batch_mean = np.mean(input_tensor, axis=0)
            self.batch_variance = np.var(input_tensor, axis=0)

            # Normalize input
            self.normalized_input = (input_tensor - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)

            # Scale and shift
            output_tensor = self.weights * self.normalized_input + self.bias

            # Update moving mean and variance for testing phase
            if self.moving_mean is None:
                self.moving_mean = self.batch_mean
                self.moving_variance = self.batch_variance
            else:
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.batch_mean
                self.moving_variance = 0.9 * self.moving_variance + 0.1 * self.batch_variance
        else:
            # Testing phase (inference)
            normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            output_tensor = self.weights * normalized_input + self.bias

        return output_tensor

    def backward(self, error_tensor):
        
        batch_size = error_tensor.shape[0]

        # Gradients w.r.t weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Gradient w.r.t inputs
        d_normalized_input = error_tensor * self.weights
        d_variance = np.sum(d_normalized_input * (self.normalized_input * -0.5) / (self.batch_variance + self.epsilon), axis=0)
        d_mean = np.sum(d_normalized_input * -1 / np.sqrt(self.batch_variance + self.epsilon), axis=0) + \
                 d_variance * np.sum(-2 * (self.normalized_input * np.sqrt(self.batch_variance + self.epsilon)), axis=0) / batch_size
        input_gradient = d_normalized_input / np.sqrt(self.batch_variance + self.epsilon) + \
                         d_variance * 2 * self.normalized_input / batch_size + d_mean / batch_size

        # Update weights and bias if optimizers are defined
        if self.optimizer:
            self.weights = self.optimizer.update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.update(self.bias, self.gradient_bias)

        return input_gradient

    def reformat(self, tensor):
        
        if len(tensor.shape) == 4:
            # Image-like to vector-like
            batch_size, channels, height, width = tensor.shape
            return tensor.reshape(batch_size, channels * height * width)
        elif len(tensor.shape) == 2:
            # Vector-like to image-like
            batch_size, vector_size = tensor.shape
            channels = self.channels
            height_width = int(np.sqrt(vector_size / channels))
            return tensor.reshape(batch_size, channels, height_width, height_width)
        else:
            raise ValueError("Invalid tensor shape for reformatting.")

"""
"""
import numpy as np

class BatchNormalization:
    def __init__(self, channels):
        self.channels = channels
        self.trainable = True
        self.weights = None
        self.bias = None
        self.moving_mean = None
        self.moving_variance = None
        self.epsilon = 1e-10
        self.optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.initialize(channels)

    def initialize(self, channels):
        
        self.weights = np.ones((channels,))
        self.bias = np.zeros((channels,))

    def forward(self, input_tensor, is_training=True):
        
        if len(input_tensor.shape) == 4:  # Convolutional input
            batch_size, channels, height, width = input_tensor.shape
            axes = (0, 2, 3)
        else:  # Fully connected input
            batch_size, channels = input_tensor.shape
            axes = 0

        if is_training:
            # Compute batch mean and variance
            self.batch_mean = np.mean(input_tensor, axis=axes, keepdims=True)
            self.batch_variance = np.var(input_tensor, axis=axes, keepdims=True)

            # Normalize input
            self.normalized_input = (input_tensor - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)

            # Scale and shift
            output_tensor = self.weights[:, None, None] * self.normalized_input + self.bias[:, None, None] if len(input_tensor.shape) == 4 else self.weights * self.normalized_input + self.bias

            # Update moving mean and variance
            if self.moving_mean is None:
                self.moving_mean = self.batch_mean
                self.moving_variance = self.batch_variance
            else:
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.batch_mean
                self.moving_variance = 0.9 * self.moving_variance + 0.1 * self.batch_variance
        else:
            # Use moving mean and variance for inference
            normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            output_tensor = self.weights[:, None, None] * normalized_input + self.bias[:, None, None] if len(input_tensor.shape) == 4 else self.weights * normalized_input + self.bias

        return output_tensor

    def backward(self, error_tensor):

        batch_size = error_tensor.shape[0]

        # Gradients w.r.t weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Gradient w.r.t inputs
        d_normalized_input = error_tensor * self.weights
        d_variance = np.sum(d_normalized_input * (self.normalized_input * -0.5) / (self.batch_variance + self.epsilon),
                            axis=0)
        d_mean = np.sum(d_normalized_input * -1 / np.sqrt(self.batch_variance + self.epsilon), axis=0) + \
                 d_variance * np.sum(-2 * (self.normalized_input * np.sqrt(self.batch_variance + self.epsilon)),
                                     axis=0) / batch_size
        input_gradient = d_normalized_input / np.sqrt(self.batch_variance + self.epsilon) + \
                         d_variance * 2 * self.normalized_input / batch_size + d_mean / batch_size

        # Update weights and bias if optimizers are defined
        if self.optimizer:
            self.weights = self.optimizer.update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.update(self.bias, self.gradient_bias)

        return input_gradient

    def reformat(self, tensor):

        if len(tensor.shape) == 4:
            # Image-like to vector-like
            batch_size, channels, height, width = tensor.shape
            return tensor.reshape(batch_size, channels * height * width)
        elif len(tensor.shape) == 2:
            # Vector-like to image-like
            batch_size, vector_size = tensor.shape
            channels = self.channels
            height_width = int(np.sqrt(vector_size / channels))
            return tensor.reshape(batch_size, channels, height_width, height_width)
        else:
            raise ValueError("Invalid tensor shape for reformatting.")
"""

#Local(3)
"""
import numpy as np

class BatchNormalization:
    def __init__(self, channels):
        self.channels = channels
        self.trainable = True
        self.weights = None
        self.bias = None
        self.moving_mean = None
        self.moving_variance = None
        self.epsilon = 1e-10
        self.optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.testing_phase = False  # Fix: Add the missing attribute
        self.initialize(channels)

    def initialize(self, channels):
        self.weights = np.ones((channels,))
        self.bias = np.zeros((channels,))

    def forward(self, input_tensor, is_training=True):
        if len(input_tensor.shape) == 4:  # Convolutional input
            batch_size, channels, height, width = input_tensor.shape
            axes = (0, 2, 3)
        else:  # Fully connected input
            batch_size, channels = input_tensor.shape
            axes = 0

        if is_training:
            # Compute batch mean and variance
            self.batch_mean = np.mean(input_tensor, axis=axes, keepdims=True)
            self.batch_variance = np.var(input_tensor, axis=axes, keepdims=True)

            # Normalize input
            self.normalized_input = (input_tensor - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)

            # Scale and shift
            output_tensor = (
                self.weights[:, None, None] * self.normalized_input + self.bias[:, None, None]
                if len(input_tensor.shape) == 4
                else self.weights * self.normalized_input + self.bias
            )

            # Update moving mean and variance
            if self.moving_mean is None:
                self.moving_mean = self.batch_mean
                self.moving_variance = self.batch_variance
            else:
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.batch_mean
                self.moving_variance = 0.9 * self.moving_variance + 0.1 * self.batch_variance
        else:
            # Use moving mean and variance for inference
            normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            output_tensor = (
                self.weights[:, None, None] * normalized_input + self.bias[:, None, None]
                if len(input_tensor.shape) == 4
                else self.weights * normalized_input + self.bias
            )

        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]

        # Gradients w.r.t weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Gradient w.r.t inputs
        d_normalized_input = error_tensor * self.weights
        d_variance = np.sum(
            d_normalized_input * (self.normalized_input * -0.5) / (self.batch_variance + self.epsilon), axis=0
        )
        d_mean = np.sum(
            d_normalized_input * -1 / np.sqrt(self.batch_variance + self.epsilon), axis=0
        ) + d_variance * np.sum(-2 * (self.normalized_input * np.sqrt(self.batch_variance + self.epsilon)), axis=0) / batch_size
        input_gradient = (
            d_normalized_input / np.sqrt(self.batch_variance + self.epsilon)
            + d_variance * 2 * self.normalized_input / batch_size
            + d_mean / batch_size
        )

        # Update weights and bias if optimizers are defined
        if self.optimizer:
            self.weights = self.optimizer.update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.update(self.bias, self.gradient_bias)

        return input_gradient

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            # Image-like to vector-like
            batch_size, channels, height, width = tensor.shape
            return tensor.reshape(batch_size, channels * height * width)
        elif len(tensor.shape) == 2:
            # Vector-like to image-like
            batch_size, vector_size = tensor.shape
            channels = self.channels
            height_width = int(np.sqrt(vector_size / channels))
            return tensor.reshape(batch_size, channels, height_width, height_width)
        else:
            raise ValueError("Invalid tensor shape for reformatting.")
"""

#local(4)
class BatchNormalization:
    def __init__(self, channels):
        self.channels = channels
        self.trainable = True
        self.weights = None
        self.bias = None
        self.moving_mean = None
        self.moving_variance = None
        self.epsilon = 1e-10
        self.optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.testing_phase = False  # Fix: Add the missing attribute
        self.initialize(channels)

    def initialize(self, channels):
        self.weights = np.ones((channels,))
        self.bias = np.zeros((channels,))

    def forward(self, input_tensor, is_training=True):
        if len(input_tensor.shape) == 4:  # Convolutional input
            batch_size, channels, height, width = input_tensor.shape
            axes = (0, 2, 3)
        else:  # Fully connected input
            batch_size, channels = input_tensor.shape
            axes = 0

        if is_training:
            # Compute batch mean and variance
            self.batch_mean = np.mean(input_tensor, axis=axes, keepdims=True)
            self.batch_variance = np.var(input_tensor, axis=axes, keepdims=True)

            # Normalize input
            self.normalized_input = (input_tensor - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)

            # Scale and shift
            output_tensor = (
                self.weights[:, None, None] * self.normalized_input + self.bias[:, None, None]
                if len(input_tensor.shape) == 4
                else self.weights * self.normalized_input + self.bias
            )

            # Update moving mean and variance
            if self.moving_mean is None:
                self.moving_mean = self.batch_mean
                self.moving_variance = self.batch_variance
            else:
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.batch_mean
                self.moving_variance = 0.9 * self.moving_variance + 0.1 * self.batch_variance
        else:
            # Use moving mean and variance for inference
            normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            output_tensor = (
                self.weights[:, None, None] * normalized_input + self.bias[:, None, None]
                if len(input_tensor.shape) == 4
                else self.weights * normalized_input + self.bias
            )

        return output_tensor


    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:  # Convolutional input
            axes = (0, 2, 3)
        else:  # Fully connected input
            axes = 0

        batch_size = error_tensor.shape[0]

        # Gradients w.r.t weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=axes)
        self.gradient_bias = np.sum(error_tensor, axis=axes)

        # Gradient w.r.t inputs
        d_normalized_input = (
            error_tensor * self.weights[:, None, None] if len(error_tensor.shape) == 4 else error_tensor * self.weights
        )
        d_variance = np.sum(
            d_normalized_input * (self.normalized_input * -0.5) / (self.batch_variance + self.epsilon), axis=axes, keepdims=True
        )
        d_mean = np.sum(
            d_normalized_input * -1 / np.sqrt(self.batch_variance + self.epsilon), axis=axes, keepdims=True
        ) + d_variance * np.sum(-2 * (self.normalized_input), axis=axes, keepdims=True) / batch_size
        input_gradient = (
            d_normalized_input / np.sqrt(self.batch_variance + self.epsilon)
            + d_variance * 2 * self.normalized_input / batch_size
            + d_mean / batch_size
        )

        # Update weights and bias if optimizers are defined
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        return input_gradient

    """def reformat(self, tensor):
        if len(tensor.shape) == 4:
            # Image-like to vector-like
            batch_size, channels, height, width = tensor.shape
            return tensor.reshape(batch_size, channels * height * width)
        elif len(tensor.shape) == 2:
            # Vector-like to image-like
            batch_size, vector_size = tensor.shape
            channels = self.channels
            height_width = int(np.sqrt(vector_size // channels))  # Fix: Correct calculation of height/width
            return tensor.reshape(batch_size, channels, height_width, height_width)
        else:
            raise ValueError("Invalid tensor shape for reformatting.")

"""

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            # Image-like to vector-like
            batch_size, channels, height, width = tensor.shape
            return tensor.reshape(batch_size, channels * height * width)
        elif len(tensor.shape) == 2:
            # Vector-like to image-like
            batch_size, vector_size = tensor.shape
            channels = self.channels
            # Validate if vector_size is divisible by channels
            if vector_size % channels != 0:
                raise ValueError(f"Vector size {vector_size} is not divisible by the number of channels {channels}.")
            # Calculate spatial size
            spatial_size = vector_size // channels
            # Handle non-perfect square spatial sizes
            height = int(np.sqrt(spatial_size))
            width = spatial_size // height
            if height * width != spatial_size:
                raise ValueError(
                    f"Spatial size {spatial_size} cannot be reshaped into a valid height and width."
                )
            return tensor.reshape(batch_size, channels, height, width)
        else:
            raise ValueError("Invalid tensor shape for reformatting.")