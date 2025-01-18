import numpy as np
from Layers import Initializers

class BatchNormalization:
    def __init__(self, channels, weights_initializer=None, bias_initializer=None):
        self.channels = channels
        self.trainable = True
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.moving_mean = None
        self.moving_variance = None
        self.epsilon = 1e-10
        self.optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.testing_phase = False
        self.weights = None
        self.bias = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        if weights_initializer is None:
            weights_initializer = Initializers.Constant(1.0) 
        if bias_initializer is None:
            bias_initializer = Initializers.Constant(0.0)  
        self.weights = weights_initializer.initialize((self.channels,), fan_in=self.channels, fan_out=None)
        self.bias = bias_initializer.initialize((self.channels,), fan_in=self.channels, fan_out=None)


    def forward(self, input_tensor, is_training=True, weights_initializer=None, bias_initializer=None):
        if self.weights is None or self.bias is None:
            self.initialize(weights_initializer, bias_initializer)

        if len(input_tensor.shape) == 4:  
            batch_size, channels, height, width = input_tensor.shape
            axes = (0, 2, 3)
        else:  
            batch_size, channels = input_tensor.shape
            axes = 0

        if is_training and not self.testing_phase:
            self.batch_mean = np.mean(input_tensor, axis=axes, keepdims=True)
            self.batch_variance = np.var(input_tensor, axis=axes, keepdims=True)


            self.normalized_input = (input_tensor - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)

            output_tensor = (
                self.weights[:, None, None] * self.normalized_input + self.bias[:, None, None]
                if len(input_tensor.shape) == 4
                else self.weights * self.normalized_input + self.bias
            )

            if self.moving_mean is None:
                self.moving_mean = self.batch_mean
                self.moving_variance = self.batch_variance
            else:
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.batch_mean
                self.moving_variance = 0.9 * self.moving_variance + 0.1 * self.batch_variance
        else:
            normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            output_tensor = (
                self.weights[:, None, None] * normalized_input + self.bias[:, None, None]
                if len(input_tensor.shape) == 4
                else self.weights * normalized_input + self.bias
            )

        return output_tensor


    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:  
            axes = (0, 2, 3) 
        else:  
            axes = 0  

        batch_size = error_tensor.shape[0]

        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=axes, keepdims=False)
        self.gradient_bias = np.sum(error_tensor, axis=axes, keepdims=False)

        d_normalized_input = error_tensor * (
            self.weights[:, None, None] if len(error_tensor.shape) == 4 else self.weights
        )

        d_variance = np.sum(
            d_normalized_input * (self.normalized_input * -0.5) / (self.batch_variance + self.epsilon) ** 1.5,
            axis=axes,
            keepdims=True,
        )

        d_mean = np.sum(
            d_normalized_input * -1 / np.sqrt(self.batch_variance + self.epsilon),
            axis=axes,
            keepdims=True,
        ) + d_variance * np.sum(-2 * (self.normalized_input), axis=axes, keepdims=True) / batch_size

        input_gradient = (
                d_normalized_input / np.sqrt(self.batch_variance + self.epsilon)
                + d_variance * 2 * (self.normalized_input) / batch_size
                + d_mean / batch_size
        )

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        return input_gradient

    def reformat(self, tensor):
        if len(tensor.shape) == 4:  # Image-like to vector-like
            batch_size, channels, height, width = tensor.shape
            if channels != self.channels:
                raise ValueError(f"Expected {self.channels} channels, but got {channels}.")
            return tensor.transpose(0, 2, 3, 1).reshape(-1, channels)

        elif len(tensor.shape) == 2:  # Vector-like to image-like
            num_elements, channels = tensor.shape
            if channels != self.channels:
                raise ValueError(f"Expected {self.channels} channels, but got {channels}.")
            
            total_image_elements = num_elements * channels
            batch_size = total_image_elements // (self.channels * 6 * 4)
            if total_image_elements % (self.channels * 6 * 4) != 0:
                raise ValueError("Input tensor shape is incompatible with the expected dimensions.")

            return tensor.reshape(batch_size, 6, 4, self.channels).transpose(0, 3, 1, 2)

        else:
            raise ValueError("Tensor must be either 4D (image format) or 2D (vector format).")
