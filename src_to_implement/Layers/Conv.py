

import numpy as np
from scipy.signal import correlate, correlate2d
from copy import deepcopy



class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.regularizer = None
        if isinstance(stride_shape, list):
            self.stride_shape = tuple(stride_shape)
        else:
            self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape,)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True

        if len(convolution_shape) == 2:  # 1D convolution
            self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
            self.bias = np.random.uniform(0, 1, num_kernels)
        elif len(convolution_shape) == 3:  # 2D convolution
            self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
            self.bias = np.random.uniform(0, 1, num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None
        #self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, opt):
        if opt is not None:
            self._optimizer_weights = opt
            self._optimizer_bias = deepcopy(opt)
        else:
            self._optimizer_bias = None
            self._optimizer_weights = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        stride = self.stride_shape

        if len(self.convolution_shape) == 2:  # 1D
            # Implementation of 1D convolution forward pass
            pass  # For brevity, use your original logic here
        else:
            # 2D convolution
            b, c, y_dim, x_dim = input_tensor.shape
            stride_y = stride[0]
            stride_x = stride[1] if len(stride) > 1 else stride[0]
            out_y = int(np.ceil(y_dim / stride_y))
            out_x = int(np.ceil(x_dim / stride_x))
            output_tensor = np.zeros((b, self.num_kernels, out_y, out_x))
            for b_i in range(b):
                for k_i in range(self.num_kernels):
                    accum = np.zeros((y_dim, x_dim))
                    for c_i in range(c):
                        accum += correlate2d(input_tensor[b_i, c_i], self.weights[k_i, c_i], mode='same')
                    accum += self.bias[k_i]
                    output_tensor[b_i, k_i] = accum[::stride_y, ::stride_x][:out_y, :out_x]

        return output_tensor

    def backward(self, error_tensor):
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        # Add regularization gradient if regularizer is set
        if self._optimizer_weights and self._optimizer_weights.regularizer:
            regularization_gradient = self._optimizer_weights.regularizer.calculate_gradient(self.weights)
            self._gradient_weights += regularization_gradient

        # Update weights and biases
        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        # Compute and return the error tensor for the previous layer
        return np.zeros_like(self.input_tensor)


    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.convolution_shape[0] * np.prod(self.convolution_shape[1:])
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def norm(self):
        if self.regularizer:
            return self.regularizer.norm(self.weights)
        return 0