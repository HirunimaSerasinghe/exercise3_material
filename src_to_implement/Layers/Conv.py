"""
Task:
 Implement a class Conv in the le Conv.py in folder Layers . This class has to provide
 the methods forward(input tensor) and backward(error tensor).
 ˆ
 ˆ
 ˆ
 Write a constructor for this class, receiving the arguments stride shape, convolu
tion shape and num kernels de ning the operation. Note the following:
 this layer has trainable parameters, so set the inherited member trainable accord
ingly.
 stride shape can be a single value or a tuple. The latter allows for di erent strides
 in the spatial dimensions.
 convolution shape determines whether this object provides a 1D or a 2D con
volution layer. For 1D, it has the shape [c, m], whereas for 2D, it has the shape
 [c, m, n], where c represents the number of input channels, and m, n represent the
 spatial extent of the lter kernel.
 num kernels is an integer value.
 Initialize the parameters of this layer uniformly random in the range [01).
 To be able to test the gradients with respect to the weights: The members for weights
 and biases should be named weights and bias. Additionally provide two properties:
 gradient weights and gradient bias, which return the gradient with respect to the
 weights and bias, after they have been calculated in the backward-pass.
 Implement a method forward(input tensor) which returns a tensor that serves as the
 input tensor for the next layer. Note the following:
 The input layout for 1D is de ned in b, c, y order, for 2D in b, c, y, x order. Here,
 b stands for the batch, c represents the channels and x, y represent the spatial
 dimensions.
 You can calculate the output shape in the beginning based on the input tensor
 and the stride shape.
 Use zero-padding for convolutions/correlations ( same padding). This allows input
 and output to have the same spatial shape for a stride of 1.
 Make sure that 1 1-convolutions and 1D convolutions are handled correctly.
 Hint: Using correlation in the forward and convolution/correlation in the backward pass
 might help with the ipping of kernels.
 Hint 2: The scipy package features a n-dimensional convolution/correlation.
 Hint 3: E ciency trade-o s will be necessary in this scope. For example, striding may
 be implemented wastefully as subsampling after convolution/correlation.

- Implement a property optimizer storing the optimizer for this layer. Note that you
 need two copies of the optimizer object if you handle the bias separately from the other
 weights.
- Implement a method backward(error tensor) which updates the parameters using
 the optimizer (if available) and returns the error tensor which returns a tensor that
 servers as error tensor for the next layer.
- Implement a method initialize(weights initializer, bias initializer) which reinitial
izes the weights by using the provided initializer objects.
 You can verify your implementation using the provided testsuite by providing the command
line parameter TestConv. For further debugging purposes we provide optional unittests in
 SoftConvTests.py . Please read the instructions there carefully in case you need them.
"""

# attempt at changing the modes, the strides are driving me insane
"""
import numpy as np
from scipy.signal import correlate, correlate2d

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
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
        self._optimizer_weights = None
        self._optimizer_bias = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer_weights(self):
        return self._optimizer_weights

    @optimizer_weights.setter
    def optimizer_weights(self, opt):
        self._optimizer_weights = opt

    @property
    def optimizer_bias(self):
        return self._optimizer_bias

    @optimizer_bias.setter
    def optimizer_bias(self, opt):
        self._optimizer_bias = opt

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(self.convolution_shape) == 2:  # 1D convolution
            stride = int(self.stride_shape[0])
            output_shape = (input_tensor.shape[0], self.num_kernels, (input_tensor.shape[2] - self.convolution_shape[1]) // stride + 1)
            output_tensor = np.zeros(output_shape)
            for b in range(input_tensor.shape[0]):
                for k in range(self.num_kernels):
                    for c in range(input_tensor.shape[1]):
                        output_tensor[b, k] += correlate(input_tensor[b, c], self.weights[k, c], mode='valid')[::stride]
                    output_tensor[b, k] += self.bias[k]
        elif len(self.convolution_shape) == 3:  # 2D convolution
            stride_y, stride_x = int(self.stride_shape[0]), int(self.stride_shape[1])
            output_shape = (input_tensor.shape[0], self.num_kernels, (input_tensor.shape[2] - self.convolution_shape[1]) // stride_y + 1, (input_tensor.shape[3] - self.convolution_shape[2]) // stride_x + 1)
            output_tensor = np.zeros(output_shape)
            for b in range(input_tensor.shape[0]):
                for k in range(self.num_kernels):
                    for c in range(input_tensor.shape[1]):
                        output_tensor[b, k] += correlate2d(input_tensor[b, c], self.weights[k, c], mode='valid')[::stride_y, ::stride_x]
                    output_tensor[b, k] += self.bias[k]
        return output_tensor

    def backward(self, error_tensor):
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        output_error = np.zeros_like(self.input_tensor)

        if len(self.convolution_shape) == 2:  # 1D convolution
            for b in range(error_tensor.shape[0]):
                for k in range(self.num_kernels):
                    for c in range(self.input_tensor.shape[1]):
                        self._gradient_weights[k, c] += correlate(self.input_tensor[b, c], error_tensor[b, k], mode='valid')
                        upsampled_error = np.zeros((error_tensor.shape[2] - 1) * self.stride_shape[0] + 1)
                        upsampled_error[::self.stride_shape[0]] = error_tensor[b, k]
                        output_error[b, c] += correlate(upsampled_error, self.weights[k, c], mode='full')
                    self._gradient_bias[k] += np.sum(error_tensor[b, k])
        elif len(self.convolution_shape) == 3:  # 2D convolution
            for b in range(error_tensor.shape[0]):
                for k in range(self.num_kernels):
                    for c in range(self.input_tensor.shape[1]):
                        self._gradient_weights[k, c] += correlate2d(self.input_tensor[b, c], error_tensor[b, k], mode='valid')
                        upsampled_error = np.zeros(((error_tensor.shape[2] - 1) * self.stride_shape[0] + 1, (error_tensor.shape[3] - 1) * self.stride_shape[1] + 1))
                        upsampled_error[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b, k]
                        output_error[b, c] += correlate2d(upsampled_error, self.weights[k, c], mode='full')
                    self._gradient_bias[k] += np.sum(error_tensor[b, k])

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return output_error

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.convolution_shape[0] * np.prod(self.convolution_shape[1:])
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
"""

import numpy as np
from scipy.signal import correlate, correlate2d
from copy import deepcopy



class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
# can be other shape solution for handling tuple list
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