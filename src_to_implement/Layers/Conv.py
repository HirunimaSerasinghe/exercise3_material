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

import numpy as np
from scipy.signal import correlate, correlate2d
from copy import deepcopy



class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
# can be other shape solution for handling  integer, tuple, or list.
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
        self._optimizer_weights = None
        self._optimizer_bias = None
        # tailored updates for bias and weights :)

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
            b, c, length = input_tensor.shape
            stride_l = stride[0]
            out_length = int(np.ceil(length / stride_l))
            output_tensor = np.zeros((b, self.num_kernels, out_length))
            for b_i in range(b):
                for k_i in range(self.num_kernels):
                    accum = np.zeros(length)
                    for c_i in range(c):
                        # each batch and kernel, computing correlation bw the input channels & kernel weights
                        accum += correlate(input_tensor[b_i, c_i], self.weights[k_i, c_i], mode='same')
                    accum += self.bias[k_i]
                    # Downsampling the result based on the stride
                    output_tensor[b_i, k_i] = accum[::stride_l][:out_length]
        else:
            # 2D
            b, c, y_dim, x_dim = input_tensor.shape
            stride_y = stride[0] # height
            stride_x = stride[1] if len(stride) > 1 else stride[0] # width
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

        if len(self.convolution_shape) == 2:  # 1D
            b, c, length = self.input_tensor.shape
            _, k, out_length = error_tensor.shape
            stride_l = self.stride_shape[0]

            err_upsampled = np.zeros((b, k, length))
            err_upsampled[:, :, ::stride_l] = error_tensor

            #error tensor for the previous layer 
            flipped_weights = np.flip(self.weights, axis=2)

            for kk in range(self.num_kernels):
                self._gradient_bias[kk] = np.sum(err_upsampled[:, kk])

            kernel_size = self.weights.shape[2]
            for b_i in range(b):
                for k_i in range(self.num_kernels):
                    for c_i in range(c):
                        full_corr_c = np.correlate(self.input_tensor[b_i, c_i], err_upsampled[b_i, k_i], mode='full')
                        center = (full_corr_c.size // 2)
                        start = center - (kernel_size // 2)
                        end = start + kernel_size
                        self._gradient_weights[k_i, c_i] += full_corr_c[start:end]

            output_error = np.zeros_like(self.input_tensor)
            for b_i in range(b):
                for k_i in range(self.num_kernels):
                    for c_i in range(c):
                        output_error[b_i, c_i] += correlate(err_upsampled[b_i, k_i], flipped_weights[k_i, c_i], mode='same')

        else:
            # 2D
            b, c, y_dim, x_dim = self.input_tensor.shape
            _, k, out_y, out_x = error_tensor.shape
            stride_y = self.stride_shape[0]
            stride_x = self.stride_shape[1] if len(self.stride_shape) > 1 else self.stride_shape[0]

            err_upsampled = np.zeros((b, k, y_dim, x_dim))
            err_upsampled[:, :, ::stride_y, ::stride_x] = error_tensor

            kernel_y, kernel_x = self.weights.shape[2], self.weights.shape[3]
            flipped_weights = np.flip(np.flip(self.weights, axis=2), axis=3)

            for kk in range(self.num_kernels):
                self._gradient_bias[kk] = np.sum(err_upsampled[:, kk])

            for b_i in range(b):
                for k_i in range(self.num_kernels):
                    for c_i in range(c):
                        full_corr = correlate2d(self.input_tensor[b_i, c_i], err_upsampled[b_i, k_i], mode='full')
                        center_y = full_corr.shape[0] // 2
                        center_x = full_corr.shape[1] // 2
                        start_y = center_y - (kernel_y // 2)
                        start_x = center_x - (kernel_x // 2)
                        end_y = start_y + kernel_y
                        end_x = start_x + kernel_x
                        self._gradient_weights[k_i, c_i] += full_corr[start_y:end_y, start_x:end_x]

            output_error = np.zeros_like(self.input_tensor)
            for b_i in range(b):
                for k_i in range(self.num_kernels):
                    for c_i in range(c):
                        output_error[b_i, c_i] += correlate2d(err_upsampled[b_i, k_i], flipped_weights[k_i, c_i], mode='same')

        # Update parameters with separate optimizers
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


