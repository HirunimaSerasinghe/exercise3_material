"""
 Task:
 Implement a class Pooling in the le Pooling.py in folder Layers . This class has to provide
 the methods forward(input tensor) and backward(error tensor).

- Write a constructor receiving the arguments stride shape and pooling shape, with
 the same ordering as speci ed in the convolutional layer.

- Implement a method forward(input tensor) which returns a tensor that serves as the
 input tensor for the next layer.
 Hint: Keep in mind to store the correct information necessary for the backward pass.

    Different to the convolutional layer, the pooling layer must be implemented only for
    the 2D case.

    Use valid-padding for the pooling layer. This means, unlike to the convolutional
    layer, dont apply any zero-padding. This may discard border elements of the
    input tensor. Take it into account when creating your output tensor.

 - Implement a method backward(error tensor) which returns a tensor that serves as
 the error tensor for the next layer.

 You can verify your implementation using the provided testsuite by providing the commandline
 parameter TestPooling
"""

import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_indices = None
        self.trainable = False  # Pooling layers are not trainable

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, input_height, input_width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1

        output_tensor = np.zeros((batch_size, channels, output_height, output_width))
        self.max_indices = np.zeros_like(output_tensor, dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * stride_height
                        start_j = j * stride_width
                        end_i = start_i + pool_height
                        end_j = start_j + pool_width

                        patch = input_tensor[b, c, start_i:end_i, start_j:end_j]
                        max_index = np.argmax(patch)
                        max_value = np.max(patch)

                        output_tensor[b, c, i, j] = max_value
                        self.max_indices[b, c, i, j] = max_index

        return output_tensor

    def backward(self, error_tensor):
        batch_size, channels, output_height, output_width = error_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        input_height = (output_height - 1) * stride_height + pool_height
        input_width = (output_width - 1) * stride_width + pool_width

        output_error = np.zeros_like(self.input_tensor)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * stride_height
                        start_j = j * stride_width
                        end_i = start_i + pool_height
                        end_j = start_j + pool_width

                        max_index = self.max_indices[b, c, i, j]
                        max_i, max_j = divmod(max_index, pool_width)

                        output_error[b, c, start_i + max_i, start_j + max_j] += error_tensor[b, c, i, j]

        return output_error