'''
Original publication: Tancik, M. et al. Fourier features let networks learn high frequency functions in low dimensional domains. Adv. Neural inf. Process. Syst. 33, 7537-7547 (2020).
Original code: https://github.com/titu1994/tf_fourier_features/blob/master/tf_fourier_features/fourier_features.py
Original license:

MIT License

Copyright (c) 2020 Somshubra Majumdar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Jihwan Kim
'''

import tensorflow as tf
import numpy as np

class positional_encoding(tf.keras.layers.Layer):

    def __init__(self, gaussian_projection: int, gaussian_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        if 'dtype' in kwargs:
            self._kernel_dtype = kwargs['dtype']
        else:
            self._kernel_dtype = None

        gaussian_projection = int(gaussian_projection)
        gaussian_scale = float(gaussian_scale)

        self.gauss_proj = gaussian_projection
        self.gauss_scale = gaussian_scale

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.gauss_proj <= 0:
            self.proj_kernel = tf.keras.layers.Dense(input_dim, use_bias=False, trainable=False,
                                                     kernel_initializer='identity', dtype=self._kernel_dtype)
        else:
            initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=self.gauss_scale)
            self.proj_kernel = tf.keras.layers.Dense(self.gauss_proj, use_bias=False, trainable=False,
                                                     kernel_initializer=initializer, dtype=self._kernel_dtype)

        self.z_proj_kernel = tf.keras.layers.Dense(self.gauss_proj, use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=self.gauss_scale))
        self.built = True

    def call(self, inputs, **kwargs):
        xy_inputs = inputs[:, :2]
        z_inputs = inputs[:, 2:3]

        x_proj = 2.0 * np.pi * xy_inputs
        x_proj = self.proj_kernel(x_proj)

        x_proj_sin = tf.sin(x_proj)
        x_proj_cos = tf.cos(x_proj)

        xy_output = tf.concat([x_proj_sin, x_proj_cos], axis=-1)

        z_output = self.z_positional_encoding(z_inputs)
        z_output = tf.reshape(z_output, [-1, self.gauss_proj * 2])

        output = tf.concat([xy_output, z_output], axis=-1)
        return output

    def z_positional_encoding(self, z):
        z_proj = 2.0 * np.pi * z
        z_proj = self.z_proj_kernel(z_proj)

        z_proj_sin = tf.sin(z_proj)
        z_proj_cos = tf.cos(z_proj)

        pos_encoding = tf.concat([z_proj_sin, z_proj_cos], axis=-1)
        return pos_encoding

    def get_config(self):
        config = {
            'gaussian_projection': self.gauss_proj,
            'gaussian_scale': self.gauss_scale
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))