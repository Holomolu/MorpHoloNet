import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from positional_encoding import positional_encoding
import matplotlib.pyplot as plt
import numpy as np
from time import time
from PIL import Image
import math
import os
from scipy.stats import multivariate_normal

# Load holographic image
file_name = 'Z:\\holograms\\image_file_name.tif'
image_array = Image.open(file_name)
width, height = image_array.size
image_array = np.array(image_array) / 255.0
image_array = image_array.reshape(width, height)
image_tensor = tf.convert_to_tensor(image_array)
U_z0 = tf.cast(image_tensor, tf.float32)
U_incident_avg_real = tf.sqrt(tf.reduce_mean(U_z0))

if os.path.exists('.\save_weights')==False:
    os.makedirs('.\save_weights')

minPX = 1
maxPX = width
minPY = 1
maxPY = height
segment_size = width

# Prior knowledge
x0, y0, z0 = 64, 64, 100  # Approximate location of target object, (x, y, z) indices on 3D Cartesian coordinate system
phase_shift = 0.5 # Initial assumption of phase shift per voxel = 2𝝿(n_obj−n_med)dz/λ
r = 3 # Depending on the object size

dz = 1.0
z_min = 0.0 + dz
z_max = 150 # Set longer than z0
z_num = (z_max/dz) + 1

x = tf.range(1, width + 1, dtype=tf.float32) / width
y = tf.range(1, height + 1, dtype=tf.float32) / height
xx, yy = tf.meshgrid(x, y)

def create_3d_gaussian(x0, y0, z0, r, width, height, z_min, z_max, dz):

    x = tf.range(1, width + 1, dtype=tf.float32)
    y = tf.range(1, height + 1, dtype=tf.float32)
    z = tf.range(z_min, z_max + dz, dz, dtype=tf.float32)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    pos = np.empty(xx.shape + (3,))
    pos[:, :, :, 0] = xx
    pos[:, :, :, 1] = yy
    pos[:, :, :, 2] = zz

    mean = np.array([x0, y0, z0])

    # Depending on the object shape
    cov = np.diag([r ** 2, r ** 2, (0.5 * r) ** 2]) # Sphere or ellipsoid
    #cov = np.diag([r ** 2, r ** 2, (0.25 * r) ** 2])  # Oblate spheroid

    rv = multivariate_normal(mean, cov)
    gaussian = rv.pdf(pos)
    gaussian_max = np.max(gaussian)
    gaussian_norm = gaussian/gaussian_max

    return gaussian_norm

gaussian_3d = create_3d_gaussian(x0, y0, z0, r, width, height, z_min, z_max, dz)

def slice_3d_gaussian(gaussian, z_value, z_max):

    z_index = int((z_value / z_max) * gaussian.shape[2])
    if z_index < 0 or z_index > gaussian.shape[2]:
        raise ValueError("z_value is out of bounds.")
    return gaussian[:, :, z_index - 1]

def angular_spectrum_propagator(image, depth):
    H = tf.signal.fft2d(tf.cast(image, tf.complex64))
    M2 = segment_size
    N2 = segment_size
    physicalLength = 0.5 # Magnified pixel length
    waveLength = 0.532 / 1.333
    k = 2 * math.pi / waveLength

    u = tf.linspace(0, M2 - 1, M2)
    v = tf.linspace(0, N2 - 1, N2)
    u = tf.where(u > M2 / 2, u - M2, u)
    v = tf.where(v > N2 / 2, v - N2, v)

    V, U = tf.meshgrid(v, u)
    U = tf.cast(waveLength * U / (M2 * physicalLength), tf.float32)
    V = tf.cast(waveLength * V / (N2 * physicalLength), tf.float32)
    U2 = tf.cast(U**2, tf.complex64)
    V2 = tf.cast(V**2, tf.complex64)
    F = 1j * k * tf.sqrt(1 - U2 - V2)

    if depth == 0:
        simulated_image = tf.cast(image, dtype=tf.complex64)
    else:
        depth = tf.cast(depth, tf.complex64)
        G = tf.exp(depth * F)
        recons = tf.signal.ifft2d(H * G)
        simulated_image = tf.cast(recons[0:M2, 0:N2], dtype=tf.complex64)

    return simulated_image

def create_z_dataset(z_min, z_max, dz, batch_size):
    z_values = tf.range(z_min, z_max + dz, dz, dtype=tf.float32)
    tf.random.shuffle(z_values)
    num_batches = int(np.ceil(len(z_values) / batch_size))
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_values = z_values[start_index:end_index]
        batches.append(batch_values)
    return np.array(batches)

class MorpHoloNet(Model):
    def __init__(self):
        super(MorpHoloNet, self).__init__()

        initializer_hidden = tf.keras.initializers.LecunNormal
        initializer_output = tf.keras.initializers.GlorotUniform
        activator = tf.keras.activations.swish

        self.h1 = positional_encoding(gaussian_projection=64, gaussian_scale=10.0) # Gaussian_scale: 5~12
        self.h2 = Dense(128, activation=activator, kernel_initializer=initializer_hidden)
        self.h3 = Dense(128, activation=activator, kernel_initializer=initializer_hidden)
        self.h4 = Dense(128, activation=activator, kernel_initializer=initializer_hidden)
        self.u = Dense(1, activation='sigmoid', kernel_initializer=initializer_output)

    def call(self, pos):
        x = self.h1(pos)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        out = self.u(x)

        return out

class twin_image_removal(object):
    def __init__(self):
            self.lr = 0.0001 # learning rate: 0.00001~0.0001
            self.opt = Adam(self.lr)

            self.lr_pre = 0.001
            self.opt_pre = Adam(self.lr_pre)

            self.phase_shift = tf.Variable(phase_shift, dtype=tf.float32, name='phase_shift')
            self.incident_light = tf.Variable(U_incident_avg_real, dtype=tf.float32, name='incident_light')

            self.NN = MorpHoloNet()
            self.NN.build(input_shape=(None, 3))

            self.train_loss_history = []
            self.iter_count = 0
            self.instant_loss = 0

    @tf.function
    def compute_loss_and_grads(self):
        with tf.GradientTape() as tape:
            tape.watch([self.phase_shift])
            loss = tf.constant(0.0, dtype=tf.float32)
            U_z_following_prop = tf.complex(tf.zeros_like(U_z0), tf.zeros_like(U_z0))

            z_values = tf.range(z_min, z_max + dz, dz, dtype=tf.float32)
            z_values = z_values[::-1]

            for z in z_values:
                if z == z_max:
                    z_following = tf.fill([width, height], z / z_max)
                    z_preceding = tf.fill([width, height], (z - dz) / z_max)
                    tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
                    tensor_array_preceding = tf.stack([xx, yy, z_preceding], axis=-1)
                    tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
                    tensor_array_preceding = tf.reshape(tensor_array_preceding, (-1, 3))

                    object_following = self.NN(tensor_array_following)
                    object_preceding = self.NN(tensor_array_preceding)
                    object_following = tf.reshape(object_following, [width, height])
                    object_preceding = tf.reshape(object_preceding, [width, height])

                    real_following_ref = tf.multiply(tf.ones([width, height], dtype=tf.float32), self.incident_light)
                    imag_following_ref = tf.zeros([width, height], dtype=tf.float32)
                    U_z_following_ref = tf.complex(real_following_ref, imag_following_ref)
                    phase_shift_complex = tf.complex(self.phase_shift, 0.0)
                    object_preceding_complex = tf.complex(object_preceding, tf.zeros_like(object_preceding))
                    phase_delay = tf.exp(tf.complex(0.0, 1.0) * phase_shift_complex * object_preceding_complex)
                    U_z_following_ref *= phase_delay

                    U_z_following_prop = angular_spectrum_propagator(U_z_following_ref, dz)

                    loss += 0.5*tf.reduce_mean(tf.square(object_following))

                elif z == z_min:
                    z_preceding = tf.fill([width, height], (z - dz) / z_max)
                    tensor_array_preceding = tf.stack([xx, yy, z_preceding], axis=-1)
                    tensor_array_preceding = tf.reshape(tensor_array_preceding, (-1, 3))

                    object_preceding = self.NN(tensor_array_preceding)
                    object_preceding = tf.reshape(object_preceding, [width, height])

                    phase_shift_complex = tf.complex(self.phase_shift, 0.0)
                    object_preceding_complex = tf.complex(object_preceding, tf.zeros_like(object_preceding))
                    phase_delay = tf.exp(tf.complex(0.0, 1.0) * phase_shift_complex * object_preceding_complex)
                    U_z_following_prop *= phase_delay

                    U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz)
                    U_z_following_prop_intensity = tf.square(tf.abs(U_z_following_prop))

                    loss += tf.reduce_mean(tf.square(U_z0 - U_z_following_prop_intensity))
                    loss += 0.5*tf.reduce_mean(tf.square(object_preceding))

                else:
                    z_preceding = tf.fill([width, height], (z - dz) / z_max)
                    tensor_array_preceding = tf.stack([xx, yy, z_preceding], axis=-1)
                    tensor_array_preceding = tf.reshape(tensor_array_preceding, (-1, 3))

                    object_preceding = self.NN(tensor_array_preceding)
                    object_preceding = tf.reshape(object_preceding, [width, height])

                    phase_shift_complex = tf.complex(self.phase_shift, 0.0)
                    object_preceding_complex = tf.complex(object_preceding, tf.zeros_like(object_preceding))
                    phase_delay = tf.exp(tf.complex(0.0, 1.0) * phase_shift_complex * object_preceding_complex)
                    U_z_following_prop *= phase_delay

                    U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz)

        grads = tape.gradient(loss, self.NN.trainable_variables + [self.phase_shift] + [self.incident_light])

        del tape

        return loss, grads

    def compute_loss_and_grads_BC(self):
        with tf.GradientTape() as tape:
            loss = tf.constant(0.0, dtype=tf.float32)

            x_BC = tf.range(1, width + 1, dtype=tf.float32) / width
            y_BC = tf.range(1, height + 1, dtype=tf.float32) / height
            z_BC = tf.range(0, z_max + dz, dz, dtype=tf.float32) / z_max

            offset = 0.0

            xx_BC, zz_BC = tf.meshgrid(x_BC, z_BC)
            yy_0 = tf.fill(xx_BC.shape, 0.0 + (offset / height))
            yy_1 = tf.fill(xx_BC.shape, 1.0 - (offset / height))

            tensor_array_BC_0 = tf.stack([xx_BC, yy_0, zz_BC], axis=-1)
            tensor_array_BC_1 = tf.stack([xx_BC, yy_1, zz_BC], axis=-1)

            tensor_array_BC_0 = tf.reshape(tensor_array_BC_0, (-1, 3))
            tensor_array_BC_1 = tf.reshape(tensor_array_BC_1, (-1, 3))

            object_BC_0 = self.NN(tensor_array_BC_0)
            object_BC_1 = self.NN(tensor_array_BC_1)

            loss += 0.5*tf.reduce_mean(tf.square(object_BC_0))
            loss += 0.5*tf.reduce_mean(tf.square(object_BC_1))

            yy_BC, zz_BC = tf.meshgrid(y_BC, z_BC)
            xx_0 = tf.fill(yy_BC.shape, 0.0 + (offset / width))
            xx_1 = tf.fill(yy_BC.shape, 1.0 - (offset / width))

            tensor_array_BC_2 = tf.stack([xx_0, yy_BC, zz_BC], axis=-1)
            tensor_array_BC_3 = tf.stack([xx_1, yy_BC, zz_BC], axis=-1)

            tensor_array_BC_2 = tf.reshape(tensor_array_BC_2, (-1, 3))
            tensor_array_BC_3 = tf.reshape(tensor_array_BC_3, (-1, 3))

            object_BC_2 = self.NN(tensor_array_BC_2)
            object_BC_3 = self.NN(tensor_array_BC_3)

            loss += 0.5*tf.reduce_mean(tf.square(object_BC_2))
            loss += 0.5*tf.reduce_mean(tf.square(object_BC_3))

        grads = tape.gradient(loss, self.NN.trainable_variables)

        del tape

        return loss, grads

    def compute_loss_and_grads_pre(self, z_batch):
        with tf.GradientTape() as tape:
            loss = tf.constant(0.0, dtype=tf.float32)
            for z in z_batch:
                z_following = tf.fill([width, height], z / z_max)
                tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
                tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
                object_following = self.NN(tensor_array_following)
                object_following = tf.reshape(object_following, [width, height])
                gaussian_slice = slice_3d_gaussian(gaussian_3d, z, z_max)

                loss += tf.reduce_mean(tf.square(object_following - gaussian_slice))

        grads = tape.gradient(loss, self.NN.trainable_variables)

        del tape

        return loss, grads

    def save_weights(self, path):
        self.NN.save_weights(path + 'NN.h5')
        np.save(path + 'phase_shift.npy', self.phase_shift.numpy())
        np.save(path + 'incident_light.npy', self.incident_light.numpy())

    def load_weights(self, path):
        self.NN.load_weights(path + 'NN.h5')
        self.phase_shift.assign(np.load(path + 'phase_shift.npy'))
        self.incident_light.assign(np.load(path + 'incident_light.npy'))

    def callback(self, arg=None):
        if self.iter_count % 1 == 0:
            print('iter=', self.iter_count, ', loss=', self.instant_loss, ', phase_shift=', self.phase_shift.numpy(), ', incident_light=', self.incident_light.numpy())
            self.train_loss_history.append([self.iter_count, self.instant_loss])
        self.iter_count += 1

    def train_with_adam(self, adam_num, batch_size):
        def learn():
            loss, grads = self.compute_loss_and_grads()
            self.opt.apply_gradients(
                zip(grads, self.NN.trainable_variables + [self.phase_shift] + [self.incident_light]))
            return loss

        def learn_BC():
            loss, grads = self.compute_loss_and_grads_BC()
            self.opt.apply_gradients(
                zip(grads, self.NN.trainable_variables))
            return loss

        def learn_pre(z_batch):
            loss, grads = self.compute_loss_and_grads_pre(z_batch)
            self.opt_pre.apply_gradients(
                zip(grads, self.NN.trainable_variables))
            return loss

        for epoch in range(adam_num):

            if epoch <= 300: # epoch for pre-training: 300~500
                loss = tf.constant(0.0, dtype=tf.float32)
                dataset = create_z_dataset(z_min, z_max, dz, batch_size)
                for z_batch in dataset:
                    if len(z_batch) == 1:
                        z_batch = [z_batch]
                    loss += learn_pre(z_batch)
                loss /= z_num
            else:
                loss = learn()
                loss += learn_BC()

            self.instant_loss = loss.numpy()
            self.callback()

    def predict(self, pos):
        output = self.NN(pos)
        return output

    def train(self, num_training, batch_size):
        t0 = time()
        self.train_with_adam(num_training, batch_size)
        print('\nComputation time of training: {} seconds'.format(time() - t0))

        self.save_weights("./save_weights/")
        np.savetxt('./save_weights/loss.txt', self.train_loss_history)
        train_loss_history = np.array(self.train_loss_history)
        plt.plot(train_loss_history[:, 0], train_loss_history[:, 1])
        plt.yscale("log")
        plt.show()

    def get_internal_values(self):
        return self.phase_shift.numpy(), self.incident_light.numpy()

def main():
    num_training = 1000
    batch_size = 16
    agent = twin_image_removal()
    agent.train(num_training, batch_size)
    phase_shift, incident_light = agent.get_internal_values()
    print("Phase shift: ", phase_shift)
    print("Incident light: ", incident_light)

if __name__ == "__main__":
    main()