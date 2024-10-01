import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
from MorpHoloNet import *

# Directories for saving results
dir = 'Z:' # Directory of Python workplace
Intensity_MorpHoloNet_dir = dir + '\\Results\\Intensity_MorpHoloNet'
obj_dir = dir + '\\Results\\obj'

def reset_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

reset_directory(Intensity_MorpHoloNet_dir)
reset_directory(obj_dir)

# Load a trained model
agent = twin_image_removal()
agent.load_weights('./save_weights/') # Directory of the trained model
phase_shift, incident_light = agent.get_internal_values()

print("Phase shift: ", phase_shift)
print("Incident Light: ", incident_light)

U_incident_avg_real = tf.cast(incident_light, tf.float32)

print(width, height)

minPX = 1
maxPX = width
minPY = 1
maxPY = height
segment_size = width

x = tf.range(1, width + 1, dtype=tf.float32) / width
y = tf.range(1, height + 1, dtype=tf.float32) / height
xx, yy = tf.meshgrid(x, y)

dz = 1.0
z_min = 0.0 + dz
z_max = 105.0

z_values = tf.range(z_min, z_max + dz, dz, dtype=tf.float32)
obj_arr_save = np.zeros((maxPY,maxPX,len(z_values)))
index = 0

# Saving object arrays reconstructed at different depths using MorpHoloNet
for z in z_values:
    z_norm = z / z_max
    z_filled = tf.fill(xx.shape, float(z_norm))
    tensor_array = tf.stack([xx, yy, z_filled], axis=-1)
    tensor_array = tf.reshape(tensor_array, (-1, 3))
    obj = agent.predict(tensor_array)

    print('Saving an object array at a depth of ', z)

    obj = np.array(obj)
    obj = obj.reshape(width, height)
    obj_arr_save[:, :, index] = obj

    obj = Image.fromarray(obj)
    file_directory = os.path.join(obj_dir, 'obj_arr_' + str(z.numpy()) + '.tif')
    obj.save(file_directory)

    index += 1

# Saving 3D object arrays as a numpy format
file_directory = os.path.join(obj_dir, 'obj_arr_save')
np.save(file_directory, obj_arr_save)

# Saving intensity maps reconstructed at different depths using MorpHoloNet
z_values = tf.range(0.0, z_max + dz, dz, dtype=tf.float32)
z_values_reverse = z_values[::-1]

real_following_ref = tf.multiply(tf.ones([width, height], dtype=tf.float32), U_incident_avg_real)
imag_following_ref = tf.zeros([width, height], dtype=tf.float32)
U_z_following_ref = tf.complex(real_following_ref, imag_following_ref)

for z in z_values_reverse:
    print('Saving an intensity map at depth of ', z)
    if z == z_max:
        U_z_following_ref_intensity = tf.square(tf.abs(U_z_following_ref)).numpy()
        U_z_following_ref_intensity = Image.fromarray(U_z_following_ref_intensity)
        file_directory = os.path.join(Intensity_MorpHoloNet_dir, 'Intensity_MorpHoloNet_' + str(z.numpy()) + '.tif')
        U_z_following_ref_intensity.save(file_directory)
    elif z == (z_max - dz):
        z_following = tf.fill([width, height], z / z_max)
        tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
        tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
        classification_following = agent.NN(tensor_array_following)
        classification_following = tf.reshape(classification_following, [width, height])
        phase_shift_complex = tf.complex(agent.phase_shift, 0.0)
        classification_following_complex = tf.complex(classification_following, tf.zeros_like(classification_following))
        phase_delay = tf.exp(tf.complex(0.0, 1.0) * phase_shift_complex * classification_following_complex)
        U_z_following_prop = U_z_following_ref * phase_delay
        U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz)
        U_z_following_prop_intensity = tf.square(tf.abs(U_z_following_prop)).numpy()
        U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
        file_directory = os.path.join(Intensity_MorpHoloNet_dir, 'Intensity_MorpHoloNet_' + str(z.numpy()) + '.tif')
        U_z_following_prop_intensity.save(file_directory)
    elif z == z_min:
        z_following = tf.fill([width, height], z / z_max)
        tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
        tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
        classification_following = agent.NN(tensor_array_following)
        classification_following = tf.reshape(classification_following, [width, height])
        phase_shift_complex = tf.complex(agent.phase_shift, 0.0)
        classification_following_complex = tf.complex(classification_following, tf.zeros_like(classification_following))
        phase_delay = tf.exp(tf.complex(0.0, 1.0) * phase_shift_complex * classification_following_complex)
        U_z_following_prop = U_z_following_prop * phase_delay
        U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz)
        U_z_following_prop_intensity = tf.square(tf.abs(U_z_following_prop)).numpy()
        U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
        file_directory = os.path.join(Intensity_MorpHoloNet_dir, 'Intensity_MorpHoloNet_' + str(z.numpy()) + '.tif')
        U_z_following_prop_intensity.save(file_directory)
        U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz)
        U_z_following_prop_intensity = tf.square(tf.abs(U_z_following_prop)).numpy()
        U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
        file_directory = os.path.join(Intensity_MorpHoloNet_dir, 'Intensity_MorpHoloNet_0.tif')
        U_z_following_prop_intensity.save(file_directory)
    else:
        z_following = tf.fill([width, height], z / z_max)
        tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
        tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
        classification_following = agent.NN(tensor_array_following)
        classification_following = tf.reshape(classification_following, [width, height])
        phase_shift_complex = tf.complex(agent.phase_shift, 0.0)
        classification_following_complex = tf.complex(classification_following, tf.zeros_like(classification_following))
        phase_delay = tf.exp(tf.complex(0.0, 1.0) * phase_shift_complex * classification_following_complex)
        U_z_following_prop = U_z_following_prop * phase_delay
        U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz)
        U_z_following_prop_intensity = tf.square(tf.abs(U_z_following_prop)).numpy()
        U_z_following_prop_intensity = Image.fromarray(U_z_following_prop_intensity)
        file_directory = os.path.join(Intensity_MorpHoloNet_dir, 'Intensity_MorpHoloNet_' + str(z.numpy()) + '.tif')
        U_z_following_prop_intensity.save(file_directory)

