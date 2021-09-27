import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import os
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Flatten, Concatenate, LSTM, MaxPool2D, Reshape, Conv2DTranspose, BatchNormalization, UpSampling2D, Cropping2D, Conv2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K
import pydot as pydot

from PIL import Image
import pandas as pd
import csv

architecture_ID = "R"
checkpoint_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/Applause/saved models/Latent Space LSTM/last-' + architecture_ID

Z_test = np.load('C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/Applause/Results/Embedding data/Z_last-' + architecture_ID + '.npy')
full_model = keras.models.load_model(checkpoint_path)  # Full model
# model = keras.models.load_model('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/Latent Space LSTM/last-' + architecture_ID)
# model.summary()

T_test = len(Z_test)
l = 20
latent_dim=3
batch_size = 1
start_frame_test = l - 1  # first frame index in the valid range
end_frame_test = T_test - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples_test = end_frame_test - start_frame_test + 1
print(samples_test)

def inputs_generator_inference_latent_LSTM(t_start, t_end, Z):
    t = t_start
    while t <= t_end:  #now we only predict up to the 2nd to last frame since that is the last frame at which the true v is computable
        ZSegment = Z[t - l + 1:t + 1, :]

        yield (ZSegment)  # 0 represents the target output of the model. the metric in .compile is computed using this. the addloss layer outputs the loss itself just for convienience.
        t += 1


ds_test = tf.data.Dataset.from_generator(
    inputs_generator_inference_latent_LSTM,
    args=[start_frame_test, end_frame_test, Z_test],
    output_types=(tf.float32),
    output_shapes=(l, latent_dim))

ds_test = ds_test.batch(batch_size, drop_remainder=False)

v = Z_test[start_frame_test+1:T_test, :] - Z_test[start_frame_test:T_test-1, :]


# detached_LSTM_input = Input((75, 3), name='detached_LSTM_input')
# w = full_model.layers[6](detached_LSTM_input)
# vhat = full_model.layers[8](w)
# model = Model(detached_LSTM_input, vhat, name='joint_LSTM')

# vhat = model.predict(ds_test)
vhat = full_model.predict(ds_test)

print(f"v shape {v.shape}, vhat shape {vhat.shape}")
MSE = np.mean(np.linalg.norm(v - vhat, axis=1)**2)
CosineSim = np.mean(tf.keras.losses.cosine_similarity(vhat, v, axis=-1).numpy())
print(f"MSE {MSE} Cos {CosineSim}")