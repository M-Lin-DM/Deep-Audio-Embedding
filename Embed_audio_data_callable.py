import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import plotly.graph_objects as go
import matplotlib.pyplot as plt

if __name__ == '__main__':

    architecture_ID = "R-F"
    dataset = "wav-file-name"

    checkpoint_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/saved models/last-' + architecture_ID
    outfile = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/variables/vars.npz'

    Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID

    model = keras.models.load_model(checkpoint_path)
    model.summary()

    varbs = np.load(outfile)
    varbs.files
    S_mag = varbs['S_mag']
    T = varbs['T']
    F = varbs['F']
    img_width = varbs['img_width']  # time axis
    img_height = varbs['img_height']  # freq axis
    img_size = (img_width, img_height)
    dt = varbs['dt']
    time_resolution = varbs['time_resolution']


    def normalize0_1(A):
        return (A - np.min(A)) / (np.max(A) - np.min(A))

    S_mag = normalize0_1(S_mag)

    l = 2
    batch_size = 1

    clip_tensor_shape = (l,) + img_size
    top_frames_shape = (2,) + img_size

    detached_encoder_input = Input(top_frames_shape, name='detached_encoder_input')
    embedded_top_frames = model.layers[2](detached_encoder_input)  # model.layers[2] is The wrapped encoder
    output = embedded_top_frames[:, 0, :]  # we only need the first vector since the prediction generator will be looping over all points

    encoder = Model(detached_encoder_input, output, name='encoder')
    encoder.summary()

    def inputs_generator_inference(t_start, t_end, S, dt_, img_height, img_width):

        t = t_start
        while t <= t_end:
            top_frames = []

            for j in range(t, t + 2 * dt_, dt_):  # the second image wont be used but we still need to feed 2 images, since the layer that will be extracted from the model expects this
                sub_img = np.transpose(S[:img_height, j:j + img_width])
                top_frames.append(sub_img.astype(np.float32))
            top_frames = np.array(top_frames)

            yield top_frames  # 0 represents the target output of the model. the metric in .compile is computed using this. the addloss layer outputs the loss itself just for convienience.
            t += dt_

    t_start = int(dt * (l - 1))  # first time bin index (in S_mag) in the valid range
    t_end = int(len(
        T) - img_width - dt - 1)  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
    samples = np.floor((t_end - t_start + 1) / dt)
    steps_per_epoch_tr = int(np.floor(samples / batch_size)) - 1  # number of batches

    ds_inference = tf.data.Dataset.from_generator(
        inputs_generator_inference,
        args=[t_start, t_end, S_mag, dt, img_height, img_width],
        output_types=(tf.float32),
        output_shapes=top_frames_shape)

    ds_inference = ds_inference.batch(batch_size, drop_remainder=False)

    Z = encoder.predict(ds_inference)
    np.save(Z_save_path, Z)
    Z.shape

    a = 7
    fig2 = plt.figure(figsize=(1.7778 * a, a))  # e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
    ax = fig2.gca(projection='3d')

    # ax.set_xlabel('$X$', fontsize=20)
    # ax.set_ylabel('$Y$')
    # disable auto rotation
    # ax.zaxis.set_rotate_label(False)
    # ax.set_zlabel('$\gamma$', fontsize=10, rotation = 0)
    ax.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], 'k-', markerfacecolor='black', markersize=1, linewidth=0.5, label='Z')
    plt.show()
