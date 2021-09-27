import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pydot as pydot
import wandb
from wandb.keras import WandbCallback

architecture_ID = "R-F"
dataset = "wav-file-name"

checkpoint_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/saved models/Latent Space MLP/last-' + architecture_ID
Z_tr = np.load('C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID + '.npy')

def normalize0_1(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

for j in range(3):
    Z_tr[:, j] = normalize0_1(Z_tr[:, j])

# wandb.login()
id = wandb.util.generate_id()
print(f"Wandb ID: {id}")
wandb.init(entity='potoo', project='Deep Audio Embedding', group='latent-MLP-Applause', name=architecture_ID, id=id, resume="allow",  # if you want to resume a crashed run, lookup the run id in W&B and paste it into id= and set resume="allow"
           config={"l": 0,
                   "batch_size": 4,
                   "epochs": 8,
                   "dropout": 0.0,
                   "L1": 0.000,
                   "L2": 0.000,
                   "hidden_units": 12,
                   "checkpoint_path": checkpoint_path})
config = wandb.config

l = config.l  # clip length, the number of frames in the video segment fed to the ConvLSTM, should be odd # if decimating by a factor 2
batch_size = config.batch_size


T_tr = len(Z_tr)  # number of images in the dataset
# T_val = len(Z_val)
decimation = 1
latent_dim = 3

print(Z_tr.shape)

def inputs_generator(t_start, t_end, Z):
    t = t_start
    while t <= t_end:
        v_true = Z[t+1, :] - Z[t, :]  # target output
        z_t = Z[t, :]

        yield (z_t, v_true)  # 0 represents the target output of the model. the metric in .compile is computed using this. the addloss layer outputs the loss itself just for convienience.
        t += 1

start_frame_tr = l - 1  # first frame inded in the valid range
end_frame_tr = T_tr - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples_tr = end_frame_tr - start_frame_tr + 1
steps_per_epoch_tr = int(np.floor((end_frame_tr - start_frame_tr + 1) / config.batch_size))  # number of batches

# start_frame_val = l - 1  # first frame inded in the valid range
# end_frame_val = T_val - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
# steps_per_epoch_val = int(np.floor((end_frame_val - start_frame_val + 1) / config.batch_size))

ds_train = tf.data.Dataset.from_generator(
    inputs_generator,
    args=[start_frame_tr, end_frame_tr, Z_tr],
    output_types=(tf.float32, tf.float32),
    output_shapes=((latent_dim,), (latent_dim,)))

# ds_val = tf.data.Dataset.from_generator(
#     inputs_generator,
#     args=[start_frame_val, end_frame_val, Z_val],
#     output_types=(tf.float32, tf.float32),
#     output_shapes=((l, latent_dim), (latent_dim,)))

# ds_validation = ds_val.batch(batch_size, drop_remainder=True).repeat(config.epochs)
ds_train = ds_train.shuffle(int(samples_tr * 0.33),
                            # turn off shuffle if you are training on a subset of the data for hyperparameter tuning and plan to visualize performance on the first steps_per_epoch_tuning batches of the TRAINING set
                            reshuffle_each_iteration=False)  # the argument into shuffle is the buffer size. this can be smaller than the number of samples, especially when using larger datasets
ds_train = ds_train.batch(batch_size, drop_remainder=True)  # insufficient data error was thrown without adding the .repeat(). I added the +1 dataset at the end for good measure
ds_train = ds_train.repeat(config.epochs + 1)

print(ds_train)

input = Input(shape=(latent_dim, ), name='input')
x = Dense(config.hidden_units, activation='relu', name='dense1')(input)
x = tf.keras.layers.Dropout(0.05, input_shape=(config.hidden_units,))(x)
x = Dense(config.hidden_units, activation='relu', name='dense2')(x)
x = tf.keras.layers.Dropout(0.05, input_shape=(config.hidden_units,))(x)
vhat = Dense(latent_dim, activation='tanh', name='vhat')(x)  # Predicted target

model = Model(input, vhat)
model.compile(loss='mse',  # compute loss internally
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False))

model.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='loss', #the loss, weighted by class weights
    # change to 'loss' instead of 'val_loss' if youre tuning hyperparams and training on a small subset of the data
    mode='min',
    save_best_only=False,
    save_freq='epoch')  # change to false if youre tuning hyperparams and training on a small subset of the data

# model_checkpoint_callback_best_validation = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path_best_validation,
#     save_weights_only=False,
#     monitor='val_loss',
#     # change to 'loss' instead of 'val_loss' if youre tuning hyperparams and training on a small subset of the data
#     mode='min',
#     save_best_only=True)  # change to false if youre tuning hyperparams and training on a small subset of the data

model.fit(ds_train,
          steps_per_epoch=steps_per_epoch_tr,
          # 5607,  # steps_per_epoch needs to equal exactly the number of batches in the Dataset generator
          epochs=config.epochs,
          verbose=2,
          callbacks=[WandbCallback(), model_checkpoint_callback])  #

exec(open(r"C:\Users\MrLin\Documents\Experiments\Deep Audio Embedding\python project\Forecast-in-latent-space-MLP.py").read())
