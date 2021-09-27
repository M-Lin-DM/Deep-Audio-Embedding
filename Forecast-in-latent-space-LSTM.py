from tensorflow import keras
import numpy as np
import plotly.graph_objects as go

dataset = "wav-file-name"
architecture_ID = "R-F"

checkpoint_path_LSTM = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/saved models/Latent Space LSTM/last-' + architecture_ID

# Load from:
Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID
Z_test = np.load(Z_save_path + '.npy')

# Save to:
ZForecast_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/ZForecast/' + architecture_ID + '.npy'

# Option 1: load Latent-space LSTM (trained independently)
model = keras.models.load_model(checkpoint_path_LSTM)

# Option 2: Extract LSTM module from trained full model (trained jointly with Encoder)
# full_model = keras.models.load_model('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/DVE/last-' + architecture_ID)  # Full model
# detached_LSTM_input = Input((75, 3), name='detached_LSTM_input')
# w = full_model.layers[6](detached_LSTM_input)
# vhat = full_model.layers[8](w)
# model = Model(detached_LSTM_input, vhat, name='joint_LSTM')

model.summary()


def normalize0_1(A):
    return (A - np.min(A)) / (np.max(A) - np.min(A))


for j in range(3):
    Z_test[:, j] = normalize0_1(Z_test[:, j])

N_steps = 170
t = 1000  # time step of initial condition
step_size = 0.1

T_test = len(Z_test)
l = 20
latent_dim = 12
batch_size = 1

ZSegment = Z_test[t - l + 1:t + 1, :]

ZSegment = np.expand_dims(ZSegment, axis=0)  # add batch axis
print(ZSegment.shape)

ZForecast = []
# march forward in the latent space, using the latent-LSTM model to predict the next velocity vector and z, and so on..


for i in range(N_steps):
    vhat_i_plus_1 = model.predict(
        ZSegment)  # shape (1,3) NOTE: vhat has only been optimized to ALIGN (in orientation) with the true v. consider stepping in small increments. the magnitude of vhat is not clear

    # using full vhat as a step:
    z_i_plus_1 = ZSegment[:, -1, :] + vhat_i_plus_1  # shape (1,3)

    # using small step size. When you take too small a step size, the LSTM may not know what to do with it. since small steps dont exist in the training data, appending it onto ZSegment would create an input outside the training distribution
    # z_i_plus_1 = ZSegment[:, -1, :] + step_size*vhat_i_plus_1/np.linalg.norm(vhat_i_plus_1, axis=1)  # shape (1,3)
    # print(step_size*np.linalg.norm(vhat_i_plus_1, axis=1))

    ZForecast.append(np.squeeze(z_i_plus_1))
    z_i_plus_1 = np.expand_dims(z_i_plus_1, axis=0)  # shape (1, 1,6)
    ZSegment = np.concatenate((ZSegment[:, 1:, :], z_i_plus_1),
                              axis=1)  # crop off first timestep and append the one just computed

ZForecast = np.array(ZForecast)
np.save(ZForecast_path, ZForecast)
print(f"ZForecast: {ZForecast.shape}")

# Plot-----------------------------------------------------------------------

ZForecast = np.concatenate((np.expand_dims(Z_test[t, :], axis=0), ZForecast), axis=0)
print(f"ZForecast: {ZForecast.shape}")

fig2 = go.Figure(data=[go.Scatter3d(
    name='z',
    x=Z_test[:, 0],
    y=Z_test[:, 1],
    z=Z_test[:, 2],
    mode='lines',
    line=dict(
        color='black',
        width=1
    )
)])

fig2.add_trace(go.Scatter3d(
    name='z forecast',
    x=ZForecast[:, 0],
    y=ZForecast[:, 1],
    z=ZForecast[:, 2],
    mode='markers+lines',
    marker=dict(
        size=3,
        color='red',
        symbol='circle'
    ),
    line=dict(
        color='red',
        width=4
    )
))

fig2.update_layout(
    width=900,
    height=700,
    scene_xaxis_title_text='z<sub>1</sub>',
    scene_yaxis_title_text="z<sub>2</sub>",
    scene_zaxis_title_text="z<sub>3</sub>",

    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.17,
        font_size=16
    ),
    scene_xaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
    scene_yaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
    scene_zaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
)

fig2.show()
