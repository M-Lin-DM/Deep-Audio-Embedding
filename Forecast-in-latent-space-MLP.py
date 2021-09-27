from tensorflow import keras
import numpy as np
import plotly.graph_objects as go

architecture_ID = "R-F"
dataset = "wav-file-name"

checkpoint_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/saved models/Latent Space MLP/last-' + architecture_ID

# Load from:
Z_test = np.load(
    'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID + '.npy')

# Save to:
ZForecast_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/ZForecast/' + architecture_ID + '.npy'

# Option 1: load Latent-space LSTM (trained independently)
model = keras.models.load_model(checkpoint_path)

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

N_steps = 100
t = 1200  # time step of initial condition
step_size = 0.1

T_test = len(Z_test)
l = 20
latent_dim = 3
batch_size = 1


z_t = Z_test[t, :]
z_t = np.expand_dims(z_t, axis=0)  # add batch axis
print(z_t.shape)

ZForecast = []
# march forward in the latent space, using the latent-LSTM model to predict the next velocity vector and z, and so on..
for i in range(N_steps):
    vhat_t_plus_1 = model.predict(
        z_t)  # shape (1,3) NOTE: vhat has only been optimized to ALIGN (in orientation) with the true v. consider stepping in small increments. the magnitude of vhat is not clear
    # print(vhat_i_plus_1)
    # using full vhat as a step:
    z_t_plus_1 = z_t + vhat_t_plus_1  # shape (1,3)
    # print(z_t_plus_1.shape)
    # using small step size. When you take too small a step size, the LSTM may not know what to do with it. since small steps dont exist in the training data, appending it onto z_t would create an input outside the training distribution
    # z_i_plus_1 = z_t[:, -1, :] + step_size*vhat_i_plus_1/np.linalg.norm(vhat_i_plus_1, axis=1)  # shape (1,3)
    # print(step_size*np.linalg.norm(vhat_i_plus_1, axis=1))
    ZForecast.append(np.squeeze(z_t_plus_1))
    z_t = z_t_plus_1
    # print(ZForecast)

ZForecast = np.array(ZForecast)
# print(ZForecast)
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
    mode='markers+lines',
    marker=dict(
        size=2,
        color='black',
        symbol='circle'
    ),
    line=dict(
        color='black',
        width=4
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

    #     font=dict( #setting global font will override the xaxis properties set
    #         family="Courier New, monospace",
    #         color="RebeccaPurple"
    #     ),
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
