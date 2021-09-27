import numpy as np
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

dataset = "wav-file-name"
architecture_ID = "R-F"

checkpoint_path_LSTM = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/saved models/Latent Space LSTM/last-' + architecture_ID

# Load from:
Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID
Z = np.load(Z_save_path + '.npy')
latent_dim = Z.shape[1]
print(latent_dim)
# Save to:
ZForecast_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/ZForecast/' + architecture_ID + '.npy'


def normalize0_1(A):
    return (A - np.min(A)) / (np.max(A) - np.min(A))


for j in range(latent_dim):
    Z[:, j] = normalize0_1(Z[:, j])

K = 5
neighbor_tree = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree').fit(Z)
V = Z[1:, :] - Z[0:-1, :]
V = np.concatenate((V, np.zeros(shape=(1, latent_dim))),
                   axis=0)  # add row to the end to make the indicies of v align with z
print(f"Z: {Z.shape}, V: {V.shape}")


def interpolate_v_at_z(neighbor_tree_, z_query, V_, latent_dim_):
    distances, indices = neighbor_tree_.kneighbors(z_query.reshape(1, -1))
    indices = indices[:, 1:]
    distances = distances[:, 1:]  # crop out distance to self, 0
    weights = distances ** (-1)
    # print(indices)
    weights /= np.sum(weights, axis=1)
    # print(weights)
    weights = np.tile(weights.reshape(K, 1),
                      (1, latent_dim_))  # broadcast weights across array with same shape as V subset
    # print(weights.shape)
    # print(np.squeeze(V[indices, :]))

    return np.sum(np.squeeze(V_[indices, :]) * weights, axis=0)  # weighted average of nearby V vectors


# v_interp = interpolate_V_at_Z(neighbor_tree, Z[0], V)
# print(v_interp)

N_steps = 200
t = 2000  # time step of initial condition
step_size = 0.1

T_test = len(Z)
l = 20
batch_size = 1

z_t = Z[t, :]
print(z_t.shape)

ZForecast = [z_t]
# march forward in the latent space, using the latent-LSTM model to predict the next velocity vector and z, and so on..
for i in range(N_steps):
    vhat_t = interpolate_v_at_z(neighbor_tree, z_t, V,
                                latent_dim)  # shape (1,3) NOTE: vhat has only been optimized to ALIGN (in orientation) with the true v. consider stepping in small increments. the magnitude of vhat is not clear
    # print(vhat_i_plus_1)
    # using full vhat as a step:
    z_t_plus_1 = z_t + vhat_t  # shape (1,3)
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

ZForecast = np.concatenate((np.expand_dims(Z[t, :], axis=0), ZForecast), axis=0)
print(f"ZForecast: {ZForecast.shape}")

fig2 = go.Figure(data=[go.Scatter3d(
    name='z',
    x=Z[:, 0],
    y=Z[:, 1],
    z=Z[:, 2],
    mode='markers+lines',
    marker=dict(
        size=0.5,
        color='black',
        symbol='circle'
    ),
    line=dict(
        color='black',
        width=2
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
        color='lime',
        symbol='circle'
    ),
    line=dict(
        color='lime',
        width=4
    )
))

fig2.add_trace(go.Scatter3d(
    name='initial condition',
    x=[ZForecast[0, 0]],
    y=[ZForecast[0, 1]],
    z=[ZForecast[0, 2]],
    mode='markers',
    marker=dict(
        size=6,
        color='blue',
        symbol='circle'
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
        x=0.02,
        font_size=16
    ),
    scene_xaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
    scene_yaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
    scene_zaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
)

plane_color = 'lightslategray'
fig2.update_scenes(bgcolor='lightslategray', xaxis_backgroundcolor=plane_color, yaxis_backgroundcolor=plane_color,
                   zaxis_backgroundcolor=plane_color)

fig2.show()
# fig2.savefig(movie_path + 'frame_' + f'{t:03}' + '.png', transparent=True, dpi='figure', bbox_inches=None)
