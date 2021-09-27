import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

dataset = "wav-file-name"
architecture_ID = "R"
# checkpoint_path_LSTM = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/saved models/Latent Space LSTM/last-' + architecture_ID

#Load from:
Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID
Z = np.load(Z_save_path + '.npy')
latent_dim = Z.shape[1]

# Save to:
Delta_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/variables/Delta_' + architecture_ID


def normalize0_1(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

for j in range(latent_dim):
    Z[:, j] = normalize0_1(Z[:, j])

N_steps = 30
# t=3030  # time step of initial condition
step_size = 0.1

T = len(Z)
l = 20

start_frame = l - 1  # first frame inded in the valid range
end_frame = T - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples = end_frame - start_frame + 1
print(f"total number of time-steps: {samples}")

# t = start_frame_test+1

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
    weights = np.tile(weights.reshape(K, 1), (1, latent_dim_))  # broadcast weights across array with same shape as V subset
    # print(weights.shape)
    # print(np.squeeze(V[indices, :]))

    return np.sum(np.squeeze(V_[indices, :]) * weights, axis=0)  # weighted average of nearby V vectors


# march forward in the latent space, using the latent-LSTM model to predict the next velocity vector and z, and so on..
Delta = []
# for t in range(start_frame, start_frame+51, 10):
for t in range(start_frame, end_frame-N_steps-2, 10):
    ZFuture = Z[t:t+N_steps, :]  # there will be N_steps+1 steps after adding the initial condition point to the start
    z_t = Z[t, :]
    ZForecast = [z_t]  # include the starting point, initial condition
    print(t)

    for step in range(N_steps-1):
        vhat_t = interpolate_v_at_z(neighbor_tree, z_t, V, latent_dim)  # shape (1,3) NOTE: vhat has only been optimized to ALIGN (in orientation) with the true v. consider stepping in small increments. the magnitude of vhat is not clear
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
    Delta.append(np.linalg.norm(ZForecast-ZFuture, axis=1))

Delta = np.array(Delta)
print(f"Delta: {Delta.shape}")
Delta_mean = np.mean(Delta, axis=0)
Delta_std = np.std(Delta, axis=0)


np.savez(Delta_path, Delta_mean=Delta_mean, Delta_std=Delta_std)


# Plot-----------------------------------------------------------------------
a = 9
fig2 = plt.figure(figsize=(1.7778*a, a))  #  e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
# ax = plt.axes(projection='3d')
ax = fig2.gca()
# ax.plot(np.arange(0, N_steps, 1), Delta_mean, 'k-', markerfacecolor='black', markersize=3, linewidth=1, label='Delta')
plt.errorbar(np.arange(0, N_steps, 1), Delta_mean, yerr=Delta_std, xerr=None, fmt='k')
plt.show()