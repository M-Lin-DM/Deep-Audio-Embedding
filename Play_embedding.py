import matplotlib.pyplot as plt
import numpy as np

architecture_ID = "R-F"
dataset = "wav-file-name"

movie_path = "D:/Datasets/Deep Audio Embedding/No Place to Go/movies3/"

outfile = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/variables/vars.npz'
Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID + '.npy'

vars = np.load(outfile)
vars.files
S_mag = vars['S_mag']
T = vars['T']
F = vars['F']
img_width = vars['img_width']  # time axis
img_height = vars['img_height']  # freq axis
img_size = (img_width, img_height)
dt = vars['dt']
time_resolution = vars['time_resolution']

Z = np.load(Z_save_path)

l = 45
# determine real time start and end of the embedding in seconds
t_start = int(dt*(l - 1))  # first time bin index (in S_mag) in the valid range
t_end = int(len(T) - img_width - dt - 1)  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
T_start = T[t_start]
T_end = T[t_end]
samples = np.floor((t_end - t_start + 1)/dt)

print(f"t_start {t_start} (bins), t_end {t_end} (bins)")
print(f"T_start {T_start} (s), T_end {T_end} (s)")
print(f" seconds per Spectrogram time bin: {1/time_resolution}")
# print(f"30fps T_start | seconds:frame | {np.floor(T_start)}:{np.round((T_start-np.floor(T_start))*30)}")
# print(f"30fps T_end | seconds:frame |  {np.floor(T_end)}:{np.round((T_end-np.floor(T_end))*30)}")
print(f"bins spanning input tensor: {(l-1)*dt}")
print(f"time spanning input tensor: {(l-1)*dt/time_resolution} (s)")  # should equal T_start
print(f"Total frames to be rendered: {samples}")

tail_points = 12
print(f"T_start including tail points {T_start+(tail_points-1)/time_resolution} (s), T_end {T_end} (s)")

def normalize0_1(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

S_mag = normalize0_1(S_mag)

a = 8
fig2 = plt.figure(figsize=(1.7778*a, a))  #  e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig
dtheta = 0.13/2.1  # 0.02  #rotation rate deg
k = 0
dk = 0  # 2*np.pi/360*0.05
theta = 0
# phi_0 = 15
phi = 35  # elevation angle
render_interval = 1

for t in range(tail_points, len(Z), render_interval):
    ax = fig2.add_subplot(projection='3d')
    ax.plot3D(Z[:t, 0], Z[:t, 1], Z[:t, 2], '-', markerfacecolor='black', markersize=1, linewidth=1, color='black', label='Z')
    ax.plot3D(Z[t-tail_points:t, 0], Z[t-tail_points:t, 1], Z[t-tail_points:t, 2], '-o', markerfacecolor='orange', mec='darkblue', markersize=12, linewidth=2, label='Z(t)')

    # phi = 20*np.sin(k) + phi_0
    # phi = 0
    ax.view_init(phi, theta)  #view_init(elev=None, azim=None)
    # ax.axis('off')  # for saving transparent gifs
    ax.dist = 8
    plt.draw()
    plt.pause(.01)
    fig2.savefig(movie_path + 'frame_' + f'{t:03}' + '.png', transparent=False, dpi='figure', bbox_inches=None)
    fig2.clear(keep_observers=True)

    theta += dtheta
    # k += dk
    print(t)
