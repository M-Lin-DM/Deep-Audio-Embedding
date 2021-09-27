import numpy as np
import matplotlib.pyplot as plt

dataset = "wav-file-name"

architecture_ID = "R-F"
Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID
Z_RF = np.load(Z_save_path + '.npy')

architecture_ID = "R"
Z_save_path = 'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/' + dataset + '/Results/Embedding data/Z_last-' + architecture_ID
Z_R = np.load(Z_save_path + '.npy')

fig = plt.figure(figsize=plt.figaspect(0.5))
#===============
#  First subplot
#===============
# set up the axes for the first plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot3D(Z_RF[:, 0], Z_RF[:, 1], Z_RF[:, 2], 'k-', markerfacecolor='black', markersize=1, linewidth=0.3, label='Z')
# ax1.set_zlim(-1.01, 1.1)
# fig.colorbar(surf, shrink=0.5, aspect=10)

#===============
# Second subplot
#===============
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot3D(Z_R[:, 0], Z_R[:, 1], Z_R[:, 2], 'k-', markerfacecolor='black', markersize=1, linewidth=0.3, label='Z')
# ax2.set_zlim(-1.01, 1.1)
#
plt.show()

