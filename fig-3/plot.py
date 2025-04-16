# -*- coding: utf-8 -*-

###################################################
# path
import os, sys

folder = os.path.basename(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

data_path = f"{folder}/data"

# packages
from src.utils import *

###################################################

x_1 = np.load(os.path.join(data_path, "x-1.npy"))
y_1 = np.load(os.path.join(data_path, "y-1.npy"))
z_1 = np.load(os.path.join(data_path, "z-1.npy"))

x_2 = np.load(os.path.join(data_path, "x-2.npy"))
y_2 = np.load(os.path.join(data_path, "y-2.npy"))
z_2 = np.load(os.path.join(data_path, "z-2.npy"))

###################################################

fig = plt.figure(figsize=cm2inch(8.5, 3.4))
fig.subplots_adjust(left=-0.05, right=1.04, top=1.1, bottom=-0.12)
fig.subplots_adjust(wspace=-0.1)

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x_1, y_1, z_1, cmap='coolwarm') 
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
z_low = ax1.get_zlim()[0] - 0.3 * abs(ax1.get_zlim()[1] - ax1.get_zlim()[0])
ax1.contour(x_1, y_1, z_1, 8, linewidths=0.7, zdir='z', offset=z_low, cmap='coolwarm')
ax1.set_zlim(z_low, ax1.get_zlim()[1])
ax1.view_init(elev=15, azim=-45)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_2, y_2, z_2, cmap='coolwarm')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
z_low = ax2.get_zlim()[0] - 0.3 * abs(ax2.get_zlim()[1] - ax2.get_zlim()[0])
ax2.contour(x_2, y_2, z_2, 8, linewidths=0.7, zdir='z', offset=z_low, cmap='coolwarm')
ax2.set_zlim(z_low, ax2.get_zlim()[1])
ax2.view_init(elev=15, azim=-45)

ax1.text2D(0.15, 0.88, "(a)", transform=ax1.transAxes, fontsize=10, va="top", ha="right")
ax2.text2D(0.15, 0.88, "(b)", transform=ax2.transAxes, fontsize=10, va="top", ha="right")

plt.savefig(f"{folder}/{folder}.pdf")
