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

h_matrix = np.load(f"{data_path}/h_matrix.npy")
J_matrix = np.load(f"{data_path}/J_matrix.npy")
m = np.load(f"{data_path}/m.npy")
v = np.load(f"{data_path}/v.npy")

fig = plt.figure(figsize=cm2inch(17.9, 3), constrained_layout=True)

gs = fig.add_gridspec(nrows=1, ncols=4)

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(h_matrix, cmap='Blues', vmax=0.22, vmin=-0.12)
ax1.set_xticks([])
ax1.set_yticks([])
cbar = fig.colorbar(ax1.images[0], ax=ax1, orientation='vertical')
cbar.set_ticks([-0.1, 0, 0.1, 0.2])

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(J_matrix, cmap='Blues', vmax=0.12, vmin=-0.22)
ax2.set_xticks([])
ax2.set_yticks([])
cbar = fig.colorbar(ax2.images[0], ax=ax2, orientation='vertical')
cbar.set_ticks([-0.2, -0.1, 0, 0.1])

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(m, cmap='coolwarm')  # , vmax=0.75, vmin=-0.25
ax3.set_xticks([])
ax3.set_yticks([])
cbar = fig.colorbar(ax3.images[0], ax=ax3, orientation='vertical')
cbar.set_ticks([-0.25, 0, 0.25, 0.5])

ax4 = fig.add_subplot(gs[0, 3])
ax4.imshow(v, cmap='coolwarm', vmax=0.0252, vmin=0.0218)
ax4.set_xticks([])
ax4.set_yticks([])
cbar = fig.colorbar(ax4.images[0], ax=ax4, orientation='vertical')
cbar.set_ticks([0.022, 0.023, 0.024, 0.025])

ax1.text(-0.04, 1, "(a)", transform=ax1.transAxes, fontsize=10, va="top", ha="right")
ax2.text(-0.04, 1, "(b)", transform=ax2.transAxes, fontsize=10, va="top", ha="right")
ax3.text(-0.04, 1, "(c)", transform=ax3.transAxes, fontsize=10, va="top", ha="right")
ax4.text(-0.04, 1, "(d)", transform=ax4.transAxes, fontsize=10, va="top", ha="right")

plt.savefig(f"{folder}/{folder}.pdf")
