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

loss_1 = np.load(os.path.join(data_path, "loss-1.npy"))
loss_2 = np.load(os.path.join(data_path, "loss-2.npy"))

epochs = range(1, loss_1.shape[0]+1)

W_1 = np.load(os.path.join(data_path, "W-1.npy"))
W_2 = np.load(os.path.join(data_path, "W-2.npy"))

m = np.load(os.path.join(data_path, "m.npy"))
v = np.load(os.path.join(data_path, "v.npy"))

###################################################

fig = plt.figure(figsize=cm2inch(8.5, 4), constrained_layout=True)


gs = fig.add_gridspec(
    nrows=2, ncols=3, 
    width_ratios=[2.4, 1, 1], 
    left=0.1, right=0.95,
    hspace=0, wspace=0
    )



ax1 = fig.add_subplot(gs[:, 0])

ax1.plot(epochs, loss_1, label=r"$P=10$", linestyle="-", linewidth=1.5)
ax1.plot(epochs, loss_2, label=r"$P=1000$", linestyle="-", linewidth=1.5)
ax1.set_ylabel("Test  Loss")
ax1.set_xlabel("Epochs", labelpad=0)
ax1.set_xlim(-15, 315)
ax1.set_ylim(-0.2, 4.2)
ax1.set_xticks([0, 300])
ax1.set_yticks([0, 2, 4])
ax1.legend(fontsize=8)

ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes, fontsize=10, va='top', ha='right')

ax2 = fig.add_subplot(gs[0, 1])

ax2.imshow(W_1, cmap="coolwarm", aspect="auto", vmax=1, vmin=-1)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.text(-0.1, 1, '(b)', transform=ax2.transAxes, fontsize=10, va='top', ha='right')

ax3 = fig.add_subplot(gs[0, 2])

ax3.imshow(W_2, cmap="coolwarm", aspect="auto", vmax=1, vmin=-1)
ax3.set_xticks([])
ax3.set_yticks([])

ax3.text(-0.1, 1, '(c)', transform=ax3.transAxes, fontsize=10, va='top', ha='right')

ax4 = fig.add_subplot(gs[1, 1])

ax4.imshow(m, cmap="coolwarm", aspect="auto", vmax=1, vmin=-1)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(-0.1, 1, '(d)', transform=ax4.transAxes, fontsize=10, va='top', ha='right')

ax5 = fig.add_subplot(gs[1, 2])

ax5.imshow(v, cmap="coolwarm", aspect="auto", vmax=1, vmin=-1)
ax5.set_xticks([])
ax5.set_yticks([])

ax5.text(-0.1, 1, '(e)', transform=ax5.transAxes, fontsize=10, va='top', ha='right')

cbar_ax = fig.add_axes([0.57, 0.1, 0.4, 0.04])  

cbar = fig.colorbar(ax2.images[0], cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=8, labelbottom=True, labeltop=False)

plt.savefig(f"{folder}/{folder}.pdf")