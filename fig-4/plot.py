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

contrast_ratio = np.load(os.path.join(data_path, "contrast.npy"))

data = np.load(os.path.join(data_path, "data-b.npy"), allow_pickle=True)

N_list = np.load(os.path.join(data_path, "N_list.npy"))
sgd_loss_mean = np.load(os.path.join(data_path, "sgd_loss_mean.npy"))
sgd_loss_std = np.load(os.path.join(data_path, "sgd_loss_std.npy"))
amp_loss_mean = np.load(os.path.join(data_path, "amp_loss_mean.npy"))
amp_loss_std = np.load(os.path.join(data_path, "amp_loss_std.npy"))

###################################################

fig, ax = plt.subplots(1, 2, figsize=cm2inch(8.5, 3.6), width_ratios=[1, 1], constrained_layout=True)

im = ax[0].imshow(contrast_ratio, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
ax[0].set_xticks([-0.5, 49.5])
ax[0].set_yticks([49.5])
ax[0].set_xticklabels(["0     ", 1000])
ax[0].set_yticklabels([100])
ax[0].get_yticklabels()[0].set_va('top')
ax[0].set_xlabel(r"$P$", labelpad=0)
ax[0].set_ylabel(r"$N$", labelpad=-2)
ax[0].invert_yaxis()
cbar = plt.colorbar(im, ax=ax[0], ticks=[0, 1], pad=-0.08)
cbar.ax.set_yticklabels(['0', '1'], va='bottom')
cbar.ax.get_yticklabels()[-1].set_va('top')

ax[1].plot(N_list, amp_loss_mean, label="AMP", color="#1c3c63")
ax[1].errorbar(N_list, sgd_loss_mean, yerr=sgd_loss_std, fmt='o', label="SGD", markersize=4, capsize=2, c="#43A3EF")
ax[1].set_xlabel(r"$N$", labelpad=0)
ax[1].set_ylabel("Loss", labelpad=-2)
ax[1].set_ylim(0, 0.6)
ax[1].set_xticks([0, 200])
ax[1].set_yticks([0.6])
ax[1].set_xticklabels(["0     ", 200])
ax[1].set_yticklabels([0.6])
ax[1].get_yticklabels()[0].set_va('top')
ax[1].legend(fontsize=8)

ax[0].text(-0.28, 1.01, "(a)", transform=ax[0].transAxes, fontsize=10, va="top", ha="right")
ax[1].text(-0.28, 1.01, "(b)", transform=ax[1].transAxes, fontsize=10, va="top", ha="right")

plt.savefig(f"{folder}/{folder}.pdf")
