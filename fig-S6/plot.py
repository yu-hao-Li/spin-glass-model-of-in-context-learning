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

repeat = 5000
losses = np.load(os.path.join(data_path, "losses.npy"))
errors = np.load(os.path.join(data_path, "errors.npy"))

plt.figure(figsize=cm2inch(8.5, 5.5), constrained_layout=True)
for i in range(0, 50):
    plt.plot(errors[i, :], lw=0.5)
plt.xlabel("Epochs")
plt.ylabel("Distance")

ax = plt.gca()
inset_ax = inset_axes(ax, width="50%", height="50%", loc="upper right")
errs = errors[:, -1]

inset_ax.hist(errs, bins=30, density=True, alpha=0.5)
inset_ax.set_xlabel("Distance", fontsize=8)
inset_ax.set_ylabel("Density", fontsize=8)
inset_ax.tick_params(axis='both', which='major', labelsize=6)
inset_ax.set_xlim(0, 0.3)

plt.savefig(f"{folder}/{folder}.pdf")
