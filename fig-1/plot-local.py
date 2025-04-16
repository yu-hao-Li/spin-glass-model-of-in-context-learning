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

J_matrix = np.load(os.path.join(data_path, "J-matrix.npy"))
h_matrix = np.load(os.path.join(data_path, "h-matrix.npy"))

J1 = np.load(os.path.join(data_path, "J1.npy"))
J2 = np.load(os.path.join(data_path, "J2.npy"))
J3 = np.load(os.path.join(data_path, "J3.npy"))

h1 = np.load(os.path.join(data_path, "h1.npy"))
h2 = np.load(os.path.join(data_path, "h2.npy"))

###################################################

def Gaussian(x, sigma, mu=0):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

###################################################

fig = plt.figure(figsize=cm2inch(10, 3), constrained_layout=True)

gs = fig.add_gridspec(nrows=1, ncols=5)

gray = "#97999a"

###################################################

ax3 = fig.add_subplot(gs[0, 0])
ax3.hist(h1, bins=400, color=gray, density=True)
ax3.set_xlim(-0.2, 0.4)
ax3.set_xticks([-0.1, 0.3])

x_1 = np.linspace(-0.2, 0.4, 100)
y_1 = 0.82 * Gaussian(x_1, 0.02) + 0.18 * Gaussian(x_1, 0.02, 0.2)
ax3.plot(x_1, y_1, color="red", lw=0.5)

ax3.text(0.95, 0.95, r"$\mathcal{A}$", color='red', transform=ax3.transAxes, fontsize=10, va="top", ha="right")

###################################################

ax4 = fig.add_subplot(gs[0, 1])
ax4.hist(h2, bins=400, color=gray, density=True)
ax4.set_xlim(-0.3, 0.3)
ax4.set_xticks([-0.2, 0.2])

x_2 = np.linspace(-0.3, 0.3, 100)
y_2 = Gaussian(x_2, 0.06)
ax4.plot(x_2, y_2, color="red", lw=0.5)

ax4.text(0.95, 0.95, r"$\mathcal{B}$", color='red', transform=ax4.transAxes, fontsize=10, va="top", ha="right")

###################################################

ax5 = fig.add_subplot(gs[0, 2])
ax5.hist(J1, bins=400, color=gray, density=True)
ax5.set_xlim(-0.1, 0.1)
ax5.set_xticks([-0.06, 0.06])

x_3 = np.linspace(-0.1, 0.1, 100)
y_3 = Gaussian(x_2, 0.024) * 3
ax5.plot(x_3, y_3, color="red", lw=0.5)

ax5.text(0.95, 0.95, r"$\mathcal{C}$", color='red', transform=ax5.transAxes, fontsize=10, va="top", ha="right")

###################################################

ax6 = fig.add_subplot(gs[0, 3])
ax6.hist(J2, bins=400, color=gray, density=True)
ax6.set_xlim(-0.3, 0.3)
ax6.set_xticks([-0.2, 0.2])

x_4 = np.linspace(-0.3, 0.3, 100)
y_4 = Gaussian(x_4, 0.027) * 1.04
ax6.plot(x_4, y_4, color="red", lw=0.5)

ax6.text(0.95, 0.95, r"$\mathcal{D}$", color='red', transform=ax6.transAxes, fontsize=10, va="top", ha="right")

###################################################

ax7 = fig.add_subplot(gs[0, 4])
ax7.hist(J3, bins=400, color=gray, density=True)
ax7.set_xlim(-0.55, 0.55)

x_5 = np.linspace(-0.55, 0.55, 100)
y_5 = Gaussian(x_5, 0.08) * 1.04
ax7.plot(x_5, y_5, color="red", lw=0.5)

ax7.text(0.95, 0.95, r"$\mathcal{E}$", color='red', transform=ax7.transAxes, fontsize=10, va="top", ha="right")

###################################################

for ax in [ax3, ax4, ax5, ax6, ax7]:
    ax.tick_params(axis='both', labelsize=8, length=2)
    ax.set_yticks([0, 1])
    ax.set_ylim(0, 1.05)

###################################################

plt.savefig("fig-1/fig-1-local.pdf")
