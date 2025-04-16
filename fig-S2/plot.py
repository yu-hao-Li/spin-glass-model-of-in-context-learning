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

h1_mean = np.load(os.path.join(data_path, "h1_mean.npy"))[1:]
h2_mean = np.load(os.path.join(data_path, "h2_mean.npy"))[1:]
J1_mean = np.load(os.path.join(data_path, "J1_mean.npy"))[1:]
J2_mean = np.load(os.path.join(data_path, "J2_mean.npy"))[1:]
J3_mean = np.load(os.path.join(data_path, "J3_mean.npy"))[1:]

h1_std = np.load(os.path.join(data_path, "h1_std.npy"))[1:]
h2_std = np.load(os.path.join(data_path, "h2_std.npy"))[1:]
J1_std = np.load(os.path.join(data_path, "J1_std.npy"))[1:]
J2_std = np.load(os.path.join(data_path, "J2_std.npy"))[1:]
J3_std = np.load(os.path.join(data_path, "J3_std.npy"))[1:]

N_list = np.array([21, 40, 60, 80, 100, 120, 140, 160, 180, 200])

N_list = N_list / 20

###################################################

plt.figure(figsize=cm2inch(8.5, 5.5), constrained_layout=True)

plt.fill_between(N_list, h1_mean-h1_std, h1_mean+h1_std, alpha=0.1)
plt.fill_between(N_list, h2_mean-h2_std, h2_mean+h2_std, alpha=0.1)
plt.fill_between(N_list, J1_mean-J1_std, J1_mean+J1_std, alpha=0.1)
plt.fill_between(N_list, J2_mean-J2_std, J2_mean+J2_std, alpha=0.1)
plt.fill_between(N_list, J3_mean-J3_std, J3_mean+J3_std, alpha=0.1)

plt.plot(N_list, h1_mean, 'o-', label=r"$\mathcal{A}$", markersize=5)
plt.plot(N_list, h2_mean, 'o-', label=r"$\mathcal{B}$", markersize=5)
plt.plot(N_list, J1_mean, 'o-', label=r"$\mathcal{C}$", markersize=5)
plt.plot(N_list, J2_mean, 'o-', label=r"$\mathcal{D}$", markersize=5)
plt.plot(N_list, J3_mean, 'o-', label=r"$\mathcal{E}$", markersize=5)

plt.legend(fontsize=9)

plt.xlabel(r"$\Delta N / N$", fontsize=10)
plt.ylabel(r"$\mathrm{D}_{\mathrm{KL}}~[\,P_N\,||~P_{N+\Delta N}\,]$", fontsize=10)


###################################################

plt.savefig(f"{folder}/{folder}.pdf")
