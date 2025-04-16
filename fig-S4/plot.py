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

c_list = np.linspace(0.05, 0.95, 19)

sgd_loss = np.load(f"{data_path}/sgd.npy")
sgd_loss_mean = sgd_loss.mean(axis=1)
sgd_loss_std = sgd_loss.std(axis=1)

amp_loss = np.load(f"{data_path}/amp.npy")
amp_loss_mean = np.nanmean(amp_loss, axis=1)
amp_loss_std = np.nanstd(amp_loss, axis=1)

plt.figure(figsize=cm2inch(8.5, 5.5), constrained_layout=True)
plt.plot(c_list, amp_loss_mean, color="#1c3c63", label="AMP", linewidth=1)
plt.errorbar(c_list, sgd_loss_mean, yerr=sgd_loss_std, fmt='o', markersize='3', label="SGD", capsize=1.6)

plt.xlabel(r"$c$", fontsize=10)
plt.ylabel("Test error", fontsize=10)
plt.legend()
plt.savefig(f"{folder}/{folder}.pdf")
