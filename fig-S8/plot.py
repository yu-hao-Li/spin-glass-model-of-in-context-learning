# -*- coding: utf-8 -*-

###################################################
# path
import os, sys

folder = os.path.basename(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

data_path = os.path.join(path, "fig-S8/data")

# packages
from src.utils import *
###################################################

AMP_N = np.load(f"{data_path}/AMP_N.npy")
AMP_mean = np.load(f"{data_path}/AMP_mean.npy")
AMP_std = np.load(f"{data_path}/AMP_std.npy")

SLA_N = np.load(f"{data_path}/SLA_N.npy")
SLA_mean = np.load(f"{data_path}/SLA_mean.npy")
SLA_std = np.load(f"{data_path}/SLA_std.npy")

FLA_N = np.load(f"{data_path}/FLA_N.npy")
FLA_mean = np.load(f"{data_path}/FLA_mean.npy")
FLA_std = np.load(f"{data_path}/FLA_std.npy")

SA_N = np.load(f"{data_path}/SA_N.npy")
SA_mean = np.load(f"{data_path}/SA_mean.npy")
SA_std = np.load(f"{data_path}/SA_std.npy")

TF_N = np.load(f"{data_path}/TF_N.npy")
TF_mean = np.load(f"{data_path}/TF_mean.npy")
TF_std = np.load(f"{data_path}/TF_std.npy")

MHLA_N = np.load(f"{data_path}/MHLA_N.npy")
MHLA_mean = np.load(f"{data_path}/MHLA_mean.npy")
MHLA_std = np.load(f"{data_path}/MHLA_std.npy")

###################################################

colors = ["#00b0F0", "#92D050", "#F2A900", "#4472C4", "#9355B0"]

fig = plt.figure(figsize=cm2inch(8.5, 5.5), constrained_layout=True)

plt.plot(AMP_N, AMP_mean, label="AMP", c="#1c3c63", linewidth=1, zorder=1)

plt.errorbar(SLA_N, SLA_mean, yerr=SLA_std, fmt='o', markersize='3', label="SLA", capsize=1.6, zorder=2, color=colors[0])
plt.errorbar(FLA_N, FLA_mean, yerr=FLA_std, fmt='o', markersize='3', label="FLA", capsize=1.6, zorder=3, color=colors[1])
plt.errorbar(MHLA_N, MHLA_mean, yerr=MHLA_std, fmt='o', markersize='3', label="MHLA", capsize=1.6, zorder=4, color=colors[2])
plt.errorbar(SA_N, SA_mean, yerr=SA_std, fmt='o', markersize='3', label="SA", capsize=1.6, zorder=5, color=colors[3])
plt.errorbar(TF_N, TF_mean, yerr=TF_std, fmt='o', markersize='3', label="TF", capsize=1.6, zorder=6, color=colors[4])

plt.legend(ncol=2, columnspacing=0.8, fontsize=9)
plt.xlabel(r"$N$", fontsize=10)
plt.ylabel("Test error", fontsize=10)
plt.savefig(f"{folder}/{folder}.pdf")

