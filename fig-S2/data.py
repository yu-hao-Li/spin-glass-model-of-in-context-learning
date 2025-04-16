# -*- coding: utf-8 -*-

###################################################
# path
import os, sys

folder = os.path.basename(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

data_path = f"{folder}/data"
os.makedirs(data_path, exist_ok=True)

# packages
import scipy
import pandas as pd

from src.utils import *
from src.model import *
from src.dataset import *
from src.theory import *

###################################################

P = 1000
D = 20
N_list = [20, 21, 40, 60, 80, 100, 120, 140, 160, 180, 200]
repeat = 1000

result_h1 = np.zeros((len(N_list), repeat, 100))
result_h2 = np.zeros((len(N_list), repeat, 100))
result_J1 = np.zeros((len(N_list), repeat, 100))
result_J2 = np.zeros((len(N_list), repeat, 100))
result_J3 = np.zeros((len(N_list), repeat, 100))

print("Start generating data...")

for n in range(len(N_list)):
    N = int(N_list[n])
    print(f"N: {N}")

    for i in tqdm(range(repeat)):
        J, h, λ = gen_field(P, D, N, 0, torch.device('cpu'))

        h1_full = h[:D, :D].clone()
        h1_no_diag = h1_full[~torch.eye(D, dtype=torch.bool)]
        h1 = h1_no_diag.reshape(-1).numpy()

        h2 = h[D, :D].reshape(-1).numpy()

        J1_full = J[:-D-1, :-D-1].clone()
        J1_no_diag = J1_full[~torch.eye(D*(D+1), dtype=bool)]
        J1_no_diag = J1_no_diag[J1_no_diag != 0]
        J1 = J1_no_diag.reshape(-1).numpy()

        J21 = J[-D-1:, :-D-1].clone()
        J22 = J[:-D-1, -D-1:].clone()
        J21 = J21[J21 != 0]
        J22 = J22[J22 != 0]
        J2 = np.concatenate((J21.reshape(-1).numpy(), J22.reshape(-1).numpy()))

        J3_full = J[-D-1:, -D-1:].clone()
        J3_no_diag = J3_full[~torch.eye(D+1, dtype=bool)]
        J3_no_diag = J3_no_diag[J3_no_diag != 0]
        J3 = J3_no_diag.reshape(-1).numpy()

        C_h1 = pd.cut(h1, bins=np.linspace(-0.2, 0.4, 101))
        C_h2 = pd.cut(h2, bins=np.linspace(-0.3, 0.3, 101))
        C_J1 = pd.cut(J1, bins=np.linspace(-0.1, 0.1, 101))
        C_J2 = pd.cut(J2, bins=np.linspace(-0.3, 0.3, 101))
        C_J3 = pd.cut(J3, bins=np.linspace(-0.5, 0.5, 101))

        result_h1[n, i, :] = np.array(C_h1.value_counts().values) + 1
        result_h2[n, i, :] = np.array(C_h2.value_counts().values) + 1
        result_J1[n, i, :] = np.array(C_J1.value_counts().values) + 1
        result_J2[n, i, :] = np.array(C_J2.value_counts().values) + 1
        result_J3[n, i, :] = np.array(C_J3.value_counts().values) + 1


print("Start calculating KL divergence...")

h1_mean = []
h2_mean = []
J1_mean = []
J2_mean = []
J3_mean = []

h1_std = []
h2_std = []
J1_std = []
J2_std = []
J3_std = []

for n in range(len(N_list)):

    KL_h1_repeat = []
    KL_h2_repeat = []
    KL_J1_repeat = []
    KL_J2_repeat = []
    KL_J3_repeat = []

    for i in range(repeat):
        KL_h1_repeat.append(scipy.stats.entropy(result_h1[0, i, :], result_h1[n, i, :], base=2))
        KL_h2_repeat.append(scipy.stats.entropy(result_h2[0, i, :], result_h2[n, i, :], base=2))
        KL_J1_repeat.append(scipy.stats.entropy(result_J1[0, i, :], result_J1[n, i, :], base=2))
        KL_J2_repeat.append(scipy.stats.entropy(result_J2[0, i, :], result_J2[n, i, :], base=2))
        KL_J3_repeat.append(scipy.stats.entropy(result_J3[0, i, :], result_J3[n, i, :], base=2))

    h1_mean.append(np.mean(KL_h1_repeat))
    h2_mean.append(np.mean(KL_h2_repeat))
    J1_mean.append(np.mean(KL_J1_repeat))
    J2_mean.append(np.mean(KL_J2_repeat))
    J3_mean.append(np.mean(KL_J3_repeat))

    h1_std.append(np.std(KL_h1_repeat))
    h2_std.append(np.std(KL_h2_repeat))
    J1_std.append(np.std(KL_J1_repeat))
    J2_std.append(np.std(KL_J2_repeat))
    J3_std.append(np.std(KL_J3_repeat))

np.save(os.path.join(data_path, "h1_mean.npy"), h1_mean)
np.save(os.path.join(data_path, "h2_mean.npy"), h2_mean)
np.save(os.path.join(data_path, "J1_mean.npy"), J1_mean)
np.save(os.path.join(data_path, "J2_mean.npy"), J2_mean)
np.save(os.path.join(data_path, "J3_mean.npy"), J3_mean)

np.save(os.path.join(data_path, "h1_std.npy"), h1_std)
np.save(os.path.join(data_path, "h2_std.npy"), h2_std)
np.save(os.path.join(data_path, "J1_std.npy"), J1_std)
np.save(os.path.join(data_path, "J2_std.npy"), J2_std)
np.save(os.path.join(data_path, "J3_std.npy"), J3_std)

print("Data generation done!")