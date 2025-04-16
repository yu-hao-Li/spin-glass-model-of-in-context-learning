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
from src.model import *
from src.dataset import *
from src.theory import *

###################################################

D = 100
N_list = [100, 500, 1000, 2000, 5000, 10000]

eigenvalues_list = np.load(os.path.join(data_path, "eigenvalues.npy"))

plt.figure(figsize=cm2inch(8.5, 5.5), constrained_layout=True)
for i, eigenvalues in enumerate(eigenvalues_list):
    a = int(N_list[i] / D)
    plt.hist(eigenvalues, bins=200, density=True, alpha=0.5, label=r"$N/D={}$".format(a))

plt.legend()
plt.xlabel("Eigenvalues")
plt.ylabel("Density")

plt.savefig(f"{folder}/{folder}.pdf")