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
from src.utils import *
from src.theory import *

###################################################

P = 1000
D = 10
N = 50
λ0 = 0.01
mc_steps = 1000

x, y, z = energy_landscape(P, D, N, λ0, mc_steps)

np.save(os.path.join(data_path, f"x-2.npy"), x)
np.save(os.path.join(data_path, f"y-2.npy"), y)
np.save(os.path.join(data_path, f"z-2.npy"), z)
