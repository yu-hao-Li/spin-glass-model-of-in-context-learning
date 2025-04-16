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
from src.model import *
from src.dataset import *
from src.theory import *

###################################################

P = 5000
D = 10
N = 100
wd = 0.01

sigma_x=1
sigma_w=1
steps=100
β=100
θ=0.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

dataset = Dataset("theory", device, P, D, N, sigma_x, sigma_w)
dataset.get_field(wd)

m_list, v_list, _, _ = AMP_iteration(dataset.J, dataset.h, dataset.λ, β, θ, steps, device, showlog=True)

m = m_list.reshape((D+1, D+1)).cpu().numpy()
v = v_list.reshape((D+1, D+1)).cpu().numpy()

np.save(os.path.join(data_path, "m.npy"), m)
np.save(os.path.join(data_path, "v.npy"), v)
