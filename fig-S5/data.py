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

P = 10000
D = 100
N_list = [100, 500, 1000, 2000, 5000, 10000]

eigenvalues_list = []

for i, N in enumerate(N_list):
    N = int(N)
    
    X = torch.normal(0, 1, (P, D, N))
    C = torch.matmul(X, X.transpose(1, 2)) / N
    eigenvalues = torch.linalg.eigvalsh(C) 
    print(eigenvalues.shape)
    eigenvalues_list.append(eigenvalues.flatten().numpy())

np.save(os.path.join(data_path, "eigenvalues.npy"), eigenvalues_list)
