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
from scipy.stats import rv_continuous
###################################################

class PDF_A(rv_continuous):
    def _pdf(self, x):
        return 0.82 * (1 / (np.sqrt(2 * np.pi) * 0.02)) * np.exp(-0.5 * ((x - 0) / 0.02) ** 2) + 0.18 * (1 / (np.sqrt(2 * np.pi) * 0.02)) * np.exp(-0.5 * ((x - 0.2) / 0.02) ** 2)
    
pdf_a = PDF_A(a=-0.2, b=0.4)

class PDF_B(rv_continuous):
    def _pdf(self, x):
        return (1 / (np.sqrt(2 * np.pi) * 0.06)) * np.exp(-0.5 * ((x - 0) / 0.06) ** 2) 

pdf_b = PDF_B(a=-0.3, b=0.3)

class PDF_C(rv_continuous):
    def _pdf(self, x):
        return 3 * (1 / (np.sqrt(2 * np.pi) * 0.024)) * np.exp(-0.5 * ((x - 0) / 0.024) ** 2) 

pdf_c = PDF_C(a=-0.1, b=0.1)

class PDF_D(rv_continuous):
    def _pdf(self, x):
        return 1.04 * (1 / (np.sqrt(2 * np.pi) * 0.027)) * np.exp(-0.5 * ((x - 0) / 0.027) ** 2) 

pdf_d = PDF_D(a=-0.3, b=0.3)

class PDF_E(rv_continuous):
    def _pdf(self, x):
        return 1.04 * (1 / (np.sqrt(2 * np.pi) * 0.08)) * np.exp(-0.5 * ((x - 0) / 0.08) ** 2) 

pdf_e = PDF_E(a=-0.5, b=0.5)

###################################################

h_matrix = np.zeros((11, 11))

samples_a = pdf_a.rvs(size=1000)

samples_a_large = samples_a[samples_a > 0.1]
if len(samples_a_large) < 10:
    print("Error")

samples_a_small = samples_a[samples_a < 0.1]
if len(samples_a_small) < 121:
    print("Error")

samples_a_diag = samples_a_large[:10]
samples_a_nondiag = samples_a_small[:100]

h_matrix[:10, :10] = samples_a_nondiag.reshape(10, 10)
h_matrix[np.arange(10), np.arange(10)] = samples_a_diag

samples_b = pdf_b.rvs(size=10)

h_matrix[-1, :10] = samples_b

np.save(f"{data_path}/h_matrix.npy", h_matrix)

###################################################

J_matrix = np.zeros((121, 121))

samples_c = pdf_c.rvs(size=12100).reshape(110, 110)
samples_d_1 = pdf_d.rvs(size=1210).reshape(110, 11)
samples_d_2 = pdf_d.rvs(size=1210).reshape(11, 110)
samples_e = pdf_e.rvs(size=121).reshape(11, 11)

J_matrix[:110, :110] = samples_c
J_matrix[:110, 110:] = samples_d_1
J_matrix[110:, :110] = samples_d_2
J_matrix[110:, 110:] = samples_e

J_matrix[np.arange(121), np.arange(121)] = np.zeros((121,))

np.save(f"{data_path}/J_matrix.npy", J_matrix)

###################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

J = torch.from_numpy(J_matrix).to(device)
h_matrix = h_matrix.flatten()
h = torch.from_numpy(h_matrix).to(device)
λ = torch.from_numpy(np.full_like(h_matrix, 1)).to(device)

steps=100
β=100
θ=0.9
showlog=True

m_list, v_list, _, _ = AMP_iteration(J, h, λ, β, θ, steps, device, showlog)

m = m_list.reshape(11, 11).cpu()
v = v_list.reshape(11, 11).cpu()

np.save(f"{data_path}/m.npy", m)
np.save(f"{data_path}/v.npy", v)