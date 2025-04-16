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

###################################################

def generate_s(P, D, N):

   w = torch.normal(0, 1, size=(P, 1, D)).to(device)
   x = torch.normal(0, 1, size=(P, D, N)).to(device)
   xt = torch.normal(0, 1, size=(P, D, 1)).to(device)
   y = w @ x / np.sqrt(D) 

   c1 = x @ y.permute(0, 2, 1) / N 
   s1 = xt @ c1.permute(0, 2, 1) 
   c2 = y.squeeze(1).pow(2).sum(dim=1, keepdim=True) / N 
   s2 = (xt.squeeze(-1) * c2).unsqueeze(1) 

   s_zero = torch.zeros((P, D+1, 1)).to(device)
   s = torch.cat((s1, s2), dim=1)
   s = torch.cat((s, s_zero), dim=2)
   
   return s

###################################################

device = torch.device('cpu')

P_list = np.linspace(25, 2000, 80).astype(int)
N = 100
D_list = np.linspace(10, 40, 7).astype(int)

repeat = 10

print("D_list:", D_list)
print("P_list:", P_list)

zero_ratios = np.zeros((len(D_list), len(P_list)))

for i, D in enumerate(D_list):
   for j, P in enumerate(P_list):
      eigenvalues_list = []
      for _ in range(repeat):
         s = generate_s(P, D, N)
         H = s.reshape(P, -1, 1) @ s.reshape(P, 1, -1)
         H = H.mean(dim=0)
         eigenvalues = torch.linalg.eigvalsh(H)
         eigenvalues_list.extend(eigenvalues.cpu().numpy())

      zero_count = (np.abs(eigenvalues_list) < 1e-5).sum()
      total_count = len(eigenvalues_list)
      zero_ratio = zero_count / total_count
      zero_ratios[i, j] = zero_ratio

      print(f"D={D}, P={P}, zero_ratio={zero_ratio}")
   
np.save(os.path.join(data_path, "data-b.npy"), zero_ratios)