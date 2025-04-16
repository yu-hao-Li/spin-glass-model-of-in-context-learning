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

N = 100
D = 20
P_list = [100, 500, 1500]
repeat = 2000


for P in P_list:

   print(f"P: {P}")

   eigenvalues_list = []

   for _ in tqdm(range(repeat)):

      s = generate_s(P, D, N)
      H = s.reshape(P, -1, 1) @ s.reshape(P, 1, -1)
      H = H.mean(dim=0)
      eigenvalues = torch.linalg.eigvalsh(H)
      eigenvalues_list.append(eigenvalues.cpu())

   eigenvalues_list = torch.stack(eigenvalues_list)
   eigenvalues_list = eigenvalues_list.flatten().numpy()

   np.save( os.path.join(data_path, f"P={P}.npy"), eigenvalues_list)
