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
from src.dataset import *

###################################################

def generate_matrix(P, D, N):
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {dev}")

    dataset = Dataset('theory', dev, P, D, N, 1, 1)
    dataset.get_field(0)

    J = dataset.J.cpu()
    h = dataset.h.cpu().reshape(D+1, D+1)

    np.save(os.path.join(data_path, "J-matrix.npy"), J.numpy())
    np.save(os.path.join(data_path, "h-matrix.npy"), h.numpy())

def generate_field(P, D, N):

    w = torch.normal(0, 1, size=(P, 1, D))
    x = torch.normal(0, 1, size=(P, D, N))
    xt = torch.normal(0, 1, size=(P, D, 1))

    y = w @ x / np.sqrt(D)    # P x 1 x N
    yt = w @ xt / np.sqrt(D)   # P x 1 x 1

    c1 = x @ y.permute(0, 2, 1) / N   # P x D x 1
    s1 = xt @ c1.permute(0, 2, 1)     # P x D x D
    h1 = yt * s1                      # P x D x D
    h1 = h1.mean(dim=0)               # D x D

    c2 = y.squeeze(1).pow(2).sum(dim=1, keepdim=True) / N  # P x 1
    s2 = xt.squeeze(-1) * c2                               # P x D
    h2 = yt.squeeze(-1) * s2                               # P x D
    h2 = h2.mean(dim=0).unsqueeze(0)                       # 1 x D

    h_zero = torch.zeros((D+1, 1))
    h = torch.cat((h1, h2), dim=0)
    h = torch.cat((h, h_zero), dim=1)

    s2 = s2.unsqueeze(1)
    s_zero = torch.zeros((P, D+1, 1))
    s = torch.cat((s1, s2), dim=1)
    s = torch.cat((s, s_zero), dim=2)

    J = - s.reshape(P, -1, 1) @ s.reshape(P, 1, -1)
    J = J.mean(dim=0)

    return J.cpu(), h.cpu()


def generate_distribution(P, D, N, repeat):

    h1 = []
    h2 = []
    J1 = []
    J2 = []
    J3 = []

    for _ in tqdm(range(repeat)):
        J, h = generate_field(P, D, N)

        h1 += h[:D, :D].reshape(-1).tolist()
        h2 += h[D, :D].reshape(-1).tolist()

        J1_full = J[:-D-1, :-D-1].clone()
        J1_no_diag = J1_full[~torch.eye(D*(D+1), dtype=bool)]
        J1_no_diag_no_zero = J1_no_diag[J1_no_diag != 0]
        J1 += J1_no_diag_no_zero.reshape(-1).tolist()

        J2_1 = J[-D-1:, :-D-1].clone()
        J2_2 = J[:-D-1, -D-1:].clone()
        J2_1_no_zero = J2_1[J2_1 != 0]
        J2_2_no_zero = J2_2[J2_2 != 0]
        J2 += J2_1_no_zero.reshape(-1).tolist()
        J2 += J2_2_no_zero.reshape(-1).tolist()

        J3_full = J[-D-1:, -D-1:].clone()
        J3_no_diag = J3_full[~torch.eye(D+1, dtype=bool)]
        J3_no_diag_no_zero = J3_no_diag[J3_no_diag != 0]
        J3 += J3_no_diag_no_zero.reshape(-1).tolist()

    np.save(os.path.join(data_path, "h1.npy"), h1)
    np.save(os.path.join(data_path, "h2.npy"), h2)
    np.save(os.path.join(data_path, "J1.npy"), J1)
    np.save(os.path.join(data_path, "J2.npy"), J2)
    np.save(os.path.join(data_path, "J3.npy"), J3)

###################################################

if __name__ == "__main__":

    set_seed(42)

    P = 1000
    D = 5
    N = 100
    repeat = 100000

    generate_matrix(P, D, N)
    generate_distribution(P, D, N, repeat)