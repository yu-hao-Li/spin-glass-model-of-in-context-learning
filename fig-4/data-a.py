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

def main(P_list, D, N_list, repeat, wd, 
         sigma_x, sigma_w, steps, β, θ, showlog):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    contrast_ratio = np.zeros((len(N_list), len(P_list)),)

    for i, N in enumerate(N_list):

        for j, P in enumerate(P_list):

            contrast_ratio_list = []

            for r in range(repeat):

                dataset = Dataset("theory", device, P, D, N, sigma_x, sigma_w)
                dataset.get_field(wd)

                m_list, _, _, error = AMP_iteration(dataset.J, dataset.h, dataset.λ, β, θ, steps, device, showlog)

                if error < 1e-6:
                    W = m_list.reshape((D+1, D+1)).cpu().numpy()
                    contrast_ratio_list.append(calculate_contrast_ratio(W))
                else:
                    contrast_ratio_list.append(np.nan)

                print(f"N: {N}, P: {P}, repeat: {r+1}, contrast ratio: {contrast_ratio_list[-1]}")
            
            contrast_ratio[i, j] = np.nanmean(contrast_ratio_list)
            print("-"*20)
            print(f"N: {N}, P: {P}, contrast ratio: {contrast_ratio[i, j]}")
            print("-"*20)

    np.save(os.path.join(data_path, "contrast.npy"), contrast_ratio)


if __name__ == "__main__":

    main(
        P_list=np.linspace(20, 1000, 50, dtype=int),
        D=20,
        N_list=np.linspace(2, 100, 50, dtype=int),
        repeat=10,
        wd=1,
        sigma_x=1,
        sigma_w=1,
        steps=100,
        β=100,
        θ=0.9,
        showlog=False
    )