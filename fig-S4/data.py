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

def generate_X(p, n, m, c, sigma, device):

    X = torch.normal(0, sigma, size=(p, n, m), device=device)

    row_cov = torch.full((m, m), c, device=device)
    row_cov.fill_diagonal_(1) 

    L = torch.linalg.cholesky(row_cov)

    X = torch.einsum('ijk,kl->ijl', X, L) 

    return X

class newDataset:
    def __init__(self,
                 P: int, D: int, N: int,
                 c: float, sigma_x=1, sigma_w=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ):

        self.device = device

        self.P = P
        self.D = D
        self.N = N

        self.sigma_x = sigma_x
        self.sigma_w = sigma_w
        self.c = c

        self.data = None
        self.weight = None
        self.input_matrix = None
        self.label = None

    def generate_dataset(self):
        
        self.data = generate_X(self.P, self.D, self.N+1, self.c, self.sigma_x, self.device)
        self.weight = torch.normal(0, self.sigma_w, size=(self.P, 1, self.D), device=self.device)
        label = torch.matmul(self.weight, self.data) / np.sqrt(self.D)
        input_matrix = torch.cat([self.data, label], dim=1)
        self.label = label[:, -1, -1].clone()
        input_matrix[:, -1, -1] = 0
        self.input_matrix = input_matrix

    def get_field(self, λ0: float):
        if self.input_matrix is None:
            self.generate_dataset()

        C = torch.bmm(self.input_matrix, self.input_matrix.permute(0, 2, 1))
        
        v1 = C[:, -1, :].unsqueeze(2)
        v2 = self.input_matrix[:, :, -1].unsqueeze(1)
        s_mn = torch.bmm(v1, v2) / (self.N + 1)
        
        s_i = s_mn.reshape(self.P, -1, 1)
        s_j = s_mn.reshape(self.P, 1, -1)
        self.J = -torch.bmm(s_i, s_j).mean(dim=0)
        
        self.h = (self.label.reshape(-1, 1, 1) * s_mn).mean(dim=0).reshape(-1)
        
        diag_J = torch.diagonal(self.J)
        self.λ = λ0 - diag_J


def main(P_train, P_test, D, N, c_list, repeat, wd, 
         sigma_x, sigma_w, lr, epochs, 
         steps, β, θ, showlog):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    amp_loss = np.zeros((len(c_list), repeat))
    sgd_loss = np.zeros((len(c_list), repeat))

    for i, c in enumerate(c_list):
        for r in range(repeat):
            train_dataset = newDataset(P_train, D, N, c, sigma_x, sigma_w, device)
            train_dataset.generate_dataset()
            test_dataset = newDataset(P_test, D, N, c, sigma_x, sigma_w, device)
            test_dataset.generate_dataset()

            criterion = nn.MSELoss()

            model = MergedSimpLinearAttn(D+1)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            _, sgd_loss[i, r] = train(model, optimizer, criterion, train_dataset.input_matrix, train_dataset.label, test_dataset.input_matrix, test_dataset.label, epochs)

            train_dataset.get_field(wd)

            m_list, _, _, error = AMP_iteration(
                train_dataset.J, train_dataset.h, train_dataset.λ,
                β, θ, steps, device, showlog)

            W = m_list.reshape(D+1, D+1)

            if error > 0.000005:
                amp_loss[i, r] = np.nan
            else:
                amp_loss[i, r] = AMP_test(W, test_dataset.input_matrix, test_dataset.label)

            del train_dataset, test_dataset

            print(f"c: {c}, r: {r+1}, AMP loss: {amp_loss[i, r]}, AMP error: {error}, SGD loss: {sgd_loss[i, r]}")
        
        print("-" * 50)
        print(f"c: {c}, AMP loss mean: {amp_loss[i].mean()}")
        print(f"c: {c}, SGD loss mean: {sgd_loss[i].mean()}")
        print("-" * 50)

    np.save(f"{data_path}/amp_loss.npy", amp_loss)
    np.save(f"{data_path}/sgd_loss.npy", sgd_loss)


if __name__ == "__main__":
    P_train = 10000
    P_test = 1000
    D = 20
    N = 200
    c_list = np.linspace(0.05, 0.95, 19)
    print(f"c_list: {c_list}")
    repeat = 20
    wd = 0.01
    sigma_x = 1
    sigma_w = 1
    lr = 0.001
    epochs = 5000
    steps = 200
    β = 100
    θ = 0.9
    showlog = False

    main(P_train, P_test, D, N, c_list, repeat, wd, 
         sigma_x, sigma_w, lr, epochs, 
         steps, β, θ, showlog)