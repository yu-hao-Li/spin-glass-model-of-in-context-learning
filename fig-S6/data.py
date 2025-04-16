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

class LinearEquationGroup(nn.Module):
    def __init__(self, D, N):
        super(LinearEquationGroup, self).__init__()
        self.W11 = nn.Parameter(torch.normal(0, 1, (D, D)))
        self.W21 = nn.Parameter(torch.normal(0, 1, (1, D)))
        X = torch.normal(0, 1, (D, N))
        C = (torch.matmul(X, X.T) / N)
        self.register_buffer("C", C)

    def forward(self, w):
        
        return ( w @ self.W21 / np.sqrt(D) + self.W11 ) @ self.C
    

def main(P, D, N, repeat, epochs, device):

    ones = torch.eye(D).to(device)
    labels = torch.eye(D).repeat(P, 1, 1).to(device)
    zeors = torch.zeros((1, D)).to(device)

    losses = np.zeros((repeat, epochs))
    errors = np.zeros((repeat, epochs))

    criterion = nn.MSELoss()

    for r in tqdm(range(repeat)):
        
        w = torch.normal(0, 1, (P, D, 1)).to(device)
        model = LinearEquationGroup(D, N).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for step in range(epochs):
            
            output = model(w)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[r, step] = loss.item()

            W_11 = model.W11.detach()
            W_21 = model.W21.detach()

            err_1 = criterion(W_11, ones)
            err_2 = criterion(W_21, zeors)

            errors[r, step] = err_1.item() + err_2.item()

    np.save(os.path.join(data_path, "losses.npy"), losses)
    np.save(os.path.join(data_path, "errors.npy"), errors)

    print("Data saved.")


if __name__ == "__main__":

    device = torch.device('cuda')
    print(f"device: {device}")

    P = 10
    D = 10
    N = 50
    repeat = 5000
    epochs = 1000

    main(P, D, N, repeat, epochs, device)
