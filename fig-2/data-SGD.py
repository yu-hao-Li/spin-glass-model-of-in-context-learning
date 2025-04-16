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

###################################################

set_seed(482)

D=10
N=100
P_list=[10, 1000]
P_test=200
sigma_x=1
sigma_w=1
lr=0.01
epochs=300
wd = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

for i, p in enumerate(P_list):
    print("--------------------")
    print(f"P: {p}")

    model = MergedSimpLinearAttn(D + 1)

    dataset = Dataset("experiment", device, (p, P_test), D, N, sigma_x, sigma_w)
    dataset.get_dataset()

    criterion = nn.MSELoss()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_losses, test_losses = train_single(model, optimizer, criterion, dataset.input_matrix_train, dataset.label_train, dataset.input_matrix_test, dataset.label_test, epochs)

    print("end train loss:", train_losses[-1])
    np.save(os.path.join(data_path, f"loss-{i}.npy"), test_losses)

    W = model.W.detach().cpu().numpy()
    np.save(os.path.join(data_path, f"W-{i}.npy"), W)
