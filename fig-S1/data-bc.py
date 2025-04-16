# -*- coding: utf-8 -*-

###################################################
# path
import os, sys

folder = os.path.basename(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

# packages
from src.utils import *
from src.model import *
from src.dataset import *

###################################################

def main(P_train, P_test, D, N, sigma_x, sigma_w, lr, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    dataset = Dataset("experiment", device, (P_train, P_test), D, N, sigma_x, sigma_w)
    dataset.get_dataset()

    criterion = nn.MSELoss()

    for model_name in ["Separate", "Merged"]:
        if model_name == "Separate":
            model = SeparateSimpLinearAttn(D+1)
            print("Separate")
        else:
            model = MergedSimpLinearAttn(D+1)
            print("Merged")

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses, test_losses = train_single(model, optimizer, criterion, dataset.input_matrix_train, dataset.label_train, dataset.input_matrix_test, dataset.label_test, epochs)

        if model_name == "Separate":
            separate_train_losses = train_losses
            separate_test_losses = test_losses

            np.save(os.path.join(data_path, "separate_W_q.npy"), model.W_q.cpu().detach().numpy())
            np.save(os.path.join(data_path, "separate_W_k.npy"), model.W_k.cpu().detach().numpy())
        else:
            merged_train_losses = train_losses
            merged_test_losses = test_losses

            np.save(os.path.join(data_path, "merged_W.npy"), model.W.cpu().detach().numpy())

    np.save(os.path.join(data_path, "separate_train_losses.npy"), separate_train_losses)
    np.save(os.path.join(data_path, "separate_test_losses.npy"), separate_test_losses)
    np.save(os.path.join(data_path, "merged_train_losses.npy"), merged_train_losses)
    np.save(os.path.join(data_path, "merged_test_losses.npy"), merged_test_losses)

    print("Training Done!")

###################################################

if __name__ == "__main__":

    set_seed(12345)

    data_path = f"{folder}/data-b"
    os.makedirs(data_path, exist_ok=True)

    main(P_train=120,
        P_test=1000, 
        D=20,
        N=100, 
        sigma_x=1, 
        sigma_w=1, 
        lr=0.008, 
        epochs=200
    )

    data_path = f"{folder}/data-c"
    os.makedirs(data_path, exist_ok=True)

    main(P_train=680,
        P_test=1000, 
        D=20,
        N=100, 
        sigma_x=1, 
        sigma_w=1, 
        lr=0.02, 
        epochs=200
    )
