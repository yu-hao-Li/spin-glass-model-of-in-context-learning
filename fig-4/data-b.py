# -*- coding: utf-8 -*-

###################################################
# path
import os, sys
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

data_path = os.path.join(path, "fig-4")
os.makedirs(data_path, exist_ok=True)

# packages
from src.utils import *
from src.model import *
from src.dataset import *
from src.theory import *

###################################################

def main(P_train, P_test, D, N_list, repeat, wd, 
         sigma_x, sigma_w, lr, epochs, 
         steps, β, θ, showlog):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    print("N_list:", N_list)

    amp_loss_mean = np.zeros((len(N_list)),)
    amp_loss_std = np.zeros((len(N_list)),)
    amp_error_mean = np.zeros((len(N_list)),)

    sgd_loss_mean = np.zeros((len(N_list)),)
    sgd_loss_std = np.zeros((len(N_list)),)

    for i, N in enumerate(N_list):

        amp_loss_list = []
        amp_error_list = []
        sgd_loss_list = []

        for r in range(repeat):

            dataset = Dataset("experiment", device, (P_train, P_test), D, N, sigma_x, sigma_w)
            dataset.get_dataset()

            criterion = nn.MSELoss()

            model = MergedSimpLinearAttn(D+1)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            _, sgd_loss = train(model, optimizer, criterion, dataset.input_matrix_train, dataset.label_train, dataset.input_matrix_test, dataset.label_test, epochs)

            input_matrix_test = dataset.input_matrix_test
            label_test = dataset.label_test

            del dataset

            dataset = Dataset("theory", device, P_train, D, N, sigma_x, sigma_w)
            dataset.get_field(wd)

            m_list, _, _, error = AMP_iteration(
                dataset.J, dataset.h, dataset.λ, 
                β, θ, steps, device, showlog)

            W = m_list.reshape((D+1, D+1))

            amp_loss = AMP_test(W, input_matrix_test, label_test)

            amp_loss_list.append(amp_loss)
            amp_error_list.append(error)
            sgd_loss_list.append(sgd_loss)

            print(f"N: {N}, Repeat: {r+1}, SGD Loss: {sgd_loss:.4f}, AMP Loss: {amp_loss:.4f}, AMP Error: {error:.4f}")

        amp_loss_mean[i] = np.mean(amp_loss_list)
        amp_loss_std[i] = np.std(amp_loss_list)
        amp_error_mean[i] = np.mean(amp_error_list)

        sgd_loss_mean[i] = np.mean(sgd_loss_list)
        sgd_loss_std[i] = np.std(sgd_loss_list)

        print("-"*20)
        print(f"N: {N}, SGD Loss: {sgd_loss_mean[i]:.4f} ± {sgd_loss_std[i]:.4f}, AMP Loss: {amp_loss_mean[i]:.4f} ± {amp_loss_std[i]:.4f}, AMP Error: {amp_error_mean[i]:.4f}")
        print("-"*20)

    np.save(os.path.join(data_path, "N_list.npy"), N_list)
    np.save(os.path.join(data_path, "sgd_loss_mean.npy"), sgd_loss_mean)
    np.save(os.path.join(data_path, "sgd_loss_std.npy"), sgd_loss_std)
    np.save(os.path.join(data_path, "amp_loss_mean.npy"), amp_loss_mean)
    np.save(os.path.join(data_path, "amp_loss_std.npy"), amp_loss_std)
    

if __name__ == "__main__":

    main(
        P_train=10000,
        P_test=1000,
        D=20,
        N_list=np.linspace(10, 200, 20, dtype=int),
        repeat=50,
        wd=0.01,
        sigma_x=1,
        sigma_w=1,
        lr=1e-3,
        epochs=5000,
        steps=100,
        β=100,
        θ=0.9,
        showlog=False
    )
