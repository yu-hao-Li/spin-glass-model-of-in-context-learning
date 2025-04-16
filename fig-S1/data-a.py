# -*- coding: utf-8 -*-

###################################################
# path
import os, sys

folder = os.path.basename(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.append(path)

data_path = f"{folder}/data-a"
os.makedirs(data_path, exist_ok=True)

# packages
from src.utils import *
from src.model import *
from src.dataset import *

###################################################

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 
          0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 
          1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.]
print(alphas)

P_test = 1000
D = 20
N = 100
sigma_x = 1
sigma_w = 1
lr = 0.01
epochs = 3000
repeat = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

###################################################

separate_test_mean = []
separate_test_std = []
merged_test_mean = []
merged_test_std = []

for alpha in alphas:
    P_train = int(alpha * D ** 2)

    print("--------------------")
    print(f"alpha: {alpha}, P_train: {P_train}")

    separate_train_losses = []
    separate_test_losses = []
    merged_train_losses = []
    merged_test_losses = []

    for model_name in ["Separate", "Merged"]:

        for _ in range(repeat):
            dataset = Dataset("experiment", device, (P_train, P_test), D, N, sigma_x, sigma_w)
            dataset.get_dataset()

            criterion = nn.MSELoss()

            if model_name == "Separate":
                model = SeparateSimpLinearAttn(D + 1)
            else:
                model = MergedSimpLinearAttn(D + 1)

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_loss, test_loss = train(model, optimizer, criterion, dataset.input_matrix_train, dataset.label_train, dataset.input_matrix_test, dataset.label_test, epochs)

            print(f"{model_name} | train_loss: {train_loss:.3f} test_loss: {test_loss:.3f}")

            if model_name == "Separate":
                separate_train_losses.append(train_loss)
                separate_test_losses.append(test_loss)    
            else:
                merged_train_losses.append(train_loss)
                merged_test_losses.append(test_loss)
        
    separate_test_mean.append(np.mean(separate_test_losses))
    separate_test_std.append(np.std(separate_test_losses))
    merged_test_mean.append(np.mean(merged_test_losses))
    merged_test_std.append(np.std(merged_test_losses))

np.save(os.path.join(data_path, "alphas.npy"), np.array(alphas))
np.save(os.path.join(data_path, "separate_test_mean.npy"), np.array(separate_test_mean))
np.save(os.path.join(data_path, "separate_test_std.npy"), np.array(separate_test_std))
np.save(os.path.join(data_path, "merged_test_mean.npy"), np.array(merged_test_mean))
np.save(os.path.join(data_path, "merged_test_std.npy"), np.array(merged_test_std))

print("Training and data saving completed!")
