from mlpkan import MLPKAN
from fastkan import FastKAN
from efficient_kan import KAN as EfficientKAN
from mlp.MLP import standardMLP
import torch
import numpy as np
import random
import pandas as pd
from pathlib import Path
import time

def R2(preds, targets):
    """
    Coefficient of Determination (R²).
    Note: R² can be negative if predictions are worse than the mean baseline.
    R² = 1 - (SS_res / SS_tot)
    """
    pred_mean = torch.mean(preds, dim=0, keepdim=True)
    target_mean = torch.mean(targets, dim=0, keepdim=True)
    SS_res = torch.sum((targets - preds)**2, dim=0)
    SS_tot = torch.sum((targets - target_mean)**2, dim=0)
    r2_score = 1 - (SS_res / (SS_tot + 1e-8))
    return torch.nan_to_num(r2_score)

def main(nodes, layers, seed=1):
    # Set random seeds for reproducibility
    device = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    easy_set = {
    "I.12.1", "I.12.4", "I.12.5", "I.14.3", "I.14.4", 
    "I.18.12", "I.18.16", "I.25.13", "I.26.2", "I.27.6", 
    "I.30.5", "I.43.16", "I.47.23", "II.2.42", "II.3.24", 
    "II.4.23", "II.8.31", "II.10.9", "II.13.17", "II.15.4", 
    "II.15.5", "II.27.16", "II.27.18", "II.34.11", "II.34.29b", 
    "II.38.3", "II.38.14", "III.7.38", "III.12.43", "III.15.27"
    }

    # Load the Feynman dataset
    folder_path = Path('./feynmanDatasetSmall')
    modelname = 'MLPKAN'

    results = []
    for file_path in folder_path.glob('train/*.csv'):
        if file_path.stem.replace('_train', '') not in easy_set:
            continue
        print(f"Processing {file_path.name}")
        try:
            function_name = file_path.stem.replace('_train', '')

            train_df = pd.read_csv(file_path, header=None)
            test_df = pd.read_csv(folder_path / 'test' / f'{function_name}_test.csv', header=None)
            X_train = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32)
            y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
            X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
            y_test = torch.tensor(test_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
            dataset = {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}
            # # Initialize KAN and fit the model

            match modelname:
                case 'MLPKAN':
                    kan = MLPKAN([X_train.size()[1], 3, 1], subnetwork_shape=[nodes] * layers)
                case 'FastKAN':
                    kan = FastKAN([X_train.size()[1], 3, 1], num_grids=10)
                case 'EfficientKAN':
                    kan = EfficientKAN([X_train.size()[1], 3, 1])
                case 'standardMLP':
                    kan = standardMLP([X_train.size()[1], 10, 10, 10, 1])

            t0 = time.perf_counter()
            kan.fit(dataset=dataset, steps=250, lr=1e-2, batch_size=64, early_stop=0.99, log_grad_stats=False, weight_decay=1e-3, reg_activation=0, reg_entropy=0);
            t_KAN = time.perf_counter() - t0

            y_pred_test = kan(dataset['test_input'])
            y_pred_train = kan(dataset['train_input'])
            mse_test = torch.nn.MSELoss()(y_pred_test, dataset['test_label']).item()
            mse_train = torch.nn.MSELoss()(y_pred_train, dataset['train_label']).item()
            # R2_score_kan = R2(y_pred_kan, dataset['test_label']).item()
            results.append([function_name, mse_train, mse_test, t_KAN])

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            results.append([function_name, None, None, None])

    return results


if __name__ == "__main__":
    nodes = [4,8,16,32,64,128]
    layers = [1,2,3,4,5]
    seed = 1
    results = []
    for i in layers:
        for j in nodes:
            layer_node_results = main(j, i, seed=seed)
            for r in layer_node_results:
                row = [i, j] + r
                results.append(row)
    results_df = pd.DataFrame(results, columns=['Layers', 'Nodes', 'Function', 'train MSE', 'test MSE', 'time'])
    results_df.to_csv(f'./feynmantest/MLPKAN_speedtestnodelayer.csv', index=False)
