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

def main(nodes, layers, grid_size, main_network_layers=1, main_network_nodesperlayer=3, seed=1):
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
    medium_set = {
    "I.10.7", "I.11.19", "I.12.11", "I.12.2", "I.13.12", 
    "I.13.4", "I.15.10", "I.16.6", "I.18.4", "I.24.6", 
    "I.29.4", "I.32.5", "I.34.10", "I.34.27", "I.34.8", 
    "I.38.12", "I.39.10", "I.39.11", "I.43.31", "I.43.43", 
    "I.48.2", "I.8.14", "II.11.3", "II.21.32", "II.34.2", 
    "II.34.29a", "II.34.2a", "II.37.1", "II.6.11", "II.8.7", 
    "III.13.18", "III.14.14", "III.15.12", "III.15.14", "III.17.37", 
    "III.19.51", "III.4.32", "III.8.54"
    }
    hard_set = {
    "I.6.20", "I.6.20a", "I.6.20b", "I.9.18", "I.15.3t", 
    "I.15.3x", "I.29.16", "I.30.3", "I.32.17", "I.34.14", 
    "I.37.4", "I.39.22", "I.40.1", "I.41.16", "I.44.4", 
    "I.50.26", "II.6.15a", "II.6.15b", "II.11.17", "II.11.20", 
    "II.11.27", "II.11.28", "II.13.23", "II.13.34", "II.24.17", 
    "II.35.18", "II.35.21", "II.36.38", "III.4.33", "III.9.52", 
    "III.10.19", "III.21.20"}

    all_sets = easy_set | medium_set | hard_set


    # Load the Feynman dataset
    folder_path = Path('./feynmanDatasetSmall')
    modelname = 'MLPKAN' # 'MLPKAN', 'FastKAN', 'EfficientKAN', 'standardMLP'

    results = []
    for file_path in folder_path.glob('train/*.csv'):
        if file_path.stem.replace('_train', '') not in all_sets:
            continue
        print(f"Processing {file_path.name}")
        # try:
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
                kan = MLPKAN([X_train.size()[1]] + [main_network_nodesperlayer] * main_network_layers + [1], subnetwork_shape=[nodes] * layers, residual_connection=True)
            case 'FastKAN':
                kan = FastKAN([X_train.size()[1]] + [main_network_nodesperlayer] * main_network_layers + [1], num_grids=grid_size)
            case 'EfficientKAN':
                kan = EfficientKAN([X_train.size()[1]] + [main_network_nodesperlayer] * main_network_layers + [1], grid_size=grid_size)
            case 'standardMLP':
                kan = standardMLP([X_train.size()[1]] + [main_network_nodesperlayer] * main_network_layers + [1])

        t0 = time.perf_counter()
        kan.fit(dataset=dataset, steps=250, lr=1e-2, batch_size=128, early_stop=0.99, weight_decay=1e-3, reg_activation=0, reg_entropy=0);
        t_KAN = time.perf_counter() - t0

        y_pred_test = kan(dataset['test_input'])
        y_pred_train = kan(dataset['train_input'])
        mse_test = torch.nn.MSELoss()(y_pred_test, dataset['test_label']).item()
        mse_train = torch.nn.MSELoss()(y_pred_train, dataset['train_label']).item()
        R2_score_kan = R2(y_pred_test, dataset['test_label']).item()
        level = 'Easy' if function_name in easy_set else 'Medium' if function_name in medium_set else 'Hard'
        results.append([function_name, mse_train, mse_test, R2_score_kan, t_KAN, level])

        # except Exception as e:
        #     print(f"Error processing {file_path.name}: {e}")
        #     results.append([function_name, None, None, None, None, None])

    return results


if __name__ == "__main__":
    # nodes = [2,4,6,8,10]
    # layers = [2]
    # seed = 1
    # results = []
    # for i in layers:
    #     for j in nodes:
    #         print(f"%%%%%%%Running with {i} layers and {j} nodes")
    #         layer_node_results = main(nodes=j, layers=i, grid_size=None, seed=seed)
    #         for r in layer_node_results:
    #             row = [i, j] + r
    #             results.append(row)
    # results_df = pd.DataFrame(results, columns=['Layers', 'Nodes', 'Function', 'train MSE', 'test MSE', 'R2 Score', 'time', 'Level'])
    # results_df.to_csv(f'./parameterTests/MLPKAN_layer2nodes200.csv', index=False)
    # grid_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # seed = 1
    # results = []
    # for grid_size in grid_sizes:
    #     print(f"%%%%%%%Running with grid size {grid_size}")
    #     grid_results = main(nodes=None, layers=None, grid_size=grid_size, main_network_layers=1, main_network_nodesperlayer=3, seed=seed)
    #     for r in grid_results:
    #         row = [grid_size] + r
    #         results.append(row)
    # results_df = pd.DataFrame(results, columns=['Grid Size', 'Function', 'train MSE', 'test MSE', 'R2 Score', 'time', 'Level'])
    # results_df.to_csv(f'./parameterTests/FastKAN_grid_size_variable2.csv', index=False)
    main_layers = [1,5,10,15,20]
    seed = 1
    results = []
    for main_network_layers in main_layers:
        print(f"%%%%%%%Running with main network layers {main_network_layers}")
        main_network_results = main(nodes=20, layers=2, grid_size=None, main_network_layers=main_network_layers, main_network_nodesperlayer=3, seed=seed)
        for r in main_network_results:
            row = [main_network_layers] + r
            results.append(row)
    results_df = pd.DataFrame(results, columns=['Main Network Layers', 'Function', 'train MSE', 'test MSE', 'R2 Score', 'time', 'Level'])
    results_df.to_csv(f'./parameterTests/MLPKAN_main_network_layers_variable_residual.csv', index=False)