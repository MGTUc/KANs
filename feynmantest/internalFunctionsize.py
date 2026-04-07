from mlp.MLP import standardMLP
import torch
import numpy as np
import random
import pandas as pd
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

    function_set = [
        ("x",lambda x: x[:]),
        ("x**2",lambda x: x[:]**2),
        ("x**3",lambda x: x[:]**3),
        ("x**4",lambda x: x[:]**4),
        ("x**5",lambda x: x[:]**5),
        ("1/x",lambda x: 1/(x[:] + 1e-8)),
        ("1/x**2",lambda x: 1/(x[:]**2 + 1e-8)),
        ("1/x**3",lambda x: 1/(x[:]**3 + 1e-8)),
        ("1/x**4",lambda x: 1/(x[:]**4 + 1e-8)),
        ("1/x**5",lambda x: 1/(x[:]**5 + 1e-8)),
        ("sin(x)",lambda x: torch.sin(x[:])),
        ("cos(x)",lambda x: torch.cos(x[:])),
        ("sin(2x)",lambda x: torch.sin(2*x[:])),
        ("cos(2x)",lambda x: torch.cos(2*x[:])),
        ("sin(x**2)",lambda x: torch.sin(x[:]**2)),
        ("cos(x**2)",lambda x: torch.cos(x[:]**2)),
        ("tan(x)",lambda x: torch.tan(x[:])),
        ("exp(x)",lambda x: torch.exp(x[:])),
        ("sqrt(x)",lambda x: torch.sqrt(torch.abs(x[:]))),
        ("log(x)",lambda x: torch.log(torch.abs(x[:])+1e-8)),
        ("exp(-x**2)",lambda x: torch.exp(-x[:]**2))
    ]


    results = []
    for function_name, function in function_set:
        # Reset RNG state for each function so every run is directly comparable.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        X_all = torch.randn(2200, 1)
        y_all = function(X_all) + 0.1 * torch.randn(2200, 1)  # Add some noise
        X_train = X_all[:2000]
        y_train = y_all[:2000]
        X_test = X_all[2000:]
        y_test = y_all[2000:]
        y_noise_free = function(X_test)
        dataset = {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}
        # # Initialize KAN and fit the model

        model = standardMLP([1] + [nodes]*(layers) + [1])

        history = model.fit(dataset=dataset, steps=250, lr=1e-3, batch_size=64, early_stop=None, weight_decay=1e-2, seed=seed);

        # y_pred_test = model(dataset['test_input'])
        # y_pred_train = model(dataset['train_input'])
        # mse_test = torch.nn.MSELoss()(y_pred_test, dataset['test_label']).item()
        # mse_train = torch.nn.MSELoss()(y_pred_train, dataset['train_label']).item()
        mse_test = history['mse_history_test'][-1]
        mse_train = history['mse_history_train'][-1]
        y_pred_noise_free = model(dataset['test_input'])
        mse_noise_free = torch.nn.MSELoss()(y_pred_noise_free, y_noise_free).item()
        # R2_score_kan = R2(y_pred_kan, dataset['test_label']).item()
        results.append([function_name, mse_train, mse_test, mse_noise_free])
        print(f"{nodes,layers} {function_name}, Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}, Noise-Free MSE: {mse_noise_free:.4f}")
    return results

    


if __name__ == "__main__":
    nodes = [4,8,16,32,64,128,256]
    layers = [1,2,3,4,5]
    seed = 1
    results = []
    for i in layers:
        for j in nodes:
            layer_node_results = main(j, i, seed=seed)
            for r in layer_node_results:
                row = [i, j] + r
                print(row)
                results.append(row)
    results_df = pd.DataFrame(results, columns=['Layers', 'Nodes', 'Function', 'train MSE', 'test MSE', 'Noise-Free target MSE'])
    results_df.to_csv(f'./feynmantest/internalFunctionSizeTestNoise.csv', index=False)
