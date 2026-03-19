from mlpkan.MLPKAN import MLPKAN
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

def main(subnetwork_shape):
    # Set random seeds for reproducibility
    seed = 500
    device = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    easy_set = {
    "I.12.1", "I.12.4", "I.12.5", "I.14.3", "I.14.4", 
    "I.18.12", "I.18.14", "I.25.13", "I.26.2", "I.27.6", 
    "I.30.5", "I.43.16", "I.47.23", "II.2.42", "II.3.24", 
    "II.4.23", "II.8.31", "II.10.9", "II.11.17", "II.15.4", 
    "II.15.5", "II.27.16", "II.27.18", "II.34.11", "II.34.29b", 
    "II.38.3", "II.38.14", "III.7.38", "III.12.43", "III.15.27"
    }

    # Load the Feynman dataset
    folder_path = Path('./feynmanDataset')

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

            # kan = MLPKAN([X_train.size()[1], 3, 1], subnetwork_shape=subnetwork_shape)
            kan = EfficientKAN([X_train.size()[1], 3, 1], grid_size=subnetwork_shape)
            # kan = standardMLP([X_train.size()[1], 8, 8, 1])

            t0 = time.perf_counter()
            kan.fit(dataset=dataset, steps=250, lr=0.001, early_stop=True);
            t_KAN = time.perf_counter() - t0
            y_pred_kan = kan(dataset['test_input'])
            R2_score_kan = R2(y_pred_kan, dataset['test_label']).item()

            results.append([function_name, R2_score_kan, t_KAN])

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            results.append([function_name, None, None])

    avg_R2 = np.mean([r[1] for r in results if r[1] is not None])
    avg_time = np.mean([r[2] for r in results if r[2] is not None and r[1] is not None and r[1] > 0.99])
    return avg_R2, avg_time, len([r for r in results if r[1] is not None and r[1] > 0.99])


if __name__ == "__main__":
    R2s = []
    times = []
    R2s_higherthan99 = []
    for i in range(2, 17):
        print(f"Testing subnetwork size: {i}")
        avg_R2, avg_time, count_higherthan99 = main(subnetwork_shape=[i])
        R2s.append(avg_R2)
        times.append(avg_time)
        R2s_higherthan99.append(count_higherthan99)


    pd.DataFrame({'subnetwork_size': range(2, 17), 'avg_R2': R2s,'#R2>0.99': R2s_higherthan99, 'avg_time R2>0.99': times}).to_csv('./parameterTests/subnetwork_size_results.csv', index=False)