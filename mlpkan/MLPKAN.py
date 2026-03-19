import random
import time
import numpy as np
import torch
from torch import nn
import pandas as pd
from pathlib import Path

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
    return torch.nan_to_num(r2_score).item()

class MLPKANlayer(nn.Module):
    def __init__(self, input_size, output_size, subnetwork_hidden_shape=[5, 5]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_nets = input_size * output_size
        self.pre_activations = None
        self.post_activations = None
        
        full_shape = [1] + subnetwork_hidden_shape + [1]
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        for i in range(len(full_shape) - 1):
            in_dim = full_shape[i]
            out_dim = full_shape[i+1]
            
            # Weight shape: [Num_Nets, Out_Dim, In_Dim]
            # Bias shape:  [Num_Nets, Out_Dim, 1]
            w = nn.Parameter(torch.randn(self.num_nets, out_dim, in_dim) * 0.1)
            b = nn.Parameter(torch.zeros(self.num_nets, out_dim, 1))
            
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x, save_activations=False):
        if save_activations:
            self.pre_activations = x.detach().cpu()
        batch_size = x.shape[0]
        
        # Result shape: [In*Out, 1, Batch]
        x = x.T.repeat_interleave(self.output_size, dim=0).unsqueeze(1)
        
        num_layers = len(self.weights)
        for i in range(num_layers):
            # Weight [N, Out, In] @ Input [N, In, Batch] -> [N, Out, Batch]
            x = torch.matmul(self.weights[i], x) + self.biases[i]
            
            if i < num_layers - 1:
                x = torch.relu(x)
        
        # x is currently [Num_Nets, 1, Batch] because the last layer output_dim is 1
        x = x.view(self.input_size, self.output_size, batch_size)
        if save_activations:
            self.post_activations = x.detach().cpu()
        # Sum over input_size (dim 0) -> [output_size, batch_size] -> Transpose to [Batch, Out]
        return x.sum(dim=0).T
    
    def plot_activations(self):
        if self.pre_activations is None or self.post_activations is None:
            print("No activations saved. Run forward with save_activations=True first.")
            return
        import matplotlib.pyplot as plt
        
        for i in range(self.input_size):
            # Sort by input activation and apply the same order to all output activations.
            sort_idx = torch.argsort(self.pre_activations[:, i])
            x_sorted = self.pre_activations[sort_idx, i].numpy()

            fig, axes = plt.subplots(self.output_size, 1, figsize=(7, 3 * self.output_size), squeeze=False)
            axes = axes.ravel()

            for j in range(self.output_size):
                y_sorted = self.post_activations[i, j, sort_idx].numpy()
                axes[j].plot(x_sorted, y_sorted, marker='o', linestyle='-', markersize=4)
                axes[j].set_title(f"Input {i} to Output {j} Activations")
                axes[j].set_xlabel(f"Input {i}")
                axes[j].set_ylabel(f"Output {j}")
                axes[j].grid(True)

            fig.suptitle(f"Activations for Input {i}")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()


class MLPKAN(nn.Module):
    def __init__(self, layerSizes, subnetwork_shape = [2,2]):
        super(MLPKAN, self).__init__()
        self.subnetwork_shape = subnetwork_shape

        self.layerSizes = layerSizes

        layers = []

        for i in range(len(layerSizes)-1):
            mlpkan_layer = MLPKANlayer(input_size=layerSizes[i], output_size=layerSizes[i+1], subnetwork_hidden_shape=self.subnetwork_shape)
            layers.append(mlpkan_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, save_activations=False):
        if save_activations:
            for layer in self.layers:
                x = layer(x, save_activations=save_activations)
        else:
            x = self.layers(x)
        return x

    def fit(self, dataset, steps, batch_size=128, lr=1, early_stop=False, optimizer_name='AdamW'):
        device = next(self.parameters()).device

        train_data = dataset['train_input'].to(device)
        train_labels = dataset['train_label'].to(device)
        test_data = dataset['test_input'].to(device)
        test_labels = dataset['test_label'].to(device)

        loss_fn = nn.MSELoss()
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif optimizer_name == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, line_search_fn="strong_wolfe")
        
        history = {'rmse_history': [], 'R2_history': []}
        n_samples = train_data.shape[0]
        
        for t in range(steps):
            self.train()
            if optimizer_name == 'AdamW':
                # Manual batching with shuffling
                indices = torch.randperm(n_samples, device=device)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X, y = train_data[batch_indices], train_labels[batch_indices]
                    optimizer.zero_grad()
                    pred = self.forward(X)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
            elif optimizer_name == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    pred = self.forward(train_data)
                    loss = loss_fn(pred, train_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    return loss
                optimizer.step(closure)
            
            self.eval()
            with torch.no_grad():
                test_pred = self(test_data)

                rmse_value = torch.sqrt(loss_fn(test_pred, test_labels)).item()
                history['rmse_history'].append(rmse_value)

                R2_value = R2(test_pred, test_labels)
                history['R2_history'].append(R2_value)
                print(f"Epoch {t+1}/{steps}, RMSE: {rmse_value:.4f}, R2: {R2_value:.4f} ", end='\r',flush=True)
                if early_stop and R2_value > 0.99:
                    print(f"\nEarly stopping at epoch {t+1} with R2: {R2_value:.4f}")
                    break
        
        return history

if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MLPKANmodel = MLPKAN([2,3,1], subnetwork_shape=[10,5])
    # MLPKANmodel = torch.compile(MLPKANmodel)

    train_path = Path('feynmanDataset/train/I.12.1_train.csv')
    test_path = Path('feynmanDataset/test/I.12.1_test.csv')

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    X_train = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(test_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
    dataset = {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}

    t0 = time.perf_counter()
    histories = MLPKANmodel.fit(dataset, steps=1000, lr=0.001, early_stop=True, optimizer_name='AdamW')
    t1 = time.perf_counter() - t0
    print(t1)

    pred = MLPKANmodel.forward(dataset['train_input'])
    R2_score = R2(pred, dataset['train_label'])
    pred2 = MLPKANmodel.forward(dataset['test_input'])
    R2_score2 = R2(pred2, dataset['test_label'])
    print("R2 score train:", R2_score, "R2 score test:", R2_score2)
        
