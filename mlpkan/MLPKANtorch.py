import random
import time
import numpy as np
import torch
from torch import nn
from torch.func import stack_module_state, functional_call, vmap
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

class subnetwork(nn.Module):
    def __init__(self, subnetworkshape = [2,2]):
        super(subnetwork, self).__init__()
        self.subnetworkshape = subnetworkshape
        
        # Network that takes 1 input and produces output_size outputs
        layers = [nn.Linear(1, subnetworkshape[0]), nn.ReLU()]
        for i in range(len(subnetworkshape)-1):
            layers.append(nn.Linear(subnetworkshape[i], subnetworkshape[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(subnetworkshape[-1], 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class MLPKANlayer(nn.Module):
    def __init__(self, input_size, output_size, subnetwork_shape=[2, 2]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # 1. Initialize subnets
        subnets = nn.ModuleList([subnetwork(subnetwork_shape) for _ in range(input_size * output_size)])
        
        # 2. Extract and sanitize keys
        # stack_module_state returns keys like "layers.0.weight"
        params, buffers = stack_module_state(subnets)
        
        # We replace '.' with '_' so ParameterDict doesn't complain
        self.params = nn.ParameterDict({
            k.replace('.', '_'): nn.Parameter(v) for k, v in params.items()
        })
        self.buffers = buffers
        
        # One template module for functional_call to use
        self.repr_subnetwork = subnets[0]

    def forward(self, x):
        batch_size = x.shape[0]

        # 3. Prepare Input
        # Expand x to [n_subnets, batch, 1]
        x_expanded = x.T.repeat_interleave(self.output_size, dim=0).unsqueeze(-1)

        # 4. Reconstruct dotted keys for the functional call
        # PyTorch needs the original names (with dots) to know where weights go
        dotted_params = {k.replace('_', '.'): v for k, v in self.params.items()}

        def single_subnet_forward(p, b, data):
            return functional_call(self.repr_subnetwork, (p, b), (data,))

        # 5. Vectorized Execution
        out_raw = vmap(single_subnet_forward)(dotted_params, self.buffers, x_expanded)
        
        # 6. Summation (KAN Logic)
        # Reshape to [in, out, batch] -> Sum over 'in' -> Transpose to [batch, out]
        out_raw = out_raw.view(self.input_size, self.output_size, batch_size)
        return out_raw.sum(dim=0).T


class MLPKAN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[3], output_size=1, subnetwork_shape = [2,2]):
        super(MLPKAN, self).__init__()
        self.subnetwork_shape = subnetwork_shape

        layerSizes = [input_size] + hidden_sizes + [output_size]
        self.layerSizes = layerSizes

        layers = []

        for i in range(len(layerSizes)-1):
            mlpkan_layer = MLPKANlayer(input_size=layerSizes[i], output_size=layerSizes[i+1], subnetwork_shape=self.subnetwork_shape)
            layers.append(mlpkan_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def fit(self, dataset, steps, batch_size=128, lr=1, early_stop=False, optimizer_name='Adam'):
        device = next(self.parameters()).device

        train_data = dataset['train_input'].to(device)
        train_labels = dataset['train_label'].to(device)
        test_data = dataset['test_input'].to(device)
        test_labels = dataset['test_label'].to(device)

        loss_fn = nn.MSELoss()
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, line_search_fn="strong_wolfe")
        
        history = {'rmse_history': [], 'R2_history': []}
        n_samples = train_data.shape[0]
        
        for t in range(steps):
            self.train()
            if optimizer_name == 'Adam':
                # Manual batching with shuffling
                indices = torch.randperm(n_samples, device=device)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X, y = train_data[batch_indices], train_labels[batch_indices]
                    pred = self.forward(X)
                    loss = loss_fn(pred, y)
                    optimizer.zero_grad()
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
    MLPKANmodel = MLPKAN(input_size=2, hidden_sizes=[3], output_size=1, subnetwork_shape=[5])
    # model = torch.compile(MLPKANmodel)

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
    histories = MLPKANmodel.fit(dataset, steps=1000, lr=1, early_stop=True, optimizer_name='LBFGS')
    t1 = time.perf_counter() - t0
    print(t1)

    pred = MLPKANmodel.forward(dataset['train_input'])
    R2_score = R2(pred, dataset['train_label'])
    pred2 = MLPKANmodel.forward(dataset['test_input'])
    R2_score2 = R2(pred2, dataset['test_label'])
    print("R2 score train:", R2_score, "R2 score test:", R2_score2)
        
