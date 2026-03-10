import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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

class Subnetwork(nn.Module):
    """
    A small MLP that transforms a single scalar input to a single scalar output.
    Used as the learnable function along each connection in MLPKANLayer.
    """
    def __init__(self, hidden_dims=[2, 2]):
        super().__init__()
        # Build MLP: 1 -> hidden_dims[0] -> ... -> hidden_dims[-1] -> 1
        layers = [nn.Linear(1, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPKANLayer(nn.Module):
    """
    MLPKAN Layer: A KAN layer where each input-output connection is parameterized by a Subnetwork.
    
    Architecture:
        - For input_size=2, output_size=3: creates 2*3=6 small MLPs
        - Input x[i] is routed through all output_size MLPs
        - Outputs from all inputs to a single output are summed (KAN combination)
    
    Performance: Direct tensor operations, no explicit Python loops over subnetworks.
    """
    
    def __init__(self, input_size, output_size, subnetwork_shape=[2, 2]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        n_connections = input_size * output_size
        
        # Create and store all subnetworks
        # This ensures proper parameter registration for optimizers
        self.subnets = nn.ModuleList([
            Subnetwork(subnetwork_shape) 
            for _ in range(n_connections)
        ])

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_size]
        Returns:
            [batch_size, output_size]
        """
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, self.output_size, device=x.device)
        
        # For each subnet, apply it to the appropriate input and accumulate to output
        for in_idx in range(self.input_size):
            for out_idx in range(self.output_size):
                subnet_idx = in_idx * self.output_size + out_idx
                # Extract single input feature: [batch_size] -> [batch_size, 1]
                x_single = x[:, in_idx:in_idx + 1]
                # Apply subnet: [batch_size, 1] -> [batch_size, 1]
                y = self.subnets[subnet_idx](x_single)
                # Accumulate to output
                outputs[:, out_idx] += y.squeeze(-1)
        
        return outputs


class MLPKAN(nn.Module):
    """Multi-layer MLPKAN network."""
    
    def __init__(self, layerSizes, subnetwork_shape=[2, 2]):
        super().__init__()
        self.layer_dims = layerSizes
        
        layers = []
        for i in range(len(layerSizes) - 1):
            layers.append(
                MLPKANLayer(
                    input_size=layerSizes[i],
                    output_size=layerSizes[i + 1],
                    subnetwork_shape=subnetwork_shape
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def fit(self, dataset, steps, batch_size=128,lr=1.0, early_stop=False, optimizer_name='AdamW'):
        """
        Train the MLPKAN model.
        
        Args:
            dataset: dict with 'train_input', 'train_label', 'test_input', 'test_label'
            steps: number of training epochs
            batch_size: batch size for Adam optimizer
            lr: learning rate
            early_stop: stop if R² > 0.99
            optimizer_name: 'Adam' or 'LBFGS'
        
        Returns:
            history: dict with 'rmse_history' and 'R2_history' lists
        """
        device = next(self.parameters()).device

        train_input = dataset['train_input'].to(device)
        train_label = dataset['train_label'].to(device)
        test_input = dataset['test_input'].to(device)
        test_label = dataset['test_label'].to(device)

        loss_fn = nn.MSELoss()
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == 'LBFGS':
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=lr,
                line_search_fn="strong_wolfe"
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        history = {'rmse_history': [], 'R2_history': []}
        n_samples = train_input.shape[0]

        for epoch in range(steps):
            self.train()
            
            if optimizer_name == 'AdamW':
                # Mini-batch SGD with shuffling
                indices = torch.randperm(n_samples, device=device)
                for i in range(0, n_samples, batch_size):
                    batch_idx = indices[i:i + batch_size]
                    X_batch = train_input[batch_idx]
                    y_batch = train_label[batch_idx]

                    optimizer.zero_grad()
                    pred = self.forward(X_batch)
                    loss = loss_fn(pred, y_batch)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    
            elif optimizer_name == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    pred = self.forward(train_input)
                    loss = loss_fn(pred, train_label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    return loss
                optimizer.step(closure)

            # Evaluate on test set
            self.eval()
            with torch.no_grad():
                test_pred = self.forward(test_input)
                rmse = torch.sqrt(loss_fn(test_pred, test_label)).item()
                r2 = R2(test_pred, test_label)
                
                history['rmse_history'].append(rmse)
                history['R2_history'].append(r2)
                
                print(f"Epoch {epoch + 1}/{steps}, RMSE: {rmse:.4f}, R²: {r2:.4f}", end='\r', flush=True)
                
                if early_stop and r2 > 0.99:
                    print(f"\nEarly stopping at epoch {epoch + 1} with R²: {r2:.4f}")
                    break

        return history


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = MLPKAN(
        input_size=2,
        hidden_sizes=[3],
        output_size=1,
        subnetwork_shape=[5]
    ).to(device)

    # Load data
    train_path = Path('feynmanDataset/train/I.12.1_train.csv')
    test_path = Path('feynmanDataset/test/I.12.1_test.csv')

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(test_df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, 1)

    dataset = {
        'train_input': X_train,
        'train_label': y_train,
        'test_input': X_test,
        'test_label': y_test
    }

    # Train
    t0 = time.perf_counter()
    history = model.fit(
        dataset,
        steps=1000,
        lr=1,
        early_stop=True,
        optimizer_name='LBFGS'
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTraining time: {elapsed:.2f}s")

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model.forward(X_train)
        train_r2 = R2(train_pred, y_train)
        
        test_pred = model.forward(X_test)
        test_r2 = R2(test_pred, y_test)

    print(f"R² score train: {train_r2:.4f}, R² score test: {test_r2:.4f}")
            
