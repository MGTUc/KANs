import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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

class standardMLP(nn.Module):
    """
    Simple MLP: A standard feedforward neural network with ReLU activations.
    
    Architecture:
        - For layerSizes=[1, 3, 1]: creates a 2-layer MLP with 1 input, 3 hidden units, and 1 output.
    
    Performance: Baseline for comparison with KAN architectures.
    """
    
    def __init__(self, layerSizes=[1, 3, 1]):
        super().__init__()
        layers = []
        hidden_dims = layerSizes[1:-1]
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(layerSizes[i], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], layerSizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def fit(
        self,
        dataset,
        steps,
        batch_size=128,
        lr=1,
        weight_decay=0.0,
        early_stop=None,
        seed=None,
    ):
        device = next(self.parameters()).device

        train_data = dataset['train_input']
        train_labels = dataset['train_label']
        test_data = dataset['test_input'].to(device)
        test_labels = dataset['test_label'].to(device)

        train_dataset = TensorDataset(train_data, train_labels)
        pin_memory = device.type == "cuda"
        data_loader_generator = None
        if seed is not None:
            data_loader_generator = torch.Generator(device="cpu")
            data_loader_generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            generator=data_loader_generator,
        )

        loss_fn = nn.MSELoss()

        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Handle common names (bias) and custom layouts (e.g. layers.0.biases.0).
            name_tokens = set(name.split("."))
            is_bias_like = ("bias" in name_tokens) or ("biases" in name_tokens)
            if is_bias_like or param.ndim == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        if len(no_decay_params) == 0:
            print("Warning: no no-decay params found. Check model.named_parameters() naming.")

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )
    
        
        history = {
            'mse_history_test': [],
            'mse_history_train': [],
            'R2_history': [],
        }
        for t in range(steps):
            self.train()
            reg_Loss = torch.tensor(0.0, device=device)
            for X, y in train_loader:
                X = X.to(device, non_blocking=pin_memory)
                y = y.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()
                pred = self.forward(X)
                loss = loss_fn(pred, y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
            
            self.eval()
            with torch.no_grad():
                test_pred = self(test_data)
                mse_value = loss_fn(test_pred, test_labels).item()
                history['mse_history_test'].append(mse_value)

                train_pred = self(train_data)
                mse_value_train = loss_fn(train_pred, train_labels).item()
                history['mse_history_train'].append(mse_value_train)

                R2_value = R2(test_pred, test_labels)
                history['R2_history'].append(R2_value)
                print(f"Epoch {t+1}/{steps}, MSE: {mse_value:.4f}, R2: {R2_value:.4f} ", end='\r',flush=True)
                if early_stop and R2_value > early_stop:
                    print(f"\nEarly stopping at epoch {t+1} with R2: {R2_value:.4f}")
                    break
        
        return history

    
