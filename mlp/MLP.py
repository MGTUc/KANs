import torch
import torch.nn as nn

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
        - For layerSizes=[2, 3, 1]: creates a 2-layer MLP with 2 inputs, 3 hidden units, and 1 output.
    
    Performance: Baseline for comparison with KAN architectures.
    """
    
    def __init__(self, layerSizes=[2, 3, 1]):
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
    
    def fit(self, dataset, steps, batch_size=128, lr=1.0, early_stop=False, optimizer_name='AdamW'):
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

    
