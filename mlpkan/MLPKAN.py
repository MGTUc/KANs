import random
import time
import numpy as np
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os

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
            firstLayer = i == 0
            lastLayer = i == len(full_shape) - 2

            w = nn.Parameter(self._init_scaled_weights(self.num_nets, out_dim, in_dim, lastLayer, self.input_size))
            b = nn.Parameter(torch.zeros(self.num_nets, out_dim, 1))
            
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _init_scaled_weights(num_nets, out_dim, in_dim, lastLayer, input_size):
        if lastLayer:
            # Final layer: Xavier-like scaling (sqrt(1/fan_in))
            
            std = np.sqrt(2.0 / (in_dim * input_size))
        else:
            # Hidden layers: He initialization (sqrt(2/fan_in))
            std = np.sqrt(2.0 / in_dim)
        
        return torch.randn(num_nets, out_dim, in_dim) * std * 0.1

    def forward(self, x, save_activations=False):
        if save_activations:
            self.pre_activations = x.detach().cpu()
        batch_size = x.shape[0]
        
        # Result shape: [In*Out N, 1, Batch]
        x = x.T.repeat_interleave(self.output_size, dim=0).unsqueeze(1)
        
        num_layers = len(self.weights)
        for i in range(num_layers):
            # Weight [N, Out, In] @ Input [N, In, Batch] -> [N, Out, Batch]
            x = torch.matmul(self.weights[i], x) + self.biases[i]
            
            if i < num_layers - 1:
                x = torch.nn.SiLU()(x)
        
        # x is currently [Num_Nets, 1, Batch] because the last layer output_dim is 1
        x = x.view(self.input_size, self.output_size, batch_size)
        if save_activations:
            self.post_activations = x.detach().cpu()
        # Sum over input_size (dim 0) -> [output_size, batch_size] -> Transpose to [Batch, Out]
        return x.sum(dim=0).T 
    
    def plot_activations(self, layer_idx):
        if self.pre_activations is None or self.post_activations is None:
            print("No activations saved. Run forward with save_activations=True first.")
            return
        import matplotlib.pyplot as plt
        
        for i in range(self.input_size):
            # Sort by input activation and apply the same order to all output activations.
            sort_idx = torch.argsort(self.pre_activations[:, i])
            x_sorted = self.pre_activations[sort_idx, i].numpy()

            fig, axes = plt.subplots(1, self.output_size, figsize=(5 * self.output_size, 4), squeeze=False)
            axes = axes.ravel()

            for j in range(self.output_size):
                y_sorted = self.post_activations[i, j, sort_idx].numpy()
                axes[j].plot(x_sorted, y_sorted, marker='o', linestyle='-', markersize=4)
                axes[j].set_title(f"Input {i} to Output {j} Activations")
                axes[j].set_xlabel(f"Input {i}")
                axes[j].set_ylabel(f"Output {j}")
                axes[j].grid(True)

            fig.suptitle(f"Activations for Input {i} Layer {layer_idx}")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()


class MLPKAN(nn.Module):
    def __init__(self, layerSizes, subnetwork_shape = [2,2]):
        super(MLPKAN, self).__init__()
        self.subnetwork_shape = subnetwork_shape

        self.layerSizes = layerSizes
        self.edge_attribution_scores = [[[1.0 for _ in range(layerSizes[l+1])] for _ in range(layerSizes[l])] for l in range(len(layerSizes)-1)]
        self.node_attribution_scores = [[1.0 for _ in range(layerSizes[i])] for i in range(len(layerSizes))]
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
    
    def plot(self, folder="./figures", attribution_score_alpha=True, scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None, edge_plot_scale=1.5):
        '''
        Plot an MLPKAN architecture with per-edge activation function thumbnails.

        Note: this function requires a prior forward pass with save_activations=True.
        '''
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        if attribution_score_alpha:
            for l in reversed(range(len(self.layers))):
                for i in range(self.layerSizes[l]):
                    for j in range(self.layerSizes[l + 1]):
                        
                        self.edge_attribution_scores[l][i][j] = self.node_attribution_scores[l+1][j] * (torch.std(self.layers[l].post_activations[i, j, :])/(torch.std((self.layers[l].post_activations[:,j,:]).sum(dim=0))) + 1e-8).item()
                        print(f"attribution for layer {l} edge {i}->{j}: {self.edge_attribution_scores[l][i][j]:.4f}")
                    self.node_attribution_scores[l][i] = sum(self.edge_attribution_scores[l][i][j] for j in range(self.layerSizes[l + 1]))
                    print(f"attribution for layer {l} node {i}: {self.node_attribution_scores[l][i]:.4f}")
                        


        # Normalize attribution scores per layer so weak->strong edges show visible opacity gradient
        # min_alpha = 0.2
        # for l in range(len(self.layers)):
        #     if l < len(self.edge_attribution_scores):
        #         scores = [self.edge_attribution_scores[l][i][j] for i in range(self.layerSizes[l]) for j in range(self.layerSizes[l + 1])]
        #         if scores:
        #             min_score = min(scores)
        #             max_score = max(scores)
        #             score_range = max_score - min_score + 1e-8
        #             for i in range(self.layerSizes[l]):
        #                 for j in range(self.layerSizes[l + 1]):
        #                     normalized = (self.edge_attribution_scores[l][i][j] - min_score) / score_range
        #                     self.edge_attribution_scores[l][i][j] = min_alpha + normalized * (1.0 - min_alpha)
        if attribution_score_alpha:
            max_node_score = max(max(layer) for layer in self.node_attribution_scores) + 1e-8
            max_edge_score = max(max(max(out_scores) for out_scores in layer) for layer in self.edge_attribution_scores) + 1e-8

            self.node_attribution_scores = [[max(0, min(score/max_node_score, 1.0)) for score in layer_scores] for layer_scores in self.node_attribution_scores]
            self.edge_attribution_scores = [[[max(0, min(score/max_edge_score, 1.0)) for score in out_scores] for out_scores in layer_scores] for layer_scores in self.edge_attribution_scores]

        missing_acts = [
            idx
            for idx, layer in enumerate(self.layers)
            if layer.pre_activations is None or layer.post_activations is None
        ]
        if missing_acts:
            print(
                "No activations saved for all layers. "
                "Run model(x, save_activations=True) before calling plot()."
            )
            return None

        os.makedirs(folder, exist_ok=True)

        depth = len(self.layerSizes) - 1
        thumbnail_zoom = max(0.06, 0.16 * scale * edge_plot_scale)
        for l in range(depth):
            for i in range(self.layerSizes[l]):
                for j in range(self.layerSizes[l + 1]):
                    rank = torch.argsort(self.layers[l].pre_activations[:, i])
                    x_vals = self.layers[l].pre_activations[:, i][rank].cpu().numpy()
                    y_vals = self.layers[l].post_activations[i, j, :][rank].cpu().numpy()

                    # Build temporary edge plots off-screen to avoid duplicate figure popups.
                    edge_plot_scale = max(0.25, float(edge_plot_scale))
                    with plt.ioff():
                        fig_edge, ax_edge = plt.subplots(
                            figsize=(2.2 * edge_plot_scale, 2.2 * edge_plot_scale)
                        )
                    fig_edge.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.96)
                    if tick:
                        ax_edge.tick_params(axis="both", labelsize=8)
                    else:
                        ax_edge.set_xticks([])
                        ax_edge.set_yticks([])

                    if attribution_score_alpha:
                        edge_alpha = max(0.0, min(1.0, self.edge_attribution_scores[l][i][j]))
                        ax_edge.plot(x_vals, y_vals, color="black", lw=2.0, alpha=edge_alpha)
                    else:
                        ax_edge.plot(x_vals, y_vals, color="black", lw=2.0)
                    if sample:
                        if attribution_score_alpha:
                            edge_alpha = max(0.0, min(1.0, self.edge_attribution_scores[l][i][j]))
                            ax_edge.scatter(x_vals, y_vals, color="black", s=max(2, int(30 * scale)), alpha=edge_alpha)
                        else:
                            ax_edge.scatter(x_vals, y_vals, color="black", s=max(2, int(30 * scale)))

                    for spine in ax_edge.spines.values():
                        spine.set_color("black")
                        spine.set_linewidth(1.0)

                    fig_edge.savefig(f"{folder}/sp_{l}_{i}_{j}.png", dpi=220)
                    plt.close(fig_edge)

        width = self.layerSizes
        n_layers = len(width)
        max_nodes = max(width)

        fig_w = max(9.0, 1.8 * max_nodes) * scale * 2.0
        fig_h = max(6.0, 2.6 * (n_layers - 1)) * scale * 2.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        y_layer_pos = np.linspace(0.0, 1.0, n_layers)
        x_node_pos = []
        x_margin = 0.08
        usable_width = 1.0 - 2.0 * x_margin
        global_step = usable_width / max(max_nodes - 1, 1)
        for n in width:
            if n == 1:
                x = np.array([0.5])
            else:
                # Keep per-layer node spacing consistent with the widest layer.
                span = (n - 1) * global_step
                left = 0.5 - span / 2.0
                right = 0.5 + span / 2.0
                x = np.linspace(left, right, n)
            x_node_pos.append(x)

        for l in range(n_layers):
            for i in range(width[l]):
                ax.scatter(x_node_pos[l][i], y_layer_pos[l], s=120 * scale, color="black", zorder=5)

                if l == 0 and in_vars is not None and i < len(in_vars):
                    ax.text(
                        x_node_pos[l][i],
                        y_layer_pos[l] - 0.04,
                        str(in_vars[i]),
                        ha="center",
                        va="top",
                        fontsize=max(8, int(10 * scale)),
                    )

                if l == n_layers - 1 and out_vars is not None and i < len(out_vars):
                    ax.text(
                        x_node_pos[l][i],
                        y_layer_pos[l] + 0.04,
                        str(out_vars[i]),
                        ha="center",
                        va="bottom",
                        fontsize=max(8, int(10 * scale)),
                    )

        for l in range(n_layers - 1):
            y_bottom = y_layer_pos[l]
            y_top = y_layer_pos[l + 1]
            y_mid = 0.5 * (y_bottom + y_top)
            n_in = width[l]
            n_out = width[l + 1]
            n_edges = n_in * n_out

            # Place all edge thumbnails for this layer in one horizontal strip.
            x_strip = np.linspace(0.08, 0.92, n_edges)
            thumb_centers = {}

            for i in range(n_in):
                for j in range(n_out):
                    edge_id = i * n_out + j
                    x_img = x_strip[edge_id]
                    y_img = y_mid
                    thumb_centers[(i, j)] = (x_img, y_img)

                    img_path = f"{folder}/sp_{l}_{i}_{j}.png"
                    im = plt.imread(img_path)

                    image_box = OffsetImage(im, zoom=thumbnail_zoom)
                    ann = AnnotationBbox(image_box, (x_img, y_img), frameon=False, xycoords='data')
                    ax.add_artist(ann)

            y_gap = 0.055
            for i in range(n_in):
                for j in range(n_out):
                    x_bottom = x_node_pos[l][i]
                    x_top = x_node_pos[l + 1][j]
                    x_img, y_img = thumb_centers[(i, j)]

                    edge_alpha = self.edge_attribution_scores[l][i][j]
                    if attribution_score_alpha:
                        ax.plot([x_bottom, x_img], [y_bottom, y_img - y_gap], color="black", lw=2, alpha=edge_alpha)
                        ax.plot([x_img, x_top], [y_img + y_gap, y_top], color="black", lw=2, alpha=edge_alpha)
                    else:
                        ax.plot([x_bottom, x_img], [y_bottom, y_img - y_gap], color="black", lw=2)
                        ax.plot([x_img, x_top], [y_img + y_gap, y_top], color="black", lw=2)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")

        if title is not None:
            ax.set_title(title, fontsize=max(10, int(12 * scale)))

        fig.tight_layout()
        plt.show()
    
    def plot_activations(self):
        for i, layer in enumerate(self.layers):
            layer.plot_activations(layer_idx=i)

    def _collect_grad_stats(self):
        grad_means = []
        grad_maxes = []
        near_zero_counts = []
        total_counts = []

        for layer in self.layers:
            if not isinstance(layer, MLPKANlayer):
                continue

            for param in list(layer.weights) + list(layer.biases):
                if param.grad is None:
                    continue

                g = param.grad.detach()
                abs_g = g.abs()
                grad_means.append(abs_g.mean().item())
                grad_maxes.append(abs_g.max().item())
                near_zero_counts.append((abs_g < 1e-10).sum().item())
                total_counts.append(abs_g.numel())

        if not grad_means:
            return {
                'grad_mean_abs': 0.0,
                'grad_max_abs': 0.0,
                'grad_near_zero_frac': 1.0,
            }

        return {
            'grad_mean_abs': float(np.mean(grad_means)),
            'grad_max_abs': float(np.max(grad_maxes)),
            'grad_near_zero_frac': float(np.sum(near_zero_counts) / max(1, np.sum(total_counts))),
        }

    def fit(
        self,
        dataset,
        steps,
        batch_size=128,
        lr=1,
        early_stop=False,
        optimizer_name='AdamW',
        log_grad_stats=False,
        grad_stats_every=1,
    ):
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
        
        history = {
            'rmse_history': [],
            'R2_history': [],
            'grad_mean_abs_history': [],
            'grad_max_abs_history': [],
            'grad_near_zero_frac_history': [],
        }
        n_samples = train_data.shape[0]
        
        for t in range(steps):
            self.train()
            epoch_grad_mean = []
            epoch_grad_max = []
            epoch_grad_zero_frac = []
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
                    if log_grad_stats:
                        grad_stats = self._collect_grad_stats()
                        epoch_grad_mean.append(grad_stats['grad_mean_abs'])
                        epoch_grad_max.append(grad_stats['grad_max_abs'])
                        epoch_grad_zero_frac.append(grad_stats['grad_near_zero_frac'])
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
            elif optimizer_name == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    pred = self.forward(train_data)
                    loss = loss_fn(pred, train_labels)
                    loss.backward()
                    if log_grad_stats:
                        grad_stats = self._collect_grad_stats()
                        epoch_grad_mean.append(grad_stats['grad_mean_abs'])
                        epoch_grad_max.append(grad_stats['grad_max_abs'])
                        epoch_grad_zero_frac.append(grad_stats['grad_near_zero_frac'])
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    return loss
                optimizer.step(closure)

            if log_grad_stats:
                mean_abs_grad = float(np.mean(epoch_grad_mean)) if epoch_grad_mean else 0.0
                max_abs_grad = float(np.max(epoch_grad_max)) if epoch_grad_max else 0.0
                near_zero_frac = float(np.mean(epoch_grad_zero_frac)) if epoch_grad_zero_frac else 1.0
                history['grad_mean_abs_history'].append(mean_abs_grad)
                history['grad_max_abs_history'].append(max_abs_grad)
                history['grad_near_zero_frac_history'].append(near_zero_frac)
            
            self.eval()
            with torch.no_grad():
                test_pred = self(test_data)

                rmse_value = torch.sqrt(loss_fn(test_pred, test_labels)).item()
                history['rmse_history'].append(rmse_value)

                R2_value = R2(test_pred, test_labels)
                history['R2_history'].append(R2_value)
                if log_grad_stats and ((t + 1) % max(1, grad_stats_every) == 0):
                    print(
                        f"Epoch {t+1}/{steps}, RMSE: {rmse_value:.4f}, R2: {R2_value:.4f}, "
                        f"|grad|_mean: {mean_abs_grad:.2e}, |grad|_max: {max_abs_grad:.2e}, "
                        f"near-zero grad frac: {near_zero_frac:.3f}",
                        end='\r',
                        flush=True,
                    )
                else:
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
        
