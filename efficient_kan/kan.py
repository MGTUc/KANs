# Copyright 2024 @Blealtan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn
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



class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.base_weight = torch.nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(1.0/self.in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        # self.base_activation = lambda x: x
        self.grid_eps = grid_eps

        self.pre_activations = None
        self.post_activations = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor, save_activations: bool = False):
        assert x.dim() == 2 and x.size(1) == self.in_features

        if not save_activations:
            # Standard fused path — fast, no per-edge materialization.
            base_output = F.linear(self.base_activation(x), self.base_weight)
            spline_output = F.linear(
                self.b_splines(x).view(x.size(0), -1),
                self.scaled_spline_weight.view(self.out_features, -1),
            )
            return base_output + spline_output

        # Plot/attribution path: build the per-edge tensor [in, out, batch].
        self.pre_activations = x

        base_act = self.base_activation(x)                       # [B, in]
        splines  = self.b_splines(x)                             # [B, in, num_basis]

        # Per-edge base contribution: base_weight[j,i] * SiLU(x)[b,i]
        base_per_edge = self.base_weight.T.unsqueeze(-1) * base_act.T.unsqueeze(1)  # [in, out, B]

        # Per-edge spline contribution: sum_k scaled_spline_weight[j,i,k] * splines[b,i,k]
        spline_per_edge = torch.einsum(
            "jik,bik->ijb", self.scaled_spline_weight, splines
        )                                                        # [in, out, B]

        post_activations = base_per_edge + spline_per_edge       # [in, out, B]
        self.post_activations = post_activations

        return post_activations.sum(dim=0).T                     # [B, out]

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-8)  # avoid 0/0
        regularization_loss_entropy = -torch.sum(p * torch.log(p + 1e-8))  # avoid 0*log(0)=-inf
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layerSizes = list(layers_hidden)

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

        self.edge_attribution_scores = [
            [[1.0 for _ in range(self.layerSizes[l + 1])]
             for _ in range(self.layerSizes[l])]
            for l in range(len(self.layerSizes) - 1)
        ]
        self.node_attribution_scores = [
            [1.0 for _ in range(self.layerSizes[i])]
            for i in range(len(self.layerSizes))
        ]

    def forward(self, x: torch.Tensor, update_grid=False, save_activations=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x, save_activations=save_activations)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
    def plot(self, folder="./figures", attribution_score_alpha=True, scale=0.5,
             tick=False, sample=False, in_vars=None, out_vars=None, title=None,
             edge_plot_scale=1.5):
        """
        Plot the KAN architecture with per-edge activation function thumbnails.
        Requires a prior forward pass with save_activations=True.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        if attribution_score_alpha:
            for l in reversed(range(len(self.layers))):
                for i in range(self.layerSizes[l]):
                    for j in range(self.layerSizes[l + 1]):
                        self.edge_attribution_scores[l][i][j] = (
                            self.node_attribution_scores[l + 1][j]
                            * (
                                torch.std(self.layers[l].post_activations[i, j, :])
                                / (
                                    torch.std(
                                        self.layers[l].post_activations[:, j, :].sum(dim=0)
                                    )
                                    + 1e-8
                                )
                            ).item()
                        )
                    self.node_attribution_scores[l][i] = sum(
                        self.edge_attribution_scores[l][i][j]
                        for j in range(self.layerSizes[l + 1])
                    )

            max_node_score = max(max(layer) for layer in self.node_attribution_scores) + 1e-8
            max_edge_score = max(
                max(max(out_scores) for out_scores in layer)
                for layer in self.edge_attribution_scores
            ) + 1e-8

            self.node_attribution_scores = [
                [max(0, min(s / max_node_score, 1.0)) for s in row]
                for row in self.node_attribution_scores
            ]
            self.edge_attribution_scores = [
                [[max(0, min(s / max_edge_score, 1.0)) for s in out_scores]
                 for out_scores in layer_scores]
                for layer_scores in self.edge_attribution_scores
            ]

        missing_acts = [
            idx for idx, layer in enumerate(self.layers)
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
                    x_vals = self.layers[l].pre_activations[:, i][rank].detach().cpu().numpy()
                    y_vals = self.layers[l].post_activations[i, j, :][rank].detach().cpu().numpy()

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
                            ax_edge.scatter(x_vals, y_vals, color="black",
                                            s=max(2, int(30 * scale)), alpha=edge_alpha)
                        else:
                            ax_edge.scatter(x_vals, y_vals, color="black",
                                            s=max(2, int(30 * scale)))

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
                span = (n - 1) * global_step
                left = 0.5 - span / 2.0
                right = 0.5 + span / 2.0
                x = np.linspace(left, right, n)
            x_node_pos.append(x)

        for l in range(n_layers):
            for i in range(width[l]):
                ax.scatter(x_node_pos[l][i], y_layer_pos[l],
                           s=120 * scale, color="black", zorder=5)
                if l == 0 and in_vars is not None and i < len(in_vars):
                    ax.text(x_node_pos[l][i], y_layer_pos[l] - 0.04, str(in_vars[i]),
                            ha="center", va="top", fontsize=max(8, int(10 * scale)))
                if l == n_layers - 1 and out_vars is not None and i < len(out_vars):
                    ax.text(x_node_pos[l][i], y_layer_pos[l] + 0.04, str(out_vars[i]),
                            ha="center", va="bottom", fontsize=max(8, int(10 * scale)))

        for l in range(n_layers - 1):
            y_bottom = y_layer_pos[l]
            y_top = y_layer_pos[l + 1]
            y_mid = 0.5 * (y_bottom + y_top)
            n_in = width[l]
            n_out = width[l + 1]
            n_edges = n_in * n_out

            if n_edges == 1:
                x_strip = np.array([0.5])
            else:
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
                    ann = AnnotationBbox(image_box, (x_img, y_img),
                                         frameon=False, xycoords='data')
                    ax.add_artist(ann)

            y_gap = 0.055
            for i in range(n_in):
                for j in range(n_out):
                    x_bottom = x_node_pos[l][i]
                    x_top = x_node_pos[l + 1][j]
                    x_img, y_img = thumb_centers[(i, j)]
                    edge_alpha = self.edge_attribution_scores[l][i][j]
                    if attribution_score_alpha:
                        ax.plot([x_bottom, x_img], [y_bottom, y_img - y_gap],
                                color="black", lw=2, alpha=edge_alpha)
                        ax.plot([x_img, x_top], [y_img + y_gap, y_top],
                                color="black", lw=2, alpha=edge_alpha)
                    else:
                        ax.plot([x_bottom, x_img], [y_bottom, y_img - y_gap],
                                color="black", lw=2)
                        ax.plot([x_img, x_top], [y_img + y_gap, y_top],
                                color="black", lw=2)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")

        if title is not None:
            ax.set_title(title, fontsize=max(10, int(12 * scale)))

        fig.tight_layout()
        plt.savefig(f"./SS_KAN_KUL/test___model_saves_simple_2/EfficientKAN2layer_architecture.pdf", dpi=220)
        plt.show()
    
    
    def fit(
        self,
        dataset,
        steps,
        batch_size=128,
        lr=1,
        weight_decay=0.0,
        reg_activation=0.0,
        reg_entropy=0.0,
        early_stop=None,
    ):
        device = next(self.parameters()).device

        train_data = dataset['train_input']
        train_labels = dataset['train_label']
        test_data = dataset['test_input'].to(device)
        test_labels = dataset['test_label'].to(device)

        train_dataset = TensorDataset(train_data, train_labels)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )

        loss_fn = nn.MSELoss()

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
    
        
        history = {
            'rmse_history': [],
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
                reg_Loss = self.regularization_loss(reg_activation, reg_entropy)
                loss = loss_fn(pred, y) + reg_Loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            
            self.eval()
            with torch.no_grad():
                test_pred = self(test_data)

                rmse_value = torch.sqrt(loss_fn(test_pred, test_labels)).item()
                history['rmse_history'].append(rmse_value)

                R2_value = R2(test_pred, test_labels)
                history['R2_history'].append(R2_value)
                print(f"Epoch {t+1}/{steps}, RMSE: {rmse_value:.4f}, R2: {R2_value:.4f} ", end='\r',flush=True)
                if early_stop and R2_value >= early_stop:
                    print(f"\nEarly stopping at epoch {t+1} with R2: {R2_value:.4f}")
                    break
        
        return history