import torch
import torch.nn as nn
from fastkan import FastKAN

class FullStateNonlinearityFastKAN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, zero_final_layer=False, **kan_kwargs):
        """
        Initializes the KAN-based nonlinearity module.
        
        Args:
            input_size (int): Total number of input features (state_dim + control input if used).
            hidden_layers (int): Hidden layer size (or can be a list of layer sizes).
            output_size (int): Output dimension (should match the state correction dimension).
            zero_final_layer (bool): If True, initializes the final layer weights to zero.
            **kan_kwargs: Additional keyword arguments for mlpkan.MLPKAN.
        """
        super(FullStateNonlinearityFastKAN, self).__init__()
        # Ensure layers_hidden is a list for KAN constructor
        if isinstance(hidden_layers, int):
             layers_config = [input_size, hidden_layers, output_size]
        else: # Assuming hidden_layers is already a list/tuple
             layers_config = [input_size] + list(hidden_layers) + [output_size]
        self.kan = FastKAN(layers_config, **kan_kwargs)

        if zero_final_layer:
            with torch.no_grad():
                final_layer = self.kan.layers[-1]
                final_layer.base_linear.weight.zero_()
                # final_layer.base_linear.bias.zero_()
                final_layer.spline_linear.weight.zero_()
                # final_layer.spline_linear.bias.zero_()

    def forward(self, state=None, u=None, v=None, update_grid=False):
        """
        Forward pass for the KAN nonlinearity.
        
        Args:
            v (torch.Tensor, optional): Intermediate output. If provided, the model processes v.
            x (torch.Tensor, optional): State. Must be provided with u if v is None.
            u (torch.Tensor, optional): Input. Must be provided with x if v is None.
            update_grid : this is not used but is kept for compatibility with the SSmodel interface. It is ignored in this implementation.
        Returns:
            Tensor: Nonlinear correction.
        """
        if v is not None:
            # Process intermediate output v
            inp = v
        elif state is not None and u is not None:            
            inp = torch.cat([state, u], dim=-1)
        elif state is not None and u is None:            
            inp = state

        return self.kan(inp)