import torch
import torch.nn as nn
from mlpkan import MLPKAN

class FullStateNonlinearityMLPKAN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, **kan_kwargs):
        """
        Initializes the KAN-based nonlinearity module.
        
        Args:
            input_size (int): Total number of input features (state_dim + control input if used).
            hidden_layers (int): Hidden layer size (or can be a list of layer sizes).
            output_size (int): Output dimension (should match the state correction dimension).
            use_input_in_nonlinearity (bool): If True, KAN receives both state and input.
            **kan_kwargs: Additional keyword arguments for mlpkan.MLPKAN.
        """
        super(FullStateNonlinearityMLPKAN, self).__init__()
        # Ensure layers_hidden is a list for KAN constructor
        if isinstance(hidden_layers, int):
             layers_config = [input_size, hidden_layers, output_size]
        else: # Assuming hidden_layers is already a list/tuple
             layers_config = [input_size] + list(hidden_layers) + [output_size]
        self.kan = MLPKAN(layers_config, **kan_kwargs)

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