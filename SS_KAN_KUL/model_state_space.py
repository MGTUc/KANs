#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:11:30 2025

@author: cruz
"""

# model.py
"""
This module defines the model architecture:
1. FullStateNonlinearityKAN: Wraps the efficient_kan.KAN module to model the nonlinear correction.
2. StateSpaceKANModel: Implements the full state-space model by combining linear dynamics
   (with matrices A, B, C, D) and the KAN-based nonlinearity.
"""

import torch
import torch.nn as nn
import efficient_kan 
# from kan import KAN 


#%% Efficient kan
class FullStateNonlinearityKAN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, zero_final_layer=False, **kan_kwargs):
        """
        Initializes the KAN-based nonlinearity module.
        
        Args:
            input_size (int): Total number of input features (state_dim + control input if used).
            hidden_layers (int): Hidden layer size (or can be a list of layer sizes).
            output_size (int): Output dimension (should match the state correction dimension).
            zero_final_layer (bool): If True, initializes the final layer weights to zero.
            **kan_kwargs: Additional keyword arguments for efficient_kan.KAN.
        """
        super(FullStateNonlinearityKAN, self).__init__()
        # Ensure layers_hidden is a list for KAN constructor
        if isinstance(hidden_layers, int):
             layers_config = [input_size, hidden_layers, output_size]
        else: # Assuming hidden_layers is already a list/tuple
             layers_config = [input_size] + list(hidden_layers) + [output_size]
        self.kan = efficient_kan.KAN(layers_config, **kan_kwargs)

        if zero_final_layer:
            with torch.no_grad():
                final_layer = self.kan.layers[-1]
                final_layer.base_weight.zero_()
                final_layer.spline_weight.zero_()

        print("HEREHERE",self.kan.forward(torch.ones(1, input_size)))  # Test forward pass with zero input to check initialization

    def forward(self, state=None, u=None, v=None, update_grid=False):
        """
        Forward pass for the KAN nonlinearity.
        
        Args:
            v (torch.Tensor, optional): Intermediate output. If provided, the model processes v.
            x (torch.Tensor, optional): State. Must be provided with u if v is None.
            u (torch.Tensor, optional): Input. Must be provided with x if v is None.
            
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

        return self.kan(inp, update_grid=update_grid)


class StateSpaceKANModel(nn.Module):
    def __init__(self, A_init, B_init, C_init, D_init, state_kan_model, output_kan_model, trainable_C=True, trainable_D=True):
        """
        Initializes the full state-space model.
        
        Args:
            A_init, B_init, C_init, D_init (Tensor): Initial values for state-space matrices.
            kan_model (nn.Module): The KAN-based nonlinearity module.
            trainable_C (bool): If True, C is learnable.
            trainable_D (bool): If True, D is learnable.
        """
        super(StateSpaceKANModel, self).__init__()
        self.A = nn.Parameter(A_init.clone().detach(), requires_grad=True)
        self.B = nn.Parameter(B_init.clone().detach(), requires_grad=True)
        # Set trainability of C and D based on configuration:
        self.C = nn.Parameter(C_init.clone().detach(), requires_grad=trainable_C)
        self.D = nn.Parameter(D_init.clone().detach(), requires_grad=trainable_D)
        print(f'Is C trainable? {trainable_C}')
        print(f'Is D trainable? {trainable_D}')
        
        
        self.state_dim = self.A.shape[0]
        
        
        self.state_kan_model = state_kan_model
        self.output_kan_model = output_kan_model


    def forward(self, state, u, update_grid=False):
        """
        Computes the next state and output of the system.

        Args:
            state (Tensor): Current state [batch_size, 2]
            u (Tensor): Current input [batch_size, 1]
        
        Returns:
            next_state (Tensor): Next state [batch_size, 2]
            y (Tensor): System output [batch_size, 1]
        """
        # --- State Update ---
        linear_next_state = state @ self.A.T + u @ self.B.T
        # Apply state KAN if it exists
        if self.state_kan_model:
            state_nonlinear_correction = self.state_kan_model(state=state, u=u, update_grid=update_grid)
            next_state = linear_next_state + state_nonlinear_correction
        else:
            # Purely linear state update
            next_state = linear_next_state
            
        # --- Output Calculation ---
        # Calculate linear part of the output equation
        y_linear = state @ self.C.T + u @ self.D.T

        # Apply output KAN if it exists
        if self.output_kan_model:
            output_nonlinear_correction = self.output_kan_model(state=state, u=u, update_grid=update_grid)
            y_final = y_linear + output_nonlinear_correction
        else:
            # Purely linear output equation
            y_final = y_linear

        return next_state, y_final
    
    # --- Add separate regularization loss method ---
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Computes the combined regularization loss from active KANs."""
        total_reg_loss = 0.0
        if self.state_kan_model and hasattr(self.state_kan_model, 'kan') and hasattr(self.state_kan_model.kan, 'regularization_loss'):
            total_reg_loss += self.state_kan_model.kan.regularization_loss(regularize_activation, regularize_entropy)
        if self.output_kan_model and hasattr(self.output_kan_model, 'kan') and hasattr(self.output_kan_model.kan, 'regularization_loss'):
             total_reg_loss += self.output_kan_model.kan.regularization_loss(regularize_activation, regularize_entropy)
        return total_reg_loss

    def get_corrections_only(self, state, u,):
        """
        proxy of forward to obtain non-linear contributions separatly
        """
        # --- State Update ---
        
        linear_next_state = state @ self.A.T + u @ self.B.T
        # Apply state KAN if it exists
        if self.state_kan_model:
            state_nonlinear_correction = self.state_kan_model(state=state, u=u)
            next_state = linear_next_state + state_nonlinear_correction
        else:
            # Purely linear state update
            state_nonlinear_correction = 0
            next_state = linear_next_state
            
        # --- Output Calculation ---
        # Calculate linear part of the output equation
        y_linear = state @ self.C.T + u @ self.D.T

        # Apply output KAN if it exists
        if self.output_kan_model:
            output_nonlinear_correction = self.output_kan_model(state=state, u=u)
            y_final = y_linear + output_nonlinear_correction
        else:
            # Purely linear output equation
            output_nonlinear_correction = 0
            y_final = y_linear
        return state_nonlinear_correction, output_nonlinear_correction


#%% PYkan 

class FullStateNonlinearityKAN_pyKAN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, **kan_kwargs):
        """
        Initializes the KAN-based nonlinearity module based ON PYKAN
        
        Args:
            input_size (int): Total number of input features (state_dim + control input if used).
            hidden_layers (int): Hidden layer size (or can be a list of layer sizes).
            output_size (int): Output dimension (should match the state correction dimension).
            use_input_in_nonlinearity (bool): If True, KAN receives both state and input.
            **kan_kwargs: Additional keyword arguments for efficient_kan.KAN.
        """
        super(FullStateNonlinearityKAN_pyKAN, self).__init__()
        # Ensure layers_hidden is a list for KAN constructor
        if isinstance(hidden_layers, int):
             layers_config = [input_size, hidden_layers, output_size]
        else: # Assuming hidden_layers is already a list/tuple
             layers_config = [input_size] + list(hidden_layers) + [output_size]
        self.kan = KAN(width=[input_size, hidden_layers[0], output_size], auto_save=False, **kan_kwargs)
        self.kan.speed()
        
    def forward(self, state=None, u=None, v=None):
        """
        Forward pass for the KAN nonlinearity.
        
        Args:
            v (torch.Tensor, optional): Intermediate output. If provided, the model processes v.
            x (torch.Tensor, optional): State. Must be provided with u if v is None.
            u (torch.Tensor, optional): Input. Must be provided with x if v is None.
            
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




class StateSpaceKANModel_pyKAN(nn.Module):
    def __init__(self, A_init, B_init, C_init, D_init, state_kan_model, output_kan_model, trainable_C=True, trainable_D=True):
        """
        Initializes the full state-space model.
        
        Args:
            A_init, B_init, C_init, D_init (Tensor): Initial values for state-space matrices.
            kan_model (nn.Module): The KAN-based nonlinearity module.
            trainable_C (bool): If True, C is learnable.
            trainable_D (bool): If True, D is learnable.
        """
        super(StateSpaceKANModel_pyKAN, self).__init__()
        self.A = nn.Parameter(A_init.clone().detach(), requires_grad=True)
        self.B = nn.Parameter(B_init.clone().detach(), requires_grad=True)
        # Set trainability of C and D based on configuration:
        self.C = nn.Parameter(C_init.clone().detach(), requires_grad=trainable_C)
        self.D = nn.Parameter(D_init.clone().detach(), requires_grad=trainable_D)
        print(f'Is C trainable? {trainable_C}')
        print(f'Is D trainable? {trainable_D}')
        
        
        self.state_dim = self.A.shape[0]
        
        
        self.state_kan_model = state_kan_model
        self.output_kan_model = output_kan_model


    def forward(self, state, u):
        """
        Computes the next state and output of the system.

        Args:
            state (Tensor): Current state [batch_size, 2]
            u (Tensor): Current input [batch_size, 1]
        
        Returns:
            next_state (Tensor): Next state [batch_size, 2]
            y (Tensor): System output [batch_size, 1]
        """
        # --- State Update ---
        linear_next_state = state @ self.A.T + u @ self.B.T
        # Apply state KAN if it exists
        if self.state_kan_model:
            state_nonlinear_correction = self.state_kan_model(state=state, u=u)
            next_state = linear_next_state + state_nonlinear_correction
        else:
            # Purely linear state update
            next_state = linear_next_state
            
        # --- Output Calculation ---
        # Calculate linear part of the output equation
        y_linear = state @ self.C.T + u @ self.D.T

        # Apply output KAN if it exists
        if self.output_kan_model:
            output_nonlinear_correction = self.output_kan_model(state=state, u=u)
            y_final = y_linear + output_nonlinear_correction
        else:
            # Purely linear output equation
            y_final = y_linear

        return next_state, y_final
    



#%% OLD AND CRAP FROM HERE
#sonnet3.5
class WienerHammersteinModel_sonnet(nn.Module):
    def __init__(self, A1_init, B1_init, C1_init, A2_init, B2_init, C2_init, NonlinearityModel):
        super(WienerHammersteinModel_sonnet, self).__init__()
        
        self.A1 = nn.Parameter(A1_init, requires_grad=True)
        self.B1 = nn.Parameter(B1_init, requires_grad=True)
        self.C1 = nn.Parameter(C1_init, requires_grad=True)
        self.A2 = nn.Parameter(A2_init, requires_grad=True)
        self.B2 = nn.Parameter(B2_init, requires_grad=True)
        self.C2 = nn.Parameter(C2_init, requires_grad=True)
        
        self.NonlinearityModel = NonlinearityModel

    def forward(self, state1, state2, u):
        # First linear system
        next_state1 = state1 @ self.A1.T + u @ self.B1.T
        v = state1 @ self.C1.T
        
        # Nonlinear transformation
        w = self.NonlinearityModel(v)
        
        # Second linear system
        next_state2 = state2 @ self.A2.T + w @ self.B2.T
        
        # Compute the system output
        y = state2 @ self.C2.T
        
        return next_state2, y
    
#%%%%%grok3
class WienerHammersteinModel(nn.Module):
    def __init__(self, A1_init, B1_init, C1_init, D1_init, 
                 A2_init, B2_init, C2_init, D2_init, 
                 CustomNonlinearity, 
                 trainable_C1=True, trainable_D1=True,
                 trainable_C2=True, trainable_D2=True):
        """
        Initializes the Wiener-Hammerstein state-space model.
        
        Args:
            A1_init, B1_init, C1_init, D1_init (Tensor): Initial values for the first linear block (Wiener).
            A2_init, B2_init, C2_init, D2_init (Tensor): Initial values for the second linear block (Hammerstein).
            CustomNonlinearity (nn.Module): The static nonlinearity module (e.g., KAN, MLP).
            trainable_C1, trainable_D1 (bool): If True, C1 and D1 are learnable.
            trainable_C2, trainable_D2 (bool): If True, C2 and D2 are learnable.
        """
        super(WienerHammersteinModel, self).__init__()

        self.state_dim1 = A1_init.shape[0]
        self.state_dim2 = A2_init.shape[0]
        self.input_dim = B1_init.shape[1] # Assuming B1 and B2 have same input dim, and u is the input
        self.nonlinearity_input_dim = C1_init.shape[0] # Output dimension of C1, input to nonlinearity
        self.output_dim = C2_init.shape[0] # Output dimension of C2, system output


        # First linear block (Wiener part)
        self.A1 = nn.Parameter(A1_init.clone().detach(), requires_grad=True)
        self.B1 = nn.Parameter(B1_init.clone().detach(), requires_grad=True)
        self.C1 = nn.Parameter(C1_init.clone().detach(), requires_grad=trainable_C1)
        self.D1 = nn.Parameter(D1_init.clone().detach(), requires_grad=trainable_D1)

        # Second linear block (Hammerstein part)
        self.A2 = nn.Parameter(A2_init.clone().detach(), requires_grad=True)
        self.B2 = nn.Parameter(B2_init.clone().detach(), requires_grad=True)
        self.C2 = nn.Parameter(C2_init.clone().detach(), requires_grad=trainable_C2)
        self.D2 = nn.Parameter(D2_init.clone().detach(), requires_grad=trainable_D2)


        # Static nonlinearity (e.g., KAN, MLP)
        self.Nonlinearity = CustomNonlinearity

        # Print trainability of C and D matrices
        print(f'Is C1 trainable? {trainable_C1}')
        print(f'Is D1 trainable? {trainable_D1}')
        print(f'Is C2 trainable? {trainable_C2}')
        print(f'Is D2 trainable? {trainable_D2}')

    def forward(self, state1, state2, u):
        """
        Computes the next states and output of the Wiener-Hammerstein system.

        Args:
            state1 (Tensor): Current state of the first linear block [batch_size, state_dim1]
            state2 (Tensor): Current state of the second linear block [batch_size, state_dim2]
            u (Tensor): Current input [batch_size, input_dim]
        
        Returns:
            next_state1 (Tensor): Next state of the first linear block [batch_size, state_dim1]
            next_state2 (Tensor): Next state of the second linear block [batch_size, state_dim2]
            y (Tensor): System output [batch_size, output_dim]
        """
        # First linear block (Wiener part)
        # State update for the first block
        next_state1 = state1 @ self.A1.T + u @ self.B1.T
        # Output of the first block
        v = state1 @ self.C1.T + u @ self.D1.T  # [batch_size, intermediate_dim]

        # Static nonlinearity
        w = self.Nonlinearity(v)  # [batch_size, intermediate_dim]

        # Second linear block (Hammerstein part)
        # State update for the second block
        next_state2 = state2 @ self.A2.T + w @ self.B2.T
        # Output of the second block (system output)
        y = state2 @ self.C2.T + w @ self.D2.T  # [batch_size, output_dim]

        return next_state1, next_state2, y
    
    
#%%%%%%%%
class WienerHammersteinModelStackedState(nn.Module):
    def __init__(self, A1_init, B1_init, C1_init, D1_init, 
                 A2_init, B2_init, C2_init, D2_init, 
                 CustomNonlinearity, 
                 trainable_C1=True, trainable_D1=True,
                 trainable_C2=True, trainable_D2=True):
        """
        Initializes the Wiener-Hammerstein state-space model.
        
        Args:
            A1_init, B1_init, C1_init, D1_init (Tensor): Initial values for the first linear block (Wiener).
            A2_init, B2_init, C2_init, D2_init (Tensor): Initial values for the second linear block (Hammerstein).
            CustomNonlinearity (nn.Module): The static nonlinearity module (e.g., KAN, MLP).
            trainable_C1, trainable_D1 (bool): If True, C1 and D1 are learnable.
            trainable_C2, trainable_D2 (bool): If True, C2 and D2 are learnable.
        """
        super(WienerHammersteinModelStackedState, self).__init__()

        self.state_dim1 = A1_init.shape[0]
        self.state_dim2 = A2_init.shape[0]
        self.input_dim = B1_init.shape[1] # Assuming B1 and B2 have same input dim, and u is the input
        self.nonlinearity_input_dim = C1_init.shape[0] # Output dimension of C1, input to nonlinearity
        self.output_dim = C2_init.shape[0] # Output dimension of C2, system output


        # Parameters for the stacked state model
        self.A1 = nn.Parameter(A1_init.clone().detach(), requires_grad=True)
        self.B1 = nn.Parameter(B1_init.clone().detach(), requires_grad=True)
        self.C1 = nn.Parameter(C1_init.clone().detach(), requires_grad=trainable_C1)
        self.D1 = nn.Parameter(D1_init.clone().detach(), requires_grad=trainable_D1) # D1 might be needed for v = C1*X1 + D1*u  if nonlinearity depends on u as well

        self.A2 = nn.Parameter(A2_init.clone().detach(), requires_grad=True)
        self.B2 = nn.Parameter(B2_init.clone().detach(), requires_grad=True)
        self.C2 = nn.Parameter(C2_init.clone().detach(), requires_grad=trainable_C2)
        self.D2 = nn.Parameter(D2_init.clone().detach(), requires_grad=trainable_D2) # D2 might be needed for y = C2*X2 + D2*w if output depends on w directly


        self.Nonlinearity = CustomNonlinearity

        self.state_dim = self.state_dim1 + self.state_dim2

        # Print trainability of C and D matrices
        print(f'Is C1 trainable? {trainable_C1}')
        print(f'Is D1 trainable? {trainable_D1}')
        print(f'Is C2 trainable? {trainable_C2}')
        print(f'Is D2 trainable? {trainable_D2}')

    def forward(self, combined_state, u):
        """
        Computes the next combined state and output of the Wiener-Hammerstein system using stacked state representation.

        Args:
            combined_state (Tensor): Current combined state [batch_size, state_dim1 + state_dim2]
            u (Tensor): Current input [batch_size, input_dim]

        Returns:
            next_combined_state (Tensor): Next combined state [batch_size, state_dim1 + state_dim2]
            y (Tensor): System output [batch_size, output_dim]
        """
        # Reconstruct combined matrices in forward pass to reflect trainable parameters
        A_combined = torch.block_diag(self.A1, self.A2)
        B_u = torch.cat((self.B1, torch.zeros(self.state_dim2, self.input_dim)), dim=0)
        B_nonlin = torch.cat((torch.zeros(self.state_dim1, self.nonlinearity_input_dim), self.B2), dim=0)
        C1_block = torch.cat((self.C1, torch.zeros(self.nonlinearity_input_dim, self.state_dim2)), dim=1)
        C_output = torch.cat((torch.zeros(self.output_dim, self.state_dim1), self.C2), dim=1)
        

        # Intermediate output v from first block (using combined state and C1_block)
        v = combined_state @ C1_block.T # [batch_size, nonlinearity_input_dim]

        # Static nonlinearity
        w = self.Nonlinearity(v) # [batch_size, nonlinearity_input_dim]

        # State update for combined state
        next_combined_state = combined_state @ A_combined.T + u @ B_u.T + w @ B_nonlin.T

        # System output (using combined state and C_output)
        y = combined_state @ C_output.T

        return next_combined_state, y
    
#%%%%%%%%%%
class WienerHammersteinModelSingleState(nn.Module):
    def __init__(self, A1_init, B1_init, C1_init, # D1 is not used in this config
                 A2_init, B2_init, C2_init, # D2 is not used in this config
                 CustomNonlinearity,
                 trainable_C1=True, trainable_C2=True): # D1 and D2 are assumed to be 0
        """
        Wiener-Hammerstein model with a single combined state and explicit non-linear term.
        """
        super(WienerHammersteinModelSingleState, self).__init__()

        self.state_dim1 = A1_init.shape[0]
        self.state_dim2 = A2_init.shape[0]
        self.input_dim = B1_init.shape[1]
        self.output_dim = C2_init.shape[0]
        self.nonlinearity_input_dim = C1_init.shape[0] # Output dimension of C1, input to nonlinearity

        # Trainable Parameters (A1, B1, C1, A2, B2, C2)
        self.A1 = nn.Parameter(A1_init.clone().detach(), requires_grad=True)
        self.B1 = nn.Parameter(B1_init.clone().detach(), requires_grad=True)
        self.C1 = nn.Parameter(C1_init.clone().detach(), requires_grad=trainable_C1)

        self.A2 = nn.Parameter(A2_init.clone().detach(), requires_grad=True)
        self.B2 = nn.Parameter(B2_init.clone().detach(), requires_grad=True)
        self.C2 = nn.Parameter(C2_init.clone().detach(), requires_grad=trainable_C2)

        self.Nonlinearity = CustomNonlinearity

        self.state_dim = self.state_dim1 + self.state_dim2

        # Fixed combined matrices (constructed in forward for dynamic update)
        # No need to store them as parameters if they are reconstructed each time

        print(f'Is C1 trainable? {trainable_C1}')
        print(f'Is C2 trainable? {trainable_C2}')
        print(f'Combined State Dimension: {self.state_dim}')


    def forward(self, combined_state, u):
        """
        Forward pass for Wiener-Hammerstein with single state and explicit non-linear term.
        """
        # Reconstruct combined matrices dynamically to reflect trainable parameters
        A_combined = torch.block_diag(self.A1, self.A2)
        B_linear = torch.cat((self.B1, torch.zeros(self.state_dim2, self.input_dim)), dim=0)
        B_nonlin_vector = torch.cat((torch.zeros(self.state_dim1, self.nonlinearity_input_dim), self.B2), dim=0) # For non-linear term
        C_combined = torch.cat((torch.zeros(self.output_dim, self.state_dim1), self.C2), dim=1)

        # Split combined state to get x1
        state1 = combined_state[:, :self.state_dim1]

        # Non-linear term calculation: N(X(k)) = [0, B2]^T * f(C1 * x1(k))
        v = state1 @ self.C1.T # C1 * x1(k)
        nonlinearity_output = self.Nonlinearity(state=None, u=None, v=v) # f(C1 * x1(k))

        # --- CORRECTED Non-linear term calculation (Matrix Multiplication) ---
        non_linear_term_x2 = nonlinearity_output @ self.B2.T #  (batch_size x nonlinearity_input_dim) @ (nonlinearity_input_dim x state_dim2).T = (batch_size x state_dim2)
        non_linear_term = torch.cat([torch.zeros(combined_state.shape[0], self.state_dim1, device=combined_state.device), non_linear_term_x2], dim=1) # Pad zeros for x1 part

        # State update: X(k+1) = A_combined * X(k) + B_linear * u(k) + N(X(k))
        # State update: X(k+1) = A_combined * X(k) + B_linear * u(k) + N(X(k))
        linear_state_update = combined_state @ A_combined.T + u @ B_linear.T # Calculate linear part first
        next_combined_state = linear_state_update + non_linear_term # Add non-linear term NON-INPLACE
        
        # Output: y(k) = C_combined * X(k)
        y = combined_state @ C_combined.T

        return next_combined_state, y


#%%%%%%%%
class WienerHammersteinModelSingleState_withD(nn.Module):
    def __init__(self, A1_init, B1_init, C1_init,D1_init, # D1 is not used in this config
                 A2_init, B2_init, C2_init,D2_init, # D2 is not used in this config
                 CustomNonlinearity,
                 trainable_C1=True, trainable_C2=True,trainable_D1=True, trainable_D2=True): # D1 and D2 are assumed to be 0
        """
        Wiener-Hammerstein model with a single combined state and explicit non-linear term.
        """
        super(WienerHammersteinModelSingleState_withD, self).__init__()

        self.state_dim1 = A1_init.shape[0]
        self.state_dim2 = A2_init.shape[0]
        self.input_dim = B1_init.shape[1]
        self.output_dim = C2_init.shape[0]
        self.nonlinearity_input_dim = C1_init.shape[0] # Output dimension of C1, input to nonlinearity

        # Trainable Parameters (A1, B1, C1, D1, A2, B2, C2, D2)
        self.A1 = nn.Parameter(A1_init.clone().detach(), requires_grad=True)
        self.B1 = nn.Parameter(B1_init.clone().detach(), requires_grad=True)
        self.C1 = nn.Parameter(C1_init.clone().detach(), requires_grad=trainable_C1)
        self.D1 = nn.Parameter(D1_init.clone().detach(), requires_grad=trainable_D1) # Trainable D1
        
        self.A2 = nn.Parameter(A2_init.clone().detach(), requires_grad=True)
        self.B2 = nn.Parameter(B2_init.clone().detach(), requires_grad=True)
        self.C2 = nn.Parameter(C2_init.clone().detach(), requires_grad=trainable_C2)
        self.D2 = nn.Parameter(D2_init.clone().detach(), requires_grad=trainable_D2)

        self.Nonlinearity = CustomNonlinearity

        self.state_dim = self.state_dim1 + self.state_dim2

        # Fixed combined matrices (constructed in forward for dynamic update)
        # No need to store them as parameters if they are reconstructed each time

        print(f'Is C1 trainable? {trainable_C1}')
        print(f'Is C2 trainable? {trainable_C2}')
        print(f'Combined State Dimension: {self.state_dim}')


    def forward(self, combined_state, u, update_grid=False):
        """
        Forward pass for Wiener-Hammerstein with single state and explicit non-linear term.
        """
        # Reconstruct combined matrices dynamically to reflect trainable parameters
        A_combined = torch.block_diag(self.A1, self.A2)
        B_linear = torch.cat((self.B1, torch.zeros(self.state_dim2, self.input_dim)), dim=0)
        B_nonlin_vector = torch.cat((torch.zeros(self.state_dim1, self.nonlinearity_input_dim), self.B2), dim=0) # For non-linear term
        C_combined = torch.cat((torch.zeros(self.output_dim, self.state_dim1), self.C2), dim=1)
    
        # Split combined state to get x1
        state1 = combined_state[:, :self.state_dim1]
    
        # Non-linear term calculation: N(X(k)) = [0, B2]^T * f(C1 * x1(k))
        v = state1 @ self.C1.T + u @ self.D1.T # C1 * x1(k) + D1*u - NOW WITH D1
        #self.Nonlinearity.kan.layers[0].update_grid(v)
        nonlinearity_output = self.Nonlinearity(state=None, u=None, v=v, update_grid=update_grid) # f(C1 * x1(k))
        w = nonlinearity_output # Alias for clarity
    
        # --- CORRECTED Non-linear term calculation (Matrix Multiplication) ---
        non_linear_term_x2 = w @ self.B2.T #  (batch_size x nonlinearity_input_dim) @ (nonlinearity_input_dim x state_dim2).T = (batch_size x state_dim2)
        non_linear_term = torch.cat([torch.zeros(combined_state.shape[0], self.state_dim1, device=combined_state.device), non_linear_term_x2], dim=1) # Pad zeros for x1 part
    
        # State update: X(k+1) = A_combined * X(k) + B_linear * u(k) + N(X(k))
        linear_state_update = combined_state @ A_combined.T + u @ B_linear.T # Calculate linear part first
        next_combined_state = linear_state_update + non_linear_term # Add non-linear term NON-INPLACE
    
        # Output: y(k) = C_combined * X(k) + D2*w - NOW WITH D2
        y = combined_state @ C_combined.T + w @ self.D2.T

        return next_combined_state, y

#%%%%%%%%
class WienerHammersteinModelSingleState_withD_scaled(nn.Module):
    def __init__(self, A1_init, B1_init, C1_init,D1_init, # D1 is not used in this config
                 A2_init, B2_init, C2_init,D2_init, # D2 is not used in this config
                 CustomNonlinearity,
                 trainable_C1=True, trainable_C2=True,trainable_D1=True, trainable_D2=True): # D1 and D2 are assumed to be 0
        """
        Wiener-Hammerstein model with a single combined state and explicit non-linear term.
        """
        super(WienerHammersteinModelSingleState_withD_scaled, self).__init__()

        self.state_dim1 = A1_init.shape[0]
        self.state_dim2 = A2_init.shape[0]
        self.input_dim = B1_init.shape[1]
        self.output_dim = C2_init.shape[0]
        self.nonlinearity_input_dim = C1_init.shape[0] # Output dimension of C1, input to nonlinearity

        # Trainable Parameters (A1, B1, C1, D1, A2, B2, C2, D2)
        self.A1 = nn.Parameter(A1_init.clone().detach(), requires_grad=True)
        self.B1 = nn.Parameter(B1_init.clone().detach(), requires_grad=True)
        self.C1 = nn.Parameter(C1_init.clone().detach(), requires_grad=trainable_C1)
        self.D1 = nn.Parameter(D1_init.clone().detach(), requires_grad=trainable_D1) # Trainable D1
        
        self.A2 = nn.Parameter(A2_init.clone().detach(), requires_grad=True)
        self.B2 = nn.Parameter(B2_init.clone().detach(), requires_grad=True)
        self.C2 = nn.Parameter(C2_init.clone().detach(), requires_grad=trainable_C2)
        self.D2 = nn.Parameter(D2_init.clone().detach(), requires_grad=trainable_D2)

        self.Nonlinearity = CustomNonlinearity

        self.state_dim = self.state_dim1 + self.state_dim2

        # Fixed combined matrices (constructed in forward for dynamic update)
        # No need to store them as parameters if they are reconstructed each time

        print(f'Is C1 trainable? {trainable_C1}')
        print(f'Is C2 trainable? {trainable_C2}')
        print(f'Combined State Dimension: {self.state_dim}')


    def forward(self, combined_state, u):
        """
        Forward pass for Wiener-Hammerstein with single state and explicit non-linear term.
        """
        # Reconstruct combined matrices dynamically to reflect trainable parameters
        A_combined = torch.block_diag(self.A1, self.A2)
        B_linear = torch.cat((self.B1, torch.zeros(self.state_dim2, self.input_dim)), dim=0)
        B_nonlin_vector = torch.cat((torch.zeros(self.state_dim1, self.nonlinearity_input_dim), self.B2), dim=0) # For non-linear term
        C_combined = torch.cat((torch.zeros(self.output_dim, self.state_dim1), self.C2), dim=1)
    
        # Split combined state to get x1
        state1 = combined_state[:, :self.state_dim1]
    
        # Non-linear term calculation: N(X(k)) = [0, B2]^T * f(C1 * x1(k))
        v = state1 @ self.C1.T + u @ self.D1.T # C1 * x1(k) + D1*u - NOW WITH D1
        v = v/1.4
        #self.Nonlinearity.kan.layers[0].update_grid(v)
        nonlinearity_output = self.Nonlinearity(state=None, u=None, v=v) # f(C1 * x1(k))
        w = nonlinearity_output*1.4 # Alias for clarity
    
        # --- CORRECTED Non-linear term calculation (Matrix Multiplication) ---
        non_linear_term_x2 = w @ self.B2.T #  (batch_size x nonlinearity_input_dim) @ (nonlinearity_input_dim x state_dim2).T = (batch_size x state_dim2)
        non_linear_term = torch.cat([torch.zeros(combined_state.shape[0], self.state_dim1, device=combined_state.device), non_linear_term_x2], dim=1) # Pad zeros for x1 part
    
        # State update: X(k+1) = A_combined * X(k) + B_linear * u(k) + N(X(k))
        linear_state_update = combined_state @ A_combined.T + u @ B_linear.T # Calculate linear part first
        next_combined_state = linear_state_update + non_linear_term # Add non-linear term NON-INPLACE
    
        # Output: y(k) = C_combined * X(k) + D2*w - NOW WITH D2
        y = combined_state @ C_combined.T + w @ self.D2.T

        return next_combined_state, y
    
    
