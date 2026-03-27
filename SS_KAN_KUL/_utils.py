#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:38:24 2025

@author: cruz
"""
import numpy as np
import torch

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

# Normalize both position (x) and velocity (x_dot), along with the input (u)
def normalize_data(
    data,
    norm_type="minmax",
    normalize=True,
    data_mean=None,
    data_std=None,
    data_min=None,
    data_max=None,
):
    """
    Normalizes data using z-score or min-max normalization.

    Args:
        data (torch.Tensor or np.ndarray): The data to normalize.
        norm_type (str, optional): 'zscore' or 'minmax'. Defaults to 'zscore'.
        normalize (bool, optional): If False, returns data as is. Defaults to True.
        data_mean (torch.Tensor, optional): Mean for z-score normalization (if pre-calculated).
        data_std (torch.Tensor, optional): Std for z-score normalization (if pre-calculated).
        data_min (torch.Tensor, optional): Min value for min-max normalization (if pre-calculated).
        data_max (torch.Tensor, optional): Max value for min-max normalization (if pre-calculated).

    Returns:
        tuple: normalized_data, norm_param1, norm_param2, None
               normalized_data (torch.Tensor or np.ndarray): The normalized data.
               norm_param1: Mean (zscore) or min_val (minmax) used for normalization.
               norm_param2: Std (zscore) or max_val (minmax) used for normalization.
               None: Placeholder for potential future parameter.
    """
    if not normalize:
        return data, None, None, None  # Return data as is if no normalization


    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # Z-score normalization
    if norm_type == "zscore":
        if data_mean is None:  # Calculate mean only if not provided
            data_mean = data.mean(dim=0)
        if data_std is None:  # Calculate std only if not provided
            data_std = data.std(dim=0)
        normalized_data = (data - data_mean) / data_std
        return normalized_data, data_mean, data_std, None

    # Min-Max normalization to [-1, 1]
    elif norm_type == "minmax":
        if data_min is None:  # Calculate min_val only if not provided
            data_min = data.min(dim=0).values
        if data_max is None:  # Calculate max_val only if not provided
            data_max = data.max(dim=0).values
        normalized_data = 2 * (data - data_min) / (data_max - data_min) - 1
        return normalized_data, data_min, data_max, None

    elif norm_type == "scaled":
        range_max = 0.65
        range_min = -0.65
        if data_min is None:
            data_min = data.min(dim=0).values
        if data_max is None:
            data_max = data.max(dim=0).values
        normalized_data = (data - data_min) / (data_max - data_min) * (range_max - range_min) + range_min
        return normalized_data, data_min, data_max, None

    elif norm_type == "nothing":
        return data, None, None, None  # Return data as is if no normalization



    else:
        raise ValueError(
            "Normalization type not recognized."
        )


def denormalize_data(
    normalized_data, norm_type, norm_param1, norm_param2, normalize=True
):
    """
    Denormalizes data back to its original scale.

    Args:
        normalized_data (torch.Tensor or np.ndarray): The normalized data.
        norm_type (str): The type of normalization used ('zscore' or 'minmax').
        norm_param1: The first normalization parameter (mean for zscore, min_val for minmax).
        norm_param2: The second normalization parameter (std for zscore, max_val for minmax).
        normalize (bool, optional): If False, returns data as is. Defaults to True.

    Returns:
        torch.Tensor or np.ndarray: The denormalized data, in the same type as input.
    """
    if not normalize:
        return (
            normalized_data  # Return data as is if no denormalization needed
        )

    if isinstance(normalized_data, np.ndarray):
        data_is_numpy = True
        normalized_data = torch.tensor(normalized_data, dtype=torch.float32)
    else:
        data_is_numpy = False

    if norm_type == "zscore":
        mean = norm_param1
        std = norm_param2
        denormalized_data = normalized_data * std + mean

    elif norm_type == "minmax":
        min_val = norm_param1
        max_val = norm_param2
        denormalized_data = (
            0.5 * (normalized_data + 1) * (max_val - min_val) + min_val
        )


    elif norm_type == "scaled":
        range_max = 0.65
        range_min = -0.65
        min_val = norm_param1
        max_val = norm_param2
        denormalized_data = ((normalized_data - range_min) / (range_max - range_min)) * (max_val - min_val) + min_val

    elif norm_type == "nothing":
        denormalized_data = normalized_data  # Return data as is if no normalization


    else:
        raise ValueError(
            "Normalization type not recognized. Choose 'zscore' or 'minmax'."
        )

    if data_is_numpy:
        return denormalized_data.numpy()
    else:
        return denormalized_data
    
    
def predict_recursive(model, initial_state_norm, input_sequence_norm):
    """
    Performs recursive prediction using the state-space KAN model.

    Args:
        model (StateSpaceKANModel): Trained state-space KAN model.
        initial_state_norm (torch.Tensor): Normalized initial state [1, state_dim].
        input_sequence_norm (torch.Tensor): Normalized input sequence [sequence_length, input_dim].

    Returns:
        tuple: (output_predictions_norm, state_predictions_norm)
               output_predictions_norm (torch.Tensor): Normalized output predictions [sequence_length, output_dim].
               state_predictions_norm (torch.Tensor): Normalized state predictions [sequence_length, state_dim].
    """
    model.eval() # Set model to evaluation mode
    output_predictions_norm_list = []
    state_predictions_norm_list = []
    current_state_norm = initial_state_norm.unsqueeze(0) if initial_state_norm.ndim == 1 else initial_state_norm # Ensure initial_state is [1, state_dim]

    with torch.no_grad(): # Disable gradient calculations for prediction
        for i in range(len(input_sequence_norm)):
            current_u_norm = input_sequence_norm[i].unsqueeze(0) # Ensure input is [1, input_dim]

            # Predict next state and output
            next_state_norm_pred, output_norm_pred = model(current_state_norm, current_u_norm)

            # Store the predictions
            output_predictions_norm_list.append(output_norm_pred)
            state_predictions_norm_list.append(next_state_norm_pred)

            # Update current state for the next step (recursive prediction)
            current_state_norm = next_state_norm_pred

    output_predictions_norm = torch.cat(output_predictions_norm_list, dim=0) # shape [sequence_length, output_dim]
    state_predictions_norm = torch.cat(state_predictions_norm_list, dim=0)   # shape [sequence_length, state_dim]
    return output_predictions_norm.cpu().numpy(), state_predictions_norm.cpu().numpy()
    
def compute_error_states(model, dataset):
    # Trainnig data
    output_predictions_norm_train, state_predictions_norm_train = predict_recursive(model, dataset.X_train_norm[0], dataset.u_train_norm[:-1]) # Use new predict function
    output_predictions_denorm_train = denormalize_data(output_predictions_norm_train, 'minmax', dataset.x_min, dataset.x_max)
    # Denormalize true output values (already done in previous responses)
    y_train_denorm = denormalize_data(dataset.y_train_norm, 'minmax', dataset.x_min, dataset.x_max).numpy()
    # Calculate error in denormalized output
    error_output_denorm = output_predictions_denorm_train - y_train_denorm[1:] # Error from the second point onwards as we predict x[t+1] from x[t]
    # Calculate Root Mean Squared Error (RMSE)
    rmse_output_denorm_train = np.sqrt(np.mean(error_output_denorm**2))
    # Calculate Mean Absolute Error (MAE)
    mae_output_denorm_train = np.mean(np.abs(error_output_denorm))

    # Testing data
    output_predictions_norm_test, state_predictions_norm_test = predict_recursive(model, dataset.X_test_norm[0], dataset.u_test_norm[:-1]) # Use new predict function
    output_predictions_denorm_test = denormalize_data(output_predictions_norm_test, 'minmax', dataset.x_min, dataset.x_max)
    y_test_denorm = denormalize_data(dataset.y_test_norm, 'minmax', dataset.x_min, dataset.x_max).numpy()

    error_output_denorm_test = output_predictions_denorm_test - y_test_denorm[1:]
    rmse_output_denorm_test = np.sqrt(np.mean(error_output_denorm_test**2))
    mae_output_denorm_test = np.mean(np.abs(error_output_denorm_test))

    print("\n--- Performance Metrics (Original Scale) ---")
    print(f"MAE (Denormalized Output - Train): {mae_output_denorm_train:.4f}")
    print(f"RMSE (Denormalized Output - Train): {rmse_output_denorm_train:.4f}")
    print(f"MAE (Denormalized Output - Test): {mae_output_denorm_test:.4f}")
    print(f"RMSE (Denormalized Output - Test): {rmse_output_denorm_test:.4f}")

    return

def simulate_model(model, dataset, device):
    """
    Evaluates the trained model on training and testing data.

    Handles both scenarios: known initial states and output-error (unknown states).
    For output-error, performs simulation with state handoff from train to test.

    Args:
        model (nn.Module): The trained PyTorch model.
        dataset (SystemIdentificationDataset): The dataset object containing data and parameters.
        device (str or torch.device): The device the model and data are on.

    Returns:
        dict: A dictionary containing performance metrics:
              {'mae_train', 'rmse_train', 'mae_test', 'rmse_test'}
    """
    print("\n--- Starting Evaluation ---")
    model.eval()

    # Access data and parameters from dataset object
    u_train_norm = dataset.u_train_norm
    y_train_norm = dataset.y_train_norm
    u_test_norm = dataset.u_test_norm
    y_test_norm = dataset.y_test_norm
    y_min = dataset.x_min 
    y_max = dataset.x_max
    norm_flag = dataset.norm_flag

    pred_train_eval_norm = None
    pred_test_eval_norm = None
    y_train_norm_target_eval = None
    y_test_norm_target_eval = None
    pred_state_train_norm = None
    pred_state_test_norm = None
    
    with torch.no_grad():
        if dataset.states_available:
            print(f"  Simulation Mode: States Available (using known initial states)")
            try:
                initial_state_train = dataset.X_train_norm[0].unsqueeze(0) if dataset.X_train_norm[0].ndim == 1 else dataset.X_train_norm[0]
                initial_state_test = dataset.X_test_norm[0].unsqueeze(0) if dataset.X_test_norm[0].ndim == 1 else dataset.X_test_norm[0]

                pred_train_norm, _ = predict_recursive(model, initial_state_train, u_train_norm)
                pred_test_norm, _ = predict_recursive(model, initial_state_test, u_test_norm)

                target_train_norm = y_train_norm.cpu().numpy()
                target_test_norm = y_test_norm.cpu().numpy()
                
                # save for plotting
                pred_state_train_norm = dataset.X_train_norm
                pred_state_test_norm = dataset.X_test_norm

                print(f"    Simulated {len(pred_train_norm)} train steps and {len(pred_test_norm)} test steps.")
            except Exception as e:
                print(f"    Error during simulation with states available: {e}")

        else: # states_available is False
            print(f"  Simulation Mode: States Unavailable (Output Error with State Handoff, No Warmup)")
            try:
                # Determine state_dim
                state_dim = getattr(model, 'state_dim', None)

                # --- Simulate Training Data ---
                print(f"    Simulating full train sequence ({len(u_train_norm)} steps)...")
                current_state_train = torch.zeros(1, state_dim, device=device)
                output_pred_list_train = []
                state_train_norm_list = []
                state_train_norm_list.append(current_state_train.clone()) # Store x(0) guess
                for t in range(len(u_train_norm)):
                    current_input = u_train_norm[t].unsqueeze(0)
                    next_state_train, output_pred_train = model(current_state_train, current_input)
                    output_pred_list_train.append(output_pred_train)
                    state_train_norm_list.append(next_state_train.clone()) # Store x(t+1)
                    current_state_train = next_state_train
                pred_train_norm = torch.cat(output_pred_list_train, dim=0).cpu().numpy()
                pred_state_train_norm = torch.cat(state_train_norm_list[1:]).cpu().numpy()
                final_state_from_train = current_state_train.detach()
                target_train_norm = y_train_norm.cpu().numpy()

                # --- Simulate Testing Data ---

                print(f"    Simulating full test sequence ({len(u_test_norm)} steps)...")
                current_state_test = final_state_from_train
                output_pred_list_test = []
                state_test_norm_list = []
                state_test_norm_list.append(current_state_test.clone()) # Store initial state x(0) for test
                for t in range(len(u_test_norm)):
                    current_input = u_test_norm[t].unsqueeze(0)
                    next_state_test, output_pred_test = model(current_state_test, current_input)
                    output_pred_list_test.append(output_pred_test)
                    current_state_test = next_state_test
                    state_test_norm_list.append(next_state_test.clone()) # Store x(t+1)
                pred_test_norm = torch.cat(output_pred_list_test, dim=0).cpu().numpy()
                target_test_norm = y_test_norm.cpu().numpy()
                pred_state_test_norm = torch.cat(state_test_norm_list[1:]).cpu().numpy()

            except Exception as e:
                 print(f"    Error during output-error simulation: {e}")
    # --- Denormalize ---
    pred_train_denorm, target_train_denorm = None, None
    pred_test_denorm, target_test_denorm = None, None
    normalize_needed = (norm_flag != 'nothing')

    if pred_train_norm is not None and target_train_norm is not None:
        pred_train_denorm = denormalize_data(pred_train_norm, norm_flag, y_min, y_max, normalize=normalize_needed)
        target_train_denorm = denormalize_data(target_train_norm, norm_flag, y_min, y_max, normalize=normalize_needed)
        # Ensure numpy
        if isinstance(pred_train_denorm, torch.Tensor): pred_train_denorm = pred_train_denorm.numpy()
        if isinstance(target_train_denorm, torch.Tensor): target_train_denorm = target_train_denorm.numpy()


    if pred_test_norm is not None and target_test_norm is not None:
        pred_test_denorm = denormalize_data(pred_test_norm, norm_flag, y_min, y_max, normalize=normalize_needed)
        target_test_denorm = denormalize_data(target_test_norm, norm_flag, y_min, y_max, normalize=normalize_needed)
        # Ensure numpy
        if isinstance(pred_test_denorm, torch.Tensor): pred_test_denorm = pred_test_denorm.numpy()
        if isinstance(target_test_denorm, torch.Tensor): target_test_denorm = target_test_denorm.numpy()

    # Return dictionary (use .copy() if arrays might be modified later)
    simulation_results = {
        'pred_train_norm': pred_train_norm,
        'target_train_norm': target_train_norm,
        'pred_test_norm': pred_test_norm,
        'target_test_norm': target_test_norm,
        'pred_train_denorm': pred_train_denorm,
        'target_train_denorm': target_train_denorm,
        'pred_test_denorm': pred_test_denorm,
        'target_test_denorm': target_test_denorm,
        'pred_state_train_norm': pred_state_train_norm,
        'pred_state_test_norm': pred_state_test_norm,
        'u_train_norm':u_train_norm,
        'u_test_norm':u_test_norm

    }
    return simulation_results


def calculate_metrics(simulation_results):
    """
    Calculates MAE and RMSE from denormalized prediction and target series.

    Args:
        simulation_results (dict): A dictionary containing denormalized time series,
                                   as returned by `simulate_model`. Expected keys:
                                   'pred_train_denorm', 'target_train_denorm',
                                   'pred_test_denorm', 'target_test_denorm'.

    Returns:
        dict: A dictionary containing performance metrics:
              {'mae_train', 'rmse_train', 'mae_test', 'rmse_test'}
              Values are float('nan') if calculation is not possible.
    """
    metrics = {'mae_train': float('nan'), 'rmse_train': float('nan'),
               'mae_test': float('nan'), 'rmse_test': float('nan')}

    pred_train = simulation_results.get('pred_train_denorm')
    target_train = simulation_results.get('target_train_denorm')
    pred_test = simulation_results.get('pred_test_denorm')
    target_test = simulation_results.get('target_test_denorm')

    # Train Metrics
    if pred_train is not None and target_train is not None and pred_train.shape[0] > 0:
        len_train = min(len(pred_train), len(target_train))
        if len_train < len(pred_train) or len_train < len(target_train):
            print(f"    Metrics Warning: Truncating train comparison to {len_train} steps due to length mismatch.")
        error_train = pred_train[:len_train] - target_train[:len_train]
        metrics['rmse_train'] = np.sqrt(np.mean(error_train**2))
        metrics['mae_train'] = np.mean(np.abs(error_train))
    else:
        print("    Skipping train metrics calculation.")

    # Test Metrics
    if pred_test is not None and target_test is not None and pred_test.shape[0] > 0:
        len_test = min(len(pred_test), len(target_test))
        if len_test < len(pred_test) or len_test < len(target_test):
             print(f"    Metrics Warning: Truncating test comparison to {len_test} steps due to length mismatch.")
        error_test = pred_test[:len_test] - target_test[:len_test]
        metrics['rmse_test'] = np.sqrt(np.mean(error_test**2))
        metrics['mae_test'] = np.mean(np.abs(error_test))
    else:
        print("    Skipping test metrics calculation.")
        

    return metrics

def Luca_jfm_error(simulation_results):
    pred_train = simulation_results.get('pred_train_denorm')
    target_train = simulation_results.get('target_train_denorm')
    pred_test = simulation_results.get('pred_test_denorm')
    target_test = simulation_results.get('target_test_denorm')
    cum_err = 0
    for i in range(8):
        error_train = pred_train[20001*i:20001*(i+1)] - target_train[20001*i:20001*(i+1)]
        e_rms_i = np.sqrt(np.mean(error_train**2)/(np.mean(pred_train[20001*i:20001*(i+1)]-np.mean(target_train[20001*i:20001*(i+1)])**2)))
        cum_err =+ e_rms_i
    
    return cum_err

def run_test_simulation_and_loss(state_train, model, dataset, device, loss_fn, update_grid_flag=False):
    """
    Runs simulation on the test set and calculates the normalized MSE loss.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (SystemIdentificationDataset): Dataset object with test data.
        device (str or torch.device): Device for computation.
        loss_fn (callable): The loss function (e.g., nn.MSELoss()).
        final_state_from_train (torch.Tensor, optional):
            The final hidden state after training simulation, used for state handoff
            when dataset.states_available is False. Required in that case.

    Returns:
        float: The calculated normalized MSE loss on the test set.
               Returns float('nan') if simulation or loss calculation fails.
    """
    model.eval()  # Set model to evaluation mode

    # Get test data (ensure it's on the right device)
    u_test_norm = dataset.u_test_norm.to(device)
    y_test_norm = dataset.y_test_norm.to(device)

    pred_test_norm_tensor = None


    with torch.no_grad():
        if dataset.states_available:
            #print(f"  Simulation Mode: States Available (using known initial states)")
            try:
                initial_state_test = dataset.X_test_norm[0].unsqueeze(0) if dataset.X_test_norm[0].ndim == 1 else dataset.X_test_norm[0]
                initial_state_test = initial_state_test.to(device)

                # Use predict_recursive (returns numpy), convert back to tensor
                pred_test_norm_np, _ = predict_recursive(model, initial_state_test, u_test_norm)
                pred_test_norm_tensor = torch.tensor(pred_test_norm_np, dtype=torch.float32, device=device)
                test_loss = loss_fn(pred_test_norm_tensor, y_test_norm)

                #print(f"    Simulated {len(pred_test_norm_np)} test steps.")
            except Exception as e:
                print(f"    Error during simulation with states available: {e}")

        else: # states_available is False
            #print(f"  Simulation Mode: States Unavailable (Output Error with State Handoff, No Warmup)")

            # --- Simulate Testing Data ---
            #print(f"    Simulating full test sequence ({len(u_test_norm)} steps)...")
            current_state_test = state_train
            output_pred_list_test = []
            for t in range(len(u_test_norm)):
                current_input = u_test_norm[t].unsqueeze(0)
                next_state_test, output_pred_test = model(current_state_test, current_input,update_grid=update_grid_flag)    
                output_pred_list_test.append(output_pred_test)
                current_state_test = next_state_test
            pred_test_norm = torch.cat(output_pred_list_test, dim=0)

            test_loss = loss_fn(pred_test_norm, y_test_norm)
            
    return test_loss


def run_test_simulation_and_loss_pyKAN(state_train, model, dataset, device, loss_fn):
    """
    Runs simulation on the test set and calculates the normalized MSE loss.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (SystemIdentificationDataset): Dataset object with test data.
        device (str or torch.device): Device for computation.
        loss_fn (callable): The loss function (e.g., nn.MSELoss()).
        final_state_from_train (torch.Tensor, optional):
            The final hidden state after training simulation, used for state handoff
            when dataset.states_available is False. Required in that case.

    Returns:
        float: The calculated normalized MSE loss on the test set.
               Returns float('nan') if simulation or loss calculation fails.
    """
    model.eval()  # Set model to evaluation mode

    # Get test data (ensure it's on the right device)
    u_test_norm = dataset.u_test_norm.to(device)
    y_test_norm = dataset.y_test_norm.to(device)

    pred_test_norm_tensor = None


    with torch.no_grad():
        if dataset.states_available:
            #print(f"  Simulation Mode: States Available (using known initial states)")
            try:
                initial_state_test = dataset.X_test_norm[0].unsqueeze(0) if dataset.X_test_norm[0].ndim == 1 else dataset.X_test_norm[0]
                initial_state_test = initial_state_test.to(device)

                # Use predict_recursive (returns numpy), convert back to tensor
                pred_test_norm_np, _ = predict_recursive(model, initial_state_test, u_test_norm)
                pred_test_norm_tensor = torch.tensor(pred_test_norm_np, dtype=torch.float32, device=device)
                test_loss = loss_fn(pred_test_norm_tensor, y_test_norm)

                #print(f"    Simulated {len(pred_test_norm_np)} test steps.")
            except Exception as e:
                print(f"    Error during simulation with states available: {e}")

        else: # states_available is False
            #print(f"  Simulation Mode: States Unavailable (Output Error with State Handoff, No Warmup)")

            # --- Simulate Testing Data ---
            #print(f"    Simulating full test sequence ({len(u_test_norm)} steps)...")
            current_state_test = state_train
            output_pred_list_test = []
            for t in range(len(u_test_norm)):
                current_input = u_test_norm[t].unsqueeze(0)
                next_state_test, output_pred_test = model(current_state_test, current_input)    
                output_pred_list_test.append(output_pred_test)
                current_state_test = next_state_test
            pred_test_norm = torch.cat(output_pred_list_test, dim=0)

            test_loss = loss_fn(pred_test_norm, y_test_norm)
            
    return test_loss



#%%%% gemini ai studio 2.5 symbolic from kan original

from sklearn.linear_model import LinearRegression
# Need to define or import SYMBOLIC_LIB and potentially denormalize_data
# Example SYMBOLIC_LIB subset (use the one from pykan or define your own)
# Make sure the functions take torch tensors as input!
SYMBOLIC_LIB_TORCH = {
    'x': lambda x: x,
    'x^2': lambda x: x**2,
    'x^3': lambda x: x**3,
    'sin': lambda x: torch.sin(x),
    'cos': lambda x: torch.cos(x),
    'exp': lambda x: torch.exp(x),
    'log': lambda x: torch.log(torch.abs(x) + 1e-8), # Add abs and epsilon for safety
    'abs': lambda x: torch.abs(x),
    'tanh': lambda x: torch.tanh(x),
    'sqrt': lambda x: torch.sqrt(torch.relu(x)), # Ensure non-negative input for sqrt
    '0': lambda x: torch.zeros_like(x),
    '1': lambda x: torch.ones_like(x),
    'gaussian': lambda x: torch.exp(-x**2),
    '1/x': lambda x: 1/(x + torch.sign(x)*1e-6 + 1e-8), # Epsilon for stability
    # Add more as needed
}

# --- Keep other utility functions ---
# ... normalize_data, denormalize_data, simulate_model, etc.

# --- NEW Symbolic Fitting Function ---
def fit_symbolic_approx(kan_layer, layer_idx, input_idx, output_idx, dataset, fun_name,
                        n_points=101, a_range=(-5, 5), b_range=(-5, 5), grid_search_points=51, verbose=True):
    """
    Fits y = c * fun(a * x + b) + d to the activation of a specific edge
    in an efficient-kan KANLinear layer.

    Args:
        kan_layer: The efficient_kan.KANLinear layer.
        layer_idx: The index of this layer in the KAN model (for range estimation).
        input_idx: The input index (i).
        output_idx: The output index (j).
        dataset: Dataset object for range estimation.
        fun_name (str): Name of the symbolic function from SYMBOLIC_LIB_TORCH to fit.
        n_points: Number of points to sample for fitting.
        a_range, b_range: Search range for affine parameters a, b.
        verbose: Print fitting results.

    Returns:
        tuple: (params, r2)
               params: torch.Tensor([a, b, c, d]) of best fit parameters.
               r2: Coefficient of determination (float). Returns -1.0 if fit fails.
    """
    if fun_name not in SYMBOLIC_LIB_TORCH:
        print(f"Error: Function '{fun_name}' not in SYMBOLIC_LIB_TORCH.")
        return None, -1.0

    symbolic_func_torch = SYMBOLIC_LIB_TORCH[fun_name]
    model_device = kan_layer.grid.device

    # --- Get input range (Simplified - uses layer 0 range for all for now) ---
    # More accurate would be to pass estimated ranges like in plot functions
    min_val, max_val = -1.0, 1.0


    x_i_norm_range = torch.linspace(min_val, max_val, n_points, device=model_device).unsqueeze(1)

    # --- Calculate target KAN edge activation y_target ---
    fixed_input_val = 0.0
    with torch.no_grad():
        kan_layer_input = torch.full((n_points, kan_layer.in_features), fixed_input_val, device=model_device)
        kan_layer_input[:, input_idx] = x_i_norm_range.squeeze()
        spline_basis_i = kan_layer.b_splines(kan_layer_input)[:, input_idx, :]
        coeffs_i = kan_layer.scaled_spline_weight[output_idx, input_idx, :]
        spline_func = spline_basis_i @ coeffs_i.T
        activated_x_i = kan_layer.base_activation(x_i_norm_range)
        base_weight_ij = kan_layer.base_weight[output_idx, input_idx]
        base_func = activated_x_i.squeeze() * base_weight_ij
        y_target = spline_func + base_func 

    # --- Grid search for best a, b (similar to pykan's fit_params) ---
    grid_num = grid_search_points # Use the argument
    a_ = torch.linspace(a_range[0], a_range[1], steps=grid_num, device=model_device)
    b_ = torch.linspace(b_range[0], b_range[1], steps=grid_num, device=model_device)
    a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij') # Shape [grid_num, grid_num]


    # Calculate f(ax+b)
    with torch.no_grad():
         try:
             x_reshaped = x_i_norm_range.unsqueeze(-1) # Shape [n_points, 1, 1]
             a_grid_reshaped = a_grid.unsqueeze(0)     # Shape [1, grid_num, grid_num]
             b_grid_reshaped = b_grid.unsqueeze(0)     # Shape [1, grid_num, grid_num]
             transformed_x = a_grid_reshaped * x_reshaped + b_grid_reshaped # Shape [n_points, grid_num, grid_num]
             post_fun_all = symbolic_func_torch(transformed_x) # Should retain shape [n_points, grid_num, grid_num]
             post_fun_all = torch.nan_to_num(post_fun_all, nan=0.0, posinf=0.0, neginf=0.0)
         except Exception as e:
              print(f"Warning: Symbolic function '{fun_name}' failed: {e}. Returning poor fit.")
              return torch.tensor([1.0, 0.0, 1.0, 0.0]), -1.0


    # Calculate R^2
    # y_target shape: [n_points]
    y_target_reshaped = y_target[:, None, None] # Shape [n_points, 1, 1]
    x_mean = torch.mean(post_fun_all, dim=0, keepdim=True) # Shape [1, grid_num, grid_num] <-- OK
    y_mean = torch.mean(y_target) # Scalar mean <-- OK

    # Sum over the 'n_points' dimension (dim=0)
    numerator = torch.sum((post_fun_all - x_mean) * (y_target_reshaped - y_mean), dim=0)**2 # Shape [grid_num, grid_num] <-- OK
    denominator_term1 = torch.sum((post_fun_all - x_mean)**2, dim=0) # Shape [grid_num, grid_num] <-- OK
    denominator_term2_scalar = torch.sum((y_target - y_mean)**2) # Scalar <-- OK

    denominator = denominator_term1 * denominator_term2_scalar # Shape [grid_num, grid_num] <-- OK

    r2_all = numerator / (denominator + 1e-8)
    r2_all = torch.nan_to_num(r2_all) # Shape [grid_num, grid_num]

    # --- Find best a, b (rest of the function is ok) ---
    # ... (shape check for r2_all, argmax, unravel_index) ...
    expected_r2_shape = (grid_num, grid_num)
    if r2_all.shape != expected_r2_shape:
        print(f"Error: Unexpected shape for r2_all. Expected {expected_r2_shape}, got {r2_all.shape}")
        return torch.tensor([1.0, 0.0, 1.0, 0.0]), -1.0
    
    best_idx_flat = torch.argmax(r2_all) # Flat index (0 to grid_num*grid_num - 1)

    # *** Use np.unravel_index for robust conversion ***
    # Needs numpy array and shape tuple
    a_id, b_id = np.unravel_index(best_idx_flat.cpu().numpy(), r2_all.shape)

    # Ensure indices are within bounds (should be guaranteed by unravel_index)
    # a_id = min(a_id, grid_num - 1)
    # b_id = min(b_id, grid_num - 1)

    a_best, b_best = a_[a_id], b_[b_id]
    r2_best = r2_all[a_id, b_id].item()

    # --- Fit c, d using Linear Regression ---
    # (Linear Regression part same as before)
    with torch.no_grad():
        post_fun_best_ab = symbolic_func_torch(a_best * x_i_norm_range + b_best)
        post_fun_best_ab = torch.nan_to_num(post_fun_best_ab, nan=0.0, posinf=0.0, neginf=0.0)
    X_fit = post_fun_best_ab.cpu().numpy(); Y_fit = y_target.cpu().numpy()
    try:
        reg = LinearRegression().fit(X_fit, Y_fit)
        c_best = torch.tensor(reg.coef_[0], device=model_device, dtype=torch.float32)
        d_best = torch.tensor(reg.intercept_, device=model_device, dtype=torch.float32)
    except ValueError: c_best=torch.tensor(1.0, device=model_device); d_best=torch.tensor(0.0, device=model_device)

    if verbose: # (Print results - same as before)
        print(f"Fit results for ({layer_idx},{input_idx},{output_idx}) ... R2 = {r2_best:.4f}")

    params = torch.stack([a_best, b_best, c_best, d_best])
    return params, r2_best
#%%%

def simulate_and_extract_states(model, dataset, data_type='train'):
    """Simulate the model and extract the internal state trajectory."""
    print(f"Simulating model on {data_type} data...")
    
    model.eval()
    
    if data_type == 'train':
        u_data = dataset.u_train_norm
        y_data = dataset.y_train_norm
        time_vector = dataset.t_train if hasattr(dataset, 't_train') else torch.arange(len(u_data)) * dataset.dt
    else:
        u_data = dataset.u_test_norm
        y_data = dataset.y_test_norm
        time_vector = dataset.t_test if hasattr(dataset, 't_test') else torch.arange(len(u_data)) * dataset.dt
    
    # Initialize state trajectory storage
    state_trajectory = []
    output_trajectory = []
    
    # Initialize first state (learned internal state)
    current_state = torch.zeros(1, model.state_dim)
    
    with torch.no_grad():
        for i in range(len(u_data)):
            # Store current state
            state_trajectory.append(current_state.clone())
            
            # Get current input
            current_input = u_data[i].unsqueeze(0)
            
            # Forward pass
            next_state, output_pred = model(current_state, current_input)
            
            # Store output
            output_trajectory.append(output_pred.clone())
            
            # Update state for next iteration
            current_state = next_state
    
    # Convert to tensors
    state_traj_tensor = torch.cat(state_trajectory, dim=0)  # [time_steps, state_dim]
    output_traj_tensor = torch.cat(output_trajectory, dim=0)  # [time_steps, output_dim]
    
    # Denormalize for interpretation
    u_denorm = _utils.denormalize_data(u_data, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
    y_denorm = _utils.denormalize_data(y_data, dataset.norm_flag, dataset.x_min, dataset.x_max, normalize=(dataset.norm_flag != 'nothing'))
    output_pred_denorm = _utils.denormalize_data(output_traj_tensor, dataset.norm_flag, dataset.x_min, dataset.x_max, normalize=(dataset.norm_flag != 'nothing'))
    
    # Convert to numpy for plotting
    if isinstance(u_denorm, torch.Tensor):
        u_denorm = u_denorm.numpy()
    if isinstance(y_denorm, torch.Tensor):
        y_denorm = y_denorm.numpy()
    if isinstance(output_pred_denorm, torch.Tensor):
        output_pred_denorm = output_pred_denorm.numpy()
    if isinstance(time_vector, torch.Tensor):
        time_vector = time_vector.numpy()
    
    # Ensure all data is properly shaped
    states_np = state_traj_tensor.numpy()
    
    # Debug prints
    print(f"Data shapes for plotting:")
    print(f"  time: {time_vector.shape}")
    print(f"  states: {states_np.shape}")
    print(f"  input: {u_denorm.shape}")
    print(f"  output_true: {y_denorm.shape}")
    print(f"  output_pred: {output_pred_denorm.shape}")
    
    results = {
        'time': time_vector,
        'states_norm': states_np,
        'states_raw': states_np,  # States are internal, no denormalization
        'input_denorm': u_denorm,
        'output_true_denorm': y_denorm,
        'output_pred_denorm': output_pred_denorm,
        'data_type': data_type
    }
    
    print(f"Simulation completed. State trajectory shape: {state_traj_tensor.shape}")
    return results

#%%
import pandas as pd
import os

def save_results_to_excel(new_df, excel_path="/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/out.xlsx"):
    """
    Appends model results to an Excel file (creates it if missing).
    """

    # If file exists, append; otherwise create new file
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_excel(excel_path, index=False)
    else:
        new_df.to_excel(excel_path, index=False)

    print(f"✅ Results saved to {excel_path}")

#%%% full data sweep and update grid

def update_full_data(model, dataset,states_available_flag, device):
    with torch.no_grad():
        u_full = dataset.u_train_norm  # shape [T, u_dim]
    
        if states_available_flag:
            # Use provided (normalized) states directly
            state_full = dataset.X_train_norm  # shape [T, state_dim]
        else:
            # Roll out model to build full state trajectory without altering grids
            T = u_full.shape[0]
            state_full = torch.empty(T, model.state_dim, device=device)
            rollout_state = (
                torch.zeros(1, model.state_dim, device=device)
                + torch.randn(1, model.state_dim, device=device) * 0.01
            )
            for t in range(T):
                state_full[t:t+1] = rollout_state
                next_state, _ = model(rollout_state, u_full[t:t+1], update_grid=False)
                rollout_state = next_state.detach()
    
        # Single grid update calls (only if modules exist)
        if hasattr(model, "state_kan_model") and model.state_kan_model is not None:
            _ = model.state_kan_model(state=state_full, u=u_full, update_grid=True)
        if hasattr(model, "output_kan_model") and model.output_kan_model is not None:
            _ = model.output_kan_model(state=state_full, u=u_full, update_grid=True)

    return model


def update_full_data_pyKAN(model, dataset,states_available_flag, device):
    with torch.no_grad():
        u_full = dataset.u_train_norm  # shape [T, u_dim]
    
        if states_available_flag:
            # Use provided (normalized) states directly
            state_full = dataset.X_train_norm  # shape [T, state_dim]
        else:
            # Roll out model to build full state trajectory without altering grids
            T = u_full.shape[0]
            state_full = torch.empty(T, model.state_dim, device=device)
            rollout_state = (
                torch.zeros(1, model.state_dim, device=device)
                + torch.randn(1, model.state_dim, device=device) * 0.01
            )
            for t in range(T):
                state_full[t:t+1] = rollout_state
                next_state, _ = model(rollout_state, u_full[t:t+1])
                rollout_state = next_state.detach()
        
        v_full = torch.cat([state_full.to(device), u_full], dim=-1)

        # Single grid update calls (only if modules exist)
        if hasattr(model, "state_kan_model") and model.state_kan_model is not None:
            _ = model.state_kan_model.kan.update_grid(v_full)
        if hasattr(model, "output_kan_model") and model.output_kan_model is not None:
            _ = model.output_kan_model.kan.update_grid(v_full)

    return model

def pass_full_data_pyKAN(model, dataset,states_available_flag, device):
    with torch.no_grad():
        u_full = dataset.u_train_norm  # shape [T, u_dim]
    
        if states_available_flag:
            # Use provided (normalized) states directly
            state_full = dataset.X_train_norm  # shape [T, state_dim]
        else:
            # Roll out model to build full state trajectory without altering grids
            T = u_full.shape[0]
            state_full = torch.empty(T, model.state_dim, device=device)
            rollout_state = (
                torch.zeros(1, model.state_dim, device=device)
                + torch.randn(1, model.state_dim, device=device) * 0.01
            )
            for t in range(T):
                state_full[t:t+1] = rollout_state
                next_state, _ = model(rollout_state, u_full[t:t+1])
                rollout_state = next_state.detach()
        
        v_full = torch.cat([state_full.to(device), u_full], dim=-1)

        # Single grid update calls (only if modules exist)
        if hasattr(model, "state_kan_model") and model.state_kan_model is not None:
             model.state_kan_model.kan(v_full)
             model.state_kan_model.kan.plot()
        if hasattr(model, "output_kan_model") and model.output_kan_model is not None:
             model.output_kan_model.kan(v_full)
             model.output_kan_model.kan.plot()

    return model


def get_kan_contribution(model, dataset,states_available_flag, device):
    with torch.no_grad():
        u_full = dataset.u_train_norm  # shape [T, u_dim]
    
        if states_available_flag:
            # Use provided (normalized) states directly
            state_full = dataset.X_train_norm  # shape [T, state_dim]
        else:
            # Roll out model to build full state trajectory without altering grids
            T = u_full.shape[0]
            state_full = torch.empty(T, model.state_dim, device=device)
            rollout_state = (
                torch.zeros(1, model.state_dim, device=device)
                + torch.randn(1, model.state_dim, device=device) * 0.01
            )
            for t in range(T):
                state_full[t:t+1] = rollout_state
                next_state, _ = model(rollout_state, u_full[t:t+1], update_grid=False)
                rollout_state = next_state.detach()
    
        state_nonlinear_correction, output_nonlinear_correction = model.get_corrections_only(state=state_full, u=u_full)

    return state_nonlinear_correction, output_nonlinear_correction, state_full, u_full
