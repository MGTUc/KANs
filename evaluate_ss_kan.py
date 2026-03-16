#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SS-KAN Evaluation Script
Load and evaluate trained models
"""

import torch
import numpy as np
import os
import _utils
import _plot
from data_class_SI import SystemIdentificationDataset
from model_definitions import get_dataset_params, create_model_from_params
from extract_model_params import get_model_params_from_saved_model

def evaluate_ss_kan_model(model_path, 
                          test_case=None, 
                          model_params=None,
                          device="cpu",
                          generate_plots=False,
                          states_dim = 2,
                          save_dir=None):
    """
    Evaluate a trained SS-KAN model
    
    Args:
        model_path: Path to saved model (.pth file)
        test_case: Dataset to use (extracted from filename if None)
        model_params: Model architecture (must match training!)
        device: PyTorch device
        generate_plots: Whether to generate visualization plots
        save_dir: Where to save plots (same dir as model if None)
        
    Returns:
        dict: Evaluation results and metrics
    """
    
    print(f"\n--- Evaluating SS-KAN Model ---")
    print(f"Model Path: {model_path}")
    
    # Check if model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Extract test case from filename if not provided
    if test_case is None:
        filename = os.path.basename(model_path)
        if "Luca-Airfoil-Exp" in filename:
            test_case = "Luca-Airfoil-Exp"
        elif "Silverbox" in filename:
            test_case = "Silverbox"
        else:
            raise ValueError("Could not determine test case from filename. Please specify test_case parameter.")
    
    print(f"Test Case: {test_case}")
    
    # Load dataset (same as training)
    dataset_params = get_dataset_params(test_case, states_dim)
    dataset = SystemIdentificationDataset(
        test_case=dataset_params['test_case_name'],
        test_flag=dataset_params['test_flag'],
        norm_flag=dataset_params['norm_flag'],
        device=device,
        states_available=dataset_params['states_available_flag'],
        init_matrices_flag=dataset_params['init_matrices_flag'],
    )
    
    # If model_params not provided, try to load from saved parameter file first
    if model_params is None:
        # Check if there's a corresponding parameter file
        params_path = model_path.replace('.pth', '_params.pth')
        
        if os.path.exists(params_path):
            print("Loading model parameters from saved parameter file...")
            try:
                saved_params = torch.load(params_path, map_location='cpu')
                model_params = saved_params['model_params']
                print("Model parameters loaded from parameter file:")
                print(f"  State KAN Hidden Layers: {model_params['state_kan_hidden_layers']}")
                print(f"  Output KAN Hidden Layers: {model_params['output_kan_hidden_layers'] if model_params['output_kan_enabled'] else 'Disabled'}")
                print(f"  Grid Size: {model_params['kan_grid_size']}")
                print(f"  Training Seed: {saved_params.get('seed', 'Unknown')}")
                print(f"  Best Epoch: {saved_params.get('epoch', 'Unknown')}")
            except Exception as e:
                print(f"Warning: Could not load parameter file {params_path}: {e}")
                model_params = None
        
        # If no parameter file or loading failed, extract from state dict
        if model_params is None:
            print("Extracting model parameters from saved state dictionary...")
            try:
                model_params, dataset_info = get_model_params_from_saved_model(model_path)
                print("Model parameters extracted successfully:")
                print(f"  State KAN Hidden Layers: {model_params['state_kan_hidden_layers']}")
                print(f"  Output KAN Hidden Layers: {model_params['output_kan_hidden_layers'] if model_params['output_kan_enabled'] else 'Disabled'}")
                print(f"  Grid Size: {model_params['kan_grid_size']}")
            except Exception as e:
                print(f"Warning: Could not extract model parameters from state dict: {e}")
                print("Falling back to default parameters...")
                from model_definitions import get_default_model_params
                model_params = get_default_model_params()
                print("Warning: Using default model parameters. Results may be incorrect if architecture doesn't match!")
    
    # Create model with same architecture as training
    model = create_model_from_params(dataset, model_params, device)
    
    # Load trained weights
    print(f"Loading model weights...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")
    
    # Set model to evaluation mode
    model.eval()


    print(f"Model Info:")
    if hasattr(dataset, 'A_init') and dataset.A_init is not None:
        print(f"  State Dim: {dataset.A_init.shape[0]}")
    print(f"  Input Dim: {dataset.u_dim}")  
    print(f"  Output Dim: {dataset.y_dim}")
    print(f"  State KAN: {model_params['state_kan_hidden_layers']}")
    print(f"  Output KAN: {model_params['output_kan_hidden_layers'] if model_params['output_kan_enabled'] else 'Disabled'}")
    print(f"  Grid Size: {model_params['kan_grid_size']}")
    
    # 1. Simulate model to get time series predictions
    print("\nRunning model simulation...")
    simulation_results = _utils.simulate_model(model, dataset, device)
    
    # 2. Calculate metrics from the simulation results
    print("Calculating evaluation metrics...")
    evaluation_metrics = _utils.calculate_metrics(simulation_results)
    
    # 3. Print the results
    print("\n--- Performance Metrics (Original Scale) ---")
    print(f"MAE (Train): {evaluation_metrics['mae_train']:.4f}")
    print(f"RMSE (Train): {evaluation_metrics['rmse_train']:.4f}")
    print(f"MAE (Test): {evaluation_metrics['mae_test']:.4f}")
    print(f"RMSE (Test): {evaluation_metrics['rmse_test']:.4f}")
    
    # Setup save directory for plots
    if save_dir is None:
        save_dir = os.path.dirname(model_path) + "/Figures"
        save_dir = None
        print('doing this here')
    
    #plot single activation - 3 inputs 1 output
    # _plot.visualize_kan_layer_option1(model.state_kan_model, 0, 0, save_dir,kan_flag='state',x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.state_kan_model, 0, 1, save_dir,kan_flag='state', x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.state_kan_model, 0, 2, save_dir,kan_flag='state', x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.state_kan_model, 1, 0, save_dir,kan_flag='state',x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.state_kan_model, 1, 1, save_dir,kan_flag='state', x_range=(-1, 1))

    # _plot.visualize_kan_layer_option1(model.output_kan_model, 0, 0, save_dir,kan_flag='output',x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.output_kan_model, 0, 1, save_dir,kan_flag='output', x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.output_kan_model, 0, 2, save_dir,kan_flag='output', x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.output_kan_model, 1, 0, save_dir,kan_flag='output',x_range=(-1, 1))
    # _plot.visualize_kan_layer_option1(model.output_kan_model, 1, 1, save_dir,kan_flag='output', x_range=(-1, 1))

    # After you have: model, simulation_results
    #save_dir = "Figures/KAN_layers_Silverbox"

    # 1) State KAN (both layers)
    state_kan = model.state_kan_model
    for layer_idx in (0, 1):
        in_feats = state_kan.kan.layers[layer_idx].in_features
        for input_dim in range(in_feats):
            _plot.visualize_kan_layer_option1_copilot(
                kan_model=state_kan,
                simulation_results=simulation_results,
                layer_idx=layer_idx,
                input_dim=input_dim,
                save_dir=save_dir,
                kan_flag="state",
                data_split="train",          
            )

    # 2) Output KAN (if enabled)
    if getattr(model, "output_kan_model", None) is not None:
        out_kan = model.output_kan_model
        for layer_idx in (0, 1):
            in_feats = out_kan.kan.layers[layer_idx].in_features
            for input_dim in range(in_feats):
                _plot.visualize_kan_layer_option1_copilot(
                    kan_model=out_kan,
                    simulation_results=simulation_results,
                    layer_idx=layer_idx,
                    input_dim=input_dim,
                    save_dir=save_dir,
                    kan_flag="output",
                    data_split="train",     
                )
        
    # 4. Generate plots if requested
    if generate_plots:
        print(f"\nGenerating visualization plots...")
        print(f"Plots will be saved to: {save_dir}")
        
        # --- Plot Simulation Results ---
        _plot.plot_simulation_results_single('train', simulation_results, dataset, save_dir=save_dir)
        _plot.plot_simulation_results_single('test', simulation_results, dataset, save_dir=save_dir)
        
        # --- Plot Input vs Output Relationship ---
        _plot.plot_input_output_comparision(simulation_results, dataset, save_dir=save_dir)
        
        # --- Plot State Relationships ---
        _plot.plot_state_relationships_separate(simulation_results, dataset, save_dir=save_dir)
        _plot.plot_state_relationships_separate_output(simulation_results, dataset, save_dir=save_dir)

        if hasattr(dataset, 'A_init') and dataset.A_init is not None:
            state_dim_plot = dataset.A_init.shape[0]
        else:
            # Fallback: extract from model parameters
            state_dim_plot = len(model_params['state_kan_hidden_layers']) if model_params['state_kan_hidden_layers'] else 3
        
        input_dim_plot = dataset.u_dim
        fixed_states_norm = [0.0] * state_dim_plot
        fixed_inputs_norm = [0.0] * input_dim_plot
        
        print(f"Using fixed norm states for KAN plots: {fixed_states_norm}")
        print(f"Using fixed norm inputs for KAN plots: {fixed_inputs_norm}")
        
        # --- Plot KAN Surfaces ---
        all_fixed_norm_values = ([0.0] * state_dim_plot) + ([0.0] * input_dim_plot)
        
        # Plot Output KAN Surface (vary state 0 and input 0)
        if hasattr(model, 'output_kan_model') and model.output_kan_model is not None:
            vary1_idx = 0  # Index for state 0
            vary2_idx = state_dim_plot + 0  # Index for input 0
            output_kan_out_idx = 0  # Index for system output y[0]
            
            fixed_vals_dict = {i: val for i, val in enumerate(all_fixed_norm_values)
                              if i != vary1_idx and i != vary2_idx}
            
            _plot.plot_kan_surface(
                model=model, dataset=dataset, kan_target='output',
                vary_dim1_idx=vary1_idx, vary_dim1_name="State x[0]",
                vary_dim2_idx=vary2_idx, vary_dim2_name="Input u[0]",
                fixed_values_norm=fixed_vals_dict,
                output_idx=output_kan_out_idx, output_name="System Output y",
                plot_normalized=False,
                num_points=100,
                save_dir=save_dir,
                plot_type='surface'
            )
        
        # Plot State KAN Surface
        if hasattr(model, 'state_kan_model') and model.state_kan_model is not None:
            vary1_idx = 0  # Index for state 0
            vary2_idx = state_dim_plot + 0  # Index for input 0
            state_kan_out_idx = 0  # Index for state correction Delta x[0]
            
            fixed_vals_dict = {i: val for i, val in enumerate(all_fixed_norm_values)
                              if i != vary1_idx and i != vary2_idx}
            
            _plot.plot_kan_surface(
                model=model, dataset=dataset, kan_target='state',
                vary_dim1_idx=vary1_idx, vary_dim1_name="State",
                vary_dim2_idx=vary2_idx, vary_dim2_name="Input",
                fixed_values_norm=fixed_vals_dict,
                output_idx=state_kan_out_idx, output_name="State Correction x",
                plot_normalized=False,
                num_points=100,
                save_dir=save_dir,
                plot_type='surface'
            )
        
        print("Visualization plots completed.")
    
    # Return evaluation results
    results = {
        'model_path': model_path,
        'test_case': test_case,
        'evaluation_metrics': evaluation_metrics,
        'simulation_results': simulation_results,
        'model_params': model_params,
        'dataset_params': dataset_params
    }
    
    return results

def find_best_models(search_dir, pattern="best_model_*.pth"):
    """
    Find all best model files in a directory
    
    Args:
        search_dir: Directory to search
        pattern: Filename pattern to match
        
    Returns:
        list: Paths to found model files
    """
    import glob
    pattern_path = os.path.join(search_dir, pattern)
    model_files = glob.glob(pattern_path)
    return sorted(model_files)

def evaluate_multiple_models(search_dir, test_case=None, generate_plots=True):
    """
    Evaluate multiple models in a directory
    
    Args:
        search_dir: Directory containing model files
        test_case: Test case (auto-detected if None)
        generate_plots: Whether to generate plots for each model
        
    Returns:
        dict: Results for each model
    """
    model_files = find_best_models(search_dir)
    
    if not model_files:
        print(f"No model files found in {search_dir}")
        return {}
    
    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  {f}")
    
    results = {}
    for model_path in model_files:
        print(f"\n{'='*60}")
        model_name = os.path.basename(model_path)
        print(f"Evaluating: {model_name}")
        
        try:
            result = evaluate_ss_kan_model(
                model_path=model_path,
                test_case=test_case,
                generate_plots=generate_plots
            )
            results[model_name] = result
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    import pandas as pd
    rows = []
    for model_name, model_info in results.items():
        metrics = model_info["evaluation_metrics"]
        params = model_info["model_params"]
        
        rows.append({
            "model_name": model_name,
            "mae_train": float(metrics["mae_train"]),
            "rmse_train": float(metrics["rmse_train"]),
            "mae_test": float(metrics["mae_test"]),
            "rmse_test": float(metrics["rmse_test"]),
            "state_layers": params.get("state_kan_hidden_layers"),
            "output_layers": params.get("output_kan_hidden_layers"),
            "grid_size": params.get("kan_grid_size"),
            "output_enabled": params.get("output_kan_enabled"),
            "batch_size": model_name.split("_batch_")[1].split("_")[0] if "_batch_" in model_name else None,
            "epoch": int(model_name.split("_epoch_")[1].split("_")[0]) if "_epoch_" in model_name else None
        })
    
    df = pd.DataFrame(rows)

    
    return df
#%%%
if __name__ == "__main__":
    # Example usage
    #evaluate_multiple_models('/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/model_saves_simple_1', test_case='Luca-Airfoil-Exp-pitchRate', generate_plots=False)
    # Single model evaluation
    model_path = 'model_saves_simple_1/best_model_Luca-Airfoil-Exp-pitchRate_epoch_99_state_[5]_output_[0]_batch_64_50.pth'
    model_path = 'model_saves_simple_2/best_model_Silverbox_epoch_99_state_[1]_output_[0]_batch_64_5.pth'
    #model_path = 'your_file_updated.pth'
    #model_params = {'state_kan_hidden_layers': [2], 'output_kan_hidden_layers': None, 'kan_grid_size': 5, 'output_kan_enabled': False, 'grid_range': [-1, 1], 'trainable_C': True, 'trainable_D': True}
    #model_path = 'model_saves_simple_2/best_model_Silverbox_epoch_99_state_[2]_output_[0]_batch_64_5.pth'
    model_path = 'test___model_saves_simple_2/model_Silverbox_epoch_19_state_[2]_output_[2]_batch_64.pth'
    if os.path.exists(model_path):
        # results = evaluate_ss_kan_model(
        #     model_path=model_path,
        #     test_case="Luca-Airfoil-Exp-pitchRate",
        #     generate_plots=False,
        #     states_dim = 1,
        #     save_dir='Figures/' + model_path
        # )
        
        results = evaluate_ss_kan_model(
            model_path=model_path,
            #model_params=model_params,
            test_case="Silverbox",
            generate_plots=False,
            states_dim = 2,
            save_dir='Figures/' + model_path
        )
        print("\nEvaluation completed!")
    else:
        print(f"Example model file not found: {model_path}")

