#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:02:16 2025

@author: cruz

Just copy pasted the plotting from silverbox for now
obv, no variable names are accurate
"""

import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.fft
import numpy as np
import os
import _utils
from _utils import denormalize_data
from torch.nn import functional as F # For interpolation

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    # "font.family": "serif",
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 30,
    "font.size": 30,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 24,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
}
plt.rcParams.update(tex_fonts)

#%%%
# --- Helper to get data safely ---
def _get_data(source, key, expected_len=None):
    """Safely gets numpy data, checks length."""
    data = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
    if data is None: return None
    if isinstance(data, torch.Tensor): data = data.cpu().numpy()
    if expected_len is not None and len(data) != expected_len:
        print(f"Warning: Length mismatch for {key}. Expected {expected_len}, got {len(data)}. Data skipped.")
        return None
    return data

# --- Simulation Time Series Plot ---
def plot_simulation_results_single(data_type, simulation_results, dataset, save_dir=None):
    """Plots input, output vs prediction, and error for Train OR Test data."""
    print(f"\n--- Generating Simulation Plot: {data_type.capitalize()} ---")
    fig, axes = None, None
    try:
        # Get data using helper
        pred = _get_data(simulation_results, f'pred_{data_type}_denorm')
        target = _get_data(simulation_results, f'target_{data_type}_denorm')
        u_norm = getattr(dataset, f'u_{data_type}_norm', None)

        if target is None or pred is None or u_norm is None:
            print(f"    Skipping plot: Missing data for {data_type}."); return None, None

        # Denormalize Input & Ensure Numpy
        u = denormalize_data(u_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        if isinstance(u, torch.Tensor): u = u.cpu().numpy()

        # Sync Lengths
        len_compare = min(len(target), len(pred), len(u))
        if len_compare <= 0: print(f"    Skipping plot: Zero length data for {data_type}."); return None, None
        target, pred, u = target[:len_compare], pred[:len_compare], u[:len_compare]
        time_axis = np.arange(len_compare) * dataset.dt

        # Create Figure & Plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12),height_ratios=[1,2,1], sharex=True)
        #fig.suptitle(f'Simulation: {dataset.test_case} - {data_type.capitalize()}', fontsize=16)

        axes[0].plot(time_axis, u, 'g-', linewidth=1)
        axes[0].set_ylabel('Input u(t)'); axes[0].grid(True); 
        axes[0].set_ylabel(r'$\alpha$(t)'); axes[0].grid(True); 

        axes[1].plot(time_axis, target, 'k-', label='Exp', linewidth=1.5)
        axes[1].plot(time_axis, pred, 'r--', label='SS-KAN', linewidth=1.5, alpha=0.8)
        axes[1].set_ylabel('Output y(t)'); axes[1].grid(True); axes[1].legend()
        axes[1].set_ylabel('$C_l$'); axes[1].grid(True); axes[1].legend()

        error = (target - pred)
        axes[2].plot(time_axis, error, 'b-', linewidth=1)
        axes[2].set_xlabel('Time (s)'); 
        axes[2].set_ylabel('Abs Error')
        axes[2].grid(True); 
        axes[2].set_ylim([-0.5, 0.5])
        axes[2].set_xlim([0, 800])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()

        # Save
        if save_dir:
            filename = f'{data_type}_simulation_plot'
            save_path = os.path.join(save_dir, filename)
            if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
            try: plt.savefig(save_path+'.pdf', dpi=300); plt.savefig(save_path+'.png', dpi=300, transparent=True);

            except Exception as e: print(f"Error saving plot: {e}")
        plt.show()
        return fig, axes

    except Exception as e:
        print(f"Error plotting simulation for {data_type}: {e}"); import traceback; traceback.print_exc()
        if fig: plt.close(fig); return None, None

# --- Input-Output Relationship Plot ---
def plot_input_output_relationship(simulation_results, dataset, save_dir=None):
    """Generates scatter plots comparing system output (y) vs input (u)."""
    print("\n--- Generating Input vs Output Plot ---")
    fig, axes = None, None
    try:
        target_train = _get_data(simulation_results, 'target_train_denorm')
        target_test = _get_data(simulation_results, 'target_test_denorm')
        u_train_norm = getattr(dataset, 'u_train_norm', None)
        u_test_norm = getattr(dataset, 'u_test_norm', None)

        if not all([target_train is not None, u_train_norm is not None, target_test is not None, u_test_norm is not None]):
            print("Warning: Missing data for input-output plot."); return None, None

        u_train = denormalize_data(u_train_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        u_test = denormalize_data(u_test_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        if isinstance(u_train, torch.Tensor): u_train = u_train.cpu().numpy()
        if isinstance(u_test, torch.Tensor): u_test = u_test.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f'Input-Output Relationship: {dataset.test_case}', fontsize=16)

        len_tr = min(len(target_train), len(u_train)); len_te = min(len(target_test), len(u_test))
        axes[0].scatter(u_train[:len_tr], target_train[:len_tr], s=1, alpha=0.5, label='Train', c='blue')
        axes[0].set(xlabel=r'$\alpha$(t)', ylabel='$C_l$', title='Training'); axes[0].grid(True); axes[0].legend()

        axes[1].scatter(u_test[:len_te], target_test[:len_te], s=1, alpha=0.5, label='Test', c='orange')
        axes[1].set(xlabel=r'$\alpha$(t)', title='Testing'); axes[1].grid(True); axes[1].legend()

        plt.tight_layout()

        if save_dir:
             save_path = os.path.join(save_dir, 'input_output_relationship.png')
             if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
             try: plt.savefig(save_path, dpi=150); print(f"Plot saved to {save_path}")
             except Exception as e: print(f"Error saving plot: {e}")
        plt.show()
        return fig, axes

    except Exception as e:
        print(f"Error generating input-output plot: {e}"); import traceback; traceback.print_exc()
        if fig: plt.close(fig); return None, None


def plot_input_output_comparision(simulation_results, dataset, save_dir=None):
    """Generates scatter plots comparing system output (y) vs input (u)."""
    print("\n--- Generating Input vs Output Plot ---")
    fig, axes = None, None
    try:
        target_train = _get_data(simulation_results, 'target_train_denorm')
        pred_train = _get_data(simulation_results, 'pred_train_denorm')
        target_test = _get_data(simulation_results, 'target_test_denorm')
        u_train_norm = getattr(dataset, 'u_train_norm', None)
        u_test_norm = getattr(dataset, 'u_test_norm', None)

        if not all([target_train is not None, u_train_norm is not None, target_test is not None, u_test_norm is not None]):
            print("Warning: Missing data for input-output plot."); return None, None

        u_train = denormalize_data(u_train_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        u_test = denormalize_data(u_test_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        if isinstance(u_train, torch.Tensor): u_train = u_train.cpu().numpy()
        if isinstance(u_test, torch.Tensor): u_test = u_test.cpu().numpy()

        for j in range(u_train.shape[1]):      

            fig, axes = plt.subplots(figsize=(12, 9))

            len_tr = min(len(target_train), len(u_train)); len_te = min(len(target_test), len(u_test))
            axes.scatter(u_train[:len_tr,j], target_train[:len_tr], s=1, alpha=0.5, label='Exp', c='black')
            axes.scatter(u_train[:len_tr,j], pred_train[:len_tr], s=1, alpha=0.5, label='SS-KAN', c='blue')
            #axes.plot(u_train[:len_tr], pred_train[:len_tr], alpha=0.99, label='Exp', c='black')
            #axes.plot(u_train[:len_tr], pred_train[:len_tr], alpha=0.2, label='SS-KAN', c='red')
            axes.set(xlabel=rf'Input {j} $\alpha$(t)', ylabel='$C_l$')
            axes.grid(True) 
            axes.legend()
    
            plt.tight_layout()
    
            if save_dir:
                 save_path = os.path.join(save_dir, f'input_output_comparision_{j}')
                 if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
                 try: plt.savefig(save_path+'.pdf', dpi=300); plt.savefig(save_path+'.png', dpi=300, transparent=True);
                 except Exception as e: print(f"Error saving plot: {e}")
            plt.show()
        return 

    except Exception as e:
        print(f"Error generating input-output plot: {e}"); import traceback; traceback.print_exc()
        if fig: plt.close(fig); return None, None

# --- State Relationships Plot ---
def plot_state_relationships(simulation_results, dataset, save_dir=None):
    """Generates scatter plots of state vs input and state vs output."""
    print("\n--- Generating State Relationship Plots ---")
    figs_axes_list = []
    try:
        # Determine state source
        state_train_key = 'pred_state_train_denorm' if dataset.states_available and 'pred_state_train_denorm' in simulation_results else 'pred_state_train_norm'
        state_test_key = 'pred_state_test_denorm' if dataset.states_available and 'pred_state_test_denorm' in simulation_results else 'pred_state_test_norm'
        state_label = state_train_key.replace('pred_', '').replace('_norm', ' (Norm)').replace('_denorm', ' (Denorm)')
        print(f"    Plotting {state_label.split('(')[0].strip()} states.")

        # Get required data
        state_train = _get_data(simulation_results, state_train_key)
        state_test = _get_data(simulation_results, state_test_key)
        target_train = _get_data(simulation_results, 'target_train_denorm')
        target_test = _get_data(simulation_results, 'target_test_denorm')
        u_train_norm = getattr(dataset, 'u_train_norm', None)
        u_test_norm = getattr(dataset, 'u_test_norm', None)

        if state_train is None and state_test is None: print("Error: No state data."); return None
        if target_train is None or u_train_norm is None or target_test is None or u_test_norm is None:
             print("Warning: Missing target/input data for state relationships."); return None

        u_train = denormalize_data(u_train_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        u_test = denormalize_data(u_test_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        if isinstance(u_train, torch.Tensor): u_train = u_train.cpu().numpy()
        if isinstance(u_test, torch.Tensor): u_test = u_test.cpu().numpy()

        state_dim = state_train.shape[1] if state_train is not None else state_test.shape[1]

        # Plotting Loop
        for data_type in ['train', 'test']:
            state = state_train if data_type == 'train' else state_test
            u = u_train if data_type == 'train' else u_test
            target = target_train if data_type == 'train' else target_test

            if state is None: print(f"    Skipping {data_type} state plots."); continue

            len_compare = min(len(state), len(u), len(target))
            if len_compare == 0: print(f"    Skipping {data_type} state plots (zero length)."); continue
            state, u, target = state[:len_compare], u[:len_compare], target[:len_compare]

            fig, axes = plt.subplots(state_dim, 2, figsize=(10, 3 * state_dim), sharex='col', sharey='row')
            if state_dim == 1: axes = np.array([axes]) # Ensure 2D indexing

            fig.suptitle(f'{state_label} Relationships ({dataset.test_case} - {data_type.capitalize()})', fontsize=16)

            for i in range(state_dim):
                axes[i, 0].scatter(u, state[:, i], s=1, alpha=0.5, c='blue')
                axes[i, 0].set(ylabel=f'State x[{i+1}]', title='State vs Input' if i==0 else ''); axes[i, 0].grid(True)
                if i == state_dim - 1: axes[i, 0].set_xlabel('Input u(t)')

                axes[i, 1].scatter(target, state[:, i], s=1, alpha=0.5, c='orange')
                axes[i, 1].set(title='State vs Output' if i==0 else ''); axes[i, 1].grid(True)
                if i == state_dim - 1: axes[i, 1].set_xlabel('Output y(t)')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_dir:
                 save_path = os.path.join(save_dir, f'state_relationships_{data_type}.png')
                 if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
                 try: plt.savefig(save_path, dpi=150); print(f"Plot saved to {save_path}")
                 except Exception as e: print(f"Error saving plot: {e}")
            plt.show()
            figs_axes_list.append((fig, axes))

        return figs_axes_list

    except Exception as e:
        print(f"Error generating state relationship plots: {e}"); import traceback; traceback.print_exc()
        plt.close('all'); 
        return None



def plot_state_relationships_separate(simulation_results, dataset, save_dir=None):
    """Generates scatter plots of state vs input and state vs output."""
    print("\n--- Generating State Relationship Plots ---")
    figs_axes_list = []
    try:
        # Determine state source
        state_train_key = 'pred_state_train_denorm' if dataset.states_available and 'pred_state_train_denorm' in simulation_results else 'pred_state_train_norm'
        state_test_key = 'pred_state_test_denorm' if dataset.states_available and 'pred_state_test_denorm' in simulation_results else 'pred_state_test_norm'
        state_label = state_train_key.replace('pred_', '').replace('_norm', ' (Norm)').replace('_denorm', ' (Denorm)')
        print(f"    Plotting {state_label.split('(')[0].strip()} states.")

        # Get required data
        state_train = _get_data(simulation_results, state_train_key)
        state_test = _get_data(simulation_results, state_test_key)
        u_train_norm = getattr(dataset, 'u_train_norm', None)
        u_test_norm = getattr(dataset, 'u_test_norm', None)

        if state_train is None and state_test is None:
            print("Error: No state data.")
            return None
        if u_train_norm is None and u_test_norm is None:
            print("Warning: Missing input data for state relationships.")
            return None

        # Prefer train split if available; otherwise use test
        use_train = state_train is not None and u_train_norm is not None
        state = state_train if use_train else state_test
        u_norm = u_train_norm if use_train else u_test_norm

        # Denormalize inputs and ensure numpy
        u = denormalize_data(u_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        if isinstance(u, torch.Tensor):
            u = u.cpu().numpy()
        # Ensure state is numpy
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if state is None or u is None:
            print("Warning: No data available for plotting in selected split.")
            return None

        state_dim = state.shape[1]

        # Ensure 2D arrays for inputs
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        u_dim = u.shape[1]

        # Align lengths with states
        n_use = min(len(state), len(u)) if hasattr(state, '__len__') and hasattr(u, '__len__') else 0
        if n_use == 0:
            print("Warning: Empty arrays after alignment.")
            return None
        state = state[:n_use]
        u = u[:n_use]

        for i in range(state_dim):
            # Make a small grid of subplots for all inputs
            ncols = min(3, u_dim)
            nrows = int(np.ceil(u_dim / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False, sharey=True)
            axes_flat = axes.flatten()

            for j in range(u_dim):
                ax = axes_flat[j]
                ax.scatter(u[:, j], state[:, i], s=1, alpha=0.5, c='black')
                ax.set_xlabel(f'Input u[{j}]')
                ax.grid(True)
                if j % ncols == 0:
                    ax.set_ylabel(f'State x[{i}]')
                if j == 0:
                    ax.set_title(f'State x[{i}] vs Inputs', fontsize=12)

            # Hide any unused subplots
            for k in range(u_dim, len(axes_flat)):
                axes_flat[k].set_visible(False)

            plt.tight_layout()

            if save_dir:
                 split_name = 'train' if use_train else 'test'
                 save_path = os.path.join(save_dir, f'state_relationships_separate_state{i}_{split_name}')
                 if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
                 try:
                     plt.savefig(save_path+'.pdf', dpi=300)
                     plt.savefig(save_path+'.png', dpi=300, transparent=True)
                 except Exception as e:
                     print(f"Error saving plot: {e}")
            plt.show()
    except Exception as e:
        print(f"Error generating state relationship plots: {e}"); import traceback; traceback.print_exc()
        plt.close('all'); 
        return fig, axes


def plot_state_relationships_separate_output(simulation_results, dataset, save_dir=None):
    """Generates scatter plots of state vs input and state vs output."""
    print("\n--- Generating State Relationship Plots ---")
    figs_axes_list = []
    try:
        # Determine state source
        state_train_key = 'pred_state_train_denorm' if dataset.states_available and 'pred_state_train_denorm' in simulation_results else 'pred_state_train_norm'
        state_test_key = 'pred_state_test_denorm' if dataset.states_available and 'pred_state_test_denorm' in simulation_results else 'pred_state_test_norm'
        state_label = state_train_key.replace('pred_', '').replace('_norm', ' (Norm)').replace('_denorm', ' (Denorm)')
        print(f"    Plotting {state_label.split('(')[0].strip()} states.")

        # Get required data
        state_train = _get_data(simulation_results, state_train_key)
        state_test = _get_data(simulation_results, state_test_key)
        target_train = _get_data(simulation_results, 'target_train_denorm')
        target_test = _get_data(simulation_results, 'target_test_denorm')
        u_train_norm = getattr(dataset, 'u_train_norm', None)
        u_test_norm = getattr(dataset, 'u_test_norm', None)

        if state_train is None and state_test is None: print("Error: No state data."); return None
        if target_train is None or u_train_norm is None or target_test is None or u_test_norm is None:
             print("Warning: Missing target/input data for state relationships."); return None

        u_train = denormalize_data(u_train_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        u_test = denormalize_data(u_test_norm, dataset.norm_flag, dataset.u_min, dataset.u_max, normalize=(dataset.norm_flag != 'nothing'))
        if isinstance(u_train, torch.Tensor): u_train = u_train.cpu().numpy()
        if isinstance(u_test, torch.Tensor): u_test = u_test.cpu().numpy()

        state_dim = state_train.shape[1] if state_train is not None else state_test.shape[1]
        
        for i in range(state_dim):
            fig, axes = plt.subplots(figsize=(12, 9))

            len_tr = min(len(target_train), len(u_train)); len_te = min(len(target_test), len(u_test))
            axes.scatter(target_train, state_train[:, i], s=1, alpha=0.5, c='blue')
            #axes.scatter(u_train[:len_tr], pred_train[:len_tr], s=1, alpha=0.5, label='SS-KAN', c='blue')
            #axes.plot(u_train[:len_tr], pred_train[:len_tr], alpha=0.99, label='Exp', c='black')
            #axes.plot(u_train[:len_tr], pred_train[:len_tr], alpha=0.2, label='SS-KAN', c='red')
            axes.set(xlabel='$C_l(t)$', ylabel=f'Internal State {i}')
            axes.grid(True) 
    
            plt.tight_layout()

            if save_dir:
                 save_path = os.path.join(save_dir, f'state_relationships_separate_output_{i}')
                 if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
                 try: plt.savefig(save_path+'.pdf', dpi=300); plt.savefig(save_path+'.png', dpi=300, transparent=True);
                 except Exception as e: print(f"Error saving plot: {e}")
            plt.show()
    except Exception as e:
        print(f"Error generating state relationship plots: {e}"); import traceback; traceback.print_exc()
        plt.close('all'); 
        return fig, axes

# --- Combined KAN Slice Plot ---
def plot_kan_slices(model, dataset, kan_target, fixed_state_norm_values, fixed_input_norm_values,
                    vary_state_idx=0, vary_input_idx=0, output_idx=0,
                    plot_normalized=False, num_points=100, save_dir=None):
    """Simplified & Combined: Plots State or Output KAN slices."""
    target_name_map = {'state': 'State KAN', 'output': 'Output KAN'}
    kan_module_name = f'{kan_target}_kan_model'
    target_name = target_name_map.get(kan_target)

    print(f"\n--- Generating {target_name} Slice Plots ---")
    kan_module = getattr(model, kan_module_name, None)
    if kan_module is None: print(f"Error: Model has no '{kan_module_name}'."); return
    if len(fixed_state_norm_values) != dataset.A_init.shape[0] or \
       len(fixed_input_norm_values) != dataset.u_dim:
       print("Error: Fixed value list length mismatch."); return
    # Add minimal index validation if desired

    model.eval(); device = next(model.parameters()).device
    state_dim = dataset.A_init.shape[0]; input_dim = dataset.u_dim
    norm_flag = dataset.norm_flag; normalize_needed = (norm_flag != 'nothing')

    fixed_state_t = torch.tensor(fixed_state_norm_values, dtype=torch.float32, device=device).unsqueeze(0)
    fixed_input_t = torch.tensor(fixed_input_norm_values, dtype=torch.float32, device=device).unsqueeze(0)

    # --- Sub-function to plot one slice ---
    def _plot_single_kan_slice(vary_type, vary_idx, fixed_state, fixed_input):
        fig, ax = None, None # Init for error handling
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            # Prepare X axis (Varying dimension)
            x_data_norm, x_label = None, "X Axis"
            if vary_type == 'input':
                x_min = dataset.u_train_norm[:, vary_idx].min().item()
                x_max = dataset.u_train_norm[:, vary_idx].max().item()
                x_data_norm = torch.linspace(x_min, x_max, num_points, device=device).unsqueeze(1)
                x_label = f"Input u[{vary_idx}]"
                norm_min, norm_max = (dataset.u_min[vary_idx], dataset.u_max[vary_idx]) if hasattr(dataset, 'u_min') else (None, None)
            elif vary_type == 'state':
                x_min, x_max = (-1.0, 1.0) # Default range
                if dataset.states_available and hasattr(dataset, 'X_train_norm'):
                    x_min = dataset.X_train_norm[:, vary_idx].min().item()
                    x_max = dataset.X_train_norm[:, vary_idx].max().item()
                x_data_norm = torch.linspace(x_min, x_max, num_points, device=device).unsqueeze(1)
                x_label = f"State x[{vary_idx}]"
                norm_min, norm_max = (None, None)
                if dataset.states_available and hasattr(dataset, 'x_min') and hasattr(dataset, 'x_dot_min'):
                    mins = torch.cat([dataset.x_min, dataset.x_dot_min]); maxs = torch.cat([dataset.x_max, dataset.x_dot_max])
                    if vary_idx < len(mins): norm_min, norm_max = mins[vary_idx], maxs[vary_idx]

            # Denormalize X axis if possible
            x_plot = x_data_norm.cpu().numpy()
            if not plot_normalized and norm_min is not None:
                _min = norm_min.item() if isinstance(norm_min, torch.Tensor) else norm_min
                _max = norm_max.item() if isinstance(norm_max, torch.Tensor) else norm_max
                x_plot = denormalize_data(x_data_norm, norm_flag, _min, _max, normalize=normalize_needed).cpu().numpy()
                x_label += " (Denorm)"
            else: x_label += " (Norm)"

            # Prepare KAN input batch
            batch_state = fixed_state.repeat(num_points, 1)
            batch_input = fixed_input.repeat(num_points, 1)
            if vary_type == 'input': batch_input[:, vary_idx] = x_data_norm.squeeze()
            if vary_type == 'state': batch_state[:, vary_idx] = x_data_norm.squeeze()

            # Evaluate KAN
            with torch.no_grad(): kan_output_norm = kan_module(state=batch_state, u=batch_input)
            output_selected_norm = kan_output_norm[:, output_idx].unsqueeze(1)

            # Prepare Y axis
            y_plot = output_selected_norm.cpu().numpy()
            y_label = ""
            if kan_target == 'state': y_label = f"State Correction $\\Delta$x[{output_idx}]"
            elif kan_target == 'output': y_label = f"System Output y[{output_idx}]"
            # Denormalize Y axis
            y_norm_min, y_norm_max = None, None
            if not plot_normalized:
                # ... (copy/adapt Y denorm logic from previous combined function) ...
                if kan_target == 'state' and dataset.states_available and hasattr(dataset, 'x_min') and hasattr(dataset, 'x_dot_min'):
                    mins = torch.cat([dataset.x_min, dataset.x_dot_min]); maxs = torch.cat([dataset.x_max, dataset.x_dot_max])
                    if output_idx < len(mins): y_norm_min, y_norm_max = mins[output_idx], maxs[output_idx]
                elif kan_target == 'output' and hasattr(dataset, 'x_min') and output_idx < len(dataset.x_min):
                    y_norm_min, y_norm_max = dataset.x_min[output_idx], dataset.x_max[output_idx]

                if y_norm_min is not None:
                    _min = y_norm_min.item() if isinstance(y_norm_min, torch.Tensor) else y_norm_min
                    _max = y_norm_max.item() if isinstance(y_norm_max, torch.Tensor) else y_norm_max
                    y_plot = denormalize_data(output_selected_norm, norm_flag, _min, _max, normalize=normalize_needed).cpu().numpy()
                    y_label += " (Denorm)"
                else: y_label += " (Norm)"
            else: y_label += " (Norm)"


            # Plotting
            ax.plot(x_plot, y_plot, linewidth=2, color='purple')
            ax.set_xlabel(x_label); ax.set_ylabel(y_label)
            fixed_info = "State Fixed" if vary_type == 'input' else "Input Fixed"
            ax.set_title(f"{target_name}: Output vs {vary_type.capitalize()}[{vary_idx}] ({fixed_info})")
            ax.grid(True); plt.tight_layout()

            # Saving
            if save_dir:
                filename = f'{kan_target}_kan_out{output_idx}_vs_{vary_type}{vary_idx}.png'
                save_path = os.path.join(save_dir, filename)
                if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
                try: plt.savefig(save_path, dpi=150); print(f"Plot saved to {save_path}")
                except Exception as e: print(f"Error saving plot: {e}")
            plt.show()
            return fig, ax
        except Exception as e:
            print(f"Error plotting {target_name} vs {vary_type.capitalize()}: {e}"); import traceback; traceback.print_exc()
            if fig: plt.close(fig); return None, None

    # --- Call the sub-function for both slices ---
    _plot_single_kan_slice(vary_type='input', vary_idx=vary_input_idx,
                           fixed_state=fixed_state_t, fixed_input=fixed_input_t)
    _plot_single_kan_slice(vary_type='state', vary_idx=vary_state_idx,
                           fixed_state=fixed_state_t, fixed_input=fixed_input_t)
    
#%%% 
   # --- NEW 3D KAN Surface Plot ---

def plot_kan_surface(model, dataset, kan_target,
                     vary_dim1_idx, vary_dim1_name,
                     vary_dim2_idx, vary_dim2_name,
                     fixed_values_norm, output_idx, output_name,
                     plot_normalized=False, num_points=30, save_dir=None, plot_type='surface'):
    """
    Visualizes a 2D slice of a KAN function as a 3D surface or heatmap.

    Varies two specified input dimensions while holding others constant,
    and plots one specified KAN output dimension.

    Args:
        model (nn.Module): The trained model containing the KAN.
        dataset (SystemIdentificationDataset): Dataset object for normalization info.
        kan_target (str): Specifies which KAN to visualize ('state' or 'output').
        vary_dim1_idx (int): Index of the first KAN input dimension to vary (e.g., a state index).
        vary_dim1_name (str): Name of the first varying dimension (for plot labels).
        vary_dim2_idx (int): Index of the second KAN input dimension to vary (e.g., an input index).
        vary_dim2_name (str): Name of the second varying dimension.
        fixed_values_norm (dict): Dict where keys are indices of KAN input dimensions
                                  *not* being varied, values are fixed normalized values.
        output_idx (int): Index of the KAN output dimension to plot (state correction or final y).
        output_name (str): Base name of the KAN output dimension (e.g., "State Correction", "System Output").
        plot_normalized (bool): If True, plots axes in normalized scale. Default False.
        num_points (int): Number of points per dimension for the grid (total points = num_points^2).
        save_dir (str, optional): Directory to save the plot.
        plot_type (str): 'surface' for 3D surface plot, 'contour' for 2D contour/heatmap.
    """
    target_name_map = {'state': 'State KAN', 'output': 'Output KAN'}
    kan_module_name = f'{kan_target}_kan_model'
    target_name = target_name_map.get(kan_target)
    print(f"\n--- Generating {target_name} {plot_type.capitalize()} Plot ---")
    print(f"    Output: {output_name}[{output_idx}]")
    print(f"    Varying: {vary_dim1_name}[{vary_dim1_idx}] and {vary_dim2_name}[{vary_dim2_idx}]")

    kan_module = getattr(model, kan_module_name, None)
    if kan_module is None: print(f"Error: Model has no '{kan_module_name}'."); return
    # Add validation for indices and fixed_values_norm keys if desired

    model.eval(); device = next(model.parameters()).device
    state_dim = dataset.A_init.shape[0]; input_dim = dataset.u_dim
    kan_input_dim = state_dim + input_dim # Assuming KAN takes state+input
    norm_flag = dataset.norm_flag; normalize_needed = (norm_flag != 'nothing')

    # --- Helper to get range and denorm params ---
    def _get_dim_info(dim_type, index):
        min_val, max_val = -1.0, 1.0
        norm_min, norm_max = None, None
        if dim_type == 'input' and hasattr(dataset, 'u_train_norm') and index < input_dim:
            min_val = dataset.u_train_norm[:, index].min().item()
            max_val = dataset.u_train_norm[:, index].max().item()
            if hasattr(dataset, 'u_min'): norm_min, norm_max = dataset.u_min[index], dataset.u_max[index]
        elif dim_type == 'state' and dataset.states_available and hasattr(dataset, 'X_train_norm') and index < state_dim:
            min_val = dataset.X_train_norm[:, index].min().item()
            max_val = dataset.X_train_norm[:, index].max().item()
            if hasattr(dataset, 'x_min') and hasattr(dataset, 'x_dot_min'):
                mins=torch.cat([dataset.x_min,dataset.x_dot_min]); maxs=torch.cat([dataset.x_max,dataset.x_dot_max])
                if index < len(mins): norm_min, norm_max = mins[index], maxs[index]
        return min_val, max_val, norm_min, norm_max

    # --- Determine type (state/input) for varying dims ---
    def _get_dim_type_name(index):
        if index < state_dim: return 'state', f"State x[{index}]"
        elif index < state_dim + input_dim: return 'input', f"Input u[{index-state_dim}]"
        else: raise ValueError("Index out of bounds")

    dim1_type, _ = _get_dim_type_name(vary_dim1_idx)
    dim2_type, _ = _get_dim_type_name(vary_dim2_idx)

    # --- Get Ranges and Denorm Params for Varying Dims ---
    min1, max1, norm_min1, norm_max1 = _get_dim_info(dim1_type, vary_dim1_idx if dim1_type=='state' else vary_dim1_idx-state_dim)
    min2, max2, norm_min2, norm_max2 = _get_dim_info(dim2_type, vary_dim2_idx if dim2_type=='state' else vary_dim2_idx-state_dim)

    # --- Create meshgrid for varying inputs (normalized) ---
    x1_norm_vec = torch.linspace(min1, max1, num_points, device=device)
    x2_norm_vec = torch.linspace(min2, max2, num_points, device=device)
    X1_norm_mesh, X2_norm_mesh = torch.meshgrid(x1_norm_vec, x2_norm_vec, indexing='ij')

    # --- Prepare batch input for KAN ---
    kan_input_flat = torch.zeros(num_points * num_points, kan_input_dim, device=device)
    # Set varying dimensions
    kan_input_flat[:, vary_dim1_idx] = X1_norm_mesh.flatten()
    kan_input_flat[:, vary_dim2_idx] = X2_norm_mesh.flatten()
    # Set fixed dimensions
    for idx, val in fixed_values_norm.items():
        if idx != vary_dim1_idx and idx != vary_dim2_idx: # Check index isn't being varied
            kan_input_flat[:, idx] = val

    # --- Evaluate KAN ---
    with torch.no_grad():
        # Reconstruct state/input parts for the KAN call
        state_flat = kan_input_flat[:, :state_dim]
        input_flat = kan_input_flat[:, state_dim:]
        kan_output_flat_norm = kan_module(state=state_flat, u=input_flat)

    # Extract desired output and reshape to grid
    output_selected_norm = kan_output_flat_norm[:, output_idx]
    Z_norm_mesh = output_selected_norm.reshape(num_points, num_points)

    # --- Prepare Axes for Plotting (Denormalize if requested) ---
    X1_plot, X2_plot, Z_plot = None, None, None
    x1_label, x2_label, z_label = f"{vary_dim1_name} (Norm)", f"{vary_dim2_name} (Norm)", f"{output_name}[{output_idx}] (Norm)"

    if plot_normalized:
        X1_plot, X2_plot = X1_norm_mesh.cpu().numpy(), X2_norm_mesh.cpu().numpy()
        Z_plot = Z_norm_mesh.cpu().numpy()
    else:
        # Denormalize X1
        if norm_min1 is not None:
             _min = norm_min1.item() if isinstance(norm_min1, torch.Tensor) else norm_min1
             _max = norm_max1.item() if isinstance(norm_max1, torch.Tensor) else norm_max1
             X1_plot = denormalize_data(X1_norm_mesh, norm_flag, _min, _max, normalize=normalize_needed).cpu().numpy()
             x1_label = f"{vary_dim1_name} (Denorm)"
        else: X1_plot = X1_norm_mesh.cpu().numpy()
        # Denormalize X2
        if norm_min2 is not None:
             _min = norm_min2.item() if isinstance(norm_min2, torch.Tensor) else norm_min2
             _max = norm_max2.item() if isinstance(norm_max2, torch.Tensor) else norm_max2
             X2_plot = denormalize_data(X2_norm_mesh, norm_flag, _min, _max, normalize=normalize_needed).cpu().numpy()
             x2_label = f"{vary_dim2_name} (Denorm)"
        else: X2_plot = X2_norm_mesh.cpu().numpy()
        # Denormalize Z (KAN Output)
        z_norm_min, z_norm_max = None, None
        # ... (Adapt Y denorm logic from plot_kan_slices based on kan_target/output_idx) ...
        if kan_target == 'state' and dataset.states_available and hasattr(dataset, 'x_min') and hasattr(dataset, 'x_dot_min'):
            mins=torch.cat([dataset.x_min,dataset.x_dot_min]); maxs=torch.cat([dataset.x_max,dataset.x_dot_max])
            if output_idx < len(mins): z_norm_min, z_norm_max = mins[output_idx], maxs[output_idx]
        elif kan_target == 'output' and hasattr(dataset, 'x_min') and output_idx < len(dataset.x_min):
            z_norm_min, z_norm_max = dataset.x_min[output_idx], dataset.x_max[output_idx]

        if z_norm_min is not None:
             _min = z_norm_min.item() if isinstance(z_norm_min, torch.Tensor) else z_norm_min
             _max = z_norm_max.item() if isinstance(z_norm_max, torch.Tensor) else z_norm_max
             Z_plot = denormalize_data(Z_norm_mesh, norm_flag, _min, _max, normalize=normalize_needed).cpu().numpy()
             z_label = f"{output_name}[{output_idx}] (Denorm)"
        else: Z_plot = Z_norm_mesh.cpu().numpy()

    # --- Create Plot ---
    fig = plt.figure(figsize=(10, 8))
    if plot_type == 'surface':
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X1_plot, X2_plot, Z_plot, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel(x1_label); ax.set_ylabel(x2_label); ax.set_zlabel(z_label)
        fig.colorbar(surf, shrink=0.5, aspect=10, label=z_label)
        # Optional: Set viewing angle
        # ax.view_init(elev=20., azim=-65)
    elif plot_type == 'contour':
        ax = fig.add_subplot(111)
        contour = ax.contourf(X1_plot, X2_plot, Z_plot, cmap='viridis', levels=20) # Filled contour
        # ax.contour(X1_plot, X2_plot, Z_plot, colors='k', levels=10, linewidths=0.5) # Add contour lines
        ax.set_xlabel(x1_label); ax.set_ylabel(x2_label)
        fig.colorbar(contour, label=z_label)
    else:
        print(f"Error: Invalid plot_type '{plot_type}'. Use 'surface' or 'contour'.")
        plt.close(fig); return

    ax.set_title(f"{target_name}: {output_name}[{output_idx}] vs ({vary_dim1_name}, {vary_dim2_name})")
    plt.tight_layout()

    # --- Save / Show ---
    if save_dir:
        fixed_inputs_str = "_".join([f"fix{k}={'%.1f' % v}" for k, v in fixed_values_norm.items()])
        filename = f"{kan_target}_kan_{plot_type}_out{output_idx}_vs_in{vary_dim1_idx}in{vary_dim2_idx}_{fixed_inputs_str}.png"
        save_path = os.path.join(save_dir, filename)
        if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
        try: plt.savefig(save_path, dpi=150); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot: {e}")
    plt.show() 
    
#%%%
def plot_kan_scatter3d(
    model,
    dataset,
    simulation_results,
    kan_target='output',
    state_idxs=(0, 1),
    input_idx=0,
    output_idx=0,
    data_split='train',
    color_by='kan',  # 'kan' | 'pred' | 'target'
    max_points=20000,
    percentiles=(1, 99),
    plot_normalized_axes=True,
    save_dir=None,
    html_path=None,
    plotly_inline_js=True,
    plotly_auto_open=False,
    plotly_colorscale='Viridis',
    point_size=3,
    alpha=0.7,
    cmap='viridis',
    # --- New flexible axes options ---
    axes_mode='auto',  # 'auto' | 'ssu' (2 states + 1 input) | 'suu' (1 state + 2 inputs)
    input_idxs=(0, 1),
):
    """
        3D scatter of either:
            - two states (x[i], x[j]) and one input (u[k])  [axes_mode='ssu']
            - one state (x[i]) and two inputs (u[j], u[k])  [axes_mode='suu']

    Uses post-training normalized samples from simulation_results to ensure
    visualization stays within active/support regions. States remain in normalized
    coordinates by default (internal to model). Inputs can be kept normalized, too.

    Args:
        model: Trained model with 'state_kan_model' or 'output_kan_model'.
        dataset: Dataset object (only used for dims and labeling).
        simulation_results: Dict with post-training arrays, e.g.:
            - 'pred_state_train_norm' or 'pred_state_test_norm'
            - 'u_train_norm' or 'u_test_norm'
            - optionally 'pred_train_denorm'/'pred_train_norm', 'target_train_denorm' for color_by.
        kan_target: 'output' (default) or 'state' — determines which KAN produces color if color_by='kan'.
    axes_mode: 'auto' chooses 'ssu' if >=2 states else 'suu' if >=2 inputs.
            'ssu' uses two state indices for X/Y and one input for Z.
            'suu' uses one state for X and two inputs for Y/Z.
    state_idxs: tuple of state indices. For 'ssu', provide two (i, j). For 'suu', provide one (i,).
    input_idx: for 'ssu', the single input index k used for Z axis (kept for backward compatibility).
    input_idxs: for 'suu', tuple of two input indices (j, k) used for Y/Z axes.
        output_idx: index into KAN output dimension used for coloring.
        data_split: 'train' or 'test' — selects which arrays from simulation_results to use.
        color_by: 'kan' to evaluate KAN at each sample; 'pred' or 'target' uses provided arrays if present.
        max_points: Subsample limit for performance.
        percentiles: Tuple (low, high) to filter outliers per axis and focus on active region.
        plot_normalized_axes: If True, axes stay normalized (recommended for internal states).
        save_dir: Optional path to save the figure.
        point_size: Scatter marker size.
        alpha: Scatter transparency.
        cmap: Matplotlib colormap for coloring.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D
        model.eval()

        # --- Resolve keys based on split ---
        split = 'train' if data_split.lower().startswith('t') else 'test'
        state_key = f'pred_state_{split}_norm'
        u_key = f'u_{split}_norm'
        pred_key_denorm = f'pred_{split}_denorm'
        pred_key_norm = f'pred_{split}_norm'
        target_key_denorm = f'target_{split}_denorm'

        # --- Fetch arrays ---
        Xn = _get_data(simulation_results, state_key)
        Un = getattr(dataset, u_key, None)
        if isinstance(Un, torch.Tensor):
            Un = Un.cpu().numpy()

        if Xn is None or Un is None:
            print("Error: Missing normalized states or inputs for scatter3d.")
            return None

        # Align lengths
        n = min(len(Xn), len(Un))
        if n == 0:
            print("Error: Empty arrays for scatter3d.")
            return None
        Xn = Xn[:n]
        Un = Un[:n]

        # --- Determine axes mode ---
        n_states = Xn.shape[1]
        n_inputs = Un.shape[1]
        mode = (axes_mode or 'auto').lower()
        if mode == 'auto':
            mode = 'ssu' if n_states >= 2 else ('suu' if n_inputs >= 2 else 'invalid')
        if mode == 'invalid':
            print("Error: Need either >=2 states or >=2 inputs for 3D scatter.")
            return None
        # Select axes data according to mode
        if mode == 'ssu':
            i, j = (state_idxs if isinstance(state_idxs, (list, tuple)) and len(state_idxs) >= 2 else (0, 0))
            k = input_idx
            if n_states < 2:
                print("Error: axes_mode='ssu' requires at least 2 states.")
                return None
            if i >= n_states or j >= n_states or k >= n_inputs:
                print("Error: axis index out of bounds.")
                return None
            xi, xj, uk = Xn[:, i], Xn[:, j], Un[:, k]
            x_label = f"State x[{i}] (Norm)"
            y_label = f"State x[{j}] (Norm)"
            z_label = f"Input u[{k}] (Norm)"
            title_axes = f"x[{i}], x[{j}], u[{k}]"
            fname = f"kan_scatter3d_{split}_x{i}_x{j}_u{k}_{color_by}_{kan_target}_out{output_idx}.png"
        elif mode == 'suu':
            i = state_idxs[0] if isinstance(state_idxs, (list, tuple)) and len(state_idxs) >= 1 else 0
            j_inp, k_inp = (input_idxs[:2] if isinstance(input_idxs, (list, tuple)) and len(input_idxs) >= 2 else (0, 1))
            if i >= n_states or max(j_inp, k_inp) >= n_inputs:
                print("Error: axis index out of bounds.")
                return None
            xi, xj, uk = Xn[:, i], Un[:, j_inp], Un[:, k_inp]
            x_label = f"State x[{i}] (Norm)"
            y_label = f"Input u[{j_inp}] (Norm)"
            z_label = f"Input u[{k_inp}] (Norm)"
            title_axes = f"x[{i}], u[{j_inp}], u[{k_inp}]"
            fname = f"kan_scatter3d_{split}_x{i}_u{j_inp}_u{k_inp}_{color_by}_{kan_target}_out{output_idx}.png"
        else:
            print("Error: axes_mode must be 'auto', 'ssu', or 'suu'.")
            return None

        # Filter to active region via per-axis percentiles
        lo, hi = percentiles
        xi_lo, xi_hi = np.percentile(xi, [lo, hi])
        xj_lo, xj_hi = np.percentile(xj, [lo, hi])
        uk_lo, uk_hi = np.percentile(uk, [lo, hi])
        keep = (xi >= xi_lo) & (xi <= xi_hi) & (xj >= xj_lo) & (xj <= xj_hi) & (uk >= uk_lo) & (uk <= uk_hi)

        xi = xi[keep]; xj = xj[keep]; uk = uk[keep]
        idx = np.arange(len(xi))
        if len(idx) == 0:
            print("Warning: no points after percentile filtering.")
            return None

        # Subsample for performance
        if len(idx) > max_points:
            sel = np.random.choice(idx, size=max_points, replace=False)
            xi = xi[sel]; xj = xj[sel]; uk = uk[sel]

        # --- Compute color values ---
        color_vals = None
        if color_by == 'kan':
            # Evaluate the chosen KAN on the corresponding full (state,u) for selected points
            # Need full state/input for each point (normalized)
            X_full = _get_data(simulation_results, state_key)
            if isinstance(X_full, torch.Tensor):
                X_full = X_full
            else:
                X_full = torch.tensor(X_full, dtype=torch.float32)
            U_full = getattr(dataset, u_key, None)
            if isinstance(U_full, torch.Tensor):
                U_full = U_full
            else:
                U_full = torch.tensor(U_full, dtype=torch.float32)

            X_full = X_full[:n][keep]
            U_full = U_full[:n][keep]
            if len(X_full) > max_points:
                X_full = X_full[sel]
                U_full = U_full[sel]

            device = next(model.parameters()).device
            X_full = X_full.to(device)
            U_full = U_full.to(device)

            with torch.no_grad():
                if kan_target == 'output':
                    kan_mod = getattr(model, 'output_kan_model', None)
                    if kan_mod is None:
                        print("Error: model has no output_kan_model")
                        return None
                    Z = kan_mod(state=X_full, u=U_full)  # normalized output
                else:  # 'state'
                    kan_mod = getattr(model, 'state_kan_model', None)
                    if kan_mod is None:
                        print("Error: model has no state_kan_model")
                        return None
                    Z = kan_mod(state=X_full, u=U_full)  # normalized delta state

            if output_idx >= Z.shape[1]:
                print("Error: output_idx out of bounds for KAN output.")
                return None
            color_vals = Z[:, output_idx].detach().cpu().numpy()

        elif color_by in ('pred', 'target'):
            key = pred_key_denorm if color_by == 'pred' else target_key_denorm
            Y = _get_data(simulation_results, key)
            if Y is None:
                # fallback to normalized predictions if available
                if color_by == 'pred':
                    Y = _get_data(simulation_results, pred_key_norm)
            if Y is None:
                print(f"Error: Missing {color_by} arrays in simulation_results.")
                return None
            Y = Y[:n][keep]
            if len(Y) > max_points:
                Y = Y[sel]
            if Y.ndim == 1:
                color_vals = Y
            else:
                if output_idx >= Y.shape[1]:
                    print("Error: output_idx out of bounds for color array.")
                    return None
                color_vals = Y[:, output_idx]
        else:
            print("Error: color_by must be 'kan', 'pred', or 'target'.")
            return None

        # --- Plot ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xi, xj, uk, c=color_vals, s=point_size, alpha=alpha, cmap=cmap)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        color_label = {
            'kan': f"KAN {'y' if kan_target=='output' else 'Δx'}[{output_idx}] (Norm)",
            'pred': f"Prediction y[{output_idx}]",
            'target': f"Target y[{output_idx}]",
        }[color_by]
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=color_label)

        title_target = 'Output KAN' if kan_target == 'output' else 'State KAN'
        fig.suptitle(f"3D Scatter: {title_axes} | Color = {title_target}[{output_idx}] ({split})")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(save_dir, fname)
            try:
                plt.savefig(fpath, dpi=200)
                print(f"Plot saved to {fpath}")
            except Exception as e:
                print(f"Error saving plot: {e}")
        # Optional: save interactive Plotly HTML
        if html_path:
            try:
                # Prefer calling the utility if present in this module
                if 'save_plotly_scatter3d_html' in globals():
                    color_label = {
                        'kan': f"KAN {'y' if kan_target=='output' else 'Δx'}[{output_idx}] (Norm)",
                        'pred': f"Prediction y[{output_idx}]",
                        'target': f"Target y[{output_idx}]",
                    }[color_by]
                    title_target = 'Output KAN' if kan_target == 'output' else 'State KAN'
                    title = f"3D Scatter: {title_axes} | Color = {title_target}[{output_idx}] ({split})"
                    save_plotly_scatter3d_html(
                        x=xi, y=xj, z=uk, color=color_vals, path=html_path,
                        xlabel=x_label, ylabel=y_label, zlabel=z_label,
                        color_label=color_label, title=title,
                        colorscale=plotly_colorscale, point_size=point_size,
                        opacity=alpha, inline_js=plotly_inline_js,
                        auto_open=plotly_auto_open,
                    )
                    print(f"Interactive HTML saved to {html_path}")
                else:
                    print("Warning: save_plotly_scatter3d_html not available; skipping HTML export.")
            except Exception as e:
                print(f"Error saving Plotly HTML: {e}")
        plt.show()
        return fig, ax

    except Exception as e:
        print(f"Error in plot_kan_scatter3d: {e}")
        import traceback
        traceback.print_exc()
        return None
#%%%%
#I think this is the closest function for the univariate plots. 
#problems: 
#introduces bias by forcing inputs to have a value (0). would be better to put weight to zero on the other inputs instead
#range of activation is default, this might the a problem in later layers where not all x_range is activated
def visualize_kan_layer_OLD(kan_layer, input_dim, n_points=1000, x_range=(-1, 1)):
    """
    Visualize both the base and spline components for a specific input dimension
    of a KAN layer.

    Args:
        kan_layer (KANLinear): The KAN layer to visualize
        input_dim (int): The input dimension to visualize
        n_points (int): Number of points to evaluate
        x_range (tuple): Range of x values to evaluate
    """
    # Create input points
    x = torch.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)

    # Create full input tensor with zeros for other dimensions
    full_input = torch.zeros((n_points, kan_layer.in_features))
    full_input[:, input_dim] = x.squeeze()

    with torch.no_grad():
        # Compute base component
        base_output = F.linear(kan_layer.base_activation(full_input), kan_layer.base_weight)

        # Compute spline component
        spline_bases = kan_layer.b_splines(full_input)
        spline_output = F.linear(
            spline_bases.view(full_input.size(0), -1),
            kan_layer.scaled_spline_weight.view(kan_layer.out_features, -1)
        )

        # Compute total output
        total_output = base_output + spline_output

    # Convert to numpy for plotting
    x_np = x.numpy()
    base_output_np = base_output.numpy()
    spline_output_np = spline_output.numpy()
    total_output_np = total_output.numpy()

    # Create plot
    fig, axes = plt.subplots(figsize=(10, 8))

    # # Plot base component
    # axes[0].plot(x_np, base_output_np)
    # axes[0].set_title(f'Base Component\nInput Dim {input_dim}')
    # axes[0].grid(True)

    # # Plot spline component
    # axes[1].plot(x_np, spline_output_np)
    # axes[1].set_title(f'Spline Component\nInput Dim {input_dim}')
    # axes[1].grid(True)

    # Plot total output
    axes.plot(x_np, total_output_np)
    axes.set_title(f'Total Output\nInput Dim {input_dim}')
    axes.grid(True)
    plt.tight_layout()
    plt.show()
    return fig

def visualize_kan_layer_option1(kan_model, simulation_results, layer_idx, input_dim, save_dir, kan_flag, n_points=1000, x_range=(-1, 1)):
    
    kan_layer = kan_model.kan.layers[layer_idx]
    device = kan_layer.grid.device
    x_min, x_max = x_range
    x_vec = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)  # [N,1]
    
    # Build an input tensor only for activation/base (need full width but we will read only column k)
    full_input = torch.zeros(n_points, kan_layer.in_features, device=device)
    full_input[:, input_dim] = x_vec.squeeze()
    
    # Base component for this dimension only with full input range
    with torch.no_grad():
        act_i = kan_layer.base_activation(full_input[:, input_dim:input_dim+1])  # [N,1]
        base_w_i = kan_layer.base_weight[:, input_dim]                          # [out_features]
        base_contrib = act_i @ base_w_i.unsqueeze(0)                            # [N, out_features]
        
        # Spline bases only for this dimension
        # We can reuse b_splines but pick slice for input_dim
        bases_all = kan_layer.b_splines(full_input)         # [N, in_features, coeff]
        bases_i = bases_all[:, input_dim, :]                # [N, coeff]
        spline_w_i = kan_layer.scaled_spline_weight[:, input_dim, :]  # [out_features, coeff]
        spline_contrib = bases_i @ spline_w_i.T             # [N, out_features]
        
        total_contrib = base_contrib + spline_contrib       # [N, out_features]

    x_np = x_vec.cpu().numpy()
    total_np = total_contrib.cpu().numpy()
    base_np = base_contrib.cpu().numpy()
    spline_np = spline_contrib.cpu().numpy()

    # call dataset and do the same for scatter ?
    

    n_outputs = total_np.shape[1]

    for j in range(n_outputs):
        fig, axes = plt.subplots(figsize=(10, 8)) 
        axes.plot(x_np, total_np[:, j], label='Total', color='black', linewidth=1.8)
        axes.plot(x_np, spline_np[:, j], label='Spline', color='red', linestyle='--', linewidth=1.0, alpha=0.8)
        axes.plot(x_np, base_np[:, j], label='Base', color='blue', linestyle=':', linewidth=1.0, alpha=0.8)
        #axes.grid(True, alpha=0.3)
        axes.set_title(f"Input dim {input_dim}, Output {j}")
        axes.legend()
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        
        # Save
        if save_dir:
            filename = f'activation_function_{kan_flag}_layer_{layer_idx}_in_{input_dim}_out_{j}'
            save_path = os.path.join(save_dir, filename)
            if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
            try: plt.savefig(save_path+'.pdf', dpi=300); plt.savefig(save_path+'.png', dpi=300, transparent=True);
            except Exception as e: print(f"Error saving plot: {e}")
        plt.close()

    return 




#% copilot
def visualize_kan_layer_option1_copilot(
    kan_model,
    simulation_results,
    layer_idx: int,
    input_dim: int,
    save_dir=None,
    kan_flag: str = "state",  # just used in filename
    n_points: int = 1000,
    x_range: tuple =None,
    data_split: str = "train",
    overlay: bool = False,
    overlay_percentiles=(1, 99),
    max_points: int = 20000,
    show_knots: bool = False,
    # --- NEW: density panel controls ---
    show_density: bool = False,
    density_kind: str = "hist",   # 'hist' | 'rug'
    density_bins: int = 200,
):
    """
    Plot the isolated univariate contribution f_i(x) for one input dim to all outputs,
    overlay (optional) active samples projected onto the same function, and show a
    small density panel (hist or rug) along x to visualize support.

    Args:
      kan_model: wrapper with .kan (sequence of KANLinear layers)
      simulation_results: dict with normalized arrays:
         - pred_state_{split}_norm
         - u_{split}_norm
      layer_idx: which KAN layer to visualize (0-based)
      input_dim: column at the input of that layer to visualize
      x_range: optional (min,max) in normalized layer-input units; if None, uses percentiles of activations
      show_density: if True, add a small histogram/rug below the curve
      density_kind: 'hist' for histogram panel, 'rug' for rug marks
      density_bins: number of bins for histogram
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    kan_layer = kan_model.kan.layers[layer_idx]
    device = kan_layer.grid.device
    split = "train" if data_split.lower().startswith("t") else "test"

    # 1) Build layer-0 inputs from simulation_results (normalized)
    Xn = simulation_results.get(f"pred_state_{split}_norm")
    Un = simulation_results.get(f"u_{split}_norm")
    if isinstance(Xn, torch.Tensor) is False and Xn is not None:
        Xn = torch.tensor(Xn, dtype=torch.float32)
    if isinstance(Un, torch.Tensor) is False and Un is not None:
        Un = torch.tensor(Un, dtype=torch.float32)
    if Xn is None or Un is None:
        print("Error: simulation_results must contain pred_state_*_norm and u_*_norm.")
        return

    n = min(len(Xn), len(Un))
    if n == 0:
        print("Error: empty arrays in simulation_results.")
        return
    a = torch.cat([Xn[:n], Un[:n]], dim=1).to(device)  # layer-0 activations

    # 2) Propagate to the input of target layer
    with torch.no_grad():
        a_l = a
        for l in range(layer_idx):
            a_l = kan_model.kan.layers[l](a_l)  # output of layer l, input to l+1
    # a_l is the input to kan_layer
    if input_dim >= a_l.shape[1]:
        print(f"Error: input_dim {input_dim} out of bounds for layer {layer_idx} (in_features={a_l.shape[1]}).")
        return

    # 3) Collect activations for selected dim and define plotting range
    x_samples_full = a_l[:, input_dim].detach().cpu().numpy()
    # Percentile crop for support-focused view
    lo, hi = overlay_percentiles
    x_lo, x_hi = np.percentile(x_samples_full, [lo, hi]) if len(x_samples_full) > 1 else (-1.0, 1.0)
    # Keep for overlay density (don’t throw away everything if tiny arrays)
    mask = (x_samples_full >= x_lo) & (x_samples_full <= x_hi)
    x_samples = x_samples_full[mask] if np.any(mask) else x_samples_full
    if len(x_samples) > max_points:
        sel = np.random.choice(len(x_samples), size=max_points, replace=False)
        x_samples = x_samples[sel]

    # Plot range
    if x_range is None:
        if len(x_samples) >= 2:
            xmin, xmax = float(np.min(x_samples)), float(np.max(x_samples))
        else:
            xmin, xmax = -1.0, 1.0
    else:
        xmin, xmax = x_range
    x_vec = torch.linspace(xmin, xmax, n_points, device=device).unsqueeze(1)  # [N,1]

    # 4) Compute isolated univariate function for this input_dim (no bias from other dims)
    with torch.no_grad():
        # Vary only this column; others are zeroed so their spline bases don’t add bias
        full_input = torch.zeros(n_points, kan_layer.in_features, device=device)
        full_input[:, input_dim] = x_vec.squeeze()

        # Base part (only this column)
        act_i = kan_layer.base_activation(full_input[:, input_dim:input_dim+1])      # [N,1]
        base_w_i = kan_layer.base_weight[:, input_dim]                              # [out_features]
        base_contrib = act_i @ base_w_i.unsqueeze(0)                                # [N, out_features]

        # Spline part (only this column)
        bases_all = kan_layer.b_splines(full_input)                                  # [N, in_features, coeff]
        bases_i = bases_all[:, input_dim, :]                                         # [N, coeff]
        spline_w_i = kan_layer.scaled_spline_weight[:, input_dim, :]                 # [out_features, coeff]
        spline_contrib = bases_i @ spline_w_i.T                                      # [N, out_features]

        total_contrib = base_contrib + spline_contrib                                # [N, out_features]

        # Overlay y-values on the same isolated function (project samples onto the curve)
        y_overlay = None
        if overlay and len(x_samples) > 0:
            x_t = torch.tensor(x_samples, device=device, dtype=torch.float32).unsqueeze(1)
            inp_t = torch.zeros(len(x_samples), kan_layer.in_features, device=device)
            inp_t[:, input_dim] = x_t.squeeze()
            act_i_t = kan_layer.base_activation(inp_t[:, input_dim:input_dim+1])     # [M,1]
            base_t = act_i_t @ base_w_i.unsqueeze(0)                                 # [M,out]
            bases_all_t = kan_layer.b_splines(inp_t)                                  # [M,in,coeff]
            bases_i_t = bases_all_t[:, input_dim, :]                                  # [M,coeff]
            spline_t = bases_i_t @ spline_w_i.T                                       # [M,out]
            y_overlay = (base_t + spline_t).detach().cpu().numpy()                   # [M,out]

    # 5) Plot per-output with optional overlay, knot lines, and density panel
    x_np = x_vec.detach().cpu().numpy()
    total_np = total_contrib.detach().cpu().numpy()
    base_np = base_contrib.detach().cpu().numpy()
    spline_np = spline_contrib.detach().cpu().numpy()

    n_outputs = total_np.shape[1]
    for j in range(n_outputs):
        if show_density:
            # Two-row layout: top = curve, bottom = density (shared x)
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            ax_den = fig.add_subplot(gs[1, 0], sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax_den = None

        # Curves
        ax.plot(x_np, total_np[:, j], label='Total', color='black', linewidth=1.8)
        ax.plot(x_np, spline_np[:, j], label='Spline', color='red', linestyle='--', linewidth=1.0, alpha=0.8)
        ax.plot(x_np, base_np[:, j], label='Base', color='blue', linestyle=':', linewidth=1.0, alpha=0.8)

        # Overlay points projected onto the same univariate function
        # if overlay and y_overlay is not None and len(x_samples) > 0:
        #     ax.scatter(x_samples, y_overlay[:, j], s=4, alpha=0.15, color='gray', label='Active zone')

        # Knot lines
        # if show_knots and hasattr(kan_layer, "grid") and kan_layer.grid is not None:
        #     try:
        #         knots = kan_layer.grid[input_dim].detach().cpu().numpy()
        #         for xv in knots:
        #             ax.axvline(xv, color='gray', linewidth=0.5, alpha=0.3)
        #     except Exception:
        #         pass

        # Titles/labels
        #ax.set_title(f"Layer {layer_idx} • In[{input_dim}] → Out[{j}]")
        ax.set_ylabel(f"Layer {layer_idx}  Out[{j}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        #plt.xlim(-1,1)
        plt.tight_layout()

        if show_density:
            # Density panel (hist or rug)
            if len(x_samples) > 0:
                if density_kind == "hist":
                    ax_den.hist(x_samples, bins=density_bins, color='gray', alpha=0.6)
                    ax_den.set_ylabel("count",)
                else:
                    # rug marks along x (no y scale)
                    ax_den.plot(x_samples, np.zeros_like(x_samples), '|', color='gray', alpha=0.6, markersize=6)
                    ax_den.set_yticks([])
                ax_den.grid(True, alpha=0.2)
            ax_den.set_xlabel("Layer-input (Norm)")
            # Hide top x tick labels to declutter
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(f"Layer {layer_idx}  In[{input_dim}]")

        # Save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"{density_kind}" if show_density else "nodensity"
            fname = f"activation_function_{kan_flag}_L{layer_idx}_in{input_dim}_out{j}_{split}_{suffix}"
            try:
                plt.savefig(os.path.join(save_dir, fname + '.png'), dpi=200)
                plt.savefig(os.path.join(save_dir, fname + '.pdf'), dpi=200)

            except Exception as e:
                print(f"Error saving plot: {e}")

        plt.show()
        plt.close(fig)

    return


#%%% gemini function
def plot_kan_layer_functions_simple(kan_layer, layer_idx, input_idx, dataset,
                                    # --- NEW: Pass calculated range ---
                                    input_range_norm=(-1.0, 1.0),
                                    # ----------------------------------
                                    n_points=100, plot_normalized_x=True,
                                    save_dir=None):
    """
    Simplified: Visualizes univariate effective functions for a KANLinear layer.
    Accepts an estimated input range.
    """
    layer_label = f"Layer {layer_idx}"
    print(f"    Plotting Input Index {input_idx} (Range: [{input_range_norm[0]:.2f}, {input_range_norm[1]:.2f}])")

    model_device = kan_layer.grid.device
    state_dim = dataset.A_init.shape[0]
    input_dim = dataset.u_dim

    # --- Use provided range ---
    min_val, max_val = input_range_norm
    axis_label = f"Layer {layer_idx} Input[{input_idx}]"
    norm_min, norm_max = None, None # Denorm params for THIS specific input activation
    allow_denorm_x = (layer_idx == 0) and (not plot_normalized_x) # Only allow Layer 0 x-denorm

    # Try to get denorm params only for layer 0
    if layer_idx == 0:
        # (Logic to find norm_min, norm_max based on input_idx mapping to state/input)
        if input_idx < state_dim: # Assume state
             axis_label = f"State x[{input_idx}]"
             if allow_denorm_x and dataset.states_available and hasattr(dataset, 'x_min') and hasattr(dataset, 'x_dot_min'):
                  mins=torch.cat([...]); maxs=torch.cat([...])
                  if input_idx < len(mins): norm_min, norm_max = mins[input_idx], maxs[input_idx]
        elif input_idx < state_dim + input_dim: # Assume input
             sub_idx = input_idx - state_dim; axis_label = f"Input u[{sub_idx}]"
             if allow_denorm_x and hasattr(dataset, 'u_min'):
                 if sub_idx < len(dataset.u_min): norm_min, norm_max = dataset.u_min[sub_idx], dataset.u_max[sub_idx]
    # --- End Range/Label Logic ---

    x_i_norm = torch.linspace(min_val, max_val, n_points, device=model_device).unsqueeze(1)

    # --- Calculate Spline and Base Function Components ---
    with torch.no_grad():
        # Need input shape [n_points, in_features] for calculations
        kan_layer_input = torch.zeros(n_points, kan_layer.in_features, device=model_device)
        kan_layer_input[:, input_idx] = x_i_norm.squeeze()

        # 1. Spline Function phi_ij = Sum_k ( B_k(x_i) * w_ijk )
        spline_basis_i = kan_layer.b_splines(kan_layer_input)[:, input_idx, :] # [n_points, n_coeffs]
        coeffs_i = kan_layer.scaled_spline_weight[:, input_idx, :]            # [out_features, n_coeffs]
        spline_funcs = spline_basis_i @ coeffs_i.T                             # [n_points, out_features]

        # 2. Base Function base_ij = base_activation(x_i) * base_weight_ji
        activated_x_i = kan_layer.base_activation(x_i_norm)                     # [n_points, 1] (element-wise activation)
        base_weight_i = kan_layer.base_weight[:, input_idx].unsqueeze(0)        # [1, out_features] (weights for input_i to all outputs j)
        base_funcs = activated_x_i * base_weight_i                              # [n_points, out_features] (Broadcast multiply)

        # 3. Total Effective Function = Spline + Base
        total_funcs = spline_funcs + base_funcs                                # [n_points, out_features]

    # --- Prepare X-axis for Plotting ---
    # (Same logic as before)
    if plot_normalized_x:
        x_plot = x_i_norm.cpu().numpy(); axis_label += " (Norm)"
    else:
        if norm_min is not None and norm_max is not None:
            _min = norm_min.item() if isinstance(norm_min, torch.Tensor) else norm_min
            _max = norm_max.item() if isinstance(norm_max, torch.Tensor) else norm_max
            x_plot = denormalize_data(x_i_norm, dataset.norm_flag, _min, _max, normalize=(dataset.norm_flag != 'nothing')).cpu().numpy()
            axis_label += " (Denorm)"
        else: x_plot = x_i_norm.cpu().numpy(); axis_label += " (Norm)"


    # --- Plotting ---
    n_outputs = kan_layer.out_features
    ncols = int(np.ceil(np.sqrt(n_outputs)))
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), sharex=True, squeeze=False) # Increased height slightly
    axes_flat = axes.flatten()
    fig.suptitle(f'KAN Effective Functions {layer_label} (Input: {axis_label})', fontsize=12)

    for j in range(n_outputs):
        ax = axes_flat[j]
        # Extract numpy arrays for plotting this output dimension j
        y_total_plot = total_funcs[:, j].cpu().numpy()
        y_spline_plot = spline_funcs[:, j].cpu().numpy()
        y_base_plot = base_funcs[:, j].cpu().numpy()

        # Plot all three components
        ax.plot(x_plot, y_total_plot, label='Total (Base+Spline)', linewidth=2, color='black')
        ax.plot(x_plot, y_spline_plot, label='Spline $\phi_{ij}$', linewidth=1.5, linestyle='--', color='red')
        ax.plot(x_plot, y_base_plot, label='Base $b_{ij}$', linewidth=1.5, linestyle=':', color='blue')

        ax.set_title(f'-> Output[{j}]', fontsize=9)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        if j == 0: ax.legend(fontsize=7) # Add legend only once or per plot? Once here.
        if j >= n_outputs - ncols : ax.set_xlabel(axis_label, fontsize=9)
        if j % ncols == 0: ax.set_ylabel('Effective Activation', fontsize=9)


    for k in range(n_outputs, len(axes_flat)): axes_flat[k].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        # (Save logic same as before)
        filename = f"kan_effective_funcs_{layer_label.replace(' ','')}_input{input_idx}.png"
        save_path = os.path.join(save_dir, filename)
        if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)
        try: plt.savefig(save_path, dpi=150); print(f"Plot saved: {save_path}")
        except Exception as e: print(f"Error saving plot: {e}")
    plt.show()

    return fig, axes


def visualize_kan_model_functions_simple(model, dataset, kan_target='state',
                                  input_indices_per_layer=None,
                                  layer_indices=None,
                                  plot_normalized_x=True, save_dir=None):
    """
    Simplified: Visualizes KAN univariate effective functions.
    Estimates activation ranges for deeper layers using training data.
    """
    kan_attr_name = f"{kan_target}_kan_model"
    kan_module_wrapper = getattr(model, kan_attr_name)
    kan_core_model = kan_module_wrapper.kan
    device = next(model.parameters()).device
    model.eval() # Ensure model is in eval mode

    # --- Prepare Full Training Input Data ---
    print("Preparing full training data for range estimation...")
    u_train_full = dataset.u_train_norm.to(device)
    state_train_full = None
    state_dim = dataset.A_init.shape[0]

    if dataset.states_available:
        state_train_full = dataset.X_train_norm.to(device)
    else:
        # --- Simulate to get state trajectory if states unavailable ---
        # This uses the main model's dynamics, not just the KAN
        print("Simulating full training sequence to estimate states for KAN input range...")
        state_trajectory = []
        current_state = torch.zeros(1, state_dim, device=device)
        with torch.no_grad():
            for i in range(len(u_train_full)):
                state_trajectory.append(current_state.clone()) # Store state x(t)
                next_state, _ = model(current_state, u_train_full[i].unsqueeze(0)) # Use main model
                current_state = next_state
        state_train_full = torch.cat(state_trajectory, dim=0) # Shape [N, state_dim]
        print("State trajectory estimated.")

    if state_train_full is None:
         print("Error: Could not obtain state data for range estimation.")
         return

    # Ensure lengths match (might be off by one depending on sim start)
    min_len = min(len(state_train_full), len(u_train_full))
    state_train_full = state_train_full[:min_len]
    u_train_full = u_train_full[:min_len]

    # Construct input sequence for the specific KAN being visualized
    kan_input_sequence = None
    if kan_target == 'state' or kan_target == 'output': # Assuming state+input KANs
         kan_input_sequence = torch.cat([state_train_full, u_train_full], dim=1)
    # Add logic here if output KAN takes y_linear

    if kan_input_sequence is None:
        print("Error: Could not construct KAN input sequence for range estimation.")
        return

    # --- Perform Forward Pass through KAN to Estimate Ranges ---
    activation_ranges = {} # {layer_idx: [(min0, max0), ...]}
    # Use batches for memory efficiency if sequence is very long
    batch_size_range_est = 4001 # Adjust batch size as needed
    print(f"Estimating activation ranges using KAN forward pass (batch size: {batch_size_range_est})...")

    # Initialize running min/max for each layer's input activations
    # Need to know the number of features entering each layer
    layer_in_features = [l.in_features for l in kan_core_model.layers]
    layer_act_min = [torch.full((f,), float('inf'), device=device) for f in layer_in_features]
    layer_act_max = [torch.full((f,), float('-inf'), device=device) for f in layer_in_features]

    with torch.no_grad():
        for i in range(0, len(kan_input_sequence), batch_size_range_est):
            batch_input = kan_input_sequence[i:i+batch_size_range_est]
            current_activations = batch_input

            # Update range for Layer 0 input
            layer_act_min[0] = torch.minimum(layer_act_min[0], torch.min(current_activations, dim=0)[0])
            layer_act_max[0] = torch.maximum(layer_act_max[0], torch.max(current_activations, dim=0)[0])

            # Pass through layers and update ranges for subsequent layers' inputs
            for layer_idx, layer in enumerate(kan_core_model.layers):
                current_activations = layer(current_activations) # Output of layer 'layer_idx'
                if layer_idx + 1 < len(kan_core_model.layers): # If this is input to next layer
                    layer_act_min[layer_idx + 1] = torch.minimum(layer_act_min[layer_idx + 1], torch.min(current_activations, dim=0)[0])
                    layer_act_max[layer_idx + 1] = torch.maximum(layer_act_max[layer_idx + 1], torch.max(current_activations, dim=0)[0])

    # Convert final min/max tensors to list of tuples
    for l_idx in range(len(kan_core_model.layers)):
         activation_ranges[l_idx] = list(zip(layer_act_min[l_idx].cpu().tolist(),
                                             layer_act_max[l_idx].cpu().tolist()))

    print("Activation range estimation complete.")

 # --- Plotting Loop ---
    all_layer_indices = range(len(kan_core_model.layers))
    target_layer_indices = layer_indices if layer_indices is not None else all_layer_indices

    for layer_idx in target_layer_indices:
        if layer_idx not in all_layer_indices: continue
        kan_layer = kan_core_model.layers[layer_idx]
        all_inputs_for_this_layer = range(kan_layer.in_features)

        # --- Determine which input indices to plot for THIS layer ---
        target_input_indices = all_inputs_for_this_layer # Default: plot all
        if input_indices_per_layer is not None:
            if layer_idx in input_indices_per_layer:
                # Use the specific list provided for this layer
                target_input_indices = input_indices_per_layer[layer_idx]
                # Filter out invalid indices for this layer
                target_input_indices = [i for i in target_input_indices if i in all_inputs_for_this_layer]
            else:
                 # If layer not in dict, plot nothing for this layer (or default to all?)
                 # Let's default to plotting nothing if specific indices were requested for other layers
                 if input_indices_per_layer: # Check if the dict is not empty
                      target_input_indices = []
                 # else: # If dict was empty/None, we already defaulted to all above

        # -----------------------------------------------------------

        print(f"\n--- Processing Layer {layer_idx} (Plotting inputs: {list(target_input_indices)}) ---")
        if layer_idx not in activation_ranges:
             print(f"    Warning: Activation range for Layer {layer_idx} input not estimated. Skipping."); continue

        for input_idx in target_input_indices:
            # --- Get estimated range (same as before) ---
            estimated_range = activation_ranges[layer_idx][input_idx]
            if estimated_range[0] >= estimated_range[1]:
                 print(f"    Warning: Skipping plot for Layer {layer_idx} Input {input_idx} due to invalid range {estimated_range}.")
                 continue

            # --- Call layer plotting function ---
            plot_kan_layer_functions_simple(
                kan_layer=kan_layer, layer_idx=layer_idx,
                input_idx=input_idx, dataset=dataset,
                input_range_norm=estimated_range,
                plot_normalized_x=plot_normalized_x,
                save_dir=save_dir
            )
            
           

            
inputs_to_plot_state = {
    0: [0, 1,2],          # For Layer 0, plot original state[0] and state[1] inputs
    1: [0,1]              # For Layer 1, plot input activation[0] (output 0 from Layer 0)
    # Add more layers if needed
}

#%%% plotly

import plotly.graph_objects as go

def save_plotly_scatter3d_html(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    color: np.ndarray,
    path: str,
    xlabel: str = "State x[i] (Norm)",
    ylabel: str = "State x[j] (Norm)",
    zlabel: str = "Input u[k] (Norm)",
    color_label: str = "Value",
    title: str = "3D Scatter",
    colorscale: str = "turbo",
    point_size: int = 3,
    opacity: float = 0.7,
    inline_js: bool = True,
    auto_open: bool = False,
):
    """
    Save an interactive 3D scatter (rotatable) as a standalone HTML.

    Args:
        x, y, z: 1D arrays for axes (prefer normalized values for internal states/inputs).
        color: 1D array for color-mapping (e.g., KAN output, prediction, target, error).
        path: Output HTML file path.
        inline_js: True embeds plotly.js for fully offline viewing; False uses CDN.
        auto_open: If True, opens the HTML in a browser after saving.
    """
    fig = go.Figure(
        data=go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=point_size,
                opacity=opacity,
                color=color,
                colorscale=colorscale,
                colorbar=dict(title=color_label),
            ),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(
        path,
        full_html=True,
        include_plotlyjs="inline" if inline_js else "cdn",
        auto_open=auto_open,
    )
    
    
#% gon
def forward_kan_maybe(
    kan_model,
    simulation_results,
    save_dir=None,
    kan_flag: str = "state",  # just used in filename
    n_points: int = 1000,

):
    """
    Plot the isolated univariate contribution f_i(x) for one input dim to all outputs,
    overlay (optional) active samples projected onto the same function, and show a
    small density panel (hist or rug) along x to visualize support.

    Args:
      kan_model: wrapper with .kan (sequence of KANLinear layers)
      simulation_results: dict with normalized arrays:
         - pred_state_{split}_norm
         - u_{split}_norm
      layer_idx: which KAN layer to visualize (0-based)
      input_dim: column at the input of that layer to visualize
      x_range: optional (min,max) in normalized layer-input units; if None, uses percentiles of activations
      show_density: if True, add a small histogram/rug below the curve
      density_kind: 'hist' for histogram panel, 'rug' for rug marks
      density_bins: number of bins for histogram
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    split = "train" 

    # 1) Build layer-0 inputs from simulation_results (normalized)
    Xn = simulation_results.get(f"pred_state_train_norm")
    Un = simulation_results.get(f"u_train_norm")
    if isinstance(Xn, torch.Tensor) is False and Xn is not None:
        Xn = torch.tensor(Xn, dtype=torch.float32)
    if isinstance(Un, torch.Tensor) is False and Un is not None:
        Un = torch.tensor(Un, dtype=torch.float32)
    if Xn is None or Un is None:
        print("Error: simulation_results must contain pred_state_*_norm and u_*_norm.")
        return

    x0_range= torch.linspace(Xn[:, 0].min(), Xn[:, 0].max(), n_points).reshape(-1, 1) # Range slightly wider than training data
    x1_range = torch.linspace(Xn[:, 1].min(), Xn[:, 1].max(), n_points).reshape(-1, 1) # Range slightly wider than training data
    u0_range = torch.linspace(Un[:, 0].min(), Un[:, 0].max(), n_points).reshape(-1, 1) # Range slightly wider than training data

    x0_range= torch.linspace(-1, 1, n_points).reshape(-1, 1) # Range slightly wider than training data
    x1_range = torch.linspace(-1, 1, n_points).reshape(-1, 1) # Range slightly wider than training data
    u0_range = torch.linspace(Un[:, 0].min(), Un[:, 0].max(), n_points).reshape(-1, 1) # Range slightly wider than training data


    n = min(len(Xn), len(Un))
    if n == 0:
        print("Error: empty arrays in simulation_results.")
        return
    x = torch.cat([x0_range, x1_range], dim=1).to('cpu')  # layer-0 activations

    with torch.no_grad():
            kan_out = kan_model(state=x,u =u0_range)

    kan_output_0 = (kan_out[:, 0]).detach().numpy()
    kan_output_1 = (kan_out[:, 1]).detach().numpy()
        
    #coeffs = np.polyfit(x[:,1].numpy(), kan_output_1, 3)
    #f = np.poly1d(coeffs)
    #y_fit = f(x[:,1].numpy())
    
    plt.figure(figsize=(10, 8))
    #plt.plot(x[:,0].numpy(), kan_output_0, label='$\delta{x_{0, k+1}}$', linewidth=2)
    plt.plot(x[:,1].numpy(), kan_output_1, label='$\delta{x_{1, k+1}}$', linewidth=2)
    plt.xlabel('$\delta{x_{1, k}}$')
    plt.ylabel('{KAN}$_f(\mathbf{x}(k),\mathbf{u}(k))$') # Assuming KAN output is correction to velocity
    plt.legend()
    plt.grid(True)
    plt.xticks()
    plt.yticks()
    #lt.ylim([-0.06,0.04])
    plt.tight_layout()
    plt.savefig(save_dir + '/kan_out.png',dpi=300)  # Adjust layout to prevent labels from being cut off
    plt.show()