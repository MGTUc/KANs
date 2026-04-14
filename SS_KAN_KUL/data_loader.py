#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:46:25 2025

@author: cruz
"""

# data_preprocessing.py
"""
This module handles data loading, computing derivatives, and normalization.
It loads the Silverbox dataset from nonlinear_benchmarks, computes the velocity
(using finite differences), and normalizes the position, velocity, and control input.
"""

import torch
import numpy as np
import nonlinear_benchmarks  # Assumes this module is in your PYTHONPATH
from _utils import normalize_data
def load_and_preprocess_data(test_case='Silverbox',test_flag=None,norm_flag='minmax'):
    if test_case == 'Silverbox':
    # Load the Silverbox dataset
        train_val, test = nonlinear_benchmarks.Silverbox(atleast_2d=True)
        dt = train_val.sampling_time  # Sampling time (seconds)
        f_s = 1 / dt  # Sampling frequency (Hz)
    
        # Unpack training data (u: control input, y: measured position)
        u_train, y_train = train_val
        num_samples = len(u_train)

    elif test_case=='Wiener-Hammerstein':
        train_val, test = nonlinear_benchmarks.WienerHammerBenchMark(atleast_2d=True)
        dt = train_val.sampling_time  # Sampling time (seconds)
        print(test.state_initialization_window_length) # = 50
        u_train, y_train = train_val

    # Convert inputs and outputs to PyTorch tensors (float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Create a time vector for training data (if needed)
    t_train = torch.arange(len(y_train), dtype=torch.float32) * dt

    # Compute velocity (x_dot) using finite differences:
    # Use central differences for interior points,
    # forward difference for the first point, and backward difference for the last point.
    x_dot_train = torch.zeros_like(y_train)
    x_dot_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2 * dt)
    x_dot_train[0] = (y_train[1] - y_train[0]) / dt
    x_dot_train[-1] = (y_train[-1] - y_train[-2]) / dt

    # Normalize each variable using min-max normalization (or z-score, as defined in norm.py)
    y_train_norm, x_min, x_max, _ = normalize_data(y_train, norm_type=norm_flag, normalize=True)
    x_dot_train_norm, x_dot_min, x_dot_max, _ = normalize_data(x_dot_train, norm_type=norm_flag, normalize=True)
    u_train_norm, u_min, u_max, _ = normalize_data(u_train, norm_type=norm_flag, normalize=True)

    # Concatenate normalized position and velocity to form the full state vector [x, x_dot]
    X_train_norm = torch.cat((y_train_norm, x_dot_train_norm), dim=-1)

    # Process test data similarly
    if test_flag=='arrow_no_extra' and test_case=='Silverbox':
        test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
        u_test, y_test = test_arrow_no_extrapolation.u, test_arrow_no_extrapolation.y
    
    elif test_flag=='arrow_extra' and test_case=='Silverbox':
        test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
        u_test, y_test = test_arrow_full.u, test_arrow_full.y

    elif test_flag=='multisine' and test_case=='Silverbox':
        test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
        u_test, y_test = test_multisine.u, test_multisine.y

    elif test_case=='Wiener-Hammerstein':
        u_test, y_test = test

    else:
        print('Gon dont be lazy and implement this!')
        return
    
    u_test = torch.tensor(u_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    x_dot_test = torch.zeros_like(y_test)
    x_dot_test[1:-1] = (y_test[2:] - y_test[:-2]) / (2 * dt)
    x_dot_test[0] = (y_test[1] - y_test[0]) / dt
    x_dot_test[-1] = (y_test[-1] - y_test[-2]) / dt

    # Normalize test data using the same min/max values from training data to maintain consistency
    if norm_flag == 'zscore':
        y_test_norm, _, _, _ = normalize_data(y_test, norm_type=norm_flag, normalize=True, data_mean=x_min, data_std=x_max)
        x_dot_test_norm, _, _, _ = normalize_data(x_dot_test, norm_type=norm_flag, normalize=True, data_mean=x_dot_min, data_std=x_dot_max)
        u_test_norm, _, _, _ = normalize_data(u_test, norm_type=norm_flag, normalize=True, data_mean=u_min, data_std=u_max)
    else:
        y_test_norm, _, _, _ = normalize_data(y_test, norm_type=norm_flag, normalize=True, data_min=x_min, data_max=x_max)
        x_dot_test_norm, _, _, _ = normalize_data(x_dot_test, norm_type=norm_flag, normalize=True, data_min=x_dot_min, data_max=x_dot_max)
        u_test_norm, _, _, _ = normalize_data(u_test, norm_type=norm_flag, normalize=True, data_min=u_min, data_max=u_max)
    X_test_norm = torch.cat((y_test_norm, x_dot_test_norm), dim=-1)

    # Return a dictionary of processed data and normalization parameters for use in training and evaluation.
    return X_train_norm, u_train_norm,y_train_norm,X_test_norm,u_test_norm,y_test_norm,dt,x_min, x_max,x_dot_min,x_dot_max,u_min,u_max
   
def vanderpol_loader(dt=0.01,epsilon=0.3,omega_s=2 * np.pi * 0.2,T=100):
    def discrete_wake_viv(x, u, dt, epsilon, omega_s):
        """
        Compute the discrete-time update for the wake dynamics.
        
        Args:
            x: State vector [x1, x2] where x1 = C_L and x2 = dC_L/dt
            u: Input (forcing term) at the current time step (now position forcing)
            dt: Time step size (Delta t)
            epsilon, omega_s: Model parameters
        
        Returns:
            x_next: Updated state vector [x1_next, x2_next]
        """
        x1, x2 = x  # Unpack state variables

        # Update equations (assuming the model accepts 'u' as the forcing term directly)
        x1_next = x1 + dt * x2
        x2_next = - dt * omega_s**2 * x1 + (1 + epsilon * dt) * x2 + dt * u -epsilon * dt * x1**2 * x2

        return [x1_next, x2_next]

    from signal_generation import generate_multisine


    # Fixed Amplitude for multisine (desired amplitude of position forcing)
    A_multisine = 1.0
    steps = int(T / dt)

    # Parameters for multisine generation
    N_sine_waves = 10  # Number of sine waves in multisine
    freq_min = 0.2 * omega_s
    freq_max = 1.5 * omega_s
    multisine_frequencies = np.linspace(freq_min, freq_max, N_sine_waves) # Frequencies for multisine

    # Generate multisine input signal
    time_multisine, multisine_signal = generate_multisine(N_sine_waves, A_multisine, multisine_frequencies, T, dt) # Generate for T+dt duration
    time_multisine_test, multisine_signal_test = generate_multisine(N_sine_waves, A_multisine, multisine_frequencies, T, dt) # Generate for T+dt duration


    warmup_duration = 30.0  # Example warmup duration (adjust as needed)
    warmup_steps = int(warmup_duration / dt)
    
    x_warmup = [0, 0] # Initial state for warmup
    time_warm, u_warmup = generate_multisine(N_sine_waves, A_multisine, multisine_frequencies, T, dt)

    for j in range(warmup_steps): # Run warmup simulation (WITHOUT recording data)
        u_warmup_e = u_warmup[j] # Get one input value (adjust input generation if needed)
        x_warmup = discrete_wake_viv(x_warmup, u_warmup_e, dt, epsilon, omega_s)
    

    # Initialize state
    x = list(x_warmup)
    x_test = list(x_warmup)

    input_sequence = []
    state_sequence = []
    input_sequence_test = []
    state_sequence_test = []

    state_sequence.append(x)
    state_sequence_test.append(x_test)


    for t_idx in range(steps):
        u_val = multisine_signal[t_idx] # Use multisine signal as input
        u_val_test = multisine_signal_test[t_idx] # Use multisine signal as input
        input_sequence.append(u_val)
        input_sequence_test.append(u_val_test)
        x = discrete_wake_viv(x, u_val, dt, epsilon, omega_s)
        x_test = discrete_wake_viv(x_test, u_val_test, dt, epsilon, omega_s)
        if np.any(np.isnan(x)):
            print(f"NaN detected at time step {t_idx}.")
            break
        state_sequence.append(x)
        state_sequence_test.append(x_test)

    state_sequence = np.array(state_sequence)
    input_sequence = np.array(input_sequence)
    
    state_sequence_test = np.array(state_sequence_test)
    input_sequence_test = np.array(input_sequence_test)
    
    obs = state_sequence[:, 0]
    obs_test = state_sequence_test[:, 0]
###### my stuff trasin
    u_train = torch.tensor(input_sequence.reshape(-1,1), dtype=torch.float32)
    y_train = torch.tensor(obs.reshape(-1,1), dtype=torch.float32)
    x_dot = torch.tensor(state_sequence[:,1].reshape(-1,1), dtype=torch.float32)

    # Normalize each variable using min-max normalization (or z-score, as defined in norm.py)
    y_train_norm, x_min, x_max, _ = normalize_data(y_train, norm_type='minmax', normalize=True)
    x_dot_train_norm, x_dot_min, x_dot_max, _ = normalize_data(x_dot, norm_type='minmax', normalize=True)
    u_train_norm, u_min, u_max, _ = normalize_data(u_train, norm_type='minmax', normalize=True)

    X_train_norm = torch.cat((y_train_norm, x_dot_train_norm), dim=-1)
###### my stuff test
    u_test = torch.tensor(input_sequence_test.reshape(-1,1), dtype=torch.float32)
    y_test = torch.tensor(obs_test.reshape(-1,1), dtype=torch.float32)
    x_dot_test = torch.tensor(state_sequence_test[:,1].reshape(-1,1), dtype=torch.float32)

    # Normalize test data using the same min/max values from training data to maintain consistency
    y_test_norm, _, _, _ = normalize_data(y_test, norm_type='minmax', normalize=True, data_min=x_min, data_max=x_max)
    x_dot_test_norm, _, _, _ = normalize_data(x_dot_test, norm_type='minmax', normalize=True, data_min=x_dot_min, data_max=x_dot_max)
    u_test_norm, _, _, _ = normalize_data(u_test, norm_type='minmax', normalize=True, data_min=u_min, data_max=u_max)
    X_test_norm = torch.cat((y_test_norm, x_dot_test_norm), dim=-1)


    return X_train_norm, u_train_norm,y_train_norm,X_test_norm,u_test_norm,y_test_norm,dt,x_min, x_max,x_dot_min,x_dot_max,u_min,u_max
