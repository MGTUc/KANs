#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:15:42 2025

@author: cruz
"""
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import random
import SS_KAN_KUL._utils as _utils
from SS_KAN_KUL.data_class_SI import SystemIdentificationDataset
from SS_KAN_KUL.model_state_space import FullStateNonlinearityKAN, StateSpaceKANModel
from mlpkan import FullStateNonlinearityMLPKAN
from fastkan import FullStateNonlinearityFastKAN
import time
import os
import pandas as pd
#import _plot

# %%% Set seed
special_flag = None #use for save in model name
device = "cpu"
seed_value = 1
# seed_value=random.randint(1,1111)
#seed_value = 311 # good seed for 10grid 2 layers Silver
#seed_value = 1004 # good seef for 5grid 2 layers Silver
print(f"\nFixed seed value at {seed_value}")
random.seed(seed_value)  # Python's built-in random module
np.random.seed(seed_value)  # NumPy
torch.manual_seed(seed_value)  # PyTorch CPU


# %%% Data
test_case_name = "Silverbox"
test_flag = "arrow_extra"
norm_flag = "minmax"
states_available_flag = False
init_matrices_flag = 'Silverbox'


# %%% Load Data
dataset = SystemIdentificationDataset(
    test_case=test_case_name,
    test_flag=test_flag,
    norm_flag=norm_flag,
    device=device,
    states_available=states_available_flag,
    init_matrices_flag=init_matrices_flag,
)


# %%% KAN
state_dim = dataset.A_init.shape[0]
input_dim = dataset.u_dim
output_dim = dataset.y_dim
# Input KAN
state_kan_input_size = state_dim + input_dim
state_kan_hidden_layers = [] # Hidden layers for state KAN
state_kan_output_size = state_dim


# Output KAN
output_kan_input_size = state_dim + input_dim
output_kan_hidden_layers = [] # Hidden layers for output KAN
output_kan_output_size = output_dim 


# %%% model call
# Instantiate state KAN (ensure params match)
nonlinearity_type = "MLPKAN" # Choose between "KAN", "MLPKAN", "FastKAN"
match nonlinearity_type:
    case "KAN":
        kan_grid_size = 5
        output_kan_grid_size = 5
        state_kan = FullStateNonlinearityKAN(
            state_kan_input_size, state_kan_hidden_layers, state_kan_output_size, # Use general kan_hidden_layers config
            grid_size=kan_grid_size,
            grid_range=[-1, 1],
            #base_activation=torch.nn.Identity
            #grid_eps=0.1
        )
        output_kan = FullStateNonlinearityKAN(
            output_kan_input_size,
            output_kan_hidden_layers, # Use specific hidden layer config
            output_kan_output_size,
            grid_size=output_kan_grid_size, # Use specific grid size config
            grid_range=[-1,1],
            #base_activation=torch.nn.Identity
            # grid_eps=0.1
        )
        extra_info_modelname = f"KAN_grid{kan_grid_size}_{seed_value}"
    case "MLPKAN":
        subnetwork_shape = [30]
        state_kan = FullStateNonlinearityMLPKAN(
            state_kan_input_size,
            state_kan_hidden_layers,
            state_kan_output_size, # Use general kan_hidden_layers config
            subnetwork_shape = subnetwork_shape
        )
        output_kan = FullStateNonlinearityMLPKAN(
            output_kan_input_size,
            output_kan_hidden_layers, # Use specific hidden layer config
            output_kan_output_size,
            subnetwork_shape = subnetwork_shape # Use specific grid size config
        )
        extra_info_modelname = f"MLPKAN_subnet{str(subnetwork_shape)}_{seed_value}_SiLU_noOut_reg"
    case "FastKAN":
        num_grids = 5
        output_num_grids = 5
        state_kan = FullStateNonlinearityFastKAN(
            state_kan_input_size, 
            state_kan_hidden_layers, 
            state_kan_output_size, # Use general kan_hidden_layers config
            num_grids=num_grids,
        )
        output_kan = FullStateNonlinearityFastKAN(
            output_kan_input_size,
            output_kan_hidden_layers, # Use specific hidden layer config
            output_kan_output_size,
            num_grids=output_num_grids, # Use specific grid size config
        )
        extra_info_modelname = f"FastKAN_grid{num_grids}_{seed_value}"


# state_kan = None
output_kan = None 
model = StateSpaceKANModel(
    dataset.A_init,
    dataset.B_init,
    dataset.C_init,
    dataset.D_init,
    state_kan,
    output_kan,
    trainable_C=True,
    trainable_D=True,
)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# %%% Saving/Loading Configuration
model_save_dir = f"./SS_KAN_KUL/test___model_saves_simple_{state_dim}"  # Directory to save model state dicts
os.makedirs(model_save_dir, exist_ok=True)
save_every = 0  # Save model state dict every N epochs (0 to disable intermediate saves)
# Set this path to load a specific state_dict before training, or set to None

load_model = False
if load_model:
    match nonlinearity_type:
        case "KAN":
            load_model_path = 'C:\\Users\\Maarten\\codeProjects\\KANs\\SS_KAN_KUL\\test___model_saves_simple_2\\best_model_Silverbox_epoch_24_state_[2]_output_[2]_batch_512_KAN.pth'
        case "MLPKAN":
            load_model_path = 'C:\\Users\\Maarten\\codeProjects\\KANs\\SS_KAN_KUL\\test___model_saves_simple_2\\best_model_Silverbox_epoch_24_state_[2]_output_[2]_batch_512_MLPKAN.pth'
        case "FastKAN":
            load_model_path = 'C:\\Users\\Maarten\\codeProjects\\KANs\\SS_KAN_KUL\\test___model_saves_simple_2\\best_model_Silverbox_epoch_24_state_[2]_output_[2]_batch_512_FastKAN.pth'
else:    
    load_model_path = None
#load_model_path = 'test___model_saves_simple_2/model_Silverbox_epoch_99_state_[2]_output_[2]_batch_64_highL1.pth'
#load_model_path = 'model_saves_simple_1/best_model_Luca-Airfoil-CFD_epoch_989_state_[3]_output_[3]_batch_512_0.pth'
print(f"loading model:{load_model_path}")
save_best_model = True # <-- Add this flag to enable saving the best model
# %%% Load model (if)
if load_model_path:
    if os.path.isfile(load_model_path):
        print(f"=> Loading model state_dict from '{load_model_path}'")
        try:
            state_dict = torch.load(load_model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("=> Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Proceeding with initialized or previously trained weights.")
    else:
        print(
            f"=> Model file not found at '{load_model_path}'. Proceeding with initialized weights."
        )


# %%% Optimizer
# learning_rate = 1e-2
# weight_decay = 1e-5
# lr_scheduler_gamma = 0.999
# num_epochs = 10
# batch_size = 512
# reg_lambda_l1 = 1e-3
# reg_lambda_l2 = 1e-5
learning_rate = 1e-3
weight_decay = 1e-4
lr_scheduler_gamma = 0.999  
num_epochs = 20
batch_size = 32
reg_lambda_l1 = 1
reg_lambda_l2 = 0


# %%% Optimization setup
# Exclude bias vectors from weight decay.
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # Handle common names (bias) and custom layouts (e.g. layers.0.biases.0).
    name_tokens = set(name.split("."))
    is_bias_like = ("bias" in name_tokens) or ("biases" in name_tokens)
    if is_bias_like or param.ndim == 1:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

if len(no_decay_params) == 0:
    print("Warning: no no-decay params found. Check model.named_parameters() naming.")

optimizer = optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
)

print(
    f"Optimizer param groups -> decay: {sum(p.numel() for p in decay_params):,} params, "
    f"no_decay: {sum(p.numel() for p in no_decay_params):,} params"
)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=lr_scheduler_gamma
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
loss_fn = nn.MSELoss()
# %%% print
print("\n--- Experiment Configuration ---")
print(f"  Device: {device}")
print(f"  Seed Value: {seed_value}")
print(
    f"  Test Case: {test_case_name}, Test Flag: {test_flag}, Norm Flag: {norm_flag}"
)
print(f"The input dimension is {input_dim}.\nThe number of states is {state_dim}.\nThe output dimension is {output_dim}")
# print(f"  Model Type: {model_type}")
print(
    f" State KAN model struct:{state_kan_input_size},{state_kan_hidden_layers},{state_kan_output_size}"
)
print(
    f" Output KAN model struct:{output_kan_input_size},{output_kan_hidden_layers},{output_kan_output_size}"
)
print(
    f"  Optimizer: AdamW, Learning Rate: {learning_rate}, Weight Decay: {weight_decay}"
)
#print(f"  LR Scheduler: ExponentialLR, Gamma: {lr_scheduler_gamma}")
print(
    f"  Epochs: {num_epochs}, Batch Size: {batch_size}, Reg Lambda L1: {reg_lambda_l1}, Reg Lambda L2: {reg_lambda_l2}"
)
print("-------------------------------\n")
# %%% Training Loop
test_eval_every = 5
if states_available_flag is True:
    train_dataset = torch.utils.data.TensorDataset(
        dataset.X_train_norm, dataset.u_train_norm, dataset.y_train_norm
    )
    # test_dataset = torch.utils.data.TensorDataset(dataset.X_test_norm, dataset.u_test_norm, dataset.y_test_norm)
elif states_available_flag is False:
    train_dataset = torch.utils.data.TensorDataset(
        dataset.u_train_norm, dataset.y_train_norm
    )
    # test_dataset = torch.utils.data.TensorDataset(dataset.u_test_norm, dataset.y_test_norm)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
print("\n--- Training Start ---")
epoch_loss_list_train = []
epoch_loss_list_test = []
epoch_time_list = []
# --- Initialize tracking for best model ---
best_test_loss = float('inf')
best_epoch = -1
current_best_model_path = None # <-- Add this to track the path

# hidden_state = torch.zeros(1, model.state_dim, device=device) + torch.randn(1, model.state_dim, device=device) * 0.01
t0 = time.perf_counter()
for epoch in range(num_epochs):
    
    # add funtion to do a full data sweep every x epochs and update grid ?
    a = 201111111111
    # if (epoch + 1) % a == 0:
    #     print(f'updating grid every {a} epochs')
    #     model.eval()
    #     model = _utils.update_full_data(model, dataset,states_available_flag, device)
    #     model.train()
    if (epoch + 1) == a:
        print(f'updating grid ONCE at {a} epoch')
        model.eval()
        model = _utils.update_full_data(model, dataset,states_available_flag, device)
        model.train()

    batch_loss_list_train = []
    batch_MSE_list_train = []
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"
    )

    current_sim_state = (
        torch.zeros(1, model.state_dim, device=device)
        + torch.randn(1, model.state_dim, device=device) * 0.01
    )

    for batch_idx, batch_data in enumerate(progress_bar):        
        #update_grid_now = (epoch %5 == 0 and epoch != 0 and batch_idx == 0) # Update every 5 epochs

        model.train()  # Set model to training mode
        optimizer.zero_grad()

        #update_grid_now = (batch_idx == 0) # Update only on the first batch of the epoch
        update_grid_now = False # no update
       # update_grid_now = True
      
    
        if states_available_flag:
            state_batch, u_batch, batch_target_output = (
                batch_data  # Unpack state, input, output from DataLoader
            )

            next_state_pred_list = []
            output_pred_list = []
            current_state = state_batch[0].unsqueeze(
                0
            )  # Initialize with the first state of the batch
            for t in range(
                len(u_batch)
            ):  # Iterate through time steps in the batch (adjust range)
                current_u = u_batch[t].unsqueeze(0)  # Current input u[t]
                next_state_pred, output_pred = model(current_state, current_u, update_grid_now)
                next_state_pred_list.append(next_state_pred)
                output_pred_list.append(output_pred)
                current_state = next_state_pred

            next_state_pred_batch = torch.cat(next_state_pred_list, dim=0)
            predicted_outputs_batch = torch.cat(output_pred_list, dim=0)

            loss_output = loss_fn(
                predicted_outputs_batch, batch_target_output
            )

        elif not states_available_flag:

            # Initialize combined state for each batch
            # combined_state_batch = current_sim_state.detach()
            output_pred_list_batch = []
            # current_combined_state = combined_state_batch[0, :].unsqueeze(0) # Initial combined state for recursive prediction within batch
            sim_state_for_batch = current_sim_state.detach()
            batch_input, batch_target_output = batch_data
            # Recursive prediction within the batch
            # print("start state:",sim_state_for_batch)
            for t in range(
                len(batch_input)
            ):  # Iterate through time steps in the batch
                current_input = batch_input[t].unsqueeze(0)

                # next_combined_state, output_pred = model(current_combined_state, current_input)
                next_sim_state, output_pred = model(
                    sim_state_for_batch, current_input, update_grid=update_grid_now
                )
                # print("current input:",current_input)
                # print("output and state prediction:",output_pred.item(), next_sim_state)
                output_pred_list_batch.append(output_pred)

                # Update current combined state for next time step
                # current_combined_state = next_combined_state
                sim_state_for_batch = next_sim_state
                if update_grid_now and t == 0:
                    update_grid_now = False # Prevent repeated updates within the same batch simulation
                
            # break
            predicted_outputs_batch = torch.cat(output_pred_list_batch, dim=0)
            current_sim_state = sim_state_for_batch.detach()

            loss_output = loss_fn(
                predicted_outputs_batch, batch_target_output
            )
        # break
        #reg_loss_l1 = kan_model.kan.regularization_loss()
        reg_loss_l1 = model.regularization_loss(regularize_activation=0, regularize_entropy=1e-5) # Gets combined loss from active KANs
        reg_loss_l2 = (
            torch.norm(model.A - dataset.A_init) ** 2
            + torch.norm(model.B - dataset.B_init) ** 2
            + torch.norm(model.C - dataset.C_init) ** 2
            + torch.norm(model.D - dataset.D_init) ** 2
        )

        reg_loss_l2 = (torch.norm(model.A) + torch.norm(model.B) +
             torch.norm(model.C) + torch.norm(model.D))

        # reg_loss_l2 = (torch.norm(model.A1) + torch.norm(model.B1) +
        #     torch.norm(model.C1) + torch.norm(model.A2) + torch.norm(model.B2) +
        #     torch.norm(model.C2) +torch.norm(model.D1)+torch.norm(model.D2))
        # --- Apply Weight ---
        # Simple linear weight: increases from min_weight to 1.0 over the epoch
        #min_weight = 0.1 # Example: start weighting at 0.5
        #max_weight = 2.5 # Example: end weighting at 1.5 (tune these)
        # Or simpler: weight = min_weight + (1.0 - min_weight) * (batch_idx / (num_batches - 1))
       # weight = min_weight + (max_weight - min_weight) * (batch_idx / max(1, 16 - 1))

        loss = (
            loss_output
            + reg_lambda_l2 * reg_loss_l2
            + reg_lambda_l1 * reg_loss_l1
        )
        # Option 2: Weight only the output MSE loss 
        #loss = (loss_output * weight) + reg_lambda_l2 * reg_loss_l2 + reg_lambda_l1 * reg_loss_l1
        
        loss.backward()

        if torch.isnan(loss) or torch.isinf(loss):
            print("Error: Loss is NaN or Inf! Stopping training.")
            break  # Stop training if loss is invalid

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Try max_norm=1.0 or 5.0

        optimizer.step()  # Optimizer step for batch

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        batch_loss_list_train.append(loss_output.item())
    # break
    avg_epoch_loss = float(np.mean(batch_loss_list_train))
    epoch_loss_list_train.append(avg_epoch_loss)
    print(f'Epoch average train loss = {avg_epoch_loss:.4f}')
    scheduler.step()
    
    # Test evaluation trick to avoid too many test evaluations which cost like shit
    test_loss = None
    if (epoch + 1) % test_eval_every == 0:
        test_loss = _utils.run_test_simulation_and_loss(current_sim_state, model, dataset, device, loss_fn)
        epoch_loss_list_test.append(test_loss)
        print(f"Epoch {epoch+1} Test Loss: {test_loss:.4f}")
    
    # test_loss = _utils.run_test_simulation_and_loss(current_sim_state, model, dataset, device, loss_fn)
    # epoch_loss_list_test.append(test_loss)
    # print(f"Epoch {epoch+1} Test Loss: {test_loss:.4f}")

    # --- Save Best Model Logic ---
    if save_best_model and test_loss is not None and not np.isnan(test_loss):
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            new_best_path = os.path.join(model_save_dir, f'best_model_{test_case_name}_epoch_{epoch}_state_{state_kan_hidden_layers}_output_{output_kan_hidden_layers}_batch_{batch_size}_{extra_info_modelname}.pth')
            try:
                torch.save(model.state_dict(), new_best_path)
                print(f"*** New best model saved with Test Loss: {best_test_loss:.4f} at Epoch {epoch+1} to {new_best_path} ***")
                path_to_delete = current_best_model_path # Store the old path *before* updating
                current_best_model_path = new_best_path 
                if os.path.isfile(path_to_delete):
                    os.remove(path_to_delete)
                    print(f"    Successfully deleted previous best model: {path_to_delete}")
            except Exception as e:
                print(f"\nError saving best model state_dict: {e}")

    
    # --- Periodic Save Model State Dict 
    save_model_flag = (save_every > 0 and (epoch + 1) % save_every == 0) or \
                      (epoch == num_epochs - 1) 

    save_model_flag = False # Disable intermediate saves for now, only save best model

    if save_model_flag and special_flag is None:
        model_name = f'model_{test_case_name}_epoch_{epoch}_state_{state_kan_hidden_layers}_output_{output_kan_hidden_layers}_batch_{batch_size}_{extra_info_modelname}.pth'
        model_save_path = os.path.join(model_save_dir, model_name)
        try:
            torch.save(model.state_dict(), model_save_path)
            print(f"\nModel state_dict saved to {model_save_path}")
        except Exception as e:
            print(f"\nError saving model state_dict: {e}")
    elif save_model_flag and special_flag is not None:
        model_name = f'model_{test_case_name}_epoch_{epoch}_state_{state_kan_hidden_layers}_output_{output_kan_hidden_layers}_batch_{batch_size}_specialflag_{special_flag}_{extra_info_modelname}.pth'
        model_save_path = os.path.join(model_save_dir, model_name)
    
    epoch_time = time.perf_counter() - t0
    epoch_time_list.append(epoch_time)

# import sys
# sys.exit("end")
t1 = time.perf_counter()
t_total = t1 - t0
print(f"\n Training finished in {t_total:.2f} seconds")
print('restart from save but with update TRUE')
trainingCSVPath = os.path.join(model_save_dir, f'training_history_{test_case_name}_state_{state_kan_hidden_layers}_output_{output_kan_hidden_layers}_batch_{batch_size}_{extra_info_modelname}.csv')
interleaved_test_loss = [None] * len(epoch_loss_list_train)

for index, test_value in enumerate(epoch_loss_list_test):
    interleaved_test_loss[((index + 1) * test_eval_every) - 1] = test_value


training_history_df = pd.DataFrame({
    'epoch': list(range(1, len(epoch_loss_list_train) + 1)),
    'time': epoch_time_list,
    'train_loss': epoch_loss_list_train,
    'test_loss': interleaved_test_loss,
})
training_history_df.to_csv(trainingCSVPath, index=False)

# %% Final metrics
# --- Final Evaluation Section ---
# Decide whether to load the best model for final evaluation or use the model from the last epoch
model_to_evaluate = model # Default: use model from last epoch
if save_best_model and best_epoch != -1:
    if os.path.isfile(new_best_path):
        print(f"\n--- Loading best model from Epoch {best_epoch+1} for final evaluation ---")
        try:
            state_dict = torch.load(new_best_path, map_location=device)
            model.load_state_dict(state_dict) # Load weights into the existing model object
            model_to_evaluate = model # Ensure we use the model with loaded best weights
            print("Best model weights loaded successfully for evaluation.")
        except Exception as e:
            print(f"Error loading best model for evaluation: {e}. Evaluating model from last epoch.")
    else:
        print("Best model file not found. Evaluating model from last epoch.")


# 1. Simulate model to get time series predictions
simulation_results = _utils.simulate_model(model_to_evaluate, dataset, device)

# 2. Calculate metrics from the simulation results
evaluation_metrics = _utils.calculate_metrics(simulation_results)

# 3. Print the results
print("\n--- Performance Metrics (Original Scale) ---")
print(f"MAE (Train): {evaluation_metrics['mae_train']:.4f}")
print(f"RMSE (Train): {evaluation_metrics['rmse_train']:.4f}")
print(f"MAE (Test): {evaluation_metrics['mae_test']:.4f}")
print(f"RMSE (Test): {evaluation_metrics['rmse_test']:.4f}")

# aaaa
# rows = []
# rows.append({
#     "model_name": new_best_path,
#     "input_dim": input_dim,
#     "state_dim":state_dim,
#     "output_dim":output_dim,
#     "mae_train": float(evaluation_metrics["mae_train"]),
#     "rmse_train": float(evaluation_metrics["rmse_train"]),
#     "mae_test": float(evaluation_metrics["mae_test"]),
#     "rmse_test": float(evaluation_metrics["rmse_test"]),
#     "state_layers": state_kan_hidden_layers,
#     "output_layers": output_kan_hidden_layers,
#     "grid_size": kan_grid_size,
#     "batch_size":batch_size,
#     "epoch": best_epoch
# })

# new_df = pd.DataFrame(rows)
# _utils.save_results_to_excel(new_df)

