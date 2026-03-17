#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:49:20 2025

@author: cruz
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import SS_KAN_KUL._utils as _utils
import nonlinear_benchmarks 
from scipy import signal
import scipy.io

class SystemIdentificationDataset:
    def __init__(self, test_case='Silverbox', test_flag=None, norm_flag='minmax',device='cpu',
                 states_available=False,init_matrices_flag=None):
        """
        Loads and preprocesses system identification datasets.

        Args:
            test_case (str): 'Silverbox' or 'Wiener-Hammerstein'.
            test_flag (str, optional): Specific test flag for 'Silverbox' (e.g., 'arrow_extra').
            norm_flag (str): Normalization type ('minmax', 'zscore', etc.).
        """
        self.states_available=states_available
        self.init_matrices_flag = init_matrices_flag
        self.device=device
        self.test_case = test_case
        self.test_flag = test_flag
        self.norm_flag = norm_flag
        self.dt = None # Sampling time
        self.x_min, self.x_max = None, None # Normalization parameters for output/state
        self.x_dot_min, self.x_dot_max = None, None # Normalization parameters for state derivative
        self.u_min, self.u_max = None, None # Normalization parameters for input
        self.X_train_norm, self.u_train_norm, self.y_train_norm = None, None, None # Normalized training data
        self.X_test_norm, self.u_test_norm, self.y_test_norm = None, None, None # Normalized test data

        self.X_dim  = 0
        self.u_dim  = 0
        self.y_dim  = 0

        self.A_init, self.B_init, self.C_init, self.D_init = None, None, None, None # For StateSpaceModel

        self._load_and_preprocess() # Call the data loading and preprocessing method in the constructor
        self._init_lin_matrices() 
        
    def _load_and_preprocess(self):
        test_case = self.test_case
        test_flag = self.test_flag
        norm_flag = self.norm_flag

        if self.test_case == 'Silverbox':
            train_val, test = nonlinear_benchmarks.Silverbox(atleast_2d=True)
            self.dt = train_val.sampling_time
            f_s = 1 / self.dt
            u_train, y_train = train_val

        elif self.test_case == 'Wiener-Hammerstein':
            train_val, test = nonlinear_benchmarks.WienerHammerBenchMark(atleast_2d=True)
            self.dt = train_val.sampling_time
            print(test.state_initialization_window_length)
            u_train, y_train = train_val
            u_test, y_test = test


        elif self.test_case == 'Luca-Airfoil-CFD':
            mat_CFD = scipy.io.loadmat('TimeSeries_Exp_CFD/DatasetCFD.mat')

            u_train = mat_CFD['uTrain'] # Angle of attack
            y_train = mat_CFD['yTrain'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_CFD['uTrain'] # Angle of attack
            y_test = mat_CFD['yTrain'] # Lift coefficient
            
        elif self.test_case == 'Luca-Airfoil-CFD-pitchRate':
            mat_CFD = scipy.io.loadmat('TimeSeries_Exp_CFD/DatasetCFD.mat')

            u_train = mat_CFD['uTrain'] # Angle of attack
            y_train = mat_CFD['yTrain'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_CFD['uTrain'] # Angle of attack
            y_test = mat_CFD['yTrain'] # Lift coefficient
            self.time = np.arange(0,32008*self.dt,self.dt)
            #very raw approach
            da_dt_train = np.gradient(u_train.ravel(),self.time)
            da_dt_train[da_dt_train > 200] = 0
            u_train = np.concatenate((u_train,da_dt_train.reshape(-1,1)),axis=1)
            da_dt_test = da_dt_train # change in future!
            u_test = np.concatenate((u_test,da_dt_test.reshape(-1,1)),axis=1)
            # I think I might have to remake Luca function and create the data since the function he uses is symbolic differentable?
            
            
        elif self.test_case == 'Luca-Airfoil-Exp':
            mat_exp = scipy.io.loadmat('/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/TimeSeries_Exp_CFD/DatasetExp_LowTI.mat')
            u_train = mat_exp['aoa'] # Angle of attack
            y_train = mat_exp['cl'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_exp['aoa'] # Angle of attack
            y_test = mat_exp['cl'] # Lift coefficient
            self.time = np.arange(0,160000*self.dt,self.dt)
            
        elif self.test_case == 'Luca-Airfoil-Exp-pitchRate':
            mat_exp = scipy.io.loadmat('/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/TimeSeries_Exp_CFD/DatasetExp_LowTI.mat')
            u_train = mat_exp['aoa'] # Angle of attack
            y_train = mat_exp['cl'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_exp['aoa'] # Angle of attack
            y_test = mat_exp['cl'] # Lift coefficient
            self.time = np.arange(0,160000*self.dt,self.dt)
            #very raw approach
            da_dt_train = np.gradient(u_train.ravel(),self.time)
            da_dt_train[da_dt_train > 200] = 0
            u_train = np.concatenate((u_train,da_dt_train.reshape(-1,1)),axis=1)
            da_dt_test = da_dt_train # change in future!
            u_test = np.concatenate((u_test,da_dt_test.reshape(-1,1)),axis=1)
            # I think I might have to remake Luca function and create the data since the function he uses is symbolic differentable?
            
            
        else:
            raise ValueError(f"Unknown test_case: {test_case}")

        u_train = torch.tensor(u_train, dtype=torch.float32,device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32,device=self.device)
        t_train = torch.arange(len(y_train), dtype=torch.float32,device=self.device) * self.dt
        self.t_train = t_train
        y_train_norm, self.x_min, self.x_max, _ = _utils.normalize_data(y_train, norm_type=norm_flag, normalize=True)
        u_train_norm, self.u_min, self.u_max, _ = _utils.normalize_data(u_train, norm_type=norm_flag, normalize=True)


        if self.states_available is True:
            x_dot_train = torch.zeros_like(y_train,device=self.device)
            x_dot_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2 * self.dt)
            x_dot_train[0] = (y_train[1] - y_train[0]) / self.dt
            x_dot_train[-1] = (y_train[-1] - y_train[-2]) / self.dt
            x_dot_train_norm, self.x_dot_min, self.x_dot_max, _ = _utils.normalize_data(x_dot_train, norm_type=norm_flag, normalize=True)
            X_train_norm = torch.cat((y_train_norm, x_dot_train_norm), dim=-1)
            self.X_train_norm = X_train_norm
            
        # Process test data
        if test_case=='Silverbox':
            if test_flag=='arrow_no_extra':
                test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
                u_test, y_test = test_arrow_no_extrapolation.u, test_arrow_no_extrapolation.y
            elif test_flag=='arrow_extra':
                test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
                u_test, y_test = test_arrow_full.u, test_arrow_full.y
            elif test_flag=='multisine':
                test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
                u_test, y_test = test_multisine.u, test_multisine.y


        u_test = torch.tensor(u_test, dtype=torch.float32,device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32,device=self.device)
        t_test = torch.arange(len(y_test), dtype=torch.float32,device=self.device) * self.dt
        self.t_test = t_test
        y_test_norm, _, _, _ = _utils.normalize_data(y_test, norm_type=norm_flag, normalize=True, data_min=self.x_min, data_max=self.x_max)
        u_test_norm, _, _, _ = _utils.normalize_data(u_test, norm_type=norm_flag, normalize=True, data_min=self.u_min, data_max=self.u_max)


        if self.states_available is True:
            x_dot_test = torch.zeros_like(y_test,device=self.device)
            x_dot_test[1:-1] = (y_test[2:] - y_test[:-2]) / (2 * self.dt)
            x_dot_test[0] = (y_test[1] - y_test[0]) / self.dt
            x_dot_test[-1] = (y_test[-1] - y_test[-2]) / self.dt
            x_dot_test_norm, _, _, _ = _utils.normalize_data(x_dot_test, norm_type=norm_flag, normalize=True, data_min=self.x_dot_min, data_max=self.x_dot_max)
            X_test_norm = torch.cat((y_test_norm, x_dot_test_norm), dim=-1)
            self.X_test_norm = X_test_norm
            self.X_dim = X_test_norm.size()[1]


        self.u_train_norm, self.y_train_norm =  u_train_norm, y_train_norm
        self.u_test_norm, self.y_test_norm = u_test_norm, y_test_norm
        self.u_dim, self.y_dim  = u_train_norm.size()[1], y_train_norm.size()[1]
    
    def _init_lin_matrices(self):
        
        def np_to_tensor(array):
            return torch.tensor(array, dtype=torch.float32, device=self.device)
        
        
        if self.test_case=='Silverbox':
            # Define system parameters
            m = 1
            c = 0.1
            k = 1
            self.A_init = torch.tensor([[1.0, self.dt],
                              [-self.dt*(k/m), 1.0 - (c/m) * self.dt]], dtype=torch.float32, device=self.device)
            
            self.B_init = torch.tensor([[0.0],
                              [self.dt]], dtype=torch.float32, device=self.device)
            self.C_init = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=self.device)
            self.D_init = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)

        elif self.test_case == 'Wiener-Hammerstein':
            if self.init_matrices_flag == 'filters':
                fs = 51200
                def create_chebyshev1_filter():
                    b, a = signal.cheby1(N=3, rp=0.5, Wn=4400, btype='low', fs=fs)
                    return signal.tf2ss(b, a)
                def create_chebyshev2_filter():
                    b, a = signal.cheby2(N=3, rs=40, Wn=5000, btype='low', fs=fs)
                    return signal.tf2ss(b, a)

                A1, B1, C1, D1 = create_chebyshev1_filter()
                A2, B2, C2, D2 = create_chebyshev2_filter()
                self.A1_init = np_to_tensor(A1)
                self.B1_init = np_to_tensor(B1)
                self.C1_init = np_to_tensor(C1)
                self.D1_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
                self.A2_init = np_to_tensor(A2)
                self.B2_init = np_to_tensor(B2)
                self.C2_init = np_to_tensor(C2)
                self.D2_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
                
            elif self.init_matrices_flag == 'matlab':
                import scipy.io
                mat_path = 'matlab22.mat' 
                mat = scipy.io.loadmat(mat_path)
                A=mat['A_train']
                B=mat['B_train']
                C=mat['C_train']
                
                self.A_init = np_to_tensor(A)
                self.B_init = np_to_tensor(B)
                self.C_init = np_to_tensor(C)
                self.D_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
        
        elif self.test_case == 'Luca-Airfoil-CFD':
            if self.init_matrices_flag == 'matlab_1':
                if self.norm_flag == 'minmax':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_1state.mat'
                elif self.norm_flag == 'nothing':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_NOTnorm_foil_1state.mat'

            elif self.init_matrices_flag == 'matlab_4':
                if self.norm_flag == 'minmax':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_4state.mat'
                elif self.norm_flag == 'nothing':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_4state.mat'

            elif self.init_matrices_flag == 'ones':
                return
            import scipy.io
            mat = scipy.io.loadmat(mat_path)
            A=mat['A_train']
            B=mat['B_train']
            C=mat['C_train']
                
            self.A_init = np_to_tensor(A)
            self.B_init = np_to_tensor(B)
            self.C_init = np_to_tensor(C)
            self.D_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
            
            
        elif self.test_case == 'Luca-Airfoil-Exp':
            if self.init_matrices_flag == 'matlab_1':
                if self.norm_flag == 'minmax':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_1state_EXP.mat'
                elif self.norm_flag == 'nothing':
                    mat_path = None
                    
            if self.init_matrices_flag == 'matlab_2':
                if self.norm_flag == 'minmax':
                    mat_path = '/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/TimeSeries_Exp_CFD/matlab_norm_foil_2state_EXP.mat'

            elif self.init_matrices_flag == 'ones':
                return
            
            import scipy.io
            mat = scipy.io.loadmat(mat_path)
            A=mat['A_train']
            B=mat['B_train']
            C=mat['C_train']
                
            self.A_init = np_to_tensor(A)
            self.B_init = np_to_tensor(B)
            self.C_init = np_to_tensor(C)
            self.D_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
            #self.B_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            #self.C_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)

        elif self.test_case == 'Luca-Airfoil-Exp-pitchRate':

            self.A_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.B_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)
            self.C_init = torch.tensor(np.array(-125).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.C_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.D_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)

        elif self.test_case == 'Luca-Airfoil-CFD-pitchRate':

            self.A_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.B_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)
            self.C_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.D_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)

        return
    
    