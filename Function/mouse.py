import numpy as np
import pandas as pd

import torch


import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
sys.path.insert(0, str(parent_dir))
import Function.learning as learning
import Function.inference as inference
import Script.INFORMATION as info
     
class Kinematics:
    def __init__(self, data, node_name, video_frame_rate, use_parameter = False):
        self.node_name = node_name
        self.mouse_pos = data[node_name]
        self.video_frame_rate = video_frame_rate
        self.use_parameter = use_parameter
        self.manual_parameters = self.Get_Manual_Parameters()
        self.parameters = {}
        self.filterRes = None
        self.smoothRes = None
        
    def Run(self):
        self.Infer_Parameters()
        self.Inference() 
        
    def Get_Manual_Parameters(self):
        dt = 1/self.video_frame_rate
        pos_x0, pos_y0 = 0, 0
        vel_x0, vel_y0 = 0.0, 0.0
        acc_x0, acc_y0 = 0.0, 0.0

        # Manual Parameters
        sigma_a = 20
        sqrt_diag_V0_value = 1e-3

        m0 = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0], dtype=np.double)
        V0 = np.diag(np.ones(len(m0))*sqrt_diag_V0_value**2)


        B = np.array([[1, dt, dt**2/2, 0, 0, 0],
                    [0, 1, dt, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, dt, dt**2/2],
                    [0, 0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 0, 1]],
                    dtype=np.double)


        Qe = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                    [dt**3/2, dt**2,   dt,      0, 0, 0],
                    [dt**2/2, dt,      1,       0, 0, 0],
                    [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                    [0, 0, 0, dt**3/2, dt**2,   dt],
                    [0, 0, 0, dt**2/2, dt,      1]],
                    dtype=np.double)
        Q = sigma_a**2 * Qe

        sigma_x = 1
        sigma_y = 1

        Z = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0]],
                    dtype=np.double)
        R = np.diag([sigma_x**2, sigma_y**2])
        
        parameters = {'sigma_a': sigma_a,
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y,
                    'sqrt_diag_V0_value': sqrt_diag_V0_value,
                    'B': B,
                    'Qe': Qe,
                    'm0': m0,
                    'V0': V0,
                    'Z': Z,
                    'R': R}
        
        if self.use_parameter:
            parameter_file_path = f'{info.data_path}/Parameters/{self.node_name}.npz'
            if Path(parameter_file_path).exists():
                loaded_params = np.load(parameter_file_path)
                parameters = {key: loaded_params[key] for key in loaded_params.files}
                print(f"Loaded parameters from {parameter_file_path}")
            else:
                print(f"Parameter file {parameter_file_path} does not exist. Using manual parameters.")
        return parameters
    
    def Learn_Parameters(self, y, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z):
        lbfgs_max_iter = 2
        lbfgs_tolerance_grad = 1e-3
        lbfgs_tolerance_change = 1e-3
        lbfgs_lr = 1.0
        lbfgs_n_epochs = 100
        lbfgs_tol = 1e-3
        
        Qe_reg_param_learned = 1e-10
        sqrt_diag_R_torch = torch.DoubleTensor([sigma_x, sigma_y])
        m0_torch = torch.from_numpy(m0.copy())
        sqrt_diag_V0_torch = torch.DoubleTensor([sqrt_diag_V0_value
                                                for i in range(len(m0))])
        if Qe_reg_param_learned is not None:
            Qe_regularized_learned = Qe + Qe_reg_param_learned * np.eye(Qe.shape[0])
        else:
            Qe_regularized_learned = Qe
        y_torch = torch.from_numpy(y.astype(np.double))
        B_torch = torch.from_numpy(B.astype(np.double))
        Qe_regularized_learned_torch = torch.from_numpy(Qe_regularized_learned.astype(np.double))
        Z_torch = torch.from_numpy(Z.astype(np.double))

        vars_to_estimate = {}
        vars_to_estimate["sigma_a"] = True
        vars_to_estimate["sqrt_diag_R"] = True
        vars_to_estimate["R"] = True
        vars_to_estimate["m0"] = True
        vars_to_estimate["sqrt_diag_V0"] = True
        vars_to_estimate["V0"] = True

        optim_res_learned = learning.torch_lbfgs_optimize_SS_tracking_diagV0(
            y=y_torch, B=B_torch, sigma_a0=sigma_a,
            Qe=Qe_regularized_learned_torch, Z=Z_torch, sqrt_diag_R_0=sqrt_diag_R_torch, m0_0=m0_torch,
            sqrt_diag_V0_0=sqrt_diag_V0_torch, max_iter=lbfgs_max_iter, lr=lbfgs_lr,
            vars_to_estimate=vars_to_estimate, tolerance_grad=lbfgs_tolerance_grad,
            tolerance_change=lbfgs_tolerance_change, n_epochs=lbfgs_n_epochs,
            tol=lbfgs_tol)
        
        sigma_a = optim_res_learned["estimates"]["sigma_a"].item()
        sigma_x = optim_res_learned["estimates"]["sqrt_diag_R"].numpy()[0]
        sigma_y = optim_res_learned["estimates"]["sqrt_diag_R"].numpy()[1]
        sqrt_diag_V0_value = optim_res_learned["estimates"]["sqrt_diag_V0"].numpy()
        m0 = optim_res_learned["estimates"]["m0"].numpy()
        V0 = np.diag(sqrt_diag_V0_value**2)
        R = np.diag(optim_res_learned["estimates"]["sqrt_diag_R"].numpy()**2)

        return sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value[0], B, m0, V0, Z, R
    
    def Get_Observations(self):
        mouse_pos = self.mouse_pos
        return mouse_pos
    
    def Infer_Parameters(self):
        mouse_pos = self.Get_Observations()
        obs = np.transpose(mouse_pos[["x", "y"]].to_numpy())
        
        params = self.manual_parameters
        sigma_a = params['sigma_a']
        sigma_x = params['sigma_x']
        sigma_y = params['sigma_y']
        sqrt_diag_V0_value = params['sqrt_diag_V0_value']
        B = params['B']
        Qe = params['Qe']
        m0 = params['m0']
        V0 = params['V0']
        Z = params['Z']
        R = params['R']

        sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, m0, V0, Z, R = self.Learn_Parameters(obs, sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, Z)
        print('Inferring LDS Parameters Completed', flush=True)
        
        parameters = {'sigma_a': sigma_a,
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y,
                    'sqrt_diag_V0_value': sqrt_diag_V0_value,
                    'B': B,
                    'Qe': Qe,
                    'm0': m0,
                    'V0': V0,
                    'Z': Z,
                    'R': R}

        self.parameters = parameters
    
    def Inference(self):
        obs = np.transpose(self.mouse_pos[["x", "y"]].to_numpy())
        
        params = self.parameters
        sigma_a = params['sigma_a']
        B = params['B']
        Qe = params['Qe']
        m0 = params['m0']
        V0 = params['V0']
        Z = params['Z']
        R = params['R']

        Q = (sigma_a**2) * Qe

        # Filtering
        filterRes = inference.filterLDS_SS_withMissingValues_np(
            y=obs, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
            
        # Smoothing
        smoothRes = inference.smoothLDS_SS( 
            B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
            xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)

        print('Inference Completed', flush = True)
            
        self.filterRes = filterRes
        self.smoothRes = smoothRes



def main():
    print('None')
    
    
if __name__ == "__main__":
    main()