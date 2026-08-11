import cv2
import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.linalg import inv, det
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import h5py
from datetime import datetime, timedelta
import torch
from dotmap import DotMap

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()

functions_dir = current_script_path.parents[0] / 'Functions'
sys.path.insert(0, str(functions_dir))
import learning
import inference

ssm_dir = current_script_path.parents[2] / 'SSM'
sys.path.insert(0, str(ssm_dir))
import ssm as ssm

aeon_mecha_dir = current_script_path.parents[2] / 'aeon_mecha' 
sys.path.insert(0, str(aeon_mecha_dir))
import aeon
import aeon.io.api as api
from aeon.io import reader, video
from aeon.schema.schemas import social02
from aeon.analysis.utils import visits, distancetravelled

nodes_name = ['nose', 'head', 'right_ear', 'left_ear', 'spine1', 'spine2','spine3', 'spine4']

class Regression:
    def __init__(self, VISITS):
        self.visits = VISITS
        self.predictor = 'duration'
        self.regressor = None
        self.X = None
        self.Y = None 
        self.split_perc = 0.5
        
    def Get_Variables(self):
        self.X = self.visits[self.regressor]
        #scaler = StandardScaler()
        #self.X = pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
        #self.X.loc[:, 'interc'] = np.ones(len(self.X))
        self.Y = self.visits[[self.predictor]]
    
    def Logistic(self):
        self.Get_Variables()
        
        split_size = int(len(self.Y) * self.split_perc)
        indices = np.arange(len(self.Y))
        
        accuracy_max = -1e10
        for i in range(100):
            np.random.shuffle(indices)
            
            train_indices = indices[:split_size]
            test_indices = indices[split_size:]

            X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
            Y_train, Y_test = self.Y.iloc[train_indices], self.Y.iloc[test_indices]
            
            model = LogisticRegression(random_state=42)
            model.fit(X_train, Y_train.values.ravel())
            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Calculate performance metrics
            accuracy = accuracy_score(Y_test, y_pred)
            '''
            precision = precision_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            '''
            

            if accuracy > accuracy_max:  
                result_valid = model
                accuracy_max = accuracy
        
        return result_valid
    
    def Poisson(self):
        self.Get_Variables()
        '''model = sm.GLM(self.Y, self.X, family=sm.families.Poisson(sm.families.links.Log()))
        result = model.fit()'''
        
        split_size = int(len(self.Y) * self.split_perc)
        indices = np.arange(len(self.Y))
        
        L_max = -1e10
        for i in range(100):
            np.random.shuffle(indices)
            
            train_indices = indices[:split_size]
            test_indices = indices[split_size:]

            X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
            Y_train, Y_test = self.Y.iloc[train_indices], self.Y.iloc[test_indices]
            
            model = sm.GLM(Y_train, X_train, family=sm.families.Poisson(sm.families.links.Log()))
            #result = model.fit_regularized(alpha=0.01, L1_wt=1.0, maxiter = 1000)
            result = model.fit()
            Y_test_pred = result.predict(X_test)
            Y_test, Y_test_pred = Y_test.to_numpy().reshape(1,-1)[0], Y_test_pred.to_numpy()
            
            epsilon = 1e-10
            Y_test_pred = np.where(Y_test_pred <= 0, epsilon, Y_test_pred)

            L = np.mean(Y_test * np.log(Y_test_pred) - Y_test_pred - np.log(factorial(Y_test)))
            #D = 2 * np.sum(Y_test * np.log(Y_test / Y_test_pred) - (Y_test - Y_test_pred))
            if L > L_max:  
                result_valid = result
                L_max = L
        
        return result_valid
    
    def Linear_Regression(self):
        self.Get_Variables()
        split_size = int(len(self.Y) * self.split_perc)
        indices = np.arange(len(self.Y))
        
        corre_max = -1
        for i in range(100):
            np.random.shuffle(indices)
            
            train_indices = indices[:split_size]
            test_indices = indices[split_size:]

            X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
            Y_train, Y_test = self.Y.iloc[train_indices], self.Y.iloc[test_indices]
            
            model = sm.GLM(Y_train, X_train, family=sm.families.Gaussian())
            result = model.fit()
            Y_test_pred = result.predict(X_test)
            Y_test, Y_test_pred = Y_test.to_numpy().reshape(1,-1)[0], Y_test_pred.to_numpy()

            corre = np.corrcoef(Y_test, Y_test_pred)[0,1]
            if corre > corre_max:  
                result_valid = result
                x_test, y_test = X_test, Y_test
                corre_max = corre
        
        return result_valid

class HMM:
    def __init__(self, mouse):
        self.mouse = mouse
        self.n_state = None
        self.feature = None
        self.features = None
        self.model = None
        self.model_period = self.mouse.active_chunk
        self.parameters = None
        self.states = None
        self.TransM = None
        self.loglikelihood = None
        self.process_states = self.Process_States(self)
        self.kl_divergence = None
        self.ConnecM = None
        
    def Get_Features(self):
        if self.feature == 'Kinematics': self.features = ['smoothed_speed', 'smoothed_acceleration']
        if self.feature == 'Body': self.features = ['spine1-spine3','head-spine3', 'right_ear-spine3', 'left_ear-spine3']
        if self.feature == 'Kinematics_and_Body': self.features = ['smoothed_speed', 'smoothed_acceleration', 'head-spine3', 'left_ear-spine3', 'right_ear-spine3', 'spine1-spine3']
    
    def Get_KL_Divergence(self):
        def kl_divergence_gaussian(p_means, p_variances, q_means, q_variances):
            k = len(p_means)
    
            q_variances_inv = inv(q_variances)
            trace_term = np.trace(np.dot(q_variances_inv, p_variances))
            
            mean_diff = q_means - p_means
            mean_term = np.dot(np.dot(mean_diff.T, q_variances_inv), mean_diff)

            det_term = np.log(det(q_variances) / det(p_variances))

            kl_div = 0.5 * (trace_term + mean_term - k + det_term)
            
            return max(0, kl_div)

        def create_kl_divergence_matrix(states):
            n_states = len(states)
            kl_matrix = np.zeros((n_states, n_states))
            
            for i in range(n_states):
                for j in range(n_states):  
                    p_means, p_variances = states[j]
                    q_means, q_variances = states[i]
                    kl_matrix[i, j] = kl_divergence_gaussian(p_means, p_variances, q_means, q_variances)

            
            return kl_matrix
        
        states_params = [(self.parameters[0].T[i], self.parameters[2][i]) for i in range(self.n_state)]
        self.kl_divergence = create_kl_divergence_matrix(states_params)
    
    def Get_ConnecM(self):
        ConnecM = np.zeros((10,10))
        states = self.states
        for i in range(len(states)-1):
            if states[i+1] != states[i]:
                ConnecM[states[i]][states[i+1]] += 1
        for i in range(10):
            ConnecM[i] = ConnecM[i]/np.sum(ConnecM[i])
            
        self.ConnecM = ConnecM
    
    def Fit_Model(self, n_state, feature):
        self.n_state = n_state
        self.feature = feature
        self.Get_Features()
        
        # Fit Models
        fitting_input = np.array(self.mouse.mouse_pos[self.model_period[0]:self.model_period[1]][self.features])
        self.model = ssm.HMM(self.n_state, len(fitting_input[0]), observations="gaussian")
        lls = self.model.fit(fitting_input, method="em", num_iters=50, init_method="kmeans")
        
        # Sort Clusters Based on Ascending Speed
        state_mean_speed = self.model.observations.params[0].T[0]
        index = np.argsort(state_mean_speed, -1) 
        
        # Sort and Save Parameters
        parameters_mean_sorted = self.model.observations.params[0][index].T
        parameters_var = np.zeros((self.n_state,len(self.features)))
        parameters_covar = self.model.observations.params[1]
        for i in range(self.n_state):
            for j in range(len(self.features)):
                parameters_var[i][j] = self.model.observations.params[1][i][j][j] # state i, feature j 
        parameters_var_sorted = parameters_var[index].T 
        parameters_covar_sorted = parameters_covar[index]
        self.parameters = [parameters_mean_sorted, parameters_var_sorted, parameters_covar_sorted]
        np.save('../../SocialData/HMMStates/Params_' + self.mouse.type + "_" + self.mouse.mouse + '.npy', self.parameters)
        
        # Sort and Save Transition Matrix
        self.TransM = self.model.transitions.transition_matrix[index].T[index].T
        np.save('../../SocialData/HMMStates/TransM_' + self.mouse.type + "_" + self.mouse.mouse + '.npy', self.TransM)
        
        # Infer, Sort and Save states
        obs = np.array(self.mouse.mouse_pos[self.features])
        self.loglikelihood = self.model.log_likelihood(obs)
        self.states = self.model.most_likely_states(obs)
        new_values = np.empty_like(self.states)
        for i, val in enumerate(index): new_values[self.states == val] = i
        self.states = new_values
        np.save('../../SocialData/HMMStates/States_' + self.mouse.type + "_" + self.mouse.mouse + ".npy", self.states)
        
    def Fit_Model_without_Saving(self, n_state, feature):
        self.n_state = n_state
        self.feature = feature
        self.Get_Features()
        
        fitting_input = np.array(self.mouse.mouse_pos[self.model_period[0]:self.model_period[1]][self.features])
        inferring_input = np.array(self.mouse.mouse_pos[self.features]) #np.array(self.mouse.mouse_pos[self.model_period[0] + pd.Timedelta('1D'):self.model_period[1]+pd.Timedelta('1D')][self.features])
        
        self.model = ssm.HMM(self.n_state, len(fitting_input[0]), observations="gaussian")
        self.loglikelihood = self.model.fit(fitting_input, method="em", num_iters=50, init_method="kmeans")
        
        ''' 
        # Sort Clusters Based on Ascending Speed
        state_mean_speed = self.model.observations.params[0].T[0]
        index = np.argsort(state_mean_speed, -1) 
        
        # Sort and Save Parameters
        parameters_mean_sorted = self.model.observations.params[0][index].T
        parameters_var = np.zeros((self.n_state,len(self.features)))
        parameters_covar = self.model.observations.params[1]
        for i in range(self.n_state):
            for j in range(len(self.features)):
                parameters_var[i][j] = self.model.observations.params[1][i][j][j] # state i, feature j 
        parameters_var_sorted = parameters_var[index].T 
        parameters_covar_sorted = parameters_covar[index]
        self.parameters = [parameters_mean_sorted, parameters_var_sorted, parameters_covar_sorted]
        
        # Sort and Save Transition Matrix
        self.TransM = self.model.transitions.transition_matrix[index].T[index].T

        # Infer States
        self.states = self.model.most_likely_states(inferring_input)
        self.loglikelihood = self.model.log_likelihood(inferring_input)
        new_values = np.empty_like(self.states)
        for i, val in enumerate(index): new_values[self.states == val] = i
        self.states = new_values
        '''
    
    def Get_States(self, n_state, feature):
        try:
            self.TransM = np.load('../../SocialData/HMMStates/TransM_' + self.mouse.type + "_" + self.mouse.mouse + ".npy", allow_pickle=True)
            self.parameters = np.load('../../SocialData/HMMStates/Params_' + self.mouse.type + "_" + self.mouse.mouse + '.npy', allow_pickle=True)
            self.states = np.load('../../SocialData/HMMStates/States_' + self.mouse.type + "_" + self.mouse.mouse + ".npy", allow_pickle=True)
        except FileNotFoundError:
            self.Fit_Model(n_state, feature)

        self.n_state = n_state
        self.feature = feature
        self.Get_Features()
        self.Get_KL_Divergence()
        self.Get_ConnecM()
        
    class Process_States:
        def __init__(self, hmm):
            self.hmm = hmm
            
        def State_Probability(self, mouse_pos, time_seconds = 10):
            if 'state' not in mouse_pos.columns: mouse_pos['state'] = self.hmm.states
            grouped = mouse_pos.groupby([pd.Grouper(freq=str(time_seconds)+'S'), 'state']).size()
            prob = grouped.groupby(level=0).apply(lambda g: g / g.sum())
            states_prob = prob.unstack(level=-1).fillna(0)
            states_prob.index = states_prob.index.get_level_values(0)
            max_columns = states_prob.idxmax(axis=1)
            return max_columns, states_prob
        
        def State_Dominance(self, mouse_pos, time_seconds = 10):
            max_columns, states_prob = self.State_Probability(mouse_pos, time_seconds)
            states_prob = pd.DataFrame(0, index=states_prob.index, columns=states_prob.columns)
            for row in range(len(states_prob)):
                col = max_columns[row]
                states_prob.at[states_prob.index[row], col] = 1
            return states_prob
        
        def State_Timewindow(self, mouse_pos, timewindow = 10):
            def Calculate_Count_Curve(states, target_state, window_size):
                count_curve = np.zeros(len(states))
                for i in range(len(states) - window_size + 1):
                    window = states[i:i+window_size]
                    count = np.count_nonzero(window == target_state)
                    count_curve[i] = count/window_size
                return count_curve
            
            N = self.hmm.n_state
            for i in range(N): 
                mouse_pos.loc[mouse_pos.index, 'State' + str(i)] = Calculate_Count_Curve(mouse_pos['state'].to_numpy(), target_state = i, window_size = timewindow)
            return mouse_pos
            
        def Event_Triggering(self, mouse_pos, Events, left_seconds, right_seconds, variable = 'state', insert_nan = 1, return_Events = False):
            left_period = pd.Timedelta(str(left_seconds+1) + 'S')
            right_period = pd.Timedelta(str(right_seconds+1) + 'S')
            
            Valid_Events = []
            VARIABLES = []
            for i in range(len(Events)):
                trigger = Events[i]
                
                latest_valid_index = mouse_pos.loc[trigger - left_period:trigger, variable].index
                latest_valid_variable = mouse_pos.loc[latest_valid_index, [variable]].values.reshape(-1)
                if len(latest_valid_variable) >= int(10*left_seconds): latest_valid_variable  = latest_valid_variable[-int(10*left_seconds):]
                
                next_valid_index = mouse_pos.loc[trigger:trigger + right_period, variable].index
                next_valid_variable = mouse_pos.loc[next_valid_index, [variable]].values.reshape(-1)
                if len(next_valid_variable) >= int(10*right_seconds): next_valid_variable  = next_valid_variable[:int(10*right_seconds)]
                
                if insert_nan == 1: Variable = np.concatenate((latest_valid_variable, np.array([np.nan]), next_valid_variable))
                else: Variable = np.concatenate((latest_valid_variable, next_valid_variable))
                
                if len(Variable) == 10*(int(left_seconds + right_seconds)) + insert_nan: 
                    VARIABLES.append(Variable)
                    Valid_Events.append(trigger)
            if return_Events: return np.array(Valid_Events), np.array(VARIABLES)
            else: return np.array(VARIABLES)
        
        def Find_Event_Sequence(self, STATES, penalty = 0):
            def most_frequent(arr):
                states, count = np.unique(arr, return_counts = True)
                if max(count)/len(arr) < penalty: return np.nan
                return states[np.argmax(count)]
            
            #valid_start = int(len(STATES.T[0])/3)
            valid_start = 0
            sequence = [most_frequent(STATES.T[i][valid_start:]) for i in range(len(STATES[0]))]
            return sequence
        
        def Compare_Sequence(self, seq1, seq2, kl_matrix, connecM = None):
            nx, ny = len(seq1), len(seq2)

            cost = np.zeros((nx + 1, ny + 1))
            cost[0, 1:] = np.inf
            cost[1:, 0] = np.inf

            for i in range(1, nx + 1):
                j_start, j_stop = 1, ny + 1
                for j in range(j_start, j_stop):
                    #cost[i, j] = kl_matrix[int(seq1[i-1]), int(seq2[j-1])] * (1 - connecM[int(seq1[i-1]), int(seq2[j-1])])
                    cost[i, j] = kl_matrix[int(seq1[i-1]), int(seq2[j-1])]
                    cost[i, j] += min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            
            return cost[-1, -1]
            
class Arena:
    def __init__(self, mouse):
        self.mouse = mouse
        self.root = self.mouse.root
        self.start = self.mouse.mouse_pos.index[0]
        self.end = self.mouse.mouse_pos.index[-1]
        self.metadata = self.Get_Metadata()
        self.starts = self.Get_Starts()
        self.origin = self.Get_Origin()
        self.radius = self.Get_Radius()
        self.entry = self.Entry()
        self.patch = ['Patch1', 'Patch2', 'Patch3']
        self.patch_location = self.Get_Patch_Location()
        self.patch_distance = self.Get_Patch_Distance()
        self.patch_r = 50
        self.wheel_r = 30

        self.move_wheel_interval_seconds = 5
        self.pre_visit_seconds = 10
        
        self.pellets_per_patch = self.Get_Pellets_per_Patch()
        self.pellets = self.Get_Pellets()
        self.visits = None
    
    def Get_Metadata(self):
        metadata = aeon.load(self.root, social02.Metadata, start=self.mouse.mouse_pos.index[0] - pd.Timedelta('1H'), end=self.mouse.mouse_pos.index[0] + pd.Timedelta('3D'))["metadata"].iloc[0]
        return metadata
    
    def Get_Starts(self):
        starts = []
        for i in range(len(self.mouse.starts)):
            start = pd.Timestamp(datetime.strptime(self.mouse.starts[i], '%Y-%m-%dT%H-%M-%S'))
            starts.append(start)
        return starts
    
    def Get_Origin(self):
        origin = self.metadata.ActiveRegion.ArenaCenter
        return [int(origin.X), int(origin.Y)]
    
    def Get_Radius(self):
        r_inner = int(self.metadata.ActiveRegion.ArenaInnerRadius)
        r_outer = int(self.metadata.ActiveRegion.ArenaOuterRadius)
        r = (r_outer + r_inner)/2
        return r-5
    
    def Get_Patch_Location(self):
        patch_loc = {}
        patch_loc['Patch1'] = np.mean([(int(point.X), int(point.Y)) for point in self.metadata.ActiveRegion.Patch1Region.ArrayOfPoint], axis = 0)
        patch_loc['Patch2'] = np.mean([(int(point.X), int(point.Y)) for point in self.metadata.ActiveRegion.Patch2Region.ArrayOfPoint], axis = 0)
        patch_loc['Patch3'] = np.mean([(int(point.X), int(point.Y)) for point in self.metadata.ActiveRegion.Patch3Region.ArrayOfPoint], axis = 0)
        return patch_loc
    
    def Get_Patch_Distance(self):
        patch1 = self.patch_location['Patch1']
        patch2 = self.patch_location['Patch2']
        patch3 = self.patch_location['Patch3']
        d1 = np.sqrt((patch1[0]-patch2[0])**2 + (patch1[1]-patch2[1])**2)
        d2 = np.sqrt((patch2[0]-patch3[0])**2 + (patch2[1]-patch3[1])**2)
        d3 = np.sqrt((patch1[0]-patch3[0])**2 + (patch1[1]-patch3[1])**2)
        return (d1+d2+d3)/3
            
    def Entry(self):
        mouse_pos_ = self.mouse.mouse_pos.copy()
        distance = np.sqrt((self.mouse.mouse_pos['smoothed_position_x'] - self.origin[0]) ** 2 + (self.mouse.mouse_pos['smoothed_position_y'] - self.origin[1]) ** 2)
        mouse_pos_['Arena'] = 0
        mouse_pos_.loc[self.mouse.mouse_pos.iloc[np.where(distance < self.radius)].index, 'Arena'] = 1
        InArena = mouse_pos_.Arena.to_numpy()
        outside_arena = mouse_pos_.iloc[np.where(InArena < 1)].index
        outside_arena_array = outside_arena.to_numpy()
        outside_inside_indices = np.where(np.diff(outside_arena_array) > np.timedelta64(2, 's'))[0]
        entry_raw = outside_arena[outside_inside_indices]
        entry_raw_position_x = mouse_pos_.loc[entry_raw, 'smoothed_position_x']
        true_entry_indices = np.where(entry_raw_position_x < (self.origin[0]-self.radius + 30), True, False)
        entry = entry_raw[true_entry_indices]
        return entry
    
    def Get_Pellets_per_Patch(self):
        social02.Patch1.DeliverPellet.value = 1
        pellets1 = aeon.load(self.root, social02.Patch1.DeliverPellet, start=self.start, end=self.end)

        social02.Patch2.DeliverPellet.value = 1
        pellets2 = aeon.load(self.root, social02.Patch2.DeliverPellet, start=self.start, end=self.end)

        social02.Patch3.DeliverPellet.value = 1
        pellets3 = aeon.load(self.root, social02.Patch3.DeliverPellet, start=self.start, end=self.end)
            
        return {'Patch1':pellets1, 'Patch2':pellets2, 'Patch3':pellets3}
        
    def Get_Pellets(self):
        def Check_Pellet(PELLET):
            valid_rows = []
            patch_loc = [self.patch_location['Patch' + str(i+1)] for i in range(3)]
            for time_index in PELLET.index:
                window_start = time_index - pd.Timedelta('0.5S')
                window_end = time_index + pd.Timedelta('0.5S')
                
                x = self.mouse.mouse_pos[window_start:window_end]['x'].to_numpy()
                y = self.mouse.mouse_pos[window_start:window_end]['y'].to_numpy()
                
                patch_distance = [np.sqrt((x[0]-patch_loc[j][0])**2 + (y[0]-patch_loc[j][1])**2) for j in range(3)]
                
                has_mouse_pos = len(x) > 0 and min(patch_distance) < self.patch_r
                
                valid_rows.append(has_mouse_pos)

            return PELLET[valid_rows]
            
        try:
            PELLET = pd.read_parquet('../../SocialData/Pellets/'+ self.mouse.type + "_" + self.mouse.mouse +'_PELLET.parquet', engine='pyarrow')
        except FileNotFoundError:
            PELLET = pd.concat([self.pellets_per_patch['Patch1'],self.pellets_per_patch['Patch2'], self.pellets_per_patch['Patch3']], ignore_index=False)
            PELLET = Check_Pellet(PELLET)
            PELLET = PELLET.sort_index()
            PELLET.to_parquet('../../SocialData/Pellets/'+ self.mouse.type + "_" + self.mouse.mouse +'_PELLET.parquet', engine='pyarrow')
        return PELLET

    
    def Move_Wheel(self, start, end, patch):
        interval_seconds = self.move_wheel_interval_seconds
        starts, ends = [],[]
        while start < end:
            if start.minute != 0:
                if start.hour != 23:
                    end_ = pd.Timestamp(year = start.year, month = start.month, day = start.day, hour = start.hour+1, minute=0, second=0) - pd.Timedelta('2S')
                else:
                    end_ = pd.Timestamp(year = start.year, month = start.month, day = start.day+1, hour = 0, minute=0, second=0) - pd.Timedelta('2S')
            else:
                end_ = start + pd.Timedelta('1H') - pd.Timedelta('2S')
            starts.append(start+ pd.Timedelta('1S'))
            ends.append(end_)
            start = end_ + pd.Timedelta('2S')

        encoders = []
        if patch == 'Patch1': 
            for i in range(len(starts)):
                start, end = starts[i], ends[i]
                encoder = aeon.load(self.root, social02.Patch1.Encoder, start=start, end=end)
                encoders.append(encoder)
        elif patch == 'Patch2': 
            for i in range(len(starts)):
                start, end = starts[i], ends[i]
                encoder = aeon.load(self.root, social02.Patch2.Encoder, start=start, end=end)
                encoders.append(encoder)
        else: 
            for i in range(len(starts)):
                start, end = starts[i], ends[i]
                encoder = aeon.load(self.root, social02.Patch3.Encoder, start=start, end=end)
                encoders.append(encoder)
        encoder = pd.concat(encoders, ignore_index=False)
        encoder = encoder[::5]
        
        encoder = encoder.sort_index()
        encoder = encoder[~encoder.index.duplicated(keep='first')]
        
        w = -distancetravelled(encoder.angle).to_numpy()
        dw = np.concatenate((np.array([0]), w[:-1]- w[1:]))
        encoder['Distance'] = pd.Series(w, index=encoder.index)
        encoder['DistanceChange'] = pd.Series(dw, index=encoder.index)
        encoder['Move'] = np.where(abs(encoder.DistanceChange) > 0.05, 1, 0)
        
        if interval_seconds < 0.01: return encoder
        
        groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
        one_groups = encoder[encoder['Move'] == 1].groupby(groups).groups
        one_groups = list(one_groups.values())
        for i in range(len(one_groups) - 1):
            end_current_group = one_groups[i][-1]
            start_next_group = one_groups[i + 1][0]
            duration = start_next_group - end_current_group

            if duration < pd.Timedelta(seconds=interval_seconds):
                encoder.loc[end_current_group:start_next_group, 'Move'] = 1
                
        return encoder

    def Visits_in_Patch(self, patch):
        def In_Patch(start, end, mouse_pos_):
            patch_ox, patch_oy = self.patch_location[patch][0], self.patch_location[patch][1]
            mouse_pos_after = mouse_pos_[start:end]

            distance = np.sqrt((mouse_pos_after['smoothed_position_x'] - patch_ox)**2 + (mouse_pos_after['smoothed_position_y'] - patch_oy)**2)
            out_of_patch = np.where(distance > self.patch_distance, 1, 0)
            if len(out_of_patch) > 0 and max(out_of_patch) == 1: return False
            else: return True
            
        def Leave_Wheel(visit_end, mouse_pos_):
            patch_ox, patch_oy = self.patch_location[patch][0], self.patch_location[patch][1]
            mouse_pos_after = mouse_pos_[visit_end:visit_end + pd.Timedelta('20S')]

            distance = np.sqrt((mouse_pos_after['smoothed_position_x'] - patch_ox)**2 + (mouse_pos_after['smoothed_position_y'] - patch_oy)**2)
            out_of_wheel = np.where(distance > self.wheel_r, 1, 0)
            first_one_index = np.argmax(out_of_wheel)
            return mouse_pos_after.index[first_one_index]
            
        visits = []
        for i in range(len(self.starts)):
            if i == len(self.starts) - 1: mouse_pos_ = self.mouse.mouse_pos[self.starts[i]:]
            else: mouse_pos_ = self.mouse.mouse_pos[self.starts[i]:self.starts[i+1]]
            
            start, end = mouse_pos_.index[0], mouse_pos_.index[-1]
            encoder = self.Move_Wheel(start, end, patch = patch)
            
            # Check duration between visits
            groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
            one_groups = encoder[encoder['Move'] == 1].groupby(groups).groups
            one_groups = list(one_groups.values())
            for i in range(len(one_groups) - 1):
                end_current_group = one_groups[i][-1]
                start_next_group = one_groups[i + 1][0]
                if In_Patch(end_current_group, start_next_group, mouse_pos_):
                    encoder.loc[end_current_group:start_next_group, 'Move'] = 1
        
            visit = {'start':[],'end':[], 'duration':[], 'speed':[], 'acceleration':[], 'entry':[], 'patch':[], 'pellet':[]}
            groups = encoder['Move'].ne(encoder['Move'].shift()).cumsum()
            moves = encoder[encoder['Move'] == 1].groupby(groups)['Move']
            for name, group in moves:
                visit_start, visit_end = group.index[0], group.index[-1]
                if (visit_end-visit_start).total_seconds() < 0: continue
                
                index = self.entry.searchsorted(visit_start, side='left') - 1
                index = max(index, 0)
                time_from_enter_arena = (visit_start - self.entry[index]).total_seconds()
                if time_from_enter_arena < 0: continue
                
                if len(mouse_pos_[visit_start: visit_end]['x']) == 0: continue
                if len(mouse_pos_[visit_end:visit_end + pd.Timedelta('20S')]['smoothed_position_x']) < 150: continue
                
                visit['start'].append(visit_start)
                visit['end'].append(Leave_Wheel(visit_end, mouse_pos_))
                visit['duration'].append((visit_end-visit_start).total_seconds())
                visit['entry'].append(time_from_enter_arena)
                    
                pre_end = visit_start
                pre_start = pre_end - pd.Timedelta(seconds = self.pre_visit_seconds)
                if pre_start < mouse_pos_.index[0]: pre_start = mouse_pos_.index[0]
                pre_visit_data = mouse_pos_.loc[pre_start:pre_end]
                visit['speed'].append(pre_visit_data['smoothed_speed'].mean())
                visit['acceleration'].append(pre_visit_data['smoothed_acceleration'].mean())
                
                visit['patch'].append(patch)
                visit['pellet'].append(len(self.pellets_per_patch[patch][visit_start:visit_end]))
            visit = pd.DataFrame(visit)
            visits.append(visit)
        visits = pd.concat(visits, ignore_index=True)
        return visits
        
    def Combine_Visits_in_Patch(self, Visits):
        def Check_In_Experiment(Visits):
            mask = pd.Series(False, index=Visits.index)
            for start, end in zip(self.mouse.experiment_starts, self.mouse.experiment_ends):
                mask |= (Visits['start'] >= start) & (Visits['end'] <= end)
            return Visits[mask].reset_index(drop=True)
    
        Visits = pd.concat(Visits, ignore_index=True)

        Visits = Visits[Visits['duration'] >= 5]
        Visits = Visits.sort_values(by='start',ignore_index=True)  
        
        Visits = Check_In_Experiment(Visits)
        
        Visits['last_pellets_self'] = 0
        Visits['last_pellets_other'] = 0
        Visits['last_interval'] = 0
        Visits['next_interval'] = 0
        Visits['last_duration'] = 0
        Visits['last_pellets_interval'] = 0
        
        for i in range(1,len(Visits)):
            start, end = Visits.start[i], Visits.end[i]
            last_end = Visits.end[i-1]
            Visits.loc[i, 'last_interval'] = (start - last_end).total_seconds()
            Visits.loc[i-1, 'next_interval'] = Visits.loc[i, 'last_interval'] 
            Visits.loc[i, 'last_duration'] = Visits.loc[i-1, 'duration']
            
            self_patch, other_patch = False, False
            self_pellet, other_pellet = 0, 0
            for j in range(i-1, -1, -1):
                if self_patch and other_patch: break
                if Visits.patch[j] != Visits.patch[i] and other_patch == False: 
                    other_pellet = Visits.pellet[j]
                    other_patch = True
                if Visits.patch[j] == Visits.patch[i] and self_patch == False:
                    self_pellet = Visits.pellet[j]
                    self_patch = True
            Visits.loc[i, 'last_pellets_self'] = self_pellet
            Visits.loc[i, 'last_pellets_other'] = other_pellet
            
            if Visits.loc[i-1, 'pellet'] > 0:
                Visits.loc[i, 'last_pellets_interval'] = Visits.loc[i, 'last_interval']
            else:
                Visits.loc[i, 'last_pellets_interval'] = Visits.loc[i, 'last_interval'] + Visits.loc[i-1, 'last_pellets_interval'] + Visits.loc[i-1, 'duration']
        
        Visits = Visits.dropna(subset=['speed'])
        Visits = Visits[Visits['last_interval'] >= 0]
        Visits = Visits[Visits['next_interval'] >= 0]
        
        Visits = Visits.sort_values(by='start',ignore_index=True) 
        return Visits
        
    def Get_Visits(self):
        try:
            Visits = pd.read_parquet('../../SocialData/VisitData/'  + self.mouse.type + "_" + self.mouse.mouse +'_Visit.parquet', engine='pyarrow')
        except FileNotFoundError:
            if self.pellets_per_patch == None: self.Get_Pellets()
            Visits = []
            for patch in self.patch:
                visits = self.Visits_in_Patch(patch)
                print('Visits for ' + patch + " Completed")
                Visits.append(visits)
            Visits = self.Combine_Visits_in_Patch(Visits)
            Visits = Visits.sort_values(by='start',ignore_index=True) 
            Visits.to_parquet('../../SocialData/VisitData/'  + self.mouse.type + "_" + self.mouse.mouse +'_Visit.parquet', engine='pyarrow')
        self.visits = Visits
        print('Get_Visits Completed')
        
class Kinematics:
    def __init__(self, session):
        self.session = session
        self.manual_parameters = self.Get_Manual_Parameters()
        self.parameters = {}
        self.filterRes = None
        self.smoothRes = None
        self.mouse = None
        
    def Run(self, mouse=None):
        self.mouse = mouse
        self.Infer_Parameters()
        self.Inference() 
        
    def Get_Manual_Parameters(self):
        dt = 0.1
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
        active_chunk = self.mouse.active_chunk
        mouse_pos = self.session.mouse_pos[active_chunk[0] + pd.Timedelta('2H'):active_chunk[0] + pd.Timedelta('4H')]
        return mouse_pos
    
    def Infer_Parameters(self):
        try:
            P = np.load('../../SocialData/LDS_Parameters/' + self.mouse.type + '_' + self.mouse.mouse + '_Parameters.npz', allow_pickle=True)
            sigma_a, sigma_x, sigma_y, sqrt_diag_V0_value, B, Qe, m0, V0, Z, R = P['sigma_a'].item(), P['sigma_x'].item(), P['sigma_y'].item(), P['sqrt_diag_V0_value'].item(), P['B'], P['Qe'], P['m0'], P['V0'], P['Z'], P['R']
        except FileNotFoundError:
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
            np.savez('../../SocialData/LDS_Parameters/'  + self.mouse.type + '_' + self.mouse.mouse + '_Parameters.npz', sigma_a = sigma_a, sigma_x = sigma_x, sigma_y = sigma_y, sqrt_diag_V0_value = sqrt_diag_V0_value, B = B, Qe = Qe, m0 = m0, V0 = V0, Z = Z, R = R)
            print('Inferring LDS Parameters Completed')
        
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
        try:
            filterRes = np.load('../../SocialData/LDS/' + self.session.start +'_filterRes.npz')
            smoothRes = np.load('../../SocialData/LDS/' + self.session.start +'_smoothRes.npz')
        except FileNotFoundError:
            obs = np.transpose(self.session.mouse_pos[["x", "y"]].to_numpy())
            
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
            np.savez_compressed('../SocialData/LDS/' + self.session.start +'_filterRes.npz', **filterRes)
                
            # Smoothing
            smoothRes = inference.smoothLDS_SS( 
                B=B, xnn=filterRes["xnn"], Vnn=filterRes["Vnn"],
                xnn1=filterRes["xnn1"], Vnn1=filterRes["Vnn1"], m0=m0, V0=V0)
            np.savez_compressed('../../SocialData/LDS/' + self.session.start +'_smoothRes.npz', **smoothRes) 
            print('Inference Completed')
            
        self.filterRes = filterRes
        self.smoothRes = smoothRes

class Body_Info:
    def __init__(self, mouse):
        self.mouse = mouse
        self.FixOutrangeData()
        self.cluster = {}
        
    def FixOutrangeData(self):
        timeindex = self.mouse.body_data_x.index
        for name in nodes_name:
            indices_x = np.where(abs(np.diff(self.mouse.body_data_x[name].to_numpy()) > 223))[0][1::2]
            indices_y = np.where(abs(np.diff(self.mouse.body_data_y[name].to_numpy()) > 170))[0][1::2]
            indices = np.unique(np.concatenate((indices_x, indices_y)))
            self.mouse.body_data_x.loc[timeindex[indices], name] = np.nan
            self.mouse.body_data_x.loc[timeindex[indices], name] = np.nan
    def Process_Body_Node_Distance(self, node1, node2): 
        dx = self.mouse.body_data_x[node1] - self.mouse.body_data_x[node2]
        dy = self.mouse.body_data_y[node1] - self.mouse.body_data_y[node2]
        distance = np.sqrt(dx**2 + dy**2)
        return distance
    def Process_Body_Node_Angle(self, node1, node0, node2): 
        position1 = np.array([self.mouse.body_data_x[node1], self.mouse.body_data_y[node1]]).T
        position0 = np.array([self.mouse.body_data_x[node0], self.mouse.body_data_y[node0]]).T
        position2 = np.array([self.mouse.body_data_x[node2], self.mouse.body_data_y[node2]]).T
        v1 = position1 - position0
        v2 = position2 - position0
        dot_product = np.einsum('ij,ij->i', v1, v2)
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        cos_theta = dot_product / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radians_theta = np.arccos(cos_theta)
        return np.degrees(radians_theta)

class Mouse:
    def __init__(self, aeon_exp, type, mouse):
        self.aeon_exp = aeon_exp
        self.type = type
        self.mouse = mouse
        self.root = self.Get_Root()
        self.INFO = self.Get_INFO()
        self.starts, self.ends = self.Get_Start_Times()
        self.experiment_starts, self.experiment_ends = self.Get_Experiment_Times()
        self.active_chunk = self.Get_Active_Chunk()
        
        self.body_data_x, self.body_data_y = self.Combine_SLEAP_Data()
        self.mouse_pos = self.Get_Mouse_Pos()
        

        self.arena = Arena(self)
        self.body_info = Body_Info(self)
        self.Add_Body_Info_to_mouse_pos()
        self.hmm = HMM(self)

    def Get_Root(self):
        if self.aeon_exp == 'AEON3': return '/ceph/aeon/aeon/data/raw/AEON3/social0.2/'
        if self.aeon_exp == 'AEON4': return '/ceph/aeon/aeon/data/raw/AEON4/social0.2/'
        
    def Get_INFO(self):
        experiment = Experiment()
        if self.aeon_exp == 'AEON3': return experiment.INFO3
        if self.aeon_exp == 'AEON4': return experiment.INFO4
        
    def Get_Start_Times(self):
        starts, ends = [], []
        for i in range(len(self.INFO["Mouse"])):
            if self.INFO['Type'][i] == self.type and self.INFO["Mouse"][i] == self.mouse:
                starts.append(self.INFO["Start"][i])
                ends.append(self.INFO["End"][i])
        return starts, ends
    
    def Get_Experiment_Times(self):
        start_time = datetime.strptime(self.starts[0], '%Y-%m-%dT%H-%M-%S')
        end_time = datetime.strptime(self.ends[-1], '%Y-%m-%dT%H-%M-%S') + pd.Timedelta('3H')
        
        experiment_times = DotMap()
        env_states = aeon.load(
            self.root,
            social02.Environment.EnvironmentState,
            start_time,
            end_time
        )

        # Use the last 'maintenance' event as end time
        end_time = (env_states[env_states.state == "Maintenance"]).index[-1]
        env_states = env_states[~env_states.index.duplicated(keep="first")]

        # Retain only events between visit start and stop times
        env_states = env_states.iloc[
            env_states.index.get_indexer([start_time], method="bfill")[
                0
            ] : env_states.index.get_indexer([end_time], method="ffill")[0] + 1
        ]

        # Retain only events where state changes (experiment-maintenance pairs)
        env_states = env_states[env_states["state"].ne(env_states["state"].shift())]
        if env_states["state"].iloc[0] == "Maintenance":
            # Pad with an "Experiment" event at the start
            env_states = pd.concat(
                [
                    pd.DataFrame(
                        "Experiment",
                        index=[start_time],
                        columns=env_states.columns,
                    ),
                    env_states,
                ]
            )
        else:
            # Use start time as the first "Experiment" event
            env_states.rename(index={env_states.index[0]: start_time}, inplace=True)
        experiment_times.start = env_states[
            env_states["state"] == "Experiment"
        ].index.values
        experiment_times.stop = env_states[
            env_states["state"] == "Maintenance"
        ].index.values
        
        return experiment_times.start, experiment_times.stop
    
    def Exclude_Maintainance(self, mouse_pos):
        filtered_data = pd.concat([mouse_pos.loc[start:stop] for start, stop in zip(self.experiment_starts, self.experiment_ends)])
        return filtered_data
    
    def Get_Active_Chunk(self):
        start = datetime.strptime(self.starts[0], '%Y-%m-%dT%H-%M-%S')
        try:
            start_ = pd.Timestamp(year = start.year, month = start.month, day = start.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            start_ = pd.Timestamp(year = start.year, month = start.month+1, day = 1, hour = 7, minute=0, second=0)
        try:
            end_ = pd.Timestamp(year = start_.year, month = start_.month, day = start_.day+1, hour = 7, minute=0, second=0)
        except ValueError:
            end_ = pd.Timestamp(year = start_.year, month = start_.month+1, day = 1, hour = 7, minute=0, second=0)
        return np.array([start_,end_])
    
    def Combine_SLEAP_Data(self):
        try:
            data_x = pd.read_parquet('../../SocialData/BodyData/' + self.type + "_" + self.mouse + '_x.parquet', engine='pyarrow')
            data_y = pd.read_parquet('../../SocialData/BodyData/' + self.type + "_" + self.mouse + '_y.parquet', engine='pyarrow')
        except FileNotFoundError:
            data_x, data_y= [], []
            for i in range(len(self.starts)):
                session = Session(aeon_exp = self.aeon_exp, type = self.type, mouse = self.mouse, start = self.starts[i], end = self.ends[i])
                x, y = session.body_data_x, session.body_data_y
                data_x.append(x)
                data_y.append(y)
                
            data_x = pd.concat(data_x, ignore_index=False)
            data_y = pd.concat(data_y, ignore_index=False)
                
            data_x.to_parquet('../../SocialData/BodyData/' + self.type + "_" + self.mouse + '_x.parquet', engine='pyarrow')
            data_y.to_parquet('../../SocialData/BodyData/' + self.type + "_" + self.mouse + '_y.parquet', engine='pyarrow')
            
            print('Body Data for Mouse Completed')
        
        return data_x, data_y
        
    def Get_Mouse_Pos(self):
        mouse_pos = []
        for i in range(len(self.starts)):
            session = Session(aeon_exp = self.aeon_exp, type = self.type, mouse = self.mouse, start = self.starts[i], end = self.ends[i])
            session.Add_Kinematics(self)
            mouse_pos.append(session.mouse_pos)
        mouse_pos = pd.concat(mouse_pos, ignore_index=False)
        '''mouse_pos = mouse_pos[mouse_pos['smoothed_speed']<2000]
        mouse_pos = mouse_pos[mouse_pos['smoothed_acceleration']<8000]'''
        mouse_pos = self.Exclude_Maintainance(mouse_pos)
        early_period_start, early_period_end = mouse_pos.index[0], mouse_pos.index[0] + pd.Timedelta('600S')
        valid_exp_start = early_period_start
        for i in range(len(mouse_pos[early_period_start: early_period_end])):
            if mouse_pos[early_period_start: early_period_end].smoothed_acceleration[i] > 2500: valid_exp_start = mouse_pos[early_period_start: early_period_end].index[i] + pd.Timedelta('20S')
        return mouse_pos[valid_exp_start:]
    
    def Fix_Body_Info_Nan(self, mouse_pos, column):
        mouse_pos[column] = mouse_pos[column].interpolate()
        mouse_pos[column] = mouse_pos[column].bfill()
        mouse_pos[column] = mouse_pos[column].ffill()
        return mouse_pos 

    def Add_Body_Info_to_mouse_pos(self):
        
        NODES = [['head', 'spine3'], ['left_ear', 'spine3'],['right_ear', 'spine3'], ['spine1', 'spine3']]
        for nodes in NODES:
            variable = nodes[0] + '-' + nodes[1]

            self.mouse_pos[variable] = pd.Series(self.body_info.Process_Body_Node_Distance(nodes[0], nodes[1]), index = self.mouse_pos.index)
            self.mouse_pos = self.Fix_Body_Info_Nan(self.mouse_pos,variable)
        
        variable = 'right_left'
        self.mouse_pos.loc[self.mouse_pos.index, variable] = abs(self.mouse_pos['left_ear-spine3'].to_numpy() - self.mouse_pos['right_ear-spine3'].to_numpy())
        self.mouse_pos = self.Fix_Body_Info_Nan(self.mouse_pos, variable)
        
    def Add_Distance_to_mouse_pos(self):
        distance = np.sqrt((self.mouse_pos['smoothed_position_x'] - self.arena.origin[0]) ** 2 + (self.mouse_pos['smoothed_position_y'] - self.arena.origin[1]) ** 2)
        self.mouse_pos['r'] = distance
    
    def Run_Visits(self):
        self.arena.Get_Visits()
        self.Add_Distance_to_mouse_pos()

class Session:
    def __init__(self, aeon_exp, type, mouse, start, end):
        self.aeon_exp = aeon_exp
        self.type = type
        self.mouse = mouse
        self.start = start
        self.end = end
        self.file_path = self.Get_Filepath()
        self.body_data_x, self.body_data_y = self.Extract_SLEAP_Data()
        self.mouse_pos = self.Get_Mouse_Pos()
        self.kinematics = Kinematics(self)
    
    def Fix_Start(self, data_x, data_y):
        start = data_x.index[0]
        
        if self.start == "2024-02-02T00-15-00": start = pd.Timestamp('2024-02-02 00:16:15.0')
        if self.start == '2024-02-25T17-22-33': start = pd.Timestamp('2024-02-25 17:32:33.0')
        if self.start == '2024-02-28T13-54-17': start = pd.Timestamp('2024-02-28 13:58:45.0')
        if self.start == '2024-01-31T10-14-14': start = pd.Timestamp('2024-01-31 10:21:35.0')
        if self.start == '2024-02-05T14-36-00': start = pd.Timestamp('2024-02-05 14:59:43.1')
        if self.start == '2024-02-25T17-24-32': start = pd.Timestamp('2024-02-25 17:30:05.0')
        if self.start == '2024-02-28T13-45-06': start = pd.Timestamp('2024-01-31 10:21:35.0')
        
        return data_x[start:], data_y[start:]
    
    def Delete_Unrecorded_Head_and_Tail(self, df, df_):
        temp_df = df.dropna(subset=['spine2'])
        first_valid_index, last_valid_index = temp_df.index[0], temp_df.index[-1]
        df = df.loc[first_valid_index:last_valid_index]
        df_ = df_.loc[first_valid_index:last_valid_index]
        
        temp_df = df_.dropna(subset=['spine2'])
        first_valid_index, last_valid_index = temp_df.index[0], temp_df.index[-1]
        df = df.loc[first_valid_index:last_valid_index]
        df_ = df_.loc[first_valid_index:last_valid_index]
        
        return df, df_

    def Get_Filepath(self):
        if self.aeon_exp == 'AEON3': 
            file_path = "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON3/analyses/CameraTop_"
        else: 
            file_path = "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON4/analyses/CameraTop_"
        return file_path
    
    def Extract_SLEAP_Data(self):
        try:
            data_x = pd.read_parquet('../../SocialData/RawData/' + self.start + '_x.parquet', engine='pyarrow')
            data_y = pd.read_parquet('../../SocialData/RawData/' + self.start + '_y.parquet', engine='pyarrow')
        except FileNotFoundError:
            start_chunk, end_chunk = datetime.strptime(self.start, '%Y-%m-%dT%H-%M-%S'), datetime.strptime(self.end, '%Y-%m-%dT%H-%M-%S')
            chunk = start_chunk.replace(minute=0, second=0, microsecond=0)
            flag = 0
            chunks = []
            while flag == 0:
                chunk_time = chunk.strftime('%Y-%m-%dT%H-%M-%S')
                chunks.append(chunk_time)
                chunk += timedelta(hours=1)
                if chunk_time == self.end: flag = 1
            
            dfs_x, dfs_y = [], []
            for j in range(len(chunks)):
                #print(self.file_path,  chunks[j])
                file_path_ = self.file_path + chunks[j] + "_full_pose.analysis.h5"
                with h5py.File(file_path_, 'r') as f: tracks_matrix = f['tracks'][:]
                num_frames = tracks_matrix.shape[3]

                if j == 0: start_time = datetime.strptime(chunks[j+1], '%Y-%m-%dT%H-%M-%S') - timedelta(seconds=num_frames*0.02)
                else: start_time = datetime.strptime(chunks[j], '%Y-%m-%dT%H-%M-%S')

                timestamps = [start_time + timedelta(seconds=k*0.02) for k in range(num_frames)]
                
                df_x = pd.DataFrame(tracks_matrix[0][0].T, index=timestamps, columns=nodes_name)
                dfs_x.append(df_x)
                
                df_y = pd.DataFrame(tracks_matrix[0][1].T, index=timestamps, columns=nodes_name)
                dfs_y.append(df_y)
            
            data_x, data_y = pd.concat(dfs_x, ignore_index=False), pd.concat(dfs_y, ignore_index=False)
            data_x, data_y = data_x[::5],  data_y[::5]
            data_x, data_y = self.Fix_Start(data_x, data_y)
            data_x, data_y = self.Delete_Unrecorded_Head_and_Tail(data_x, data_y)
            
            data_x.to_parquet('../../SocialData/RawData/' + self.start + '_x.parquet', engine='pyarrow')
            data_y.to_parquet('../../SocialData/RawData/' + self.start + '_y.parquet', engine='pyarrow')
        
        return data_x, data_y

    def Get_Mouse_Pos(self):
        x, y = self.body_data_x['spine2'], self.body_data_y['spine2']
        mouse_pos = pd.DataFrame({'x': x, 'y': y})
        mouse_pos['x'] = mouse_pos['x'].interpolate()
        mouse_pos['y'] = mouse_pos['y'].interpolate()
        return mouse_pos
    
    def Add_Kinematics(self, mouse = None):
        self.kinematics.Run(mouse)
        smoothRes = self.kinematics.smoothRes
        self.mouse_pos['smoothed_position_x'] = pd.Series(smoothRes['xnN'][0][0], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_position_y'] = pd.Series(smoothRes['xnN'][3][0], index=self.mouse_pos.index)
    
        x_vel, y_vel = smoothRes['xnN'][1][0], smoothRes['xnN'][4][0]
        vel = np.sqrt(x_vel**2 + y_vel**2)
        self.mouse_pos['smoothed_speed'] = pd.Series(vel, index=self.mouse_pos.index)
            
        x_acc, y_acc = smoothRes['xnN'][2][0], smoothRes['xnN'][5][0]
        acc = np.sqrt(x_acc**2 + y_acc**2)
        '''
        speed_diff = np.diff(vel)
        mask = np.concatenate(([0], np.sign(speed_diff)))
        '''
        self.mouse_pos['smoothed_acceleration'] = pd.Series(acc, index=self.mouse_pos.index)
        

        self.mouse_pos['smoothed_velocity_x'] = pd.Series(smoothRes['xnN'][1][0], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_velocity_y'] = pd.Series(smoothRes['xnN'][4][0], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_acceleration_x'] = pd.Series(smoothRes['xnN'][2][0], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_acceleration_y'] = pd.Series(smoothRes['xnN'][5][0], index=self.mouse_pos.index)
        
        self.mouse_pos['smoothed_position_x_var'] = pd.Series(smoothRes['VnN'][0][0], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_position_y_var'] = pd.Series(smoothRes['VnN'][3][3], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_velocity_x_var'] = pd.Series(smoothRes['VnN'][1][1], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_velocity_y_var'] = pd.Series(smoothRes['VnN'][4][4], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_acceleration_x_var'] = pd.Series(smoothRes['VnN'][2][2], index=self.mouse_pos.index)
        self.mouse_pos['smoothed_acceleration_y_var'] = pd.Series(smoothRes['VnN'][5][5], index=self.mouse_pos.index)
        
        start = self.mouse_pos.index[0] + pd.Timedelta('60S')
        end = self.mouse_pos.index[-1] - pd.Timedelta('60S')
        self.mouse_pos = self.mouse_pos[start:end]

class Experiment:
    def __init__(self):
        self.INFO3 =  { "Type": ["Pre", "Pre", "Pre", "Pre", "Post", "Post", "Post"],
                        "Mouse": ["BAA-1104045", "BAA-1104045", "BAA-1104045", "BAA-1104047","BAA-1104045", "BAA-1104047", "BAA-1104047"],
                        "Start": ["2024-01-31T11-28-39", "2024-02-01T22-36-47", "2024-02-02T00-15-00", "2024-02-05T15-43-07", "2024-02-25T17-22-33", "2024-02-28T13-54-17", "2024-03-01T16-46-12"],
                        "End": ["2024-02-01T20-00-00", "2024-02-01T23-00-00", "2024-02-03T16-00-00", "2024-02-08T14-00-00", "2024-02-28T12-00-00", "2024-03-01T15-00-00", "2024-03-02T15-00-00"]}
        self.INFO4 =  {"Type": ["Pre", "Pre", "Pre", "Pre", "Post", "Post"],
                        "Mouse": ["BAA-1104048", "BAA-1104048", "BAA-1104048", "BAA-1104049","BAA-1104048", "BAA-1104049"],
                        "Start": ["2024-01-31T10-14-14", "2024-02-01T20-46-44", "2024-02-01T23-40-29", "2024-02-05T14-36-00", "2024-02-25T17-24-32", "2024-02-28T13-45-06"],
                        "End": ["2024-02-01T19-00-00", "2024-02-01T22-00-00", "2024-02-03T16-00-00", "2024-02-08T14-00-00", "2024-02-28T12-00-00", "2024-03-02T15-00-00"]}

def main():
    print('None')
    
    
if __name__ == "__main__":
    main()