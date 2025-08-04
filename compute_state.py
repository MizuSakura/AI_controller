import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from itertools import chain
import torch


class compute_process:
    def __init__(self, state_frame=10, error_frame=10, reward_frame=10, action_frame=10,
                 max_state=10,min_state= 0,user_normalization=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_frame = state_frame
        self.error_frame = error_frame
        self.reward_frame = reward_frame
        self.action_frame = action_frame

        self.max_state = max_state
        self.min_state = min_state
        self.max_action = 10
        self.min_action = 0

        self.user_normalization = user_normalization

        self.setpoint = 5

        self.state = deque([0.0] * self.state_frame, maxlen= self.state_frame)
        self.error = deque([0.0] * self.error_frame, maxlen= self.error_frame)
        self.reward = deque([0.0] * self.reward_frame, maxlen= self.reward_frame)
        self.action = deque([0.0] * self.action_frame, maxlen= self.action_frame)
        self.delta_memory = deque(maxlen=3)

        # parameter of static such as  mean,standard-deviation,deviation
        self.ema_alpha_state = 0.5 #--> sentivity EMA 
        self.ema_mean_state = 0.0
        self.ema_std_state = 1e-10

        self.ema_alpha_error = 0.5 #--> sentivity EMA 
        self.ema_mean_error = 0.0
        self.ema_std_error = 1e-10

        self.ema_alpha_action = 0.9 #--> sentivity EMA 
        self.ema_mean_action = 0.0
        self.ema_std_action = 1e-10

        # self.ema_alpha_delta_error = 0.5
        # self.ema_mean_delta_error = 0
        # self.ema_std_delta_error = 1e-10

        self.delta_penalize = 0
        self.max_delta = 10e-15


    def _init_memory_state(self,state):
        self.state = deque([state] * self.state_frame, maxlen= self.state_frame)

    def _init_memory_error(self,setpoint,state):
        error = setpoint - state
        self.error = deque([error] * self.error_frame, maxlen= self.error_frame)

    def _init_memory_reward(self,reward):
        self.reward = deque([reward] * self.reward_frame, maxlen= self.reward_frame)

    def _init_memory_action(self,action):
        self.action = deque([action] * self.action_frame, maxlen= self.action_frame)

    def update_ema_stats_error(self, new_delta):
        diff = new_delta - self.ema_mean_error
        self.ema_mean_error += self.ema_alpha_error * diff
        self.ema_std_error += self.ema_alpha_error * (abs(diff) - self.ema_std_error)

    def update_ema_stats_action(self, new_delta):
        diff = new_delta - self.ema_mean_action
        self.ema_mean_action += self.ema_alpha_action * diff
        self.ema_std_action += self.ema_alpha_action * (abs(diff) - self.ema_std_action)

    def update_ema_stats_state(self, new_delta):
        diff = new_delta - self.ema_mean_state
        self.ema_mean_state += self.ema_alpha_state * diff
        self.ema_std_state += self.ema_alpha_state * (abs(diff) - self.ema_std_state)

    # def update_ema_delta_error(self,new_delta):
    #     diff = new_delta - self.ema_mean_delta_error
    #     self.ema_mean_delta_error += self.ema_alpha_delta_error * diff
    #     self.ema_std_delta_error += self.ema_alpha_delta_error * (abs(diff) - self.ema_std_delta_error)

    def max_min_normalization(self, value, max, min):
        
        result = (value - min) / (max - min)
        return result
    
    def z_score_mromalization(self, value,mean,deviation):
        
        result = (value - mean) / deviation
        return result
    
    def mix_abs_normalization(self, value,max):
        result = value / abs(max)
        return result
    
    def tanh_normalization(self, value, mean, deviation):
        result = (np.tanh((value - mean)/deviation) + 1) * 0.5
        return result
    
    def normalization_state_manager(self, type_normal="max_min", value=0):
        type_normal = type_normal.lower()

        if type_normal == "max_min":
            return self.max_min_normalization(value=value, max=self.max_state, min=self.min_state)
        elif type_normal == "z_score":
            return self.z_score_mromalization(value=value, mean=self.ema_mean_state, deviation=self.ema_std_state)
        elif type_normal == "mix_abs":
            return self.mix_abs_normalization(value=value, max=self.max_state)
        elif type_normal == "tanh":
            return self.tanh_normalization(value=value, mean=self.ema_mean_state, deviation=self.ema_std_state)
        else:
            print(f"[Warning] Unknown normalization type: {type_normal}. Returning raw value.")
            return value
        
    def normalization_action_manager(self,type_normal="max_min", value=0):
        type_normal = type_normal.lower()

        if type_normal == "max_min":
            return self.max_min_normalization(value=value, max=self.max_action, min=self.min_state)
        elif type_normal == "z_score":
            return self.z_score_mromalization(value=value, mean=self.ema_mean_action, deviation=self.ema_std_action)
        elif type_normal == "mix_abs":
            return self.mix_abs_normalization(value=value, max=self.max_action)
        elif type_normal == "tanh":
            return self.tanh_normalization(value=value, mean=self.ema_mean_action, deviation=self.ema_std_action)
        else:
            print(f"[Warning] Unknown normalization type: {type_normal}. Returning raw value.")
            return value
        
    def normalization_error_manager(self, type_normal="max_min", value=0):
        type_normal = type_normal.lower()

        error_max = self.setpoint - self.min_state  
        error_min = self.setpoint - self.max_state  

        if type_normal == "max_min":
            result = ((value - error_min) / (error_max - error_min)) * 2 - 1
            return result
        elif type_normal == "z_score":
            return self.z_score_mromalization(value=value, mean=self.ema_mean_error, deviation=self.ema_std_error)
        elif type_normal == "mix_abs":
            return self.mix_abs_normalization(value=value, max=abs(error_max))
        elif type_normal == "tanh":
            return self.tanh_normalization(value=value, mean=self.ema_mean_error, deviation=self.ema_std_error)
        else:
            print(f"[Warning] Unknown normalization type: {type_normal}. Returning raw value.")
            return value
        
    def add_data_manager(self, state_method="max_min", error_method="max_min", action_method="max_min", **kwargs):
        """
        Example:
        add_data_manager(state=1.2, setpoint=2.0, reward=0.5, action=0.3,
                        normalized=True, state_method="max_min", error_method="tanh", action_method="z_score")
        """

        supported_keys = ['state', 'setpoint', 'reward', 'action']

        for key in kwargs:
            if key not in supported_keys:
                print(f"[Warning] Unsupported key: '{key}' will be ignored.")

        # 1. Handle state
        if 'state' in kwargs and kwargs['state'] is not None:
            raw_state = kwargs['state']
            self.update_ema_stats_state(raw_state)  # Update EMA
            if self.user_normalization:
                norm_state = self.normalization_state_manager(type_normal=state_method, value=raw_state)
                self.state.append(norm_state)
            else:
                self.state.append(raw_state)

        # 2. Handle error
        if 'state' in kwargs and 'setpoint' in kwargs:
            if kwargs['state'] is not None and kwargs['setpoint'] is not None:
                raw_error = kwargs['setpoint'] - kwargs['state']
                self.update_ema_stats_error(raw_error)  # Update EMA
                if self.user_normalization:
                    norm_error = self.normalization_error_manager(type_normal=error_method, value=raw_error)
                    self.error.append(norm_error)
                else:
                    self.error.append(raw_error)
            else:
                print("[Warning] Cannot compute error: 'state' or 'setpoint' is None")

        # 3. Handle reward
        if 'reward' in kwargs and kwargs['reward'] is not None:
            self.reward.append(kwargs['reward'])

        # 4. Handle action
        if 'action' in kwargs and kwargs['action'] is not None:
            raw_action = kwargs['action']
            self.update_ema_stats_action(raw_action)  # Update EMA
            if self.user_normalization:
                norm_action = self.normalization_action_manager(type_normal=action_method, value=raw_action)
                self.action.append(norm_action)
            else:
                self.action.append(raw_action)


    def __len__(self, name: str = None):
        if name is None:
            return (len(self.state) + len(self.error) + len(self.action) + len(self.reward))

        name = name.lower()
        names = [n.strip() for n in name.split(',')]
        total = 0

        for n in names:
            if n == 'state':
                total += len(self.state)
            elif n == 'error':
                total += len(self.error)
            elif n == 'reward':
                total += len(self.reward)
            elif n == 'action':
                total += len(self.action)
            else:
                print(f"Unknown name: {n}")
                print("Please specify one or more names: 'state', 'error', 'reward', 'action'")
        return total
    
    def reward_funtion_state(self, reward_range=(0, 1.0)):
        error = self.error[-1]

        value = 1 - abs(error)
        parameter = 3.33
        raw_reward = np.exp(parameter * abs(value))

        raw_min = np.exp(0)
        raw_max = np.exp(parameter)

        min_val, max_val = reward_range
        rescaled_reward = min_val + (max_val - min_val) * ((raw_reward - raw_min) / (raw_max - raw_min))
        if error >= 0.7:
            return -10
        
        return rescaled_reward
    
    def reward_funtion_delta_error(self, reward_range=(-0.1, 1.0)):
        pass
        self.delta_penalize = np.clip(self.delta_penalize,-1,1)
        if len(self.error) < 2:
            return 0
        delta = np.sum(np.diff(np.abs(self.error)))
        self.delta_memory.append(delta)
        
        if self.delta_memory[-1] < 0:
            self.delta_penalize -= 0.1
            if self.delta_memory[-1] < self.max_delta:
                self.max_delta = self.delta_memory[-1]
            return delta / self.max_delta * (-self.delta_penalize)
        
        else:
            self.delta_penalize += 0.1
            return -1 * self.delta_penalize

    def reward_function_action(self, reward_range=(-0.3, 1.0), 
                            smooth_threshold=0.1,
                            impact_weight=0.5,
                            overreact_penalty=-0.2):
        if len(self.action) < 2 or len(self.error) < 2:
            return 0.0

        diffs = np.abs(np.diff(self.action))
        max_diff = np.max(diffs)

        # smooth reward
        if max_diff <= smooth_threshold:
            smooth_reward = reward_range[1]
        else:
            delta = np.sum(diffs)
            raw_reward = np.exp(-2.5 * delta)
            smooth_reward = reward_range[0] + (reward_range[1] - reward_range[0]) * raw_reward

        # impact reward: ถ้า action เปลี่ยน → error ควรลด
        delta_action = self.action[-1] - self.action[-2]
        delta_error = self.error[-1] - self.error[-2]

        if np.sign(delta_action) == np.sign(-delta_error):  # หาก action ทำให้ error ลด
            impact_reward = impact_weight
        else:
            impact_reward = overreact_penalty

        total_reward = smooth_reward + impact_reward
        return np.clip(total_reward, reward_range[0], reward_range[1])

    
    def return_state_manager(self,name:str=None, return_type: str = 'list'):
        name = name.lower()
        if name is None:
            print("[Warning] 'name' parameter is required. Returning empty list.")
            return [] 
        
        names = [n.strip() for n in name.split(',')]
        buffer_data = []
        for n in names:
            if n == 'state':
                buffer_data.append(list(self.state))
            elif n == 'error':
                buffer_data.append(list(self.error))
            elif n == 'reward':
                buffer_data.append(list(self.reward))
            elif n == 'action':
                buffer_data.append(list(self.action))
            else:
                print("the name does not match the information")
        
        flat_array = np.concatenate(buffer_data).astype(np.float32)

        if return_type == 'list':
            return flat_array.tolist()
        elif return_type == 'numpy':
            return flat_array
        elif return_type == 'tensor':
            return torch.from_numpy(flat_array).to(self.device)
        else:
            return flat_array
    
    def compute_all_rewards(self):
        rs = 0.7 * self.reward_funtion_state()
        re = 0.3 * self.reward_funtion_delta_error()
        ra = 0.2 * self.reward_function_action()
        return rs + re + ra, rs, re, ra
    
    def reward_manager(self, name='state'):
        if name == 'state':
            return self.reward_funtion_state()
        elif name == 'delta_error':
            return self.reward_funtion_delta_error()
        elif name == 'action':
            return self.reward_function_action()
        elif name == 'all':
            return self.compute_all_rewards()
        else:
            raise ValueError(f"Unknown reward type: {name}")
            

