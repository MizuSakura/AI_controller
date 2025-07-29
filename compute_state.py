import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class compute_process:
    def __init__(self, state_frame=3, error_frame=3, reward_frame=3, action_frame=3):
        self.state_frame = state_frame
        self.error_frame = error_frame
        self.reward_frame = reward_frame
        self.action_frame = action_frame

        self.state = deque([0.0] * self.state_frame, maxlen= self.state_frame)
        self.error = deque([0.0] * self.error_frame, maxlen= self.error_frame)
        self.reward = deque([0.0] * self.reward_frame, maxlen= self.reward_frame)
        self.action = deque([0.0] * self.action_frame, maxlen= self.action_frame)

        self.delta_error = deque(maxlen= 3)

        # สำหรับ EMA
        self.ema_alpha_state = 0.5 #--> sentivity EMA 
        self.ema_mean_state = 0.0
        self.ema_std_state = 1e-10

        self.ema_alpha_error = 0.5 #--> sentivity EMA 
        self.ema_mean_error = 0.0
        self.ema_std_error = 1e-10

        self.ema_alpha_action = 0.9 #--> sentivity EMA 
        self.ema_mean_action = 0.0
        self.ema_std_action = 1e-10

    def _init_memory_state(self,state):
        self.state = deque([state] * self.state_frame, maxlen= self.state_frame)

    def _init_memory_error(self,setpoint,state):
        error = setpoint - state
        self.error = deque([error] * self.error_frame, maxlen= self.error_frame)

    def _init_memory_reward(self,reward):
        self.reward = deque([reward] * self.reward_frame, maxlen= self.reward_frame)

    def _init_memory_action(self,action):
        self.action = deque([action] * self.action_frame, maxlen= self.action_frame)

    def add_data_manager(self, **kwargs):
  
        # example  add_data_manager(state=1.2, setpoint=2.0, reward=0.5, action=0.3)

        supported_keys = ['state', 'setpoint', 'reward', 'action']

        for key in kwargs:
            if key not in supported_keys:
                print(f"[Warning] Unsupported key: '{key}' will be ignored.")

        if 'state' in kwargs and kwargs['state'] is not None:
            self.state.append(kwargs['state'])

        if 'state' in kwargs and 'setpoint' in kwargs:
            if kwargs['state'] is not None and kwargs['setpoint'] is not None:
                error = kwargs['setpoint'] - kwargs['state']
                self.error.append(error)
            else:
                print("[Warning] Cannot compute error: 'state' or 'setpoint' is None")

        if 'reward' in kwargs and kwargs['reward'] is not None:
            self.reward.append(kwargs['reward'])

        if 'action' in kwargs and kwargs['action'] is not None:
            self.action.append(kwargs['action'])


    def return_state_manager(self,name:str=None):
        name = name.lower()
        if name == 'state':
            return list(self.state)
        elif name == 'error':
            return list(self.error)
        elif name == 'reward':
            return list(self.reward )
        elif name == 'action':
            return list(self.action)
        else:
            print("the name does not match the information")

    def return_normalized_manager(self, name: str = None):
        name = name.lower()
        eps = 1e-10

        if name == 'state':
            values = np.array(self.state)
            norm = (np.tanh((values - self.ema_mean_state) / (self.ema_std_state + eps)) + 1) / 2
            return norm.tolist()

        elif name == 'error':
            values = np.array(self.error)
            norm = (np.tanh((values - self.ema_mean_error) / (self.ema_std_error + eps)) + 1) / 2
            return norm.tolist()

        elif name == 'action':
            values = np.array(self.action)
            norm = (np.tanh((values - self.ema_mean_action) / (self.ema_std_action + eps)) + 1) / 2
            return norm.tolist()

        else:
            raise ValueError("Name must be one of ['state', 'error', 'action']")

    def return_all_normalized(self):
        return self.return_normalized_manager('state') + self.return_normalized_manager('error') + self.return_normalized_manager('action')

    def __len__(self, name: str = None):
        if name is None:
            return (len(self.state) + len(self.error) + len(self.action))

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
    
    def reward_manager(self,name):
        name_type = name
        if name_type == 'state':
            return self.reward_state()
        elif name_type == 'error':
            return self.reward_error()
        elif name_type == 'action':
            return self.reward_action()
        else:
            raise ValueError(f"Unknown reward name: {name}")

    def reward_state(self,reward_range=(0, 1.0)):
        mean_error = np.mean(self.error)
        self.update_ema_stats_state(mean_error)

        normalized = (np.tanh((mean_error - self.ema_mean_state) / (self.ema_std_state + 1e-10)) + 1) / 2

        base_reward = 1 - normalized
        parameter = 3.33
        raw_reward = np.exp(parameter * base_reward)

        raw_min = np.exp(0)
        raw_max = np.exp(parameter)

        min_val, max_val = reward_range
        rescaled_reward = min_val + (max_val - min_val) * ((raw_reward - raw_min) / (raw_max - raw_min))

        #print(f"[State] mean_error: {mean_error:.4f} | normalized: {normalized:.4f} | reward: {rescaled_reward:.4f}")
        return rescaled_reward

    def reward_error(self,reward_range=(0, 1.0),tolerance_delta:float=1e-10,tolerance_error=0.1):
       
        delta = np.sum(np.diff(np.abs(self.error)))
        self.update_ema_stats_error(delta)
        last_error = np.mean(np.abs(self.error))
        self.delta_error.append(delta)

        if last_error <= tolerance_error:
            return reward_range[1]
        
        if np.mean(self.delta_error) == 0:
            return reward_range[0]
        
        normalized = (np.tanh((delta - self.ema_mean_error) / (self.ema_std_error + 1e-10)) + 1) / 2
        base_reward = 1.0 - normalized
        parameter = 3.33
        raw_reward = np.exp(parameter * base_reward)

        raw_min = np.exp(0)
        raw_max = np.exp(parameter)

        min_val, max_val = reward_range
        rescaled_reward = min_val + (max_val - min_val) * ((raw_reward - raw_min) / (raw_max - raw_min))

        return rescaled_reward

    def reward_action(self, reward_range=(0, 1.0), smooth_threshold=0.1):
        # 1. ตรวจสอบข้อมูล action
        if len(self.action) < 2:
            return 0.0

        # 2. วัดความเปลี่ยนแปลงของ action
        diffs = np.abs(np.diff(self.action))
        max_diff = np.max(diffs)

        # 3. ถ้าเปลี่ยนแปลงน้อย → ให้ reward เต็ม
        if max_diff <= smooth_threshold:
            return reward_range[1]  # Full reward = 1.0

        # 4. ถ้าเปลี่ยนมาก → ลงโทษโดยใช้ Exponential
        delta = np.sum(np.diff(self.action))
        self.update_ema_stats_action(delta)

        normalized = (np.tanh((delta - self.ema_mean_action) / (self.ema_std_action + 1e-10)) + 1) / 2
        parameter = 3.33
        raw_penalty = -np.exp(parameter * normalized)  # ค่าติดลบเสมอ

        # 5. Rescale penalty → ให้อยู่ในช่วง [min, 0]
        raw_min = -np.exp(parameter)
        raw_max = -1.0
        min_val, max_val = reward_range
        reward = min_val + (0 - min_val) * ((raw_penalty - raw_min) / (raw_max - raw_min))

        return reward
        
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

class TanhNormalizerEMA:
    def __init__(self, alpha=0.01, eps=1e-8):
        self.alpha = alpha
        self.mean = None
        self.var = None
        self.eps = eps
        self.initialized = False

    def update(self, value):
       
        if not self.initialized:
            self.mean = value
            self.var = 0.0
            self.initialized = True
        else:
            delta = value - self.mean
            self.mean = self.alpha * value + (1 - self.alpha) * self.mean
            self.var = self.alpha * (delta ** 2) + (1 - self.alpha) * self.var

    def transform(self, value):
       
        if not self.initialized:
            raise ValueError("not update value")

        std = np.sqrt(self.var) + self.eps
        z = (value - self.mean) / std
        return 0.5 * (np.tanh(z) + 1)

    def update_and_transform(self, value):
        self.update(value)
        return self.transform(value)


# --- Define different error trends for testing ---
# data = []
# for i in range(0,100):
#     data.append(np.random.uniform(0,1))
#     print(i)

# data_plot = []
# time = []
# cp = compute_process(error_frame=10)
# for i in range(1,len(data)):
#     cp.error.append(data[i])
#     reward = cp.reward_error()
#     data_plot.append(reward)
#     time.append(i)

# plt.figure(figsize=(12, 6))
# plt.bar(time, data_plot, color='mediumseagreen')
# plt.plot(data,'o-')
# plt.title("Reward Over Time from Sliding Error Window", fontsize=16)
# plt.xlabel("Time Step", fontsize=14)
# plt.ylabel("Reward", fontsize=14)
# plt.ylim(0, 1.1)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

# # --- Simulate longer error trends (50 steps) ---
# length = 100
# error_scenarios = {
#     "Decreasing error":    np.linspace(10, 0, length),
#     "Increasing error":    np.linspace(0, 10, length),
#     "High Constant error": np.full(length, 10.0),
#     "Low Constant error":  np.full(length, 0.05),
#     "Oscillating error":   2.0 + np.sin(np.linspace(0, 10 * np.pi, length)) * 2.0  # range ~ [0,4]
# }

# # --- Evaluate reward for each scenario ---
# results = {}
# for name, error_seq in error_scenarios.items():
#     cp = compute_process(error_frame=10)  # sliding window length = 10
#     for e in error_seq:
#         cp.error.append(e)  # simulate real-time update
#     reward = cp.reward_error()
#     results[name] = reward

# # --- Plot the result ---
# plt.figure(figsize=(10, 6))
# bars = plt.bar(results.keys(), results.values(), color='salmon')
# plt.title("Reward from Long Error Sequences (Length = 50)", fontsize=16)
# plt.ylabel("Reward", fontsize=14)
# plt.ylim(0, 1.1)
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# # Add text labels
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}",
#              ha='center', va='bottom', fontsize=12)

# plt.tight_layout()
# plt.show()