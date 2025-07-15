import numpy as np
from collections import deque
from itertools import chain
class RC_environment:
    def __init__(self, R=1.0, C=1.0, dt=0.01,setpoint = 5):
        self.R = R  # Resistance in ohms
        self.C = C  # Capacitance in farads
        self.dt = dt  # Time step in seconds
        self.voltage_capacitor = 0.0  # Initial voltage across the capacitor
        

        self.setpoint = setpoint
        self.maximumn_volt = 10

        self.time = 0.0
        self.round_reset = 0
        self.per_error = 0
        self.per_action = 0
        self.intergal_error = 0


        self.reset()

    def reset(self):
        min_diff = 0.5 
        round = 5
        spread = (self.round_reset / 100) * self.maximumn_volt

        block = (self.round_reset // round)  

        if block % 2 == 0:
            low = max(0, self.setpoint - spread)
            high = self.setpoint - min_diff
        else:
            low = self.setpoint + min_diff
            high = min(self.maximumn_volt, self.setpoint + spread)

        self.voltage_capacitor = np.random.uniform(low=low, high=high)
        self.per_error = self.setpoint - self.voltage_capacitor
        self.round_reset += 1
        self.time = 0.0
        Done = False
        self.per_action = 0

        return self.voltage_capacitor, self.per_action, Done
    
    def step(self, voltage_source=0):
        deltal_volt = (voltage_source - self.voltage_capacitor) / (self.R * self.C)
        self.voltage_capacitor += deltal_volt * self.dt
        self.time += self.dt

        error = self.setpoint - self.voltage_capacitor

        Done = abs(error) <= 0.1

        self.per_error = error
        self.per_action = voltage_source

        return float(self.voltage_capacitor),voltage_source,Done
    
class ComputeState:
    def __init__(self, c_retroactive_action=5, c_retroactive_error=5,
                 c_retroactive_state=5, c_retroactive_setpoint=5):
        self.action_memory = deque([0.0] * c_retroactive_action, maxlen=c_retroactive_action)
        self.error_memory = deque([0.0] * c_retroactive_error, maxlen=c_retroactive_error)
        self.state_memory = deque([0.0] * c_retroactive_state, maxlen=c_retroactive_state)
        self.setpoint_memory = deque([0.0] * c_retroactive_setpoint, maxlen=c_retroactive_setpoint)

        self.span_action = 10
        self.max_action_scale = 1
        self.min_action_scale = -1
        self.max_error_scale = 2
        self.min_error_scale = 0
        self.parameter_reward_action = 0.33
        self.max_state = 10
        self.min_state = 0

        self.init_status = False

    def _init_memory(self, action, error, state, setpoint):
        self.action_memory = deque([action] * self.action_memory.maxlen, maxlen=self.action_memory.maxlen)
        self.error_memory = deque([error] * self.error_memory.maxlen, maxlen=self.error_memory.maxlen)
        self.state_memory = deque([state] * self.state_memory.maxlen, maxlen=self.state_memory.maxlen)
        self.setpoint_memory = deque([setpoint] * self.setpoint_memory.maxlen, maxlen=self.setpoint_memory.maxlen)
        self.init_status = True

    def reset(self, action=None, state=None, setpoint=None):
        if action is not None and state is not None and setpoint is not None:
            error = setpoint - state
            self._init_memory(action, error, state, setpoint)
        else:
            def reset_mem(mem): return deque([0.0] * len(mem), maxlen=mem.maxlen)
            self.action_memory = reset_mem(self.action_memory)
            self.error_memory = reset_mem(self.error_memory)
            self.state_memory = reset_mem(self.state_memory)
            self.setpoint_memory = reset_mem(self.setpoint_memory)
            self.init_status = False

    def return_state(self, action=0.0, state=0.0, setpoint=0.0):
        error = setpoint - state

        if not self.init_status:
            self._init_memory(action, error, state, setpoint)

        self.action_memory.append(action)
        self.error_memory.append(error)
        self.state_memory.append(state)
        self.setpoint_memory.append(setpoint)

        return list(chain(self.action_memory, self.error_memory,self.state_memory, self.setpoint_memory))
    

    def reward_funtion_action(self):
        actions = list(self.action_memory)
        reward = 0
        for i in range(1,len(actions)):
            delta = abs(actions[i] - actions[i -1])
            reward += (self.max_action_scale - self.min_action_scale) * np.exp(-self.parameter_reward_action * (delta)) + self.min_action_scale
        reward = reward/len(actions)
        
        return reward

        

    def reward_function_error(self):
        error = list(self.error_memory)
        sp = list(self.setpoint_memory)
        stack_reward = 0
        delta = 0
        stack = 0
        for i in range(1,len(error)):
            parameter  = self.Auto_parameter_setpoint(sp=sp[i])
            delta = abs(error[i]) - abs(error[i-1])
            if delta < 0 :
                stack_reward += (self.max_error_scale - self.min_error_scale) * np.exp(-parameter * abs(error[i])) + self.min_error_scale

            else:
                stack_reward+= -( abs((self.max_error_scale - self.min_error_scale) * np.exp(-parameter * abs(error[i])) + self.min_error_scale)) * stack
                stack +=1
        
        return stack_reward/ len(self.error_memory),stack
    
    def reward_function_error(self):
        error = list(self.error_memory)
        sp = list(self.setpoint_memory)

        stack_reward = 0
        for i in range(1,len(error)):
            parameter  = self.Auto_parameter_setpoint(sp=sp[i])
            delta = abs(error[i]) - abs(error[i - 1])
            if delta < 0:
                pass
                
            else:
                pass

    def Auto_parameter_setpoint(self,sp):
        parameter = sp / (0.3 * sp**2)
        return parameter
    
    
    
    def count_dim(self):
        return (len(self.action_memory) + len(self.error_memory) 
                +len(self.state_memory) + len(self.setpoint_memory))
    
    def return_reward(self):
       
        reward_error,stack = self.reward_function_error() 
        reward_action = self.reward_funtion_action()
        if stack > 1:
            reward = -abs(reward_action + reward_error)
        else:
            reward = reward_action + reward_error

        return reward
        