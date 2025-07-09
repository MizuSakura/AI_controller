from Agent import DDPGAgent
from RLC_evaronment import RC_environment
from comunucation import ModbusTCP
from Logging_andplot import Logger

import torch
import sys
from datetime import datetime, timedelta
import time
import numpy as np
class compute_state_input:
    def __init__(self):
        pass
        self.per_action = 0
        self.per_error = 0
        self.setpoint = 0
        self.penalize_count=0
        self.intergal_error = 0


    def reward_function(self,error):

        parameter = self.setpoint / (0.3 * self.setpoint**2)
        reward = np.exp(-(parameter)*(error))
        reward_delta = self.reward_delta(error=error,per_error=self.per_error)

        return reward+(reward_delta)

    def reward_delta(self,error,per_error):
        delta_error = abs(per_error) - abs(error)
        waight = 101
        if delta_error > 0:
           
            self.penalize_count = 0
            delta_error = delta_error * waight
        else:
            self.penalize_count +=0.1
            delta_error = -abs(delta_error* waight * (1 + self.penalize_count))
        return delta_error

    def compute(self,state,action):
        error = self.setpoint - state
        reward = self.reward_function(error)
        Done =  abs(error) <= 0.1

        self.per_error = error
        self.per_action = action
        self.intergal_error += error
        deltal = abs(error - self.per_error)

        return state,reward,Done,self.per_error,self.per_action,self.intergal_error,deltal
    
state_dim_list = ["state","reward","Done","per_error","per_action","intergal_error","deltal_volt","setpoint"]
Agent = DDPGAgent(state_dim=len(state_dim_list),action_dim=1,min_action=0,max_action=10,replay_buffer='n_step')
env = RC_environment(R=2153,C=0.01,setpoint=5)
modbus = ModbusTCP(host='192.168.1.100',port=502)
logger = Logger()
change_state = compute_state_input()


def clear_lines(n=2):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    sys.stdout.flush()

Agent.load_model(r'D:\Project_end\DDPG_NEW_git\Auto_save_data_log\n_step_round_4\model\test_Agent_329.pth')
change_state.setpoint = 5
time.sleep(2)  # Wait for the Modbus connection to stabilize
step = 0
modbus.connect()
action = 0
try:
    while True :
        analog_read = modbus.analog_read(1)
        state = float(analog_read) / 27647 * Agent.max_action 
        state,reward,Done,per_error,per_action,intergla,delta = change_state.compute(state=state,action=action)
        state_tensor = torch.tensor([state,reward,Done,per_error,per_action,intergla,delta,5] ,dtype= torch.float32 ,device=Agent.device)
        action = Agent.select_action(state_tensor,Add_Noise=False)
        
        analog_value = (action/ Agent.max_action) * 27647
        modbus.write_holding_register(1025, int(analog_value))
        step +=1

        if step % 10 == 0:
            line1 = f'\r| Action: {action:.3f}| State: {state:.3f} Voltage'

            clear_lines(5)
            sys.stdout.write(f'{line1}\n')
            sys.stdout.flush()
except KeyboardInterrupt:
    modbus.write_holding_register(1025, 0)
    modbus.disconnect()
    print("\nTraining interrupted by user.")



