from Agent import DDPGAgent
from RLC_evaronment import RC_environment
from comunucation import ModbusTCP
from Logging_andplot import Logger

import torch
import sys
from datetime import datetime, timedelta
import time
import numpy as np

Agent = DDPGAgent(state_dim=1,action_dim=1,min_action=0,max_action=10,replay_buffer='n_step')
env = RC_environment(R=2153,C=0.01,setpoint=5)
modbus = ModbusTCP(host='192.168.1.100',port=502)
logger = Logger()

def clear_lines(n=2):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    sys.stdout.flush()

Agent.load_model(r'D:\Project_end\DDPG_NEW_git\Auto_save_data_log\round_1\model\test_Agent_100.pth')
time.sleep(2)  # Wait for the Modbus connection to stabilize
step = 0
modbus.connect()
try:
    while True :
        analog_read = modbus.analog_read(1)
        state = float(analog_read) / 27647 * Agent.max_action 
        state_tensor = torch.tensor([state] ,dtype= torch.float32 ,device=Agent.device)
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