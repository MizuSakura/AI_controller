from PID_Evironment import PID_control_watel
from All_Agent import DDPG_Agent
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import torch

Agent = DDPG_Agent(State_dimension=9)
PID = PID_control_watel(characteristics_valve='Equal_Percentage')#'Equal_Percentage','linear','Quick_Opening'
List_key_ax = ['Start_Fall','End_Fall',
                'Start_Rise','End_Rise',
                'Where_Max_Overshoot','Where_Max_Undershoot','Settling_Time'
                ]

List_key_ah = ['Steady_State', 'Upper_Tolerance', 'Lower_Tolerance',]

color_list={'Start_Fall':"#00ff04",'End_Fall':"#ff0000",'Start_Rise':"#2fff00",'End_Rise':"#ff0000",
            'Where_Max_Overshoot':"#ff0000",'Where_Max_Undershoot':"#0084ff",'Settling_Time':"#00fbff",
            'Steady_State':"#00ffee", 'Upper_Tolerance':"#fff200", 'Lower_Tolerance':"#00ff11"
            }

Kp, Ki, Kd = 1,0.002,0
#Agent.load_model('D:/Project_end/DDPG_new/DDPG_control_PID/model3/Agent1000.pth')
Episode = 1000
reward_list = []
for j in range(Episode):
    info = {}
    water_level,reward,state =PID.reset()
    state.insert(0,water_level)
    state.insert(0,Kd)
    state.insert(0,Ki)
    state.insert(0,Kp)
    #print(state)
    State_input = torch.tensor([state],dtype=torch.float32).to(Agent.device)
    print(State_input.shape[1])
    
    Action =  Agent.Select_Action(State_input,True)
    #print(f'type : {type(Action)} dim : {Action.ndim} shape :{Action.shape}')
    Kp,Ki,Kd = Action[0],Action[1],Action[2]

    water_history,t = [],[]
    for i in range(50000):
        
        water_level,reward,state_output = PID.step(Kp,Ki,Kd)
        
        state_output.insert(0,water_level)
        state_output.insert(0,Kd)
        state_output.insert(0,Ki)
        state_output.insert(0,Kp)
        Next_State = torch.tensor([state_output],dtype=torch.float32).to(Agent.device)
        water_history.append(water_level)
        t.append(i)
        reward_list.append(reward)
    print(f'Kp: {Kp} | Ki: {Ki} | Kd: {Kd} reward: {reward} Ep {j+1}')
    Done = PID.check_done()
    Agent.Replay_memory.Store(State=State_input,Action=Action,Reward=reward,Next_State=Next_State,Done=Done)
    Agent.Optimize_model()
    Agent.save_model('D:/Project_end/DDPG_new/DDPG_control_PID/model4',f'Agent{j+1}.pth')
print(Done)


info = PID.stepinfo(water_history,PID.setpoint)


# for key, value in info.items():
#       print(f"{key.capitalize()}: {value}")
# print(f'reward: {reward}')

plt.plot(t,water_history)
plt.axhline(PID.Water_tank.setpoint,linestyle='--',color="#ff0000",label='setpoint' )

for key in List_key_ax:
    if info.get(key) is not None:
        plt.axvline(info[key], label=key,linestyle='-.',color=color_list.get(key,'-'),alpha=0.4)
for key in List_key_ah:
    plt.axhline(info[key],label=key,linestyle='-.',color=color_list.get(key,'-'),alpha=0.4)

plt.title(f'reward: {reward}')
plt.grid()
plt.legend()
plt.show()