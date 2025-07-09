from Agent import DDPGAgent
from RLC_evaronment import RC_environment
from comunucation import ModbusTCP
from Logging_andplot import Logger

import torch
import sys
from datetime import datetime, timedelta
import time
state_dim_list = ["state","reward","Done","per_error","per_action","intergal_error","deltal_volt","setpoint"]

Agent = DDPGAgent(state_dim=len(state_dim_list),action_dim=1,min_action=0,max_action=10,replay_buffer='n_step',Noise_type='ou_decay')
env = RC_environment(R=2153,C=0.01,setpoint=5)
modbus = ModbusTCP(host='192.168.1.100',port=502)
logger = Logger()
episode = 100
MAX_RUNTIME = timedelta(hours = 0, minutes = 10,seconds = 0) 
Foldor = r'D:\Project_end\DDPG_NEW_git\Auto_save_data_log/n_step_round_4'
#Agent.load_model(r'D:\Project_end\DDPG_NEW_git\Auto_save_data_log\n_step_round_1\model\test_Agent_51.pth')
def clear_lines(n=2):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    sys.stdout.flush()


try:
    for ep in range(0,1000):
        state,reward,Done,per_error,per_action,intergal,deltal_volt= env.reset()
        done = False
        step = 0
        data_log_state,data_log_action,data_log_reward,data_log_actor_loss,data_log_critc_loss = [],[],[],[],[]
        run_time = []
        Critic_loss, Actor_loss = None, None
        start_time = datetime.now()
        Agent.noise_manager.reset()
        while not done:
            data_log_state.append(state)

            state_tensor = torch.tensor([state,reward,Done,per_error,per_action,intergal,deltal_volt,env.setpoint] ,dtype= torch.float32 ,device=Agent.device)
            action = Agent.select_action(state_tensor,Add_Noise=True)
            next_state,reward,Done,per_error,per_action,intergal,deltal_volt = env.step(action)
            next_state_tensor = torch.tensor([next_state,reward,Done,per_error,per_action,intergal,deltal_volt,env.setpoint], dtype= torch.float32, device= Agent.device)
            state = next_state

            data_log_action.append(action)
            data_log_reward.append(reward)
            if Done:
                done =True
                break
            if Agent.replay_buffer.buffer_type == 'per':
                td_error = Agent.compute_td_error(state_tensor, action, reward, next_state_tensor, done)
                Agent.replay_buffer.add_transition(state_tensor,action,reward,next_state_tensor,Done,td_error=td_error)
                
            else:
                Agent.replay_buffer.add_transition( state_tensor, action, reward, next_state_tensor, Done)
            Actor_loss, Critic_loss = Agent.optimize_model()
            data_log_actor_loss.append(Actor_loss)
            data_log_critc_loss.append(Critic_loss)
            now = datetime.now()
            formatted_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            run_time.append(formatted_time)
            
        

            step +=1
            if step % 10:
                critic_str = f"{Critic_loss:.6f}" if Critic_loss is not None else "N/A"
                actor_str = f"{Actor_loss:.6f}" if Actor_loss is not None else "N/A"
                
                line1 = f'\rEpisode: {ep}| Action: {action:.3f}| Reward: {reward:.3}| Done: {Done}| state: {state:.3f}'
                line2 = f'Actor_loss: {actor_str} | Critic_loss: {critic_str}'

                clear_lines(5)
                sys.stdout.write(f'{line1}\n{line2}\n')
                sys.stdout.flush()

            
            elapsed_time = datetime.now() - start_time
            if elapsed_time >= MAX_RUNTIME:
                    clear_lines(5)
                    sys.stdout.write('--------------- Time out ----------------\n')
                    time.sleep(5)
                    break

        Agent.save_model(file_name=f'test_Agent_{ep}.pth',folder_name=Foldor+r'/model')
        logger.add_data_log(columns_name=['time','train_status','action', 'reward', 'state', 'actor_loss', 'critic_loss'],
                            data_list=[run_time,done,data_log_action, data_log_reward, data_log_state, data_log_actor_loss, data_log_critc_loss])
        logger.save_to_csv(f'data_log{ep}_.csv',folder_name=Foldor+r'/data_log')
        logger.clear_data()

except KeyboardInterrupt:
     print("keyborad interrup")

except Exception as e:
    print(e)
