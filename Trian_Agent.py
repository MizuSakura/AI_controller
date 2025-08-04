from Agent import DDPGAgent
from RLC_evaronment import RC_environment
from comunucation import ModbusTCP
from Logging_andplot import Logger
from compute_state import compute_process
#from test_state_compute import compute_process
import torch
import sys
from datetime import datetime, timedelta
import time

def clear_lines(n=2):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    sys.stdout.flush()

compute_state = compute_process(state_frame=3, error_frame=3, reward_frame=5, action_frame=3,
                 max_state=10,min_state= 0,user_normalization=True)

dimansion = compute_state.__len__(name="state,error,action")
print(dimansion)

Agent = DDPGAgent(state_dim=dimansion, action_dim=1, min_action=0, max_action=10, replay_buffer='per', Noise_type='ou_decay')
env = RC_environment(R=2153, C=0.01, setpoint=5)
modbus = ModbusTCP(host='192.168.1.100', port=502)
logger = Logger()
step_max = 10000
episode = 1000
MAX_RUNTIME = timedelta(minutes=5)
Folder = r'D:\Project_end\DDPG_NEW_git\Auto_save_data_log\normalized5'

try:
    for ep in range(episode):
        Agent.noise_manager.reset()
        state, per_action, Done = env.reset()
        compute_state._init_memory_state(state)
        compute_state._init_memory_action(per_action)
        compute_state._init_memory_error(setpoint=env.setpoint, state=state)
        compute_state.add_data_manager(state=state, setpoint=env.setpoint, reward=0, action=per_action)
        state_tensor = compute_state.return_state_manager(name="state,error,action",return_type='tensor')

        done = False
        step = 0
        run_time = []
        data_log_state, data_log_action, data_log_reward, data_log_actor_loss, data_log_critc_loss = [], [], [], [], []
        reward_state,reward_error,reward_action = [],[],[]
        start_time = datetime.now()

        while not done:
            action = Agent.select_action(state=state_tensor,Add_Noise=True)
            state, per_action, Done = env.step(voltage_source= action)
            reward,rs,re,ra = compute_state.compute_all_rewards()

            compute_state.add_data_manager(state=state, setpoint=env.setpoint, reward=reward, action=per_action)
            next_state_tensor = compute_state.return_state_manager(name="state,error,action",return_type='tensor')    

            if Agent.replay_buffer.buffer_type == 'per':
                td_error = Agent.compute_td_error(state_tensor, action, reward, next_state_tensor, Done)
                Agent.replay_buffer.add_transition(state_tensor, action, reward, next_state_tensor, Done, td_error=td_error)
            else:
                Agent.replay_buffer.add_transition(state_tensor, action, reward, next_state_tensor, Done)

            Actor_loss, Critic_loss = Agent.optimize_model()
            state_tensor = next_state_tensor

            # Logging
            now = datetime.now()
            run_time.append(now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            data_log_state.append(state)
            data_log_action.append(action)
            data_log_reward.append(reward)
            data_log_actor_loss.append(Actor_loss)
            data_log_critc_loss.append(Critic_loss)
            reward_state.append(rs)
            reward_error.append(re)
            reward_action.append(ra)

            if Done :
                done = True
                break

            if (datetime.now() - start_time) >= MAX_RUNTIME or step > step_max:
                print('Episode Done or Timeout')
                break

            step += 1
            if step % 100 == 0:
                clear_lines(15)
                line1 = (f"Episode {ep} | Step {step} | Reward {reward:.3f} | Action {action:.3f} | state: {env.voltage_capacitor:3f}")
                line2 = (f"Actor Loss: {Actor_loss} | Critic Loss: {Critic_loss}")
                line3 = (f"reward state: {rs} | reward error: {re} | reward action: {ra}")
                line4 = (f'state tensor: {next_state_tensor}')
                
                sys.stdout.write(f'{line1}\n{line2}\n{line3}\n{line4}\n')

        # Save model and log
        Agent.save_model(file_name=f'test_Agent_{ep}.pth', folder_name=Folder + r'\model')
        logger.add_data_log(
            columns_name=['time', 'train_status', 'action', 'reward','reward_state','reward_error','reward_action','state', 'actor_loss', 'critic_loss'],
            data_list=[run_time, done, data_log_action, data_log_reward,reward_state,reward_error,reward_action, data_log_state, data_log_actor_loss, data_log_critc_loss]
        )
        logger.save_to_csv(f'data_log{ep}_.csv', folder_name=Folder + r'\data_log')
        logger.clear_data()

except KeyboardInterrupt:
    print("Keyboard interrupt")

except Exception as e:
    print("Exception occurred:", e)


