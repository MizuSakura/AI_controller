from Agent import DDPGAgent
from RLC_evaronment import RC_environment
from comunucation import ModbusTCP
from Logging_andplot import Logger
from compute_state import compute_process

import torch
import sys
from datetime import datetime, timedelta
import time

def clear_lines(n=2):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    sys.stdout.flush()

compute_state = compute_process()
dimansion = compute_state.__len__()

Agent = DDPGAgent(state_dim=dimansion, action_dim=1, min_action=0, max_action=10, replay_buffer='per', Noise_type='ou')
env = RC_environment(R=2153, C=0.01, setpoint=5)
modbus = ModbusTCP(host='192.168.1.100', port=502)
logger = Logger()

episode = 1000
MAX_RUNTIME = timedelta(minutes=5)
Folder = r'D:\Project_end\DDPG_NEW_git\Auto_save_data_log\normalized'

try:
    for ep in range(episode):
        state, per_action, Done = env.reset()
        compute_state._init_memory_state(state)
        compute_state._init_memory_action(per_action)
        compute_state._init_memory_error(setpoint=env.setpoint, state=state)
        compute_state.add_data_manager(state=state, setpoint=env.setpoint, reward=0, action=per_action)

        done = False
        step = 0
        Agent.noise_manager.reset()
        run_time = []
        data_log_state, data_log_action, data_log_reward, data_log_actor_loss, data_log_critc_loss = [], [], [], [], []
        start_time = datetime.now()

        while not done:
            normalized = compute_state.return_all_normalized()
            state_norm = normalized
            state_tensor = torch.tensor(state_norm, dtype=torch.float32, device=Agent.device)

            action = Agent.select_action(state_tensor,Add_Noise=True)
            next_state, per_action, Done = env.step(action)
            reward_state = compute_state.reward_state(reward_range=(0, 1.0))
            reward_error = compute_state.reward_error(reward_range=(0, 1.0),tolerance_delta=1e-10,tolerance_error=0.1)
            reward_action = compute_state.reward_action(reward_range=(0, 1.0), smooth_threshold=0.1)

            reward = reward_error + reward_action + reward_state
           

            compute_state.add_data_manager(state=next_state, setpoint=env.setpoint, reward=reward, action=action)
            next_state_norm = compute_state.return_all_normalized()

            next_state_tensor = torch.tensor(next_state_norm, dtype=torch.float32, device=Agent.device)

            if Agent.replay_buffer.buffer_type == 'per':
                td_error = Agent.compute_td_error(state_tensor, action, reward, next_state_tensor, Done)
                Agent.replay_buffer.add_transition(state_tensor, action, reward, next_state_tensor, Done, td_error=td_error)
            else:
                Agent.replay_buffer.add_transition(state_tensor, action, reward, next_state_tensor, Done)

            Actor_loss, Critic_loss = Agent.optimize_model()

            # Logging
            now = datetime.now()
            run_time.append(now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            data_log_state.append(next_state)
            data_log_action.append(action)
            data_log_reward.append(reward)
            data_log_actor_loss.append(Actor_loss)
            data_log_critc_loss.append(Critic_loss)

            if Done :
                done = True
                break

            if (datetime.now() - start_time) >= MAX_RUNTIME:
                print('Episode Done or Timeout')
                break

            step += 1
            if step % 10 == 0:
                clear_lines(5)
                line1 = (f"Episode {ep} | Step {step} | Reward {reward:.3f} | Action {action:.3f} | state: {env.voltage_capacitor:3f}")
                line2 = (f"Actor Loss: {Actor_loss} | Critic Loss: {Critic_loss}")
                line3 = (f"reward state: {reward_state} | reward error: {reward_error} | reward action: {reward_action}")
                sys.stdout.write(f'{line1}\n{line2}\n{line3}\n')

        # Save model and log
        Agent.save_model(file_name=f'test_Agent_{ep}.pth', folder_name=Folder + r'\model')
        logger.add_data_log(
            columns_name=['time', 'train_status', 'action', 'reward', 'state', 'actor_loss', 'critic_loss'],
            data_list=[run_time, done, data_log_action, data_log_reward, data_log_state, data_log_actor_loss, data_log_critc_loss]
        )
        logger.save_to_csv(f'data_log{ep}_.csv', folder_name=Folder + r'\data_log')
        logger.clear_data()

except KeyboardInterrupt:
    print("Keyboard interrupt")

except Exception as e:
    print("Exception occurred:", e)


