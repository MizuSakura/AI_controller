import sys
from datetime import datetime, timedelta
import torch
from pathlib import Path

from Agent import DDPGAgent
from RLC_evaronment import RC_environment
from compute_state import compute_process

def clear_lines(n=2):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
    sys.stdout.flush()


class MultiAgentLogger:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.logs = {
            i: {
                'time': [],
                'reward': [],
                'action': [],
                'state': [],
                'actor_loss': [],
                'critic_loss': []
            } for i in range(num_agents)
        }

    def add(self, agent_id, time_stamp, reward, action, state, actor_loss, critic_loss):
        self.logs[agent_id]['time'].append(time_stamp)
        self.logs[agent_id]['reward'].append(reward)
        self.logs[agent_id]['action'].append(action)
        self.logs[agent_id]['state'].append(state)
        self.logs[agent_id]['actor_loss'].append(actor_loss)
        self.logs[agent_id]['critic_loss'].append(critic_loss)

    def save_to_csv(self, folder_path):
        import pandas as pd
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        for agent_id, data in self.logs.items():
            df = pd.DataFrame(data)
            file_path = folder / f"agent_{agent_id}_log.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved log for Agent {agent_id} to {file_path}")

    def clear(self):
        for agent_id in range(self.num_agents):
            for key in self.logs[agent_id]:
                self.logs[agent_id][key] = []


def main():
    num_agents = 20  # จำนวน agent ที่ต้องการเทรนพร้อมกัน
    episode_max = 500
    step_max = 10000
    MAX_RUNTIME = timedelta(minutes=5)
    Folder = Path(r'D:\Project_end\DDPG_NEW_git\Auto_save_data_log\normalized3_multiagent')

    # สร้าง agent, environment, compute_state สำหรับแต่ละ agent
    agents = []
    envs = []
    compute_states = []

    for _ in range(num_agents):
        compute_state = compute_process(state_frame=3, error_frame=3, reward_frame=5, action_frame=3,
                                        max_state=10, min_state=0, user_normalization=True)
        dim = compute_state.__len__(name="state,error,action")
        agent = DDPGAgent(state_dim=dim, action_dim=1, min_action=0, max_action=10, replay_buffer='vanilla', Noise_type='ou')
        env = RC_environment(R=2153, C=0.01, setpoint=5)

        agents.append(agent)
        envs.append(env)
        compute_states.append(compute_state)

    logger = MultiAgentLogger(num_agents=num_agents)

    try:
        for ep in range(episode_max):
            done_flags = [False] * num_agents
            steps = 0
            start_time = datetime.now()

            # Reset environment and state memory for each agent
            states = []
            for i in range(num_agents):
                agents[i].noise_manager.reset()
                state, per_action, Done = envs[i].reset()
                compute_states[i]._init_memory_state(state)
                compute_states[i]._init_memory_action(per_action)
                compute_states[i]._init_memory_error(setpoint=envs[i].setpoint, state=state)
                compute_states[i].add_data_manager(state=state, setpoint=envs[i].setpoint, reward=0, action=per_action)
                state_tensor = compute_states[i].return_state_manager(name="state,error,action", return_type='tensor')
                states.append(state_tensor)

            while not all(done_flags):
                actions = []
                for i in range(num_agents):
                    if not done_flags[i]:
                        action = agents[i].select_action(states[i], Add_Noise=True)
                    else:
                        action = 0.0  # ถ้า agent เสร็จแล้ว ให้ action เป็น 0
                    actions.append(action)

                for i in range(num_agents):
                    if not done_flags[i]:
                        state, per_action, done = envs[i].step(voltage_source=actions[i])
                        reward, rs, re, ra = compute_states[i].compute_all_rewards()

                        compute_states[i].add_data_manager(state=state, setpoint=envs[i].setpoint, reward=reward, action=per_action)
                        next_state_tensor = compute_states[i].return_state_manager(name="state,error,action", return_type='tensor')

                        if agents[i].replay_buffer.buffer_type == 'per':
                            td_error = agents[i].compute_td_error(states[i], actions[i], reward, next_state_tensor, done)
                            agents[i].replay_buffer.add_transition(states[i], actions[i], reward, next_state_tensor, done, td_error=td_error)
                        else:
                            agents[i].replay_buffer.add_transition(states[i], actions[i], reward, next_state_tensor, done)

                        Actor_loss, Critic_loss = agents[i].optimize_model()

                        # Logging
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        logger.add(
                            agent_id=i,
                            time_stamp=now_str,
                            reward=reward,
                            action=actions[i],
                            state=state,
                            actor_loss=Actor_loss,
                            critic_loss=Critic_loss
                        )

                        states[i] = next_state_tensor
                        done_flags[i] = done

                steps += 1
                if steps > step_max or (datetime.now() - start_time) >= MAX_RUNTIME:
                    print(f"Episode {ep} done or timeout.")
                    break

                if steps % 100 == 0:
                    clear_lines(15)
                    for i in range(num_agents):
                        print(f"Agent {i} | Episode {ep} | Step {steps} | Reward {logger.logs[i]['reward'][-1]:.3f} | Action {logger.logs[i]['action'][-1]:.3f} | State {logger.logs[i]['state'][-1]:.3f}")

            # Save model and log per episode for all agents
            for i in range(num_agents):
                agents[i].save_model(file_name=f'test_Agent_{i}_ep{ep}.pth', folder_name=Folder / 'model')

            logger.save_to_csv(Folder / 'data_log')
            logger.clear()

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    except Exception as e:
        print("Exception occurred:", e)


if __name__ == "__main__":
    main()
