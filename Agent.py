from Actor_and_Critic import Actor,Critic
from Replay_buffer import ReplayBufferManager
from Noise import NoiseManager

import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
class PID_Agent:
    def __init__(self):
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, error, Kp, Ki, Kd, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error
        u = Kp * error + Ki * self.integral + Kd * derivative
        return np.clip(u,0,1)
    
    
class DDPGAgent:
    def __init__(self, state_dim=1, action_dim=1,min_action = 0,max_action=1,Noise_type = 'ou',replay_buffer='vanilla', device= None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action, hidden_size=128).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_size=128).to(self.device)

        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBufferManager(replay_buffer)
        self.noise_manager = NoiseManager(Noise_type, action_dim=action_dim, actor_model=self.actor)

        self.gamma = 0.99

    def select_action(self, state, Add_Noise=True):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state = state.detach().clone().to(self.device).unsqueeze(0)

        action = self.actor(state).cpu().data.numpy().flatten()
        if Add_Noise:
            action += self.noise_manager.sample(state=state)

        action = self.rescale_action(action, self.min_action, self.max_action)
        action = np.clip(action, self.min_action, self.max_action)
        return action

    def rescale_action(self,raw_action,min_action,max_action):
        raw_ar_action = np.array(raw_action)
        min_ar_action = np.array(min_action)
        max_ar_action = np.array(max_action)
        scale_Action = (raw_ar_action -  (-1) )/(1- (-1)) * (max_ar_action - min_ar_action) + min_ar_action
        return scale_Action 
    
    def optimize_model(self):
        sample = self.replay_buffer.sample()
        if sample is None:
            return None, None

        if self.replay_buffer.buffer_type == 'vanilla' or self.replay_buffer.buffer_type == 'n_step':
            state, action, reward, next_state, done = sample
        elif self.replay_buffer.buffer_type == 'per':
            state, action, reward, next_state, done, is_weight, indexes = sample
       
       
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target_q = reward + self.gamma * (1 - done)  * target_q
            
        current_q = self.critic(state, action)
        td_error = target_q - current_q
        critic_loss = (is_weight * td_error ** 2).mean() if self.replay_buffer.buffer_type == 'per' else td_error.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_action = self.actor(state)
        actor_loss = (-self.critic(state, policy_action)).mean()
        actor_loss = (is_weight * actor_loss).mean() if self.replay_buffer.buffer_type == 'per' else actor_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.replay_buffer.buffer_type == 'per':
            self.replay_buffer.update_priorities(indexes, td_error.abs().cpu().numpy())

        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)
        
        return actor_loss.item(), critic_loss.item()


    def soft_update(self, net, net_target, tau=0.005):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)