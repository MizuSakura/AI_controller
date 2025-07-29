from Actor_and_Critic import Actor,Critic
from Replay_buffer import ReplayBufferManager
from Noise import NoiseManager

import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import os

class PID_Agent:
    def __init__(self):
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, error, Kp=1, Ki=0, Kd=0, dt=0.01):
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

        self.actor = Actor(state_dim, action_dim, max_action, hidden_size=256).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_size=256).to(self.device)

        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBufferManager(replay_buffer,batch_size =256)
        self.noise_manager = NoiseManager(Noise_type, action_dim=action_dim, actor_model=self.actor)

        self.gamma = 0.99
        self.current_path = Path(os.getcwd())

    def select_action(self, state, Add_Noise=True):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                state = state.detach().clone().to(self.device).unsqueeze(0)
            
            action = self.actor(state).cpu().data.numpy().flatten()

            if Add_Noise and self.noise_manager.noise_type == 'parameter':
                action += self.noise_manager.sample(state=state).cpu().data.numpy().flatten()
            else:
                action = self.actor(state).cpu().data.numpy().flatten()
                if Add_Noise:
                    action += self.noise_manager.sample()

            action = self.rescale_action(action, self.min_action, self.max_action)
            
            return np.clip(action, self.min_action, self.max_action).item()

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
            is_weight, indexes = None, None
        elif self.replay_buffer.buffer_type == 'per':
            state, action, reward, next_state, done, is_weight, indexes = sample

        with torch.no_grad():
            next_action = self.target_actor.act(next_state)
            target_q = self.target_critic.evaluate(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * target_q

        current_q = self.critic(state, action)
        td_error = target_q - current_q

        if is_weight is not None:
            critic_loss = (is_weight * td_error.pow(2)).mean()
        else:
            critic_loss = td_error.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_action = self.actor(state)
        actor_loss = -self.critic(state, policy_action).mean()

        if is_weight is not None:
            actor_loss = (is_weight * actor_loss).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if indexes is not None:
            self.replay_buffer.buffer.update_priorities(indexes, td_error.detach().cpu().numpy())

        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)

        return actor_loss.item(), critic_loss.item()



    def soft_update(self, net, net_target, tau=0.005):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def compute_td_error(self, state, action, reward, next_state, done):
        with torch.no_grad():
            if not isinstance(action, torch.Tensor):
                action = torch.tensor([[action]], dtype=torch.float32, device=self.device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(next_state.shape) == 1:
                next_state = next_state.unsqueeze(0)

            next_action = self.target_actor.act(next_state)
            target_q = self.target_critic.evaluate(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * target_q
            current_q = self.critic(state, action)
            td_error = (target_q - current_q).cpu().item()
            return td_error


    def save_model(self, file_name ='Agent.pth', folder_name=None,path_name = None):
        pass
        if not file_name.endswith('.pth'):
            file_name += '.pth'
        if path_name is not None:
            path_to_save = Path(path_name)
        else:
            path_to_save = self.current_path
        if folder_name is None:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            folder_to_save = path_to_save / today
        else:
            folder_to_save = path_to_save / folder_name
        folder_to_save.mkdir(parents=True, exist_ok=True)
        path_file = folder_to_save / file_name

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path_file)

    def load_model(self,path_file):
        path = Path(path_file)
        if path.suffix.lower() == ".pth":
            try:
                checkpoint = torch.load(path_file, map_location=self.device)
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                print(f"Model loaded successfully from {path_file}")
            except Exception as e:
                print(f"File not found. Please check the file path and file name. {e}")
        else:
            raise ValueError("pth file not found. Please check the file extension ")
