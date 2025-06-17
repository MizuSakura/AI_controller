import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from copy import deepcopy

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
    

class Actor(nn.Module):
    def __init__(self, state_dim=4, action_dim=3, max_action=1, hidden_size=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  
        )
        self.max_action = max_action
    def forward(self, state):
        return self.net(state) 
    

class Critic(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden_size=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, state, action):
        if state.dim() == 3:
            state = state.squeeze(1)
        action = action.view(-1, self.action_dim)
        x = torch.cat([state, action], dim=1)
        return self.net(x)
    
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
    
Transition_N_Step = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class Replay_Buffer_N_step:
    def __init__(self, buffer_size = 1e6, batch_size = 256,N_step = 3, gamma = 0.99,device=None):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.N_step = N_step
        self.N_step_Buffer = deque(maxlen=self.N_step)
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Store(self,State, Action, Reward, Next_State,Done):
        pass
        self.N_step_Buffer.append((State, Action, Reward, Next_State, Done))
        
        if len(self.N_step_Buffer) < self.N_step:
            return
        
        State_N_step , Action_N_Step = self.N_step_Buffer[0][0], self.N_step_Buffer[0][1]
        Reward_N_Step, Next_State_N_Step, Done_N_Step = self.get_N_Step_info()

        self.buffer.append(Transition_N_Step(State_N_step, Action_N_Step, Reward_N_Step, Next_State_N_Step , float(bool(Done_N_Step) ) ) )
    
    def get_N_Step_info(self):
            Reward , Next_State , Done = 0.0 , self.N_step_Buffer[-1][3] , self.N_step_Buffer[-1][4]
            for idx,(_, _, R, _, D) in enumerate(self.N_step_Buffer):
                Reward += (self.gamma ** idx) * R
                if D :
                    break
            return Reward , Next_State ,Done


    def Sample(self):
        if self.Size() < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        State, Action, Reward, Next_State, Done = zip(*batch)

        state = torch.stack(State).to(self.device)
        action = torch.tensor( Action, dtype= torch.float32, device= self.device).unsqueeze(1)
        reward = torch.tensor(Reward, dtype= torch.float32, device= self.device).unsqueeze(1)
        next_state = torch.stack( Next_State).to(self.device)
        done = torch.tensor(Done, dtype= torch.float32, device= self.device).unsqueeze(1)

        return state , action , reward , next_state ,done

    def Size(self):
        return len(self.buffer)
    
class DDPG_Agent:
    def __init__(self, replay_buffer_size = 1e6, batch_size = 256 ,State_dimension = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.replay_buffer_size  = int(replay_buffer_size) 

        self.state_dimension = State_dimension
        self.action_dimension = 3
        self.Max_action = 1
        self.min_action = [0,0,0]
        self.max_action = [10,1,0]

        self.Actor_network = Actor(state_dim= self.state_dimension,action_dim= self.action_dimension,max_action=self.Max_action ).to(self.device)
        self.Actor_Target = deepcopy(self.Actor_network).to(self.device)
        self.Actor_optim = optim.Adam(self.Actor_network.parameters(), lr= 1e-4)

        self.Critic_network = Critic(self.state_dimension, self.action_dimension).to(self.device)
        self.Critic_Target =  deepcopy(self.Critic_network).to(self.device)
        self.Critic_optim = optim.Adam(self.Critic_network.parameters(), lr= 1e-5)

        self.n_step = 3
        self.Gamma = 0.99
        self.Gamma_N_step = 0.5
        self.tau = 0.005
        self.Replay_memory = Replay_Buffer_N_step( buffer_size= self.replay_buffer_size, batch_size=self.batch_size, N_step=self.n_step, gamma= self.Gamma_N_step, device=self.device)

        self.Noise = OUNoise(action_dim= self.action_dimension)

    def Select_Action(self, State ,Add_Noise = False):
        if not isinstance(State, torch.Tensor):
            state = torch.tensor(State, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state = State.detach().clone().to(self.device).unsqueeze(0)
        action = self.Actor_network(state).cpu().data.numpy().flatten()
        if Add_Noise:
            action += self.Noise.sample()
        action =self.rescale_action(action,self.min_action,self.max_action)

        action = np.clip(action,self.min_action,self.max_action)

        return action
    
    def rescale_action(self,raw_action,min_action,max_action):
        raw_ar_action = np.array(raw_action)
        min_ar_action = np.array(min_action)
        max_ar_action = np.array(max_action)
        scale_Action = (raw_ar_action -  (-1) )/(1- (-1)) * (max_ar_action - min_ar_action) + min_ar_action
        return scale_Action 
    
    def Optimize_model(self):
        sample = self.Replay_memory.Sample()
        if sample is None:
            return None, None

        states, actions, rewards, next_states, dones = sample

        next_actions = self.Actor_Target(next_states)
        target_Q = self.Critic_Target(next_states, next_actions)
        Target_Q = rewards + self.Gamma ** self.n_step * (1 - dones) * target_Q

        current_Q = self.Critic_network(states, actions)

        Critic_loss = F.mse_loss(current_Q, Target_Q.detach())

        self.Critic_optim.zero_grad()
        Critic_loss.backward()
        self.Critic_optim.step()

        actor_loss = -self.Critic_network(states, self.Actor_network(states)).mean()
        self.Actor_optim.zero_grad()
        actor_loss.backward()
        self.Actor_optim.step()

        self.soft_update(self.Critic_network, self.Critic_Target)
        self.soft_update(self.Actor_network, self.Actor_Target)

        return Critic_loss.item(), actor_loss.item()

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def create_save_path(self,base_dir="Evaluation_Phase2/models", filename="Agent.pth"):
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, filename)

    def save_model(self,base_dir="Evaluation_Phase2/models",file_name = "Agent.pth"):
        save_path = self.create_save_path(base_dir = base_dir, filename = file_name)
        torch.save({
                    'Actor_network': self.Actor_network.state_dict(),
                    'Critic_network': self.Critic_network.state_dict(),
                    'Actor_optimizer': self.Actor_optim.state_dict(),
                    'Critic_optimizer': self.Critic_optim.state_dict(),
                    },save_path)
        
    def load_model(self, load_path="Evaluation_Phase2/models/Agent.pth"):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        self.Actor_network.load_state_dict(checkpoint['Actor_network'])
        self.Critic_network.load_state_dict(checkpoint['Critic_network'])
        self.Actor_optim.load_state_dict(checkpoint['Actor_optimizer'])
        self.Critic_optim.load_state_dict(checkpoint['Critic_optimizer'])

        # อัปเดต target network ให้ตรงกับ main network ทันที
        self.Actor_Target.load_state_dict(self.Actor_network.state_dict())
        self.Critic_Target.load_state_dict(self.Critic_network.state_dict())

        print(f"✅ Loaded model from: {load_path}")