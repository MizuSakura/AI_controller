import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, max_action=1, hidden_size=128):
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
    
    @torch.no_grad()
    def act(self, state):
        """Use this for fast inference without gradient tracking"""
        return self.forward(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_size=128):
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
        action = action.view(-1, self.action_dim)
        x = torch.cat([state, action], dim=1)
        return self.net(x)
    
    @torch.no_grad()
    def evaluate(self, state, action):
        """Use this for fast critic evaluation without gradient tracking"""
        return self.forward(state, action)
    