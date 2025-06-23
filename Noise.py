import numpy as np
import copy
import torch

class OUNoise:
    def __init__(self, action_dim = 1, mu=0.0, theta=0.15, sigma=0.2):
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
    
class GaussianNoise:
    def __init__(self, action_dim, mean=0.0, std=0.1, min_std=0.01, decay=0.995):
        self.std = std
        self.action_dim = action_dim
        self.mean = mean
        self.min_std = min_std
        self.decay = decay

        self.init_std = std
    
    def reset(self):
        self.std = self.init_std

    def sample(self):
        noise = np.random.normal(self.mean, self.std, self.action_dim)
        self.std = max(self.min_std, self.std * self.decay)
        return noise
    
class ParameterNoise:
    def __init__(self, actor_model, std_init=0.1):
        self.actor_model = actor_model
        self.std = std_init
        self.perturbed_model = copy.deepcopy(actor_model)
        self.apply_noise()

    def apply_noise(self):
        with torch.no_grad():
            for param, param_perturbed in zip(self.actor_model.parameters(), self.perturbed_model.parameters()):
                if param.requires_grad:
                    noise = torch.normal(0, self.std, size=param.shape).to(param.device)
                    param_perturbed.data.copy_(param.data + noise)

    def get_perturbed_action(self, state):
        return self.perturbed_model(state)

    def reset(self):
        self.perturbed_model.load_state_dict(copy.deepcopy(self.actor_model.state_dict()))
        self.apply_noise()