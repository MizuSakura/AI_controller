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

class OU_Noise_decay:
    def __init__(self, action_dim=1, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.01, decay_rate=0.99995):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.initial_sigma = sigma
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate  # ค่าที่ใช้ลด sigma
        self.action_dim = action_dim
        self.state = np.ones(self.action_dim) * self.mu
        self.step = 0

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.step = 0
        self.sigma = self.initial_sigma

    def sample(self):
        # ลด sigma แบบ exponential decay
        self.sigma = max(self.sigma_min, self.initial_sigma * (self.decay_rate ** self.step))
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        self.step += 1
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

class NoiseManager:
    def __init__(self, noise_type='ou', action_dim=1, actor_model=None):
        self.noise_type = noise_type
        self.action_dim = action_dim
        self.actor_model = actor_model

        self.noise = self._create_noise()

    def _create_noise(self,**kwargs):
        if self.noise_type == 'ou':
            return OUNoise(action_dim=self.action_dim, **kwargs)
        elif self.noise_type == 'gaussian':
            return GaussianNoise(action_dim=self.action_dim, **kwargs)
        elif self.noise_type == 'ou_decay':
            return OU_Noise_decay(action_dim=self.action_dim,**kwargs)
        elif self.noise_type == 'parameter':
            if self.actor_model is None:
                raise ValueError("actor_model must be provided for ParameterNoise")
            return ParameterNoise(actor_model=self.actor_model, **kwargs)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def reset(self):
        self.noise.reset()

    def sample(self, state=None):
        if self.noise_type == 'parameter':
            if state is None:
                raise ValueError("state must be provided for parameter noise sampling")
            return self.noise.get_perturbed_action(state)
        else:
            return self.noise.sample()

    def change_noise_type(self, new_type):
        self.noise_type = new_type
        self.noise = self._create_noise()
            