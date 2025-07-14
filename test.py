import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from itertools import chain


class ComputeState:
    def __init__(self, c_retroactive_action=5, c_retroactive_error=5,
                 c_retroactive_state=5, c_retroactive_setpoint=5):
        self.action_memory = deque([0.0] * c_retroactive_action, maxlen=c_retroactive_action)
        self.error_memory = deque([0.0] * c_retroactive_error, maxlen=c_retroactive_error)
        self.state_memory = deque([0.0] * c_retroactive_state, maxlen=c_retroactive_state)
        self.setpoint_memory = deque([0.0] * c_retroactive_setpoint, maxlen=c_retroactive_setpoint)

        self.max_action_scale = 1
        self.min_action_scale = -1
        self.max_error_scale = 2
        self.min_error_scale = 0

        self.parameter_action = 0.15
        self.init_status = False

    def return_state(self, action=0.0, state=0.0, setpoint=0.0):
        error = setpoint - state

        if not self.init_status:
            self._init_memory(action, error, state, setpoint)

        self.action_memory.append(action)
        self.error_memory.append(error)
        self.state_memory.append(state)
        self.setpoint_memory.append(setpoint)

        return list(chain(self.action_memory, self.error_memory,
                          self.state_memory, self.setpoint_memory))

    def _init_memory(self, action, error, state, setpoint):
        self.action_memory = deque([action] * self.action_memory.maxlen, maxlen=self.action_memory.maxlen)
        self.error_memory = deque([error] * self.error_memory.maxlen, maxlen=self.error_memory.maxlen)
        self.state_memory = deque([state] * self.state_memory.maxlen, maxlen=self.state_memory.maxlen)
        self.setpoint_memory = deque([setpoint] * self.setpoint_memory.maxlen, maxlen=self.setpoint_memory.maxlen)
        self.init_status = True

    def reset(self, action=None, state=None, setpoint=None):
        if action is not None and state is not None and setpoint is not None:
            error = setpoint - state
            self._init_memory(action, error, state, setpoint)
        else:
            def reset_mem(mem): return deque([0.0] * len(mem), maxlen=mem.maxlen)
            self.action_memory = reset_mem(self.action_memory)
            self.error_memory = reset_mem(self.error_memory)
            self.state_memory = reset_mem(self.state_memory)
            self.setpoint_memory = reset_mem(self.setpoint_memory)
            self.init_status = False

    def reward_function_calculate(self):
        actions = list(self.action_memory)
        errors = list(self.error_memory)
        setpoints = list(self.setpoint_memory)

        # Smoothness of action
        delta = sum(abs(actions[i] - actions[i - 1]) for i in range(1, len(actions)))
        reward_action = (self.max_action_scale - self.min_action_scale) * np.exp(
            -self.parameter_action * delta) + self.min_action_scale

        # Accuracy of error
        stack_reward = 0.0
        for e, sp in zip(errors, setpoints):
            parameter = self.auto_parameter(sp)
            exponent = np.clip(-parameter * abs(e), -100, 0)
            stack_reward += (self.max_error_scale - self.min_error_scale) * np.exp(
                exponent) + self.min_error_scale

        reward_action /= len(actions)
        reward_error = stack_reward / len(errors)
        reward_total = reward_action + reward_error

        return reward_action, reward_error, reward_total

    def auto_parameter(self, sp):
        sp_abs = abs(sp)
        if sp_abs == 0:
            return 1e-6
        return 1.0 / (0.3 * sp_abs)

    @property
    def count_dim(self):
        return (len(self.action_memory) + len(self.error_memory) +
                len(self.state_memory) + len(self.setpoint_memory))


if __name__ == "__main__":
    compute = ComputeState()

    setpoint = 5.0
    state = 0.0

    state_history = []
    action_history = []
    reward_action_history = []
    reward_error_history = []
    reward_total_history = []

    for t in range(50):
        error = setpoint - state
        action = 0.5 * error

        # จำลองระบบ
        state += action + np.random.normal(0, 0.05)

        compute.return_state(action=action, state=state, setpoint=setpoint)
        reward_action, reward_error, reward_total = compute.reward_function_calculate()

        # เก็บประวัติ
        state_history.append(state)
        action_history.append(action)
        reward_action_history.append(reward_action)
        reward_error_history.append(reward_error)
        reward_total_history.append(reward_total)

    # Plot ค่า state, action, rewards
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    axs[0, 0].plot(state_history, label='State')
    axs[0, 0].axhline(setpoint, color='gray', linestyle='--', label='Setpoint')
    axs[0, 0].set_title("State vs Setpoint")
    axs[0, 0].legend()

    axs[0, 1].plot(action_history, color='purple', label='Action')
    axs[0, 1].set_title("Action Over Time")
    axs[0, 1].legend()

    axs[1, 0].plot(reward_action_history, color='blue', label='Reward Action')
    axs[1, 0].plot(reward_error_history, color='red', label='Reward Error')
    axs[1, 0].set_title("Individual Rewards")
    axs[1, 0].legend()

    axs[1, 1].plot(reward_total_history, color='green', label='Total Reward')
    axs[1, 1].set_title("Total Reward Over Time")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
