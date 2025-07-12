import numpy as np
from collections import deque
from itertools import chain

class ComputeState:
    def __init__(self, c_retroactive_action=5, c_retroactive_error=5,
                 c_retroactive_state=5, c_retroactive_setpoint=5):
        self.action_memory = deque([0.0] * c_retroactive_action, maxlen=c_retroactive_action)
        self.error_memory = deque([0.0] * c_retroactive_error, maxlen=c_retroactive_error)
        self.state_memory = deque([0.0] * c_retroactive_state, maxlen=c_retroactive_state)
        self.setpoint_memory = deque([0.0] * c_retroactive_setpoint, maxlen=c_retroactive_setpoint)

        # scaling parameters
        self.max_action_scale = 1
        self.min_action_scale = -1

        self.max_error_scale = 1
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

        # fast flatten with chain
        return list(chain(self.action_memory, self.error_memory,
                          self.state_memory, self.setpoint_memory))

    def _init_memory(self, action, error, state, setpoint):
        """Initialize all memories with the current input values."""
        self.action_memory = deque([action] * self.action_memory.maxlen, maxlen=self.action_memory.maxlen)
        self.error_memory = deque([error] * self.error_memory.maxlen, maxlen=self.error_memory.maxlen)
        self.state_memory = deque([state] * self.state_memory.maxlen, maxlen=self.state_memory.maxlen)
        self.setpoint_memory = deque([setpoint] * self.setpoint_memory.maxlen, maxlen=self.setpoint_memory.maxlen)
        self.init_status = True

    @property
    def count_dim(self):
        return (len(self.action_memory) + len(self.error_memory) +
                len(self.state_memory) + len(self.setpoint_memory))

    def reward_function_calculate(self):
        # Convert deque to list once
        actions = list(self.action_memory)
        errors = list(self.error_memory)
        setpoints = list(self.setpoint_memory)

        # Smoothness of action
        delta = sum(abs(actions[i] - actions[i - 1]) for i in range(1, len(actions)))
        raw_reward_action = (self.max_action_scale - self.min_action_scale) * np.exp(
            -self.parameter_action * delta) + self.min_action_scale

        # Accuracy of error
        stack_reward = 0.0
        for e, sp in zip(errors, setpoints):
            parameter = self.auto_parameter(sp)
            stack_reward += (self.max_error_scale - self.min_error_scale) * np.exp(
                -parameter * abs(e)) + self.min_error_scale

        reward_action = raw_reward_action / len(actions)
        reward_error = stack_reward / len(errors)
        return reward_action + reward_error

    def auto_parameter(self, sp):
        # Avoid division by zero
        if sp == 0:
            return 1e-6
        return sp / (0.3 * sp**2)

    def reset(self):
        """Reset memory to initial zero states."""
        def reset_memory(mem):
            return deque([0.0] * len(mem), maxlen=mem.maxlen)

        self.action_memory = reset_memory(self.action_memory)
        self.error_memory = reset_memory(self.error_memory)
        self.state_memory = reset_memory(self.state_memory)
        self.setpoint_memory = reset_memory(self.setpoint_memory)
        self.init_status = False

import matplotlib.pyplot as plt
import numpy as np

# สมมุติว่าใช้ ComputeState เวอร์ชันล่าสุดที่คุณมีแล้ว
cs = ComputeState(c_retroactive_action=10, c_retroactive_error=10)

reward_list = []
setpoint = 5.0
state = 0.0

# จำลองการกระทำที่เปลี่ยนแปลงจาก smooth ไป chaotic
for t in range(50):
    # สร้าง action แบบมี noise เพิ่มขึ้นเรื่อยๆ
    action = np.sin(t * 0.2) + np.random.normal(scale=t * 0.01)
    state += action * 0.1  # สมมุติว่ามีผลกับ state
    reward_state = cs.return_state(action=action, state=state, setpoint=setpoint)
    reward = cs.reward_function_calculate()
    reward_list.append(reward)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(reward_list, label='Reward')
plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('Reward Function over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
