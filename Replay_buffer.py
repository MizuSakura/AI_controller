import numpy as np
import random
from collections import deque, namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class Vanilla_replay_buffer():
    def __init__(self, capacity=1e6, batch_size = 256,state_dimension=1, action_dimension=1, device="cuda"):
        pass
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.capacity = int(capacity)
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.capacity)

        self.state_dim = state_dimension
        self.action_dim = action_dimension

    def Store(self, State, Action, Reward, Next_State,Done):
        self.buffer.append((State, Action, Reward, Next_State, Done))

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
    
    def Count_size(self):
        return len(self.buffer)
    
class N_step_replay_buffer():
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

        self.buffer.append(Transition(State_N_step, Action_N_Step, Reward_N_Step, Next_State_N_Step , float(bool(Done_N_Step) ) ) )
    
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
    
class Sumtree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(index, priority)
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, index, priority):
        delta = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += delta

    def get_leaf(self):
        pass
    
class PER_replay_buffer():
    pass