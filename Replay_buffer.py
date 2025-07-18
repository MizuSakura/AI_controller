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
        if self.__len__() < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        State, Action, Reward, Next_State, Done = zip(*batch)

        state = torch.stack(State).to(self.device)
        action = torch.tensor( Action, dtype= torch.float32, device= self.device).unsqueeze(1)
        reward = torch.tensor(Reward, dtype= torch.float32, device= self.device).unsqueeze(1)
        next_state = torch.stack( Next_State).to(self.device)
        done = torch.tensor(Done, dtype= torch.float32, device= self.device).unsqueeze(1)

        return state , action , reward , next_state ,done

    def __len__(self):
        return len(self.buffer)
    
class N_step_replay_buffer():
    def __init__(self, buffer_size = 1e6, batch_size = 256,N_step = 3, gamma = 0.99,device=None):
        self.buffer = deque(maxlen=int(buffer_size))
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
                    Done = True
                    break
            return Reward , Next_State ,Done


    def Sample(self):
        if self.__len__() < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        State, Action, Reward, Next_State, Done = zip(*batch)

        state = torch.stack(State).to(self.device)
        action = torch.tensor( Action, dtype= torch.float32, device= self.device).unsqueeze(1)
        reward = torch.tensor(Reward, dtype= torch.float32, device= self.device).unsqueeze(1)
        next_state = torch.stack( Next_State).to(self.device)
        done = torch.tensor(Done, dtype= torch.float32, device= self.device).unsqueeze(1)

        return state , action , reward , next_state ,done
    
    def __len__(self):
        return len(self.buffer)
    
class Sumtree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.current_size = 0

    def add(self, priority, data):
        index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(index, priority)
        self.data_pointer += 1
       

        if self.current_size < self.capacity:
            self.current_size += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, index, priority):
        delta = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += delta

    def get_leaf(self,value):
        index = 0
        while index < self.capacity - 1:
            left = 2 * index + 1 #make left leaves to  index one 
            right = 2 * index + 2 #make right leaves to index tw
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = right
        data_index = index - (self.capacity - 1)
        return index, self.tree[index], self.data[data_index]
    
    def total_priority(self):
        return self.tree[0] if self.capacity > 0 else 0.0
    
    def __len__(self):
        return self.current_size
    
class PER_replay_buffer():
    def __init__(self, capacity=1e6, batch_size=256,  alpha=0.6, beta=0.4, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.capacity = int(capacity)
        self.batch_size = batch_size
        self.buffer = Sumtree(self.capacity)

        #hyperparameters for PER
        self.beta = beta
        self.alpha = alpha
        self.epsilon = 1e-6
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha
    
    def Store(self, State, Action, Reward, Next_State, Done, td_error):
        transition = Transition(State, Action, Reward, Next_State, Done)
        priority = self._get_priority(td_error)
        self.buffer.add(priority, transition)

    def Sample(self):
        indexes, samples, priorities = [], [], []

        total_priority = self.buffer.total_priority()

        segment = total_priority / self.batch_size

        for i in range(self.batch_size):
            start_segment = segment * i
            end_segment = segment * (i + 1)
            sample_value = random.uniform(start_segment, end_segment)
            
            index, priority, data = self.buffer.get_leaf(sample_value)

            samples.append(data)
            indexes.append(index)
            priorities.append(priority)

        priorities_batch = np.array(priorities) / total_priority
        is_weights_sumtree = (self.buffer.__len__() * priorities_batch) ** (-self.beta)
        is_weights_sumtree /= is_weights_sumtree.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        State, Action, Reward, Next_State, Done = zip(*samples)

        state = torch.stack(State).to(self.device)
        action = torch.tensor( Action, dtype= torch.float32, device= self.device).unsqueeze(1)
        reward = torch.tensor(Reward, dtype= torch.float32, device= self.device).unsqueeze(1)
        next_state = torch.stack( Next_State).to(self.device)
        done = torch.tensor(Done, dtype= torch.float32, device= self.device).unsqueeze(1)
        is_weight = torch.tensor(is_weights_sumtree, dtype= torch.float32, device= self.device).unsqueeze(1)

        return state, action, reward, next_state, done, is_weight, indexes
    
    def update_priorities(self, index, TD_error):
        for idx, error in zip(index, TD_error):
            priority = self._get_priority(error)
            self.buffer.update(idx, priority)

    def __len__(self):
        return self.buffer.current_size

        
class ReplayBufferManager:
    def __init__(self, buffer_type: str, **kwargs):
        self.buffer_type = buffer_type.lower()
        self.buffer = self._init_replaybuffer(**kwargs)

    def _init_replaybuffer(self,**kwargs):
        if self.buffer_type == 'vanilla':
            return Vanilla_replay_buffer(**kwargs)
        elif self.buffer_type == 'n_step':
            return N_step_replay_buffer(**kwargs)
        elif self.buffer_type == 'per':
            return PER_replay_buffer(**kwargs)
        else:
            raise ValueError(f"Unknown buffer type: {self.buffer_type}")
        
    def add_transition(self, *args, **kwargs):
        if self.buffer_type == 'per':
            # args ควรจะเป็น (State, Action, Reward, Next_State, Done)
            td_error = kwargs.get('td_error')
            if td_error is None:
                raise ValueError("td_error must be provided for PER buffer")
            return self.buffer.Store(*args, td_error=td_error) # ส่ง args และ td_error
        else:
            return self.buffer.Store(*args)
            
    def sample(self):
        if len(self.buffer) < self.buffer.batch_size:
            return None
        return self.buffer.Sample()
    
    def __len__(self):
        return len(self.buffer)