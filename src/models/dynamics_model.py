from abc import ABC, abstractmethod

import torch
import numpy as np
import random
import pandas as pd
from collections import deque, namedtuple

class DynamicsDataset:
    pass


class ReplayBuffer(torch.utils.data.Dataset, DynamicsDataset):

    def __init__(self, capacity=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "next_state"])

    def add(self, state, action, next_state):
        e = self.experience(state, action, next_state)
        self.buffer.append(e)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):

        experience = self.buffer[index]
        state = torch.tensor(experience.state)
        action = torch.tensor(experience.action)
        next_state = torch.tensor(experience.next_state)

        return torch.cat([state, action], dim=-1), next_state

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, next_states = zip(*batch)

        states = torch.tensor(states)
        actions = torch.tensor(actions)
        next_states = torch.tensor(next_states)

        return states, actions, next_states
    
    def reload_data(self):
        pass

    def get_data(self):
        states = torch.stack([torch.cat([e.state, e.action], dim=-1) for e in self.buffer])
        next_states = torch.stack([e.next_state for e in self.buffer])

        return states, next_states
    
    def save_to_csv(self, file_path):
        data = []
        for e in self.buffer:
            row = torch.cat([e.state, e.action, e.next_state], dim=-1).cpu().detach().numpy()
            data.append(row)
        
        columns = []
        state_dim = len(self.buffer[0].state)
        action_dim = len(self.buffer[0].action)
        next_state_dim = len(self.buffer[0].next_state)

        # Column names for states
        for i in range(state_dim):
            columns.append(f"st{i}")

        # Column names for actions
        for i in range(action_dim):
            columns.append(f"ac{i}")

        # Column names for next states
        for i in range(next_state_dim):
            columns.append(f"nxst{i}")

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(file_path, index=False)

    def load_csv(self, file_path):

        data = pd.read_csv(file_path)

        num_states = len([c for c in data.columns if c.startswith('st')])
        num_actions = len([c for c in data.columns if c.startswith('ac')])
        num_next_states = len([c for c in data.columns if c.startswith('nxst')])

        data = data.dropna()
        data = data.to_numpy(dtype=np.float32)
        data = torch.from_numpy(data).to('cuda').to(torch.float32)
        for row in data:
            self.add(
                state=row[0:num_states],
                action=row[num_states:num_states+num_actions],
                next_state=row[-num_next_states:]
            )


class DynamicsModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, current_state, control_input):
        """
        This is an abstract method for calculating the next state of the system.
        
        :param current_state: Current state of the system.
        :param control_input: Control input applied to the system.
        :return: The next state of the system.
        """
        pass

    @abstractmethod
    def learn(self, train_dataset: DynamicsDataset, val_dataset: DynamicsDataset):
        """
        This is an abstract method for learning the model from a data set.

        :param data_path: Path to data.
        """
        pass

    @abstractmethod
    def evaluate(self, dataset: DynamicsDataset):
        """
        This is an abstract method to evaluate the model's performance on a given dataset.
        """
        pass

    def simulate(self, initial_state, control_inputs):

        current_state = initial_state
        state_history = [current_state]
        for control in control_inputs:

            new_state = self.predict(current_state, control)
            state_history.append(new_state)
            current_state = new_state

        return state_history

    # @abstractmethod
    # def reset(self):
    #     """
    #     This is an abstract method for resetting the dynamics model to an initial state.
    #     """
    #     pass

    @abstractmethod
    def save(self, save_dir: str):
        """
        This is an abstract method for saving the model (and any accompanying artifacts).

        :param save_dir: Path where to store the model.
        """
        pass

    @abstractmethod
    def load(self, persistence_dir: str):
        """
        This is an abstract method to initialize the model from a previously trained one.

        :param persistence_dir: Path where the saved model can be found.
        """
        pass
