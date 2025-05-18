import os
import pickle

import torch
from torch import nn, Tensor


class ActionFlowMatching(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, next_state_dim: int, hidden_dim: int = 64, device: str = 'cuda'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.next_state_dim = next_state_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self._init_action_encoder()
        self._init_dynamics_encoder()
        self._init_flow_network()        

    def _init_action_encoder(self):
        self.action_encoder = nn.Sequential(
            nn.Linear(1 + self.action_dim, self.hidden_dim), nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ).to(self.device)

    def _init_dynamics_encoder(self):
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim + self.next_state_dim, self.hidden_dim), nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ).to(self.device)

    def _init_flow_network(self):
        self.flow_net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim), nn.ELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ELU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        ).to(self.device)

    def forward(self, t: Tensor, x_t: Tensor, state: Tensor, action: Tensor, state_error: Tensor) -> Tensor:
        """
        Forward pass to predict the velocity field for action modification.

        :param t: Time input (Tensor of shape [batch_size, 1]).
        :param action: Current action input (Tensor of shape [batch_size, action_dim]).
        :param state: Current state (Tensor of shape [batch_size, state_dim]).
        :param state_error: Error between predicted and observed state transitions 
                            (Tensor of shape [batch_size, state_dim]).
        :return: Modified action (Tensor of shape [batch_size, action_dim]).
        """
        # Encode action input with its encoder
        action_encoded = self.action_encoder(torch.cat((t, x_t), dim=-1))  # Shape: [batch_size, hidden_dim]

        # Encode state and state_error with the dynamics encoder
        state_transition = torch.cat((state, action, state_error), dim=-1).to(torch.float32)  # Shape: [batch_size, state_dim + action_dim + state_dim]
        dynamics_encoded = self.dynamics_encoder(state_transition)  # Shape: [batch_size, hidden_dim]

        # Concatenate both encodings
        joint_representation = torch.cat((action_encoded, dynamics_encoded), dim=-1)  # Shape: [batch_size, 2 * hidden_dim]

        # Pass through the flow network to predict the velocity field (action modification)
        velocity_field = self.flow_net(joint_representation)  # Shape: [batch_size, action_dim]
        # print('vel field', velocity_field.tolist())

        return velocity_field

    def step(self, state: Tensor, action: Tensor, state_error: Tensor, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        # t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        t_start = t_start.unsqueeze(0)
        t_end = t_end.unsqueeze(0)
        # print('stepping', t_start.shape, x_t.shape, state.shape, action.shape, state_error.shape)
        return x_t + (t_end - t_start) * self(
                t=t_start + (t_end - t_start) / 2,
                x_t= x_t + self(t=t_start, x_t=x_t, state=state, action=action, state_error=state_error) * (t_end - t_start) / 2,
                state=state,
                action=action,
                state_error=state_error
            )

    def save_model(self, save_path: str):
        """
        Save the model weights and parameters.

        :param save_path: Path to save the model file (e.g., "best_model.pth").
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'parameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'next_state_dim': self.next_state_dim,
                'hidden_dim': self.hidden_dim
            }
        }, save_path)

    @classmethod
    def load_model(cls, load_path: str):
        """
        Load a model from a file.

        :param load_path: Path to the model file (e.g., "best_model.pth").
        :return: The loaded FlowMatchingModel instance.
        """
        checkpoint = torch.load(load_path)
        params = checkpoint['parameters']
        model = cls(**params)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model




class ActionFlowMatchingDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, device='cuda'):
        if os.path.isdir(data_path):
            states = []
            planned_actions = []
            intended_actions = []
            error = []
            for file in sorted(os.listdir(data_path)):
                if file.endswith(".pt"):
                    data = torch.load(os.path.join(data_path, file))
                    
                    states.append(data['states'])
                    planned_actions.append(data['planned_actions'])
                    intended_actions.append(data['intended_actions'])
                    error.append(data['error'])
            self.states = torch.cat(states, dim=0)
            self.planned_actions = torch.cat(planned_actions, dim=0)
            self.intended_actions = torch.cat(intended_actions, dim=0)
            self.error = torch.cat(error, dim=0)
        else:
            data = torch.load(data_path)
            self.states = data['states']
            self.planned_actions = data['planned_actions']
            self.intended_actions = data['intended_actions']
            self.error = data['error']

        self.device = device
        self.states = self.states.to(self.device).to(torch.float32)
        self.planned_actions = self.planned_actions.to(self.device).to(torch.float32)
        self.intended_actions = self.intended_actions.to(self.device).to(torch.float32)
        self.error = self.error.to(self.device).to(torch.float32)

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        return self.states[idx], self.planned_actions[idx], self.intended_actions[idx], self.error[idx]
