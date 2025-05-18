import os

import torch
from torch import nn

from models import ActionFlowMatching, ActionFlowMatchingDataset


def train_afm_model(flow: ActionFlowMatching, num_epochs: int = 100_000):

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    data_path = './data/ugv/afm_data_dubins.pt'
    persistence_path = './artifacts/ugv/afm'

    dataset = ActionFlowMatchingDataset(
        data_path=data_path
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):

        average_loss = 0
        for batch_idx, (states, planned_actions, intended_actions, error) in enumerate(dataloader):

            # Combine inputs for flow training
            t = torch.rand((len(states), 1)).to('cuda')  # Time input
            x_t = (1 - t) * planned_actions + t * intended_actions
            dx_t = intended_actions - planned_actions  # Desired modification to actions
            actions_modified = flow(t=t, x_t=x_t, state=states[:, -1].unsqueeze(1), action=planned_actions, state_error=error)

            # Train the flow model
            optimizer.zero_grad()
            loss = loss_fn(actions_modified, dx_t)  # Loss between predicted and desired corrections
            loss.backward()
            optimizer.step()

            if batch_idx % 10_000 == 0: print('\tBatch', batch_idx)

            average_loss += loss.item()

        average_loss /= len(dataloader)
        if average_loss < best_loss:
            best_loss = average_loss
            flow.save_model(
                os.path.join(
                    persistence_path,
                    f'epoch={epoch}_loss={average_loss}.pth'
                )
            )

        flow.save_model(
            os.path.join(
                persistence_path,
                f'latest_model.pth'
            )
        )

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {average_loss}")

        checkpoint_rate = 5_000
        if epoch % checkpoint_rate == 0:
            flow.save_model(
                    os.path.join(
                        persistence_path,
                        f'epoch={epoch}_loss={average_loss}.pth'
                    )
                )

    flow.save_model(
        os.path.join(
            persistence_path,
            f'epoch={epoch}_loss={average_loss}.pth'
        )
    )

