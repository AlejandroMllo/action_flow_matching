import time
import os

import numpy as np
import torch

from mbrl.util.math import Normalizer, gaussian_nll

from models.dynamics_model import DynamicsModel, DynamicsDataset


class GaussianMLPDynamics(torch.nn.Module, DynamicsModel):
    """
    Based on: https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/gaussian_mlp.py
    """

    _MODEL_FNAME = 'model.pth'
    _INPUT_NORMALIZER_FNAME = 'input_normalizer_stats.pickle'
    _OUTPUT_NORMALIZER_FNAME = 'output_normalizer_stats.pickle'

    def __init__(self, state_size, action_size, output_size, device):
        super(GaussianMLPDynamics, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.in_size = state_size + action_size
        self.out_size = output_size
        self.device = device

        self.UPDATE_NORMALIZER_STATS = True

        self._init_model()

        self._min_logvar = torch.nn.Parameter(-10 * torch.ones(1, self.out_size), requires_grad=False)
        self._max_logvar = torch.nn.Parameter(0.5 * torch.ones(1, self.out_size), requires_grad=False)

        self._input_normalizer = Normalizer(self.in_size, device)
        self._input_normalizer._STATS_FNAME = self._INPUT_NORMALIZER_FNAME
        self._output_normalizer = Normalizer(self.out_size, device)
        self._output_normalizer._STATS_FNAME = self._OUTPUT_NORMALIZER_FNAME

        self.loss_history = []

        self.to(device)

    def _init_model(self):
        self.__model = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 2 * self.out_size)
        ).to(self.device)

    def forward(self, x):

        # Pre-process input
        # print('forward raw', x, x.mean(dim=1))
        x = self._input_normalizer.normalize(x)
        # print('forward normalized', x, x.mean(dim=1))

        # Predict
        pred = self.__model(x)
        means, logvars = pred[..., : self.out_size], pred[..., self.out_size :]

        # Post-process
        logvars = self._max_logvar - torch.nn.functional.softplus(self._max_logvar - logvars)
        logvars = self._min_logvar + torch.nn.functional.softplus(logvars - self._min_logvar)

        return means, logvars

    def predict(self, current_state, control_input):

        with torch.no_grad():

            self.eval()

            x = torch.cat([current_state, control_input], dim=1).to(self.device)

            means, logvars = self.forward(x)

            stds = torch.exp(0.5 * logvars)
            eps = torch.randn_like(stds)
            preds = eps.mul_(stds).add_(means)

            preds = self._output_normalizer.denormalize(preds)

            return preds

    def learn(self, train_dataset: DynamicsDataset, val_dataset: DynamicsDataset, persistence_path: str, max_epochs = 32):

        # Update datasets
        train_dataset.reload_data()
        val_dataset.reload_data()

        # Update Normalizer stats
        if self.UPDATE_NORMALIZER_STATS:
            self._update_normalizer_stats(train_dataset)
    
        # Hyperparameters
        shuffle_data = True
        # max_epochs = 32
        batch_size = 256
        early_stopping_threshold = 1e-3

        # Load data loader
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_data
        )

        optimizer = torch.optim.Adam(self.parameters())

        prev_val_loss = float('inf')

        for epoch in range(max_epochs):
            if self.UPDATE_NORMALIZER_STATS: print('started epoch', epoch)
            self.train()

            epoch_loss = 0.0

            for i, batch in enumerate(data_loader):
                if self.UPDATE_NORMALIZER_STATS:
                    if i % 20 == 0: print(f'\t\tbatch {i} out of {len(data_loader)}')
                x, y = batch

                x, y = x.to(self.device), y.to(self.device)

                # x = self._input_normalizer.normalize(x).to(self.device) # Forward func is normalizing already, no need for this.
                y = self._output_normalizer.normalize(y).to(self.device)

                optimizer.zero_grad()

                means, logvars = self.forward(x)
                loss = gaussian_nll(
                        pred_mean=means,
                        pred_logvar=logvars,
                        target=y,  # .repeat(10, 1),  #y,
                        reduce=True,
                    )

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_dataset)
            val_loss = self.evaluate(val_dataset)
            if self.UPDATE_NORMALIZER_STATS: print('Epoch {} > Average loss (Gaussian NLL): {:.4f}. Val loss: {:.4f}'.format(epoch, avg_loss, val_loss))

            if persistence_path is not None:
                self.save(
                    save_dir=os.path.join(persistence_path, 'epoch{}_{}_valloss={:.4f}'.format(epoch, time.time_ns(), val_loss))
                )

            if abs(val_loss - prev_val_loss) < early_stopping_threshold:
                print('Early stopping due to minimal change...')
                break   # early stopping

        self.eval()

    def fast_learn(self, x, y):

        self.train()

        optimizer = torch.optim.Adam(self.parameters())
        optimizer.zero_grad()

        means, logvars = self.forward(x)
        loss = gaussian_nll(
                pred_mean=means,
                pred_logvar=logvars,
                target=self._output_normalizer.normalize(y),
                reduce=True,
            )
        loss.backward()
        optimizer.step()
        self.loss_history.append(loss.item())

        self.eval()

    def evaluate(self, dataset):

        dataset.reload_data()

        with torch.no_grad():

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=256,
                shuffle=True
            )

            running_error = 0.0
            num_batches = 0
            for x, y in data_loader:

                x, y = x.to(self.device), y.to(self.device)
                state, action = x[:, :self.state_size], x[:, self.state_size:]

                predictions = self.predict(state, action)

                error = torch.nn.functional.mse_loss(predictions, y)
                running_error += error.item()
                num_batches += 1

            return running_error / max(num_batches, 1)

    def _update_normalizer_stats(self, dataset: DynamicsDataset):

        x, y = dataset.get_data()

        self._input_normalizer.update_stats(x)
        self._output_normalizer.update_stats(y)

    def save(self, save_dir):

        os.makedirs(save_dir)

        torch.save(self.state_dict(), os.path.join(save_dir, self._MODEL_FNAME))
        self._input_normalizer.save(save_dir)
        self._output_normalizer.save(save_dir)

    def load(self, persistence_dir):

        self.load_state_dict(torch.load(os.path.join(persistence_dir, self._MODEL_FNAME)))
        self._input_normalizer.load(persistence_dir)
        self._output_normalizer.load(persistence_dir)

    def get_normalizers(self):
        return self._input_normalizer, self._output_normalizer
