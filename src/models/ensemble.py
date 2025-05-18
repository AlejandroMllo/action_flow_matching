from typing import List
import time
import copy
import sys
import torch

from models.dynamics_model import DynamicsModel, DynamicsDataset


class Ensemble(DynamicsModel):

    def __init__(self, models: List[DynamicsModel], state_preprocessing=None, control_preprocessing=None, pred_transformation=None):
        """
        Initializes an ensemble of dynamics models.

        :param models: A list of models inheriting from DynamicsModel.
        """
        super().__init__()
        self.models = models
        self.state_preprocessing = state_preprocessing
        self.control_preprocessing = control_preprocessing
        self.pred_transformation = pred_transformation

    def predict(self, current_state, control_input):
        """
        Predicts the next state by averaging predictions from all models in the ensemble.

        :param current_state: Current state of the system.
        :param control_input: Control input applied to the system.
        :return: Averaged next state prediction.
        """

        state_copy = current_state.clone() 

        # Apply preprocessing if available
        if self.state_preprocessing:
            current_state = self.state_preprocessing(current_state)
        if self.control_preprocessing:
            control_input = self.control_preprocessing(control_input)

        # Predict
        stacked_inputs = torch.stack([current_state] * len(self.models), dim=0)  # Repeat for each model
        predictions = torch.stack([model.predict(stacked_inputs[i], control_input) for i, model in enumerate(self.models)])
        predictions = torch.mean(predictions, dim=0)
        # print('ENSEMBLE PREDICT loop predict', time.time() - s, predictions, predictions.shape)

        # Apply transformation if available
        if self.pred_transformation:
            predictions = self.pred_transformation(predictions, state_copy, control_input)

        return predictions

    def predict_slow(self, current_state, control_input):
        """
        Predicts the next state by averaging predictions from all models in the ensemble.

        :param current_state: Current state of the system.
        :param control_input: Control input applied to the system.
        :return: Averaged next state prediction.
        """
        # print('got state for ensemble', current_state.shape)
        state_copy = copy.deepcopy(current_state)
        if self.state_preprocessing is not None:
            current_state = self.state_preprocessing(current_state)
        if self.control_preprocessing is not None:
            control_input = self.control_preprocessing(control_input)

        predictions = [model.predict(current_state, control_input) for model in self.models]
        predictions = sum(predictions) / len(predictions)  # Average prediction

        if self.pred_transformation is not None:
            predictions = self.pred_transformation(predictions, state_copy, control_input)
        
        return predictions

    def learn(self, train_dataset: DynamicsDataset, val_dataset: DynamicsDataset, persistence_path: str, max_epochs = 32):
        """
        Trains each model in the ensemble using the provided datasets.

        :param train_dataset: Training dataset for the models.
        :param val_dataset: Validation dataset for the models.
        """
        for model in self.models:
            model.learn(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                persistence_path=persistence_path,
                max_epochs=max_epochs
            )
    
    def fast_learn(self, x, y):
        for model in self.models:
            model.fast_learn(x, y)

    def evaluate(self, dataset):
        """
        Evaluates the ensemble by averaging the performance metrics from each model.

        :param dataset: Dataset for evaluation.
        :return: Averaged evaluation metric.
        """
        evaluations = [model.evaluate(dataset) for model in self.models]
        return sum(evaluations) / len(evaluations)  # Average evaluation

    def simulate(self, initial_state, control_inputs):
        """
        Simulates the system by applying control inputs and averaging predictions across the ensemble.

        :param initial_state: Initial state of the system.
        :param control_inputs: A sequence of control inputs to simulate.
        :return: Averaged state history from all ensemble members.
        """
        current_state = initial_state
        state_history = [current_state]

        for control in control_inputs:
            predictions = [model.predict(current_state, control) for model in self.models]
            new_state = sum(predictions) / len(predictions)  # Average state prediction
            state_history.append(new_state)
            current_state = new_state

        return state_history

    def save(self, save_dir: str):
        """
        Saves each model in the ensemble to the specified directory.

        :param save_dir: Directory where the ensemble models will be saved.
        """
        for i, model in enumerate(self.models):
            model_save_dir = f"{save_dir}/model_{i}"
            model.save(model_save_dir)

    def load(self, persistence_dir: str):
        """
        Loads each model in the ensemble from the specified directory.

        :param persistence_dir: Directory where the ensemble models are saved.
        """
        for i, model in enumerate(self.models):
            model_load_dir = f"{persistence_dir}/model_{i}"
            model.load(model_load_dir)
