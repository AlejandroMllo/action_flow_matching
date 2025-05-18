#!/usr/bin/python3
import torch
import numpy as np

from mbrl.planning.trajectory_opt import TrajectoryOptimizerAgent

from models.dynamics_model import DynamicsModel
from models.action_flow_matching import ActionFlowMatching


class SamplingBasedController:

    def __init__(self, 
                 dynamics: DynamicsModel,
                 controller: TrajectoryOptimizerAgent,
                 state_preprocessor,
                 goal_preprocessor,
                 action_preprocessor,
                 device='cuda'):

        # Dynamics Model
        self._dynamics = dynamics

        # Controller
        self._controller = controller
        if isinstance(controller, TrajectoryOptimizerAgent):
            self._controller.set_trajectory_eval_fn(self._cost_function) 

        # Current Goal
        self._current_goal_state = None

        # Preprocessing
        self._state_preprocessor = state_preprocessor
        self._goal_preprocessor = goal_preprocessor
        self._action_preprocessor = action_preprocessor

        self._device = device

    def set_goal(self, goal):

        goal = self._goal_preprocessor(goal).to(self._device)
        self._current_goal_state = goal

    def _reset_controller(self):

        if isinstance(self._controller, TrajectoryOptimizerAgent):
            self._controller.reset()
            self._controller.set_trajectory_eval_fn(self._cost_function)

    def update(self, state, goal):

        state = self._state_preprocessor(state).to(self._device)
        self.set_goal(goal)
        action = self._controller.act(state)

        # self._reset_controller()

        return self._action_preprocessor(action)

    def update_model(self, arg):
        
        pass
        # if isinstance(self._controller, ControlWrapperAFM):
        #     return self._controller.update_model(arg)

    def _cost_function(self, state, candidates):
        # Candidates expected shape: ``B x H x A``
        batch, horizon, _ = candidates.shape

        running_cost = torch.zeros((batch,)).to(self._device)
        if self._current_goal_state is None:
            return running_cost - 1e10
        else:
            curr_state = torch.tile(state, (batch, 1)).to(self._device)
            for h in range(horizon):

                actions = candidates[:, h, :]
                curr_state = self._dynamics.predict(curr_state, actions)
                running_cost -= torch.norm(100_000*(self._current_goal_state[:3] - curr_state[:, :3]), dim=1)

            return running_cost 
    
    def _cost_function_masked(self, state, candidates, weights=torch.tensor([10, 10, 0]), threshold=1e-5):
        # Candidates expected shape: ``B x H x A``
        batch, horizon, _ = candidates.shape
        weights = weights.to(self._device)

        running_cost = torch.zeros((batch,)).to(self._device)
        if self._current_goal_state is None:
            return running_cost - 1e10
        else:
            curr_state = torch.tile(state, (batch, 1)).to(self._device)
            update_mask = torch.ones((batch,), dtype=torch.bool).to(self._device)  # Mask to track updates

            for h in range(horizon):
                actions = candidates[:, h, :]
                curr_state = self._dynamics.predict(curr_state, actions)
                
                # Compute norm only for items where update_mask is True
                cost_update = torch.norm(100 * (self._current_goal_state - curr_state[:, :6]), dim=1)
                running_cost[update_mask] -= cost_update[update_mask]
                
                # Update the mask
                norm_diff = torch.norm(self._current_goal_state[:2] - curr_state[:, :2], dim=1)
                update_mask = update_mask & (norm_diff >= threshold)

        return running_cost
    

class ActionFlowMatchingSamplingBasedController(SamplingBasedController):

    def __init__(self, 
                 action_flow: ActionFlowMatching,
                 dynamics_misalignment_threshold: float,
                 dynamics: DynamicsModel,
                 controller: TrajectoryOptimizerAgent,
                 state_preprocessor,
                 goal_preprocessor,
                 action_preprocessor,
                 device='cuda'):
        super(ActionFlowMatchingSamplingBasedController, self).__init__(
            dynamics=dynamics,
            controller=controller,
            state_preprocessor=state_preprocessor,
            goal_preprocessor=goal_preprocessor,
            action_preprocessor=action_preprocessor,
            device=device
        )
        """
        If error bw transition and transformed action is above treshold,
        run raw action and get new dynamics regime representation.
        Otherwise, keep the current dynamics regime.
        """

        self.action_flow = action_flow
        
        self.dynamics_misalignment_threshold = dynamics_misalignment_threshold
        
        self.misaligned_dynamics = True
        self.previous_state = None
        self.planned_action = None
        self.executed_action = None
        self.planned_next_state = None
        self.realized_next_state = None
        self.dynamics_regime = None

    def update(self, state, goal):

        with torch.no_grad():

            self.previous_state = self.realized_next_state
            state = self._state_preprocessor(state).to(self._device)
            self.realized_next_state = state.clone()

            if self.realized_next_state is None or self.planned_next_state is None:
                self.misaligned_dynamics = False   # Still exploring
                state_error = None
            else:
                state_error = (self.realized_next_state - self.planned_next_state).squeeze(0)

                if torch.norm(state_error) < self.dynamics_misalignment_threshold:
                    state_error = torch.zeros_like(state_error)

            if self.misaligned_dynamics and state_error is not None:   # New dynamics regime
                self.dynamics_representation = self.register_dynamics_regime(
                    state=self.previous_state, 
                    action=self.planned_action,
                    state_error=state_error
                )

            self.set_goal(goal)
            self.planned_action = self._controller.act(state).to(self._device)
            # print('planned action', self.planned_action)

            self.planned_next_state = self._dynamics.predict(state.unsqueeze(0), self.planned_action.unsqueeze(0))

            if self.misaligned_dynamics:    # Execute raw action to identify new regime
                self.executed_action = self.planned_action.clone()
                self.misaligned_dynamics = False
            else:                           # Execute corrected/intended action
                self.executed_action = self.transform_action(self.planned_action)
                if state_error is None:
                    self.misaligned_dynamics = True
                else:
                    self.misaligned_dynamics = torch.norm(state_error) > self.dynamics_misalignment_threshold

            return self._action_preprocessor(self.executed_action)

    def transform_action(self, action):

        init_action = action.clone()
        action = action.clone()

        if self.dynamics_regime is None:
            return action

        time_steps = torch.linspace(0, 1.0, 11).to(self._device)  # From 0 to 1 (inclusive) with 0.1 step size
        for i in range(len(time_steps)-1):
            action = self.action_flow.step(
                state=self.dynamics_regime['state'],
                action=self.dynamics_regime['action'],
                state_error=self.dynamics_regime['state_error'],
                x_t=action, t_start=time_steps[i], t_end=time_steps[i + 1]
            )
        
        # print('transformed', init_action, 'into', action)

        return action

    def register_dynamics_regime(self, state, action, state_error):

        self.dynamics_regime = dict(
            state=state[-1].unsqueeze(0),
            action=action,
            state_error=state_error
        )
