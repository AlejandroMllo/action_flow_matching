import os
import time

import numpy as np
import torch

from models import Ensemble, GaussianMLPDynamics
from tools import load_waypoints, save_results


class UGV(object):

    def __init__(self, dt):
        self.v_gain = 1
        self.w_gain = 1
        self.dt = dt

    def intervene(self, v_gain=1, w_gain=1):
        assert isinstance(v_gain, float) and isinstance(w_gain, float)

        self.v_gain = v_gain
        self.w_gain = w_gain

    def update(self, state, control, t_step):
        # Unpack the state and control tensors directly without repeating computations
        theta = state[2]
        v, w = control.clone()

        # Apply gains to control values directly
        v = v * self.v_gain
        w = w * self.w_gain

        # Simulate aleatoric noise with the optimized torch.normal for vectors
        noise = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1e-5), size=(3,)).to('cuda')

        new_state = state.clone() + noise
        new_state[0] += v * torch.cos(theta) * self.dt
        new_state[1] += v * torch.sin(theta) * self.dt
        new_state[2]  = ( (new_state[2] + w * self.dt + torch.pi) % (2 * torch.pi) ) - torch.pi

        return new_state


def ugv_mission(controller, map, waypoints_rate, v_gain, w_gain, save_path, online=False, persistence_path=None):

    if online and isinstance(controller._dynamics, Ensemble):
        for model in controller._dynamics.models:
            if isinstance(model, GaussianMLPDynamics):
                model.UPDATE_NORMALIZER_STATS = False
                print('Froze normalizing stats.')

    init_state, waypoints = load_waypoints(map, waypoints_rate)

    state = [init_state]
    intervention_steps = []
    waypoints = waypoints

    TOTAL_STEPS = 5_000
    dt = 1/5
    goal_idx = 0
    simulator = UGV(dt=dt)
    for step, timestamp in enumerate(np.arange(start=0, stop=TOTAL_STEPS * dt, step=dt)):
        step_start_time = time.time()

        if goal_idx > 0.8 * len(waypoints):
            if len(intervention_steps) == 1: intervention_steps.append(step)
            simulator.intervene(v_gain=1.0, w_gain=1.0)              # Reset
        elif goal_idx > 0.15 * len(waypoints):
            if len(intervention_steps) == 0: intervention_steps.append(step)
            simulator.intervene(v_gain=v_gain, w_gain=w_gain)    # Simulate new dynamics

        action = controller.update(state[-1], waypoints[goal_idx])

        state.append(
            simulator.update(
                state=state[-1],
                control=action,
                t_step=timestamp
            )
        )
        dist = torch.norm(state[-1][:2] - waypoints[goal_idx][:2])
        if dist < 0.5:
            goal_idx += 1

        if goal_idx >= len(waypoints):
            break

        if online and len(state) >= 2:
            controller._dynamics.fast_learn(
                x=torch.cat([state[-2][-1].unsqueeze(0), action], dim=0),
                y=state[-1] - state[-2]
            )

        if step % 50 == 0: 
            print('Step:', step, '| Exec time (s):', time.time() - step_start_time, '| Goal idx:', goal_idx, '| Dist:', dist.item())
            controller._reset_controller()

    results = dict(
        steps=step,
        max_steps=TOTAL_STEPS,
        dt=dt,
        success_rate=goal_idx / len(waypoints),
        reached_waypoints=goal_idx,
        max_waypoints=len(waypoints),
        state_history=state,
        map=map,
        waypoints_rate=waypoints_rate,
        intervention_steps=intervention_steps,
        interventions=[(v_gain, w_gain)]
    )
    if isinstance(controller._dynamics, Ensemble):
        history = []
        for model in controller._dynamics.models:
            if hasattr(model, 'loss_history'):
                history.append(torch.tensor(model.loss_history).unsqueeze(0))
        if len(history) > 0:
            history = torch.cat(history, dim=0)
            results['loss_history_mean'] = history.mean(dim=0)
            results['loss_history_std'] = history.std(dim=0)

    if online:
        controller._dynamics.save(save_dir=os.path.join(save_path, 'fine_tuned'))
    save_results(results, save_path)
