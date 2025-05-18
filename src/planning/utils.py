#!/usr/bin/python3
import omegaconf
from mbrl.planning.trajectory_opt import TrajectoryOptimizerAgent
from typing import Sequence


def init_controller(
        planning_horizon: int,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        replan_freq=1
    ) -> TrajectoryOptimizerAgent:

    optimizer_cfg = {
            '_target_': 'mbrl.planning.MPPIOptimizer',
            'num_iterations': 1,
            'population_size': 1_000,
            'gamma': 0.9,   # reward scaling term      
            'sigma': 0.4,   # noise scaling term used in action sampling
            'beta': 0.6,    # correlation term between time steps
            'device': 'cuda'
        }

    optimizer_cfg['lower_bound'] = action_lb
    optimizer_cfg['upper_bound'] = action_ub

    optimizer_cfg = omegaconf.OmegaConf.create(optimizer_cfg)

    controller = TrajectoryOptimizerAgent(
        optimizer_cfg=optimizer_cfg,
        action_lb=action_lb,
        action_ub=action_ub,
        planning_horizon=planning_horizon,
        replan_freq=replan_freq,
        verbose=False,
        keep_last_solution=True
    )
    
    return controller
