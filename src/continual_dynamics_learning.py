import os
import math

from planning import init_controller, ActionFlowMatchingSamplingBasedController
from models import Ensemble, GaussianMLPDynamics, ActionFlowMatching
from simulation import ugv_mission
from training import find_best_model
from tools import set_global_seed


def ugv_action_flow_matching(save_path, map, waypoints_rate, v_gain, w_gain, misalignment_threshold=1.0):

    def state_preprocessor(state):
        return state

    def goal_preprocessor(state):
        return state

    def action_preprocessor(action):
        return action

    def state_preprocessing(state):
        return state[:, -1].unsqueeze(1)
    
    def pred_transformation(preds, state, control):
        state[:, :2] += preds[:, :2]
        state[:, 2] = ((state[:, 2] + preds[:, 2] + math.pi) % (2*math.pi)) - math.pi  # wrap from -pi to pi
        return state
    
    artifact_base_path = './artifacts/ugv/dynamics/'
    artifacts = [
        find_best_model(os.path.join(
            artifact_base_path,
            f'model_{i}'
        )) for i in range(5)
    ]
    print('Artifacts', artifacts)
    models=[
            GaussianMLPDynamics(
                state_size=1, action_size=2, output_size=3, device='cuda'
            ) for _ in range(len(artifacts))
        ]
    for i in range(len(models)):
        models[i].load(persistence_dir=artifacts[i])
    dynamics = Ensemble(
        models=models,
        state_preprocessing=state_preprocessing,
        pred_transformation=pred_transformation
    )

    # Init AFM planner
    model_path = './artifacts/ugv/afm/latest_model.pth'
    action_flow = ActionFlowMatching.load_model(model_path)
    action_flow.eval()
    sbc = ActionFlowMatchingSamplingBasedController(
        action_flow=action_flow,
        dynamics_misalignment_threshold=misalignment_threshold,
        dynamics=dynamics,
        controller=init_controller(
            planning_horizon=5,
            action_lb=[-1, -math.pi/2],
            action_ub=[ 1,  math.pi/2],
            replan_freq=1
        ),
        state_preprocessor=state_preprocessor,
        action_preprocessor=action_preprocessor,
        goal_preprocessor=goal_preprocessor
    )

    # Simulate experimental setting.
    ugv_mission(
        controller=sbc, 
        map=map,
        waypoints_rate=waypoints_rate,
        v_gain=v_gain,
        w_gain=w_gain,
        save_path=save_path,
        online=True
    )


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    maps = [1, 2]
    wp_rate = 0.05
    gains = [
        (-1.0, 2.0), (-2.5, 1.0), (2.5, 0.05), (1.0, -1.0), (1.0, -0.5), (-1.0, 1.0), 
        (-0.5, 1.0), (2.0, 2.0), (-1.0, -1.0), (0.1, -1.5), (-0.5, 0.5)
    ]
    base_path = './results/ugv'

    for seed in range(5):
        for map_id in maps:
            for v_gain, w_gain in gains:

                exp_path = os.path.join(
                        base_path, 'afm', f'map{map_id}', f'wp_rate={wp_rate}',
                        f'v_gain={v_gain}*w_gain={w_gain}'
                    )
                print('Executing', exp_path, 'with seed', seed)

                save_path = os.path.join(exp_path, f'seed={seed}')

                if not os.path.exists(save_path): os.makedirs(save_path)

                set_global_seed(seed)

                ugv_action_flow_matching(
                    save_path=save_path,
                    map=map_id,
                    waypoints_rate=wp_rate,
                    v_gain=v_gain,
                    w_gain=w_gain
                )
