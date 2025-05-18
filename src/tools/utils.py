import os
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_global_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_waypoints(map, wp_rate):

    dir = './src/tools/maps'
    filename = f'ethz_{wp_rate}wpts_{map}.csv'

    wps = pd.read_csv(os.path.join(dir, filename))
    wps['theta'] = 0.0
    wps = torch.from_numpy(wps.to_numpy()).to('cuda')
    wps = torch.cat([wps, wps[0].unsqueeze(0)], dim=0)       # Close the trajectory
    init_state = torch.normal(0, 0.02, size=(3,)).to('cuda')
    init_state[0] += wps[0][0]
    init_state[1] += wps[0][1]

    return init_state, wps


def save_results(results, save_path):

    state_history = [
        state.detach().cpu().tolist() for state in results['state_history']
    ]
    del results['state_history']

    state_history = pd.DataFrame(state_history, columns=['x', 'y', 'theta'])
    state_history.to_csv(os.path.join(save_path, 'state_history.csv'), index=False)
    plt.scatter(state_history['x'], state_history['y'])
    plt.savefig(os.path.join(save_path, 'trajectory.pdf'))
    plt.close()

    # print(results)
    if 'loss_history_mean' in results:
        loss_history = pd.DataFrame(results['loss_history_mean'], columns=['mean_loss'])
        loss_history.to_csv(os.path.join(save_path, 'loss_history.csv'), index=False)
        plt.plot(results['loss_history_mean'], lw=4, c='g')
        plt.fill_between(
            x=list(range(len(results['loss_history_mean']))),
            y1=results['loss_history_mean'] - results['loss_history_std'],
            y2=results['loss_history_mean'] + results['loss_history_std'],
            color='g', alpha=0.3
        )
        for step in results['intervention_steps']:
            plt.axvline(x=step, label=f'Dynamics Change at t={step}', color='r')
        plt.legend()
        plt.ylabel('NLL Loss', fontsize=16)
        plt.xlabel('Timestep', fontsize=16)
        plt.grid(True, linestyle='--')
        plt.savefig(os.path.join(save_path, 'loss_history.pdf'))
        plt.tight_layout()
        plt.close()
        del results['loss_history_mean']
        del results['loss_history_std']

    json_file_path = os.path.join(save_path, "results_summary.json")
    with open(json_file_path, mode="w") as jsonfile:
        json.dump(results, jsonfile, indent=4)
