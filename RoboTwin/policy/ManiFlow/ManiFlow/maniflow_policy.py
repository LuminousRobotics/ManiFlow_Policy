if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from maniflow.workspace.train_maniflow_robotwin2_workspace import TrainManiFlowRoboTwinWorkspace
import numpy as np

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'maniflow', 'config'))
)
def main(cfg):
    workspace = TrainManiFlowRoboTwinWorkspace(cfg)
    workspace.eval()

class ManiFlow:

    def __init__(self, cfg, usr_args, run_dir) -> None:
        self.policy, self.env_runner, self.epoch = self.get_policy_and_runner(cfg, usr_args, run_dir)
        self.run_dir = run_dir

        self.temporal_agg = False  # Set to True if you want to enable temporal aggregation
        self.num_queries = 15
        self.max_timesteps = 3000  # Large enough for deployment
        self.query_frequency = 1  # Query every step for temporal aggregation
        self.state_dim = 14 # Example state dimension, adjust as needed
        
        # Initialize with zeros matching imitate_episodes.py format
        self.all_time_actions = torch.zeros([
            self.max_timesteps,
            self.max_timesteps + self.num_queries,
            self.state_dim,
        ])
        if self.temporal_agg:
            print(f"Temporal aggregation enabled with {self.num_queries} queries")
        self.t = 0  # Current timestep
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.all_time_actions = self.all_time_actions.to(self.device)

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self, observation=None):
        # Only query the policy at specified intervals (matching ACT logic)
        if self.t % self.query_frequency == 0:
            self.all_actions = self.env_runner.get_action(self.policy, observation)
            
            # Convert numpy array to torch tensor with correct shape [action_length, action_dim]
            self.all_actions = torch.from_numpy(self.all_actions).to(self.device)
        
        if self.temporal_agg:
            # Match temporal aggregation exactly from imitate_episodes.py
            # self.all_actions has shape [action_length, action_dim]
            self.all_time_actions[self.t, self.t:self.t + self.num_queries] = self.all_actions
            actions_for_curr_step = self.all_time_actions[:, self.t]
            actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]

            # Use same weighting factor as in imitate_episodes.py
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)

            raw_action = torch.sum(actions_for_curr_step * exp_weights, dim=0, keepdim=True)
        else:
            # Direct action selection, same as imitate_episodes.py
            # raw_action = self.all_actions[self.t % self.query_frequency].unsqueeze(0)
            raw_action = self.all_actions
        
        self.t += 1  # Increment timestep

        raw_action = raw_action.detach().cpu().numpy()
        
        return raw_action

    def get_policy_and_runner(self, cfg, usr_args, run_dir):
        workspace = TrainManiFlowRoboTwinWorkspace(cfg, output_dir=run_dir)
        policy, env_runner, epoch = workspace.get_policy_and_runner(cfg, usr_args, mode='latest')
        return policy, env_runner, epoch

if __name__ == "__main__":
    main()