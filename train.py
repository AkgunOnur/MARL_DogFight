import sys
import os
import torch
import gym
import time
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng

from moving_target_env import MovingTarget
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
# from monitor_new import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import *

        
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.size()) == 3:
            observations = torch.reshape(observations, (1, *observations.size()))
        return self.linear(self.cnn(observations))



def main(args):
    np.random.seed(args.seed)
    
    # train_env = SubprocVecEnv([make_env(easy_map) for j in range(args.n_procs)])
    # train_env = AgentFormation(generated_map=easy_map)
    # train_env = VecMonitor(train_env, filename = model_dir)
    model_name = "PPO"
    model_def = PPO
    N_eval = 1000

    activation_list = [nn.Tanh]
    gamma_list = [0.9]
    bs_list = [64]
    lr_list = [3e-4]
    net_list = [[64, 64]]
    ns_list = [2048]
    ne_list = [10]

    model_dir = args.out + "/saved_models"

    level = "moving_target"

    current_folder = args.out + "/outputs_" + level
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    train_env = make_vec_env(lambda: MovingTarget(), n_envs=args.n_procs, monitor_dir=current_folder, vec_env_cls=SubprocVecEnv)
    # train_env = MovingTarget()
    # train_env = AgentFormation(generated_map=gen_map, map_lim=args.map_lim, max_steps=250)
    # train_env = Monitor(train_env, current_folder + "/monitor.csv")
    # train_env = VecNormalize(train_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)
    train_env.reset()
    # model = model_def('MlpPolicy', train_env, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
    policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])
    model = model_def('MlpPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                    n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
    
    # policy_kwargs = dict(features_extractor_class=CustomCNN, net_arch=net_list[0], features_extractor_kwargs=dict(features_dim=net_list[0][0]), activation_fn=activation_list[0])
    # model = model_def('CnnPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
    #             n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
    
    eval_env = MovingTarget()

    callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval // args.n_procs, log_path  = args.out + "/" + model_name + "_" + level +"_log",
                            best_model_save_path = model_dir + "/best_model_" + level, deterministic=False, verbose=1)

    start = time.time()
    model.learn(total_timesteps=args.train_timesteps, tb_log_name=model_name + "_run_" + level, callback=callback)
    model.save(model_dir + "/last_model_" + level)
    elapsed_time = time.time() - start
    print (f"Elapsed time: {elapsed_time:.5}")

    
            

    train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_timesteps', default=1000, type=int, help='number of test iterations')
    # parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=8, type=int, help='number of processes to execute')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--out', default="output", type=str, help='the output folder')
    args = parser.parse_args()
    
    main(args)

