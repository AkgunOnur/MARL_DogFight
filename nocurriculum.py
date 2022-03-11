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
from dog_fight_env import DogFight
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C, PPO, DQN, SAC, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import *


def make_env(env_id: str, level: str, rank=int, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init():
        env = None
        if env_id == "moving_target":   
            env = MovingTarget(level = level)
        elif env_id == "dog_fight":
            env = DogFight(level = level)
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init

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



def train(train_timesteps: int, n_procs: int, folder_postfix: str, scenario: str, visualize: bool):
    
    # train_env = SubprocVecEnv([make_env(easy_map) for j in range(n_procs)])
    # train_env = AgentFormation(generated_map=easy_map)
    # train_env = VecMonitor(train_env, filename = model_dir)
    N_eval = 1000

    activation_list = [nn.Tanh]
    net_list = [[64, 64]]

    model_list = [SAC]
    model_names = ["SAC"]
    curriculum_list = ["level1", "level2"]
    policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])

    model_dir = "output/saved_models"

    print ("No Curriculum Approach \n")

    for index, current_model in enumerate(model_list): 
        curriculum = curriculum_list[-1]
        print (f"Curriculum Level: {curriculum} \n\n")
        level = scenario + "_" + model_names[index] + "_" + curriculum + "_" + folder_postfix

        current_folder = "output/" + level
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)

        n_process = n_procs            

        if scenario == "moving_target":
            # train_env = make_vec_env(lambda: MovingTarget(), n_envs=n_procs, monitor_dir=current_folder, vec_env_cls=SubprocVecEnv)
            train_env = make_vec_env(env_id=MovingTarget, n_envs=n_process, monitor_dir=current_folder, env_kwargs=dict(level=curriculum, visualization=visualize), vec_env_cls=SubprocVecEnv)
            eval_env = make_vec_env(env_id=MovingTarget, n_envs=n_process, env_kwargs=dict(level=curriculum), vec_env_cls=SubprocVecEnv)
        elif scenario == "dog_fight":
            train_env = make_vec_env(lambda: DogFight(visualization=visualize, level = curriculum), n_envs=n_process, monitor_dir=current_folder, vec_env_cls=SubprocVecEnv)
            eval_env = make_vec_env(env_id=DogFight, n_envs=n_process, env_kwargs=dict(level=curriculum), vec_env_cls=SubprocVecEnv)

        # train_env = Monitor(train_env, current_folder + "/monitor.csv")
        # train_env = VecNormalize(train_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)
        train_env.reset()
        if os.path.exists(model_dir + "/" +  level + "/best_model"):
            model = current_model.load(model_dir + "/" +  level + "/best_model", verbose=1)
        else:
            model = current_model('MlpPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log="./" + current_folder + "_tensorboard/")
        
        model.set_env(train_env)
        
        callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval, log_path = current_folder +"_log",
                                best_model_save_path = model_dir + "/" + level, deterministic=False, verbose=1)

        start = time.time()
        model.learn(total_timesteps=train_timesteps, tb_log_name=model_names[index] + "_run_" + level, callback=callback)
        model.save(model_dir + "/last_model_" + level)
        elapsed_time = time.time() - start
        print (f"Elapsed time: {elapsed_time:.5}")

        train_env.close()

