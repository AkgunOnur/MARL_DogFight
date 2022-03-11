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
from curriculum import train as curriculum_train
from nocurriculum import train as nocurriculum_train



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_timesteps', default=5000, type=int, help='number of test iterations')
    # parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=1, type=int, help='number of processes to execute')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--cur_postfix', default="cur_200k", type=str, help='the output folder')
    parser.add_argument('--nocur_postfix', default="nocur_200k", type=str, help='the output folder')
    parser.add_argument('--load', default="", type=str, help='model to be loaded')
    parser.add_argument('--scenario', default="moving_target", type=str, help='the output folder')
    parser.add_argument('--visualize', default = True, action='store_true', help='to visualize')
    args = parser.parse_args()
    
    curriculum_train(train_timesteps = args.train_timesteps, n_procs=args.n_procs, folder_postfix = args.cur_postfix, scenario = args.scenario, visualize = args.visualize)
    nocurriculum_train(train_timesteps = args.train_timesteps, n_procs=args.n_procs, folder_postfix = args.nocur_postfix, scenario = args.scenario, visualize = args.visualize)

