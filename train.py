from random import random
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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C, PPO, DQN, SAC, DDPG, TD3
from sb3_contrib import ARS, TRPO, TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import *
from curriculum_approaches import curriculum_train, noncurriculum_train, random_train, cma_train, level_train
from level_test import train as level_test
from pendulum_curriculum import curriculum_train as pendulum_curriculum
from pendulum_curriculum import nocurriculum_train as pendulum_nocurriculum



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_timesteps', default=200000, type=int, help='number of test iterations')
    parser.add_argument('--n_eval_freq', default=500, type=int, help='evaluation interval')
    parser.add_argument('--map_lim', default=50, type=int, help='map size')
    # parser.add_argument('--grid_range', default=20, type=int, help='grid range for random trainings')
    # parser.add_argument('--n_iter', default=1000, type=int, help='iteration number of random trainings')
    parser.add_argument('--n_procs', default=4, type=int, help='number of processes to execute')
    parser.add_argument('--n_population', default=10, type=int, help='number of population in each generation for CMA')
    parser.add_argument('--seed', default=100, type=int, help='seed number for test')
    parser.add_argument('--algo', default=SAC, help='name of the algorithm')
    parser.add_argument('--cur_postfix', default="", type=str, help='the output folder')
    parser.add_argument('--nocur_postfix', default="", type=str, help='the output folder')
    parser.add_argument('--level_postfix', default="", type=str, help='the output folder')
    parser.add_argument('--load', default="", type=str, help='model to be loaded')
    parser.add_argument('--output_folder', default="output_CMA_16Mayis", type=str, help='the output folder')
    parser.add_argument('--visualize', default = False, action='store_true', help='to visualize')
    args = parser.parse_args()


    cma_train(folder_postfix = "16Mayis_SAC_CMA_200k", train_timesteps=args.train_timesteps, algorithm=args.algo, n_procs=args.n_procs, map_lim=args.map_lim, N_eval_freq=args.n_eval_freq, N_population=args.n_population, output_folder = args.output_folder, seed = args.seed, visualize = args.visualize)

    # level_train(folder_postfix = "25Nisan_SAC_test_250k", init_list = best_init_list, train_timesteps=args.train_timesteps, algorithm=args.algo, n_procs=args.n_procs,  N_eval_freq=args.n_eval_freq, scenario=args.scenario, seed=args.seed, visualize=args.visualize)    
    # pendulum_curriculum(train_timesteps = args.train_timesteps, n_procs=args.n_procs, folder_postfix = args.cur_postfix, scenario = args.scenario, load_folder=args.load, N_eval_freq=args.n_eval_freq, seed=args.seed, visualize = args.visualize)
    # pendulum_nocurriculum(train_timesteps = args.train_timesteps, n_procs=args.n_procs, folder_postfix = args.nocur_postfix, scenario = args.scenario, load_folder=args.load, N_eval_freq=args.n_eval_freq, seed=args.seed, visualize = args.visualize)
    # random_train(folder_postfix = "20Nisan_SAC_random_200k", train_timesteps=args.train_timesteps, algorithm=args.algo, n_procs=args.n_procs, map_lim=args.map_lim, grid_range = args.grid_range, N_eval_freq=args.n_eval_freq, N_iter=args.n_iter, scenario = args.scenario, seed = args.seed, visualize = args.visualize)
    # curriculum_train(train_timesteps = args.train_timesteps, algorithm=args.algo, level_list=level_list, n_procs=args.n_procs, level_time_coef_list = level_time_coef_list, folder_postfix = args.cur_postfix, scenario = args.scenario, load_folder=args.load, N_eval_freq=args.n_eval_freq, seed=args.seed, visualize = args.visualize)
    # noncurriculum_train(train_timesteps = args.train_timesteps, algorithm=args.algo, level_list=level_list, n_procs=args.n_procs, folder_postfix = args.nocur_postfix, scenario = args.scenario, load_folder=args.load, N_eval_freq=args.n_eval_freq, seed=args.seed, visualize = args.visualize)
    # level_test(train_timesteps = args.train_timesteps, algorithm=args.algo, level_list=level_list, n_procs=args.n_procs, folder_postfix = args.nocur_postfix, scenario = args.scenario, load_folder=args.load, N_eval_freq=args.n_eval_freq, seed=args.seed, visualize = args.visualize)
