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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



def train(train_timesteps: int, algorithm, level_list: list, n_procs: int, folder_postfix: str, scenario: str, load_folder: str, N_eval:int, seed:int, visualize: bool):

    model_dir = "output/saved_models"
    print ("\n" + scenario + " Level Test")

    for curriculum in level_list:
        print (f"\nLevel Test: {curriculum}")
        level = scenario + "_" + folder_postfix +  "_" + curriculum

        current_folder = "output/" + level
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
    
        n_process = n_procs                        
        train_env = make_vec_env(env_id=MovingTarget, n_envs=n_process, monitor_dir=current_folder, env_kwargs=dict(level=curriculum, visualization=visualize), vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(env_id=MovingTarget, n_envs=n_process, env_kwargs=dict(level=curriculum), vec_env_cls=SubprocVecEnv)

        train_env.reset()
        # if os.path.exists(model_dir + "/" +  load_folder):
        #     model = current_model.load(model_dir + "/" +  load_folder + "/best_model", verbose=1)
        # else:
        model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")
            

        callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval, log_path = current_folder +"_log",
                                best_model_save_path = model_dir + "/" + level, deterministic=False, verbose=1)

        start = time.time()
        model.learn(total_timesteps=train_timesteps, tb_log_name= level, callback=callback)
        # model.save(model_dir + "/last_model_" + level)
        elapsed_time = time.time() - start
        print (f"Elapsed time: {elapsed_time:.5} s.")
        
        train_env.close()
