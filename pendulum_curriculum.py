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


def curriculum_train(train_timesteps: int, n_procs: int, folder_postfix: str, scenario: str, load_folder: str, N_eval:int, seed:int, visualize: bool):
    
    # train_env = SubprocVecEnv([make_env(easy_map) for j in range(n_procs)])
    # train_env = AgentFormation(generated_map=easy_map)
    # train_env = VecMonitor(train_env, filename = model_dir)

    activation_list = [nn.Tanh]
    net_list = [[64, 64]]

    model_def = PPO
    model_name = "PPO"
    curriculum_list = ["level1", "level2", "level3"]
    m_list = [0.7, 0.8, 1.0]
    g_list = [6., 8., 10.]
    train_time_coeff = {"level1":0.15, "level2":0.3, "level3":0.55}

    policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])

    model_dir = "output/saved_models"
    print ("\nCurriculum Approach")
    prev_curriculum = ""

    torch.manual_seed(seed)
    np.random.seed(seed)


    for index, curriculum in enumerate(curriculum_list):
        print (f"Curriculum Level: {curriculum} \n\n")
        level = scenario + "_" + model_name + "_" + folder_postfix +  "_" + curriculum

        current_folder = "output/" + level
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)

        n_process = n_procs              
        train_env = make_vec_env(env_id="Pendulum-v0", n_envs=n_process, monitor_dir=current_folder, env_kwargs=dict(m=m_list[index], g=g_list[index]), vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(env_id="Pendulum-v0", n_envs=n_process, env_kwargs=dict(m=m_list[index], g=g_list[index]), vec_env_cls=SubprocVecEnv)


        # train_env = Monitor(train_env, current_folder + "/monitor.csv")
        # train_env = VecNormalize(train_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)
        train_env.reset()
        # if os.path.exists(model_dir + "/" +  load_folder):
        #     model = current_model.load(model_dir + "/" +  load_folder + "/best_model", verbose=1)
        # else:
        model = model_def('MlpPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log="./" + current_folder + "_tensorboard/")
        current_train_timesteps = train_timesteps * train_time_coeff[curriculum]
        if curriculum != curriculum_list[0]:
            previous_level = scenario + "_" + model_name  + "_" + folder_postfix + "_" + prev_curriculum
            model = model_def.load(path=model_dir + "/" +  previous_level + "/best_model", env=train_env, verbose=0, only_weights = True) # + "/best_model
            model.tensorboard_log ="./" + current_folder + "_tensorboard/"
            model.set_env(train_env)
            

        callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval, log_path = current_folder +"_log",
                                best_model_save_path = model_dir + "/" + level, deterministic=False, verbose=1)

        start = time.time()
        model.learn(total_timesteps=current_train_timesteps, tb_log_name= level, callback=callback)
        # model.save(model_dir + "/last_model_" + level)
        elapsed_time = time.time() - start
        print (f"Elapsed time: {elapsed_time:.5} s.")
        
        prev_curriculum = curriculum
        train_env.close()


def nocurriculum_train(train_timesteps: int, n_procs: int, folder_postfix: str, scenario: str, load_folder: str, N_eval:int, seed:int, visualize: bool):
    
    # train_env = SubprocVecEnv([make_env(easy_map) for j in range(n_procs)])
    # train_env = AgentFormation(generated_map=easy_map)
    # train_env = VecMonitor(train_env, filename = model_dir)

    activation_list = [nn.Tanh]
    net_list = [[64, 64]]

    model_def = PPO
    model_name = "PPO"
    curriculum_list = ["level1", "level2", "level3"]
    m_list = [0.7, 0.8, 1.0]
    g_list = [6., 8., 10.]

    policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])

    model_dir = "output/saved_models"
    print ("\nNo-Curriculum Approach")

    torch.manual_seed(seed)
    np.random.seed(seed)

    curriculum = curriculum_list[-1]
    index = len(curriculum_list) - 1

    print (f"Level: {curriculum} \n\n")
    level = scenario + "_" + model_name + "_" + folder_postfix +  "_" + curriculum

    current_folder = "output/" + level
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    n_process = n_procs              
    train_env = make_vec_env(env_id="Pendulum-v0", n_envs=n_process, monitor_dir=current_folder, env_kwargs=dict(m=m_list[index], g=g_list[index]), vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_id="Pendulum-v0", n_envs=n_process, env_kwargs=dict(m=m_list[index], g=g_list[index]), vec_env_cls=SubprocVecEnv)

    train_env.reset()
    model = model_def('MlpPolicy', train_env, policy_kwargs=policy_kwargs, tensorboard_log="./" + current_folder + "_tensorboard/")        

    callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval, log_path = current_folder +"_log",
                            best_model_save_path = model_dir + "/" + level, deterministic=False, verbose=1)

    start = time.time()
    model.learn(total_timesteps=train_timesteps, tb_log_name= level, callback=callback)
    # model.save(model_dir + "/last_model_" + level)
    elapsed_time = time.time() - start
    print (f"Elapsed time: {elapsed_time:.5} s.")
    
    train_env.close()


