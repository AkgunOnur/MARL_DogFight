import sys
import os
import torch
import gym
import time
import pickle
import argparse
import operator
import numpy as np
from cmaes import CMA
from matplotlib import pyplot as plt
from numpy.random import default_rng
from gym.utils import seeding
from moving_target_env import MovingTarget
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoRemarkableImprovement, StopTrainingOnNoModelImprovement
from stable_baselines3 import A2C, PPO, DQN, SAC, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import *


level_map_ranges = {"level1": 20.0, "level2": 30.0, "level3": 60.0, "level4": 60.0, "level5":90.0}

def create_init_pos_angles(map_lim: int, N_iter: int, seed:int, folder_postfix: str):
    np.random.seed(seed)
    agent_init_pos = np.random.uniform(low=-map_lim, high=map_lim, size=(N_iter,2))
    enemy_init_pos = np.random.uniform(low=-map_lim, high=map_lim, size=(N_iter,2))
    angle_init_list = np.random.uniform(low=-np.pi, high=np.pi, size=(N_iter,2))

    with open(folder_postfix + "_init_lists.pickle", 'wb') as handle:
        pickle.dump([agent_init_pos, enemy_init_pos, angle_init_list], handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_init_grids_angles(map_lim: int, grid_range: int, N_iter: int, seed:int, folder_postfix: str):
    map_grid_list = []
    N_grids = 2*map_lim // grid_range
    for row in range(N_grids):
        for col in range(N_grids):
            map_grid_list.append([-map_lim + (row+0.5)*grid_range,
                                  -map_lim + (col+0.5)*grid_range])
            # map_grid_list.append([-map_lim + row*grid_range, -map_lim + (row+1)*grid_range,
            #                       -map_lim + col*grid_range, -map_lim + (col+1)*grid_range])

    rng = default_rng(seed)
    pos_init_list = []
    for i in range(N_iter):
        pos_init_list.append(list(rng.choice(N_grids**2, size=(2,), replace=False)))
    angle_init_list = np.random.uniform(low=-np.pi, high=np.pi, size=(N_iter,2))

    with open(folder_postfix + "_init_lists.pickle", 'wb') as handle:
        pickle.dump([map_grid_list, pos_init_list, angle_init_list], handle, protocol=pickle.HIGHEST_PROTOCOL)


def level_train(train_timesteps:int, algorithm, n_procs:int, init_list: list, N_eval_freq:int, folder_postfix: str, output_folder: str, seed:int, visualize: bool):
    deterministic = True
    N_eval_episodes = 10
    results_list = []
    agent_target = [-45, -45]
    enemy_target = [45, 45]
    angle_target = [-np.pi/2, np.pi/2]
    N_eval_freq = N_eval_freq // n_procs

    model_dir = output_folder + "/saved_models"
    
    cnt = 0
    for init in init_list:
        agent_init = init[0:2]
        enemy_init = init[2:4]
        angle_init = init[4:]
        print ("Agent init location: ", agent_init)
        print ("Enemy init location: ", enemy_init)
        print ("Angle init: ", angle_init)

        current_folder = output_folder + "/" +  folder_postfix + "_index_" + str(cnt)
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)


        train_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, monitor_dir=current_folder, env_kwargs=dict(agent_init=agent_init, opponent_init=enemy_init, angle_init=angle_init ,seed=seed, visualization=visualize), vec_env_cls=SubprocVecEnv)
        # eval_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, env_kwargs=dict(agent_init=agent_init, opponent_init=enemy_init, angle_init=angle_init ,seed=seed), vec_env_cls=SubprocVecEnv)
        target_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, env_kwargs=dict(agent_init=agent_target, opponent_init=enemy_target, angle_init=angle_target ,seed=seed), vec_env_cls=SubprocVecEnv)

        train_env.reset()
        # eval_env.reset()
        model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")

        if cnt > 0:
            previous_level = folder_postfix + "_index_" + str(cnt - 1)
            model = algorithm.load(path=model_dir + "/" +  previous_level + "/best_model", env=train_env, verbose=0, only_weights = False) # + "/best_model
            model.tensorboard_log ="./" + current_folder + "_tensorboard/"
            model.set_env(train_env)

        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals = 150, verbose = 1)

        callback = EvalCallback(eval_env=target_env, callback_after_eval=stop_callback, n_eval_episodes = N_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                                best_model_save_path = model_dir + "/" + folder_postfix + "_index_" + str(cnt), deterministic=deterministic, verbose=1)

        

        start = time.time()
        model.learn(total_timesteps=train_timesteps, tb_log_name = folder_postfix + "_index_" + str(cnt), callback=callback)
        # model.save(model_dir + "/last_model_" + level)
        elapsed_time = time.time() - start
        print (f"Elapsed time: {elapsed_time:.5} s.")        
        train_env.close()


        # Load the model
        # model = model.load(path=model_dir + "/" +  folder_postfix + "_index_" + str(cnt) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
        # eval_env.reset()
        # episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=N_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
        # mean_reward_current = np.mean(episode_rewards)
        # std_reward_current = np.std(episode_rewards)
        # print (f"Mean reward in current env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
        
        model = model.load(path=model_dir + "/" +  folder_postfix + "_index_" + str(cnt) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
        target_env.reset()
        episode_rewards, episode_lengths = evaluate_policy(model, target_env, n_eval_episodes=N_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
        mean_reward_target = np.mean(episode_rewards)
        std_reward_target = np.std(episode_rewards)
        print (f"Mean reward in target env: {mean_reward_target:.4f} Std reward in target env: {std_reward_target:.4f} \n")

        reward_list = [0, mean_reward_target]
        results_list.append(reward_list)
        cnt += 1
        # eval_env.close()
        target_env.close()

        with open(folder_postfix + "_results.pickle", 'wb') as handle:
            pickle.dump(results_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cma_train(train_timesteps:int, algorithm, n_procs:int, map_lim: int, N_eval_freq:int, N_population: int, folder_postfix: str, output_folder: str, seed:int, visualize: bool):
    deterministic = True
    N_eval_episodes = 10
    N_iteration = 20
    N_generation = 20
    agent_target = [-45, -45]
    enemy_target = [45, 45]
    angle_target = [-np.pi/2, np.pi/2]
    N_eval_freq = N_eval_freq // n_procs

    # enemy_init = [10, 10]
    # angle_init = [np.pi/2, np.pi/2]

    for iteration in range(N_iteration): # How many times this algorithm will work
        start = time.time()
        bounds = np.array([[-map_lim, map_lim], [-map_lim, map_lim], [-map_lim, map_lim], [-map_lim, map_lim], [-np.pi, np.pi], [-np.pi, np.pi]])
        # bounds = np.array([[-map_lim, map_lim], [-map_lim, map_lim]])
        lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

        mean = lower_bounds + (np.random.rand(6) * (upper_bounds - lower_bounds))
        sigma = 1.0 #upper_bounds / 4.0 
        optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, population_size=N_population, seed=seed)

        print (f"Iteration: {iteration} - CMA Train \n")
        best_reward_list = []
        best_init_list = []
        best_pop_index_list = []

        for generation in range(N_generation):
            solutions = []
            reward_dict = dict()
            init_dict = dict()

            current_folder = output_folder + "/" +  folder_postfix + "_index_" + str(generation)
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)

            print ("Generation: ", generation)

            for pop_ind in range(optimizer.population_size):
                start = time.time()
                x = optimizer.ask()
                agent_init = [x[0], x[1]]
                enemy_init = [x[2], x[3]]
                angle_init = [x[4], x[5]]
                print ("Agent init location: ", agent_init)
                print ("Enemy init location: ", enemy_init)
                print ("Angle init location: ", angle_init)
                train_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, monitor_dir=current_folder, env_kwargs=dict(agent_init=agent_init, opponent_init=enemy_init, angle_init=angle_init, seed=seed, visualization=visualize), vec_env_cls=SubprocVecEnv)
                eval_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, env_kwargs=dict(agent_init=agent_init, opponent_init=enemy_init, angle_init=angle_init, seed=seed), vec_env_cls=SubprocVecEnv)
                target_env = make_vec_env(env_id=MovingTarget, n_envs=2, env_kwargs=dict(agent_init=agent_target, opponent_init=enemy_target, angle_init=angle_target), vec_env_cls=SubprocVecEnv)

                train_env.reset()
                eval_env.reset()

                if generation > 0:
                    model = algorithm.load(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation - 1) + "_best" + "/best_model", verbose=0, only_weights = False) # + "/best_model"
                    model.tensorboard_log ="./" + current_folder + "_tensorboard/"
                    model.set_env(train_env)
                    print ("Best model in gen #", (generation - 1), " is uploaded!")
                else:
                    model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")
                
                stop_callback = StopTrainingOnNoRemarkableImprovement(max_no_improvement_evals = 100, check_percentage=0.9, verbose = 1)

                callback = EvalCallback(eval_env=eval_env, callback_after_eval=stop_callback, n_eval_episodes = N_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                                        best_model_save_path = output_folder + "/saved_models/" + folder_postfix + "_index_" + str(generation) + "_pop_" + str(pop_ind), deterministic=deterministic, verbose=1)
                
                model.learn(total_timesteps=train_timesteps, tb_log_name = folder_postfix + "_index_" + str(generation) + "_pop_" + str(pop_ind), callback=callback)
                elapsed_time = time.time() - start
                print (f"Elapsed time: {elapsed_time:.5} s.")        
                train_env.close()


                # Load the model
                model = model.load(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation) + "_pop_" + str(pop_ind) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
                eval_env.reset()
                episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=N_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
                mean_reward_current = np.mean(episode_rewards)
                std_reward_current = np.std(episode_rewards)
                print (f"Mean reward in current env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
                
                target_env.reset()
                episode_rewards, episode_lengths = evaluate_policy(model, target_env, n_eval_episodes=100, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
                mean_reward_target = np.mean(episode_rewards)
                std_reward_target = np.std(episode_rewards)
                print (f"Mean reward in target env: {mean_reward_target:.4f} Std reward in target env: {std_reward_target:.4f} ")

                init_list = [agent_init[0], agent_init[1], enemy_init[0], enemy_init[1], angle_init[0], angle_init[1]]
                eval_env.close()
                target_env.close()

                reward_dict[pop_ind] = mean_reward_target
                init_dict[pop_ind] = init_list

                current_cost = (mean_reward_target - 100) ** 2
                solutions.append((x, current_cost))
                print(f"#{generation} Reward: {mean_reward_target} (x1={x[0]}, x2 = {x[1]}, x3={x[2]}, x4 = {x[3]}, x5={x[4]}, x6 = {x[5]}) \n")
                # print(f"#{generation} Reward: {mean_reward_target} (x1={x[0]}, x2 = {x[1]}) ")

            optimizer.tell(solutions)

            reward_sorted = sorted(reward_dict.items(), key=operator.itemgetter(1), reverse=True)
            best_index = reward_sorted[0][0]
            best_reward_list.append(reward_sorted[0][1])
            best_init_list.append(init_dict[best_index])
            best_pop_index_list.append(best_index)

            model = model.load(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation) + "_pop_" + str(best_index) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
            model.save(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation) + "_best/best_model")

            with open(folder_postfix + "_curriculum_iter_" + str(iteration) + ".pickle", 'wb') as handle:
                pickle.dump([best_reward_list, best_init_list, best_pop_index_list], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if optimizer.should_stop():
                # popsize multiplied by 2 (or 3) before each restart.
                # popsize = optimizer.population_size * 2
                mean = lower_bounds + (np.random.rand(6) * (upper_bounds - lower_bounds))
                optimizer = CMA(mean=mean, sigma=sigma, population_size=N_population)
                print(f"Restart CMA-ES with popsize={N_population}")
                
        elapsed_time = time.time() - start
        print (f"Iteration: {iteration} Elapsed time: {elapsed_time:.5} s.")  

    


def random_train(train_timesteps:int, algorithm, n_procs:int, map_lim: int, grid_range: int, N_eval_freq:int, N_iter: int, folder_postfix: str, output_folder: str, seed:int, visualize: bool):
    deterministic = True
    N_eval_episodes = 10
    results_list = []
    agent_target = [-45, -45]
    enemy_target = [45, 45]
    angle_target = [-np.pi/2, np.pi/2]
    N_eval_freq = N_eval_freq // n_procs
    model_dir = output_folder + "/saved_models"

    create_init_pos_angles(map_lim, N_iter, seed, folder_postfix)
    with open(folder_postfix + "_init_lists.pickle", 'rb') as handle:
        agent_init_pos, enemy_init_pos, angle_init_list = pickle.load(handle)
    

    cnt = 0
    for agent_init, enemy_init, angle_init in zip(agent_init_pos,enemy_init_pos,angle_init_list):
        print ("Agent init location: ", agent_init)
        print ("Enemy init location: ", enemy_init)

        current_folder = output_folder + "/" +  folder_postfix + "_index_" + str(cnt)
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)


        train_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, monitor_dir=current_folder, env_kwargs=dict(agent_init=agent_init, opponent_init=enemy_init, angle_init=angle_init ,seed=seed, visualization=visualize), vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, env_kwargs=dict(agent_init=agent_init, opponent_init=enemy_init, angle_init=angle_init ,seed=seed), vec_env_cls=SubprocVecEnv)
        target_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, env_kwargs=dict(agent_init=agent_target, opponent_init=enemy_target, angle_init=angle_target ,seed=seed), vec_env_cls=SubprocVecEnv)

        train_env.reset()
        eval_env.reset()
        model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")

        stop_callback = StopTrainingOnNoRemarkableImprovement(max_no_improvement_evals = 100, check_percentage=0.9, verbose = 1)

        callback = EvalCallback(eval_env=eval_env, callback_after_eval=stop_callback, n_eval_episodes = N_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                                best_model_save_path = model_dir + "/" + folder_postfix + "_index_" + str(cnt), deterministic=deterministic, verbose=1)

        

        start = time.time()
        model.learn(total_timesteps=train_timesteps, tb_log_name = folder_postfix + "_index_" + str(cnt), callback=callback)
        # model.save(model_dir + "/last_model_" + level)
        elapsed_time = time.time() - start
        print (f"Elapsed time: {elapsed_time:.5} s.")        
        train_env.close()


        # Load the model
        model = model.load(path=model_dir + "/" +  folder_postfix + "_index_" + str(cnt) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
        eval_env.reset()
        episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=N_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
        mean_reward_current = np.mean(episode_rewards)
        std_reward_current = np.std(episode_rewards)
        print (f"Mean reward in current env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
        
        target_env.reset()
        episode_rewards, episode_lengths = evaluate_policy(model, target_env, n_eval_episodes=N_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
        mean_reward_target = np.mean(episode_rewards)
        std_reward_target = np.std(episode_rewards)
        print (f"Mean reward in target env: {mean_reward_target:.4f} Std reward in target env: {std_reward_target:.4f} \n")

        init_list = [agent_init[0], agent_init[1], enemy_init[0], enemy_init[1], angle_init[0], angle_init[1]]
        reward_list = [mean_reward_current, mean_reward_target]
        result = "index: " + str(cnt) + ", inits: " + str(init_list) + ", rewards: " + str(reward_list)
        results_list.append(result)
        cnt += 1
        eval_env.close()
        target_env.close()

        with open(folder_postfix + "_results.pickle", 'wb') as handle:
            pickle.dump(results_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



def curriculum_train(train_timesteps: int, algorithm, level_list: list, level_time_coef_list: dict, n_procs: int, folder_postfix: str, output_folder: str, load_folder: str, N_eval:int, seed:int, visualize: bool):

    # activation_list = [nn.Tanh]
    # net_list = [[64, 64]]
    # policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])
    model_dir = output_folder + "/saved_models"

    print ("\n Curriculum Approach")
    prev_curriculum = ""

    for curriculum in level_list:
        current_range = level_map_ranges[curriculum] / 2.0
        agent_init_loc = [-current_range - 5.0, -current_range, -current_range - 5.0, -current_range]
        enemy_init_loc = [current_range, current_range + 5.0, current_range, current_range + 5.0]

        print (f"\nCurriculum Level: {curriculum}")
        level =  folder_postfix +  "_" + curriculum

        current_folder = output_folder + "/" + level
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
    
        train_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, monitor_dir=current_folder, env_kwargs=dict(agent_init_range=agent_init_loc, opponent_init_range=enemy_init_loc, visualization=visualize), vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, env_kwargs=dict(agent_init_range=agent_init_loc, opponent_init_range=enemy_init_loc), vec_env_cls=SubprocVecEnv)

        # train_env = Monitor(train_env, current_folder + "/monitor.csv")
        # train_env = VecNormalize(train_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)
        train_env.reset()
        # if os.path.exists(model_dir + "/" +  load_folder):
        #     model = current_model.load(model_dir + "/" +  load_folder + "/best_model", verbose=1)
        # else:
        model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")
        current_train_timesteps = train_timesteps * level_time_coef_list[curriculum]
        if curriculum != level_list[0]:
            previous_level =  folder_postfix + "_" + prev_curriculum
            model = algorithm.load(path=model_dir + "/" +  previous_level + "/best_model", env=train_env, verbose=0, only_weights = False) # + "/best_model
            model.tensorboard_log ="./" + current_folder + "_tensorboard/"
            model.set_env(train_env)
            

        callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                                best_model_save_path = model_dir + "/" + level, deterministic=True, verbose=1)

        start = time.time()
        model.learn(total_timesteps=current_train_timesteps, tb_log_name= level, callback=callback)
        # model.save(model_dir + "/last_model_" + level)
        elapsed_time = time.time() - start
        print (f"Elapsed time: {elapsed_time:.5} s.")
        
        prev_curriculum = curriculum
        train_env.close()



def noncurriculum_train(train_timesteps: int, algorithm, level_list: list, n_procs: int, folder_postfix: str, output_folder: str, load_folder: str, N_eval_freq:int, seed:int, visualize: bool):

    model_dir = output_folder + "/saved_models"
    print ("\n Non-Curriculum Approach")

    curriculum = level_list[-1]
    print (f"\nCurriculum Level: {curriculum}")
    level_folder =  folder_postfix +  "_" + curriculum

    current_folder = output_folder + "/" + level_folder
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    
    current_range = level_map_ranges[curriculum] / 2.0
    agent_init_loc = [-current_range - 5.0, -current_range, -current_range - 5.0, -current_range]
    enemy_init_loc = [current_range, current_range + 5.0, current_range, current_range + 5.0]

    train_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, monitor_dir=current_folder, env_kwargs=dict(agent_init_range=agent_init_loc, opponent_init_range=enemy_init_loc, visualization=visualize), vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_id=MovingTarget, n_envs=n_procs, env_kwargs=dict(agent_init_range=agent_init_loc, opponent_init_range=enemy_init_loc), vec_env_cls=SubprocVecEnv)

    train_env.reset()
    # if os.path.exists(model_dir + "/" +  load_folder):
    #     model = current_model.load(model_dir + "/" +  load_folder + "/best_model", verbose=1)
    # else:
    model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")
        
    callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                            best_model_save_path = model_dir + "/" + level_folder, deterministic=True, verbose=1)

    start = time.time()
    model.learn(total_timesteps=train_timesteps, tb_log_name= level_folder, callback=callback)
    # model.save(model_dir + "/last_model_" + level_folder)
    elapsed_time = time.time() - start
    print (f"Elapsed time: {elapsed_time:.5}")

    train_env.close()
