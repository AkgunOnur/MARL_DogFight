import os
import torch
import time
import pickle
import argparse
import numpy as np
from env_util import make_vec_env
from numpy.random import default_rng
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from moving_target_env import MovingTarget
from dog_fight_env import DogFight

def main(args):
    total_reward_list = []
    deterministic = False
    seed = 100
    # curriculum_list = ["level1", "level2", "level3", "level4", "level5"]
    agent_target = [-45, -45]
    enemy_target = [45, 45]
    angle_target = [-np.pi/2, np.pi/2]

    model = args.model.load(args.main_folder + "/" + args.model_folder + "/best_model", verbose=1, only_weights = False) # + "/best_model"

    # target_env = make_vec_env(env_id=MovingTarget, n_envs=1, env_kwargs=dict(agent_init=agent_target, opponent_init=enemy_target, angle_init=angle_target ,seed=5), vec_env_cls=SubprocVecEnv)
    # episode_rewards, episode_lengths = evaluate_policy(model, target_env, n_eval_episodes=args.eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
    # mean_reward_target = np.mean(episode_rewards)
    # std_reward_target = np.std(episode_rewards)
    # print (f"Mean reward in target env: {mean_reward_target:.4f} Std reward in target env: {std_reward_target:.4f} \n")
    
    for episode in range(args.eval_episodes):
        env = args.scenario(agent_init=agent_target, opponent_init=enemy_target, angle_init=angle_target ,seed=episode, visualization=args.visualize)
        obs = env.reset()
        total_reward = 0
        iteration = 0
        iter_list = []
        
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)            
            obs, reward, done, info = env.step(action)
            # print ("reward: ", reward)
            total_reward += reward
            iteration += 1
        
            if done:
                print ("Episode: {0}, Reward: {1:.3f} in iteration: {2}".format(episode, total_reward, iteration))
                total_reward_list.append(total_reward)
                iter_list.append(iteration)
                break
    
    print ("Mean reward: {0:.3f}, Std. reward: {1:.3f}, in {2} episodes".format(np.mean(total_reward_list), np.std(total_reward_list), args.eval_episodes))

    
    with open(args.model_folder + '.pickle', 'wb') as handle:
        pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.visualize:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--model', default=SAC, help='model type')
    parser.add_argument('--scenario', default=MovingTarget, help='scenario type')
    parser.add_argument('--eval_episodes', default=100, type=int, help='number of test iterations')
    parser.add_argument('--model_folder', default="13Mayis_SAC_CMA_200k_index_17_best", type=str, help='the model folder')
    parser.add_argument('--main_folder', default="output_CMA_13Mayis/saved_models/", type=str, help='the main folder')
    parser.add_argument('--visualize', default = True, action='store_true', help='to visualize')

    args = parser.parse_args()
    main(args)

