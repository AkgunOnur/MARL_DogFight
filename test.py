import os
import torch
import time
import pickle
import argparse
import numpy as np
from numpy.random import default_rng
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from moving_target_env import MovingTarget
from dog_fight_env import DogFight

def main(args):
    scenario_name = str(args.scenario).split('\'')[1].split('.')[-1]
    total_reward_list = []
    model = args.model.load(args.main_folder + "/" + args.model_folder + "/best_model", verbose=1) # + "/best_model"
    env = args.scenario(visualization=args.visualize, max_timesteps=1000, level="level2")
    # episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes,
    #         render=False,
    #         deterministic=False,
    #         return_episode_rewards=True,
    #         warn=True)
    # print ("Episode Rewards: ", episode_rewards, " Episode length: ", episode_lengths, "Mean reward: ", np.mean(episode_rewards), "Mean length: ", np.mean(episode_lengths))

    for episode in range(args.eval_episodes):
        obs = env.reset()
        total_reward = 0
        iteration = 0
        iter_list = []
        while True:
            action, _states = model.predict(obs)
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

    
    with open('new_results_' + scenario_name + '.pickle', 'wb') as handle:
        pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.visualize:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--model', default=SAC, help='model type')
    parser.add_argument('--scenario', default=MovingTarget, help='scenario type')
    parser.add_argument('--eval_episodes', default=10, type=int, help='number of test iterations')
    parser.add_argument('--model_folder', default="moving_target_SAC_level1_cur_200k", type=str, help='the output folder')
    parser.add_argument('--main_folder', default="output/saved_models/", type=str, help='the output folder')
    parser.add_argument('--visualize', default = True, action='store_true', help='to visualize')

    args = parser.parse_args()
    main(args)

