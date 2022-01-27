import torch
import time
import pickle
import argparse
import numpy as np
from numpy.random import default_rng


import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from moving_target_env import MovingTarget



def main():
    scenario = "moving_target"
    model_dir = 'output/saved_models/'
    visualization = True

    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--eval_episodes', default=10, type=int, help='number of test iterations')

    args = parser.parse_args()
    os.makedirs(model_dir, exist_ok=True)

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], reward_range=reward_range)
    
    # with open('saved_maps_' + str(map_lim) + '.pickle', 'rb') as handle:
    #     easy_list, medium_list, gen_list = pickle.load(handle)


    total_reward_list = []
    model = PPO.load(model_dir + "/best_model_" + scenario + "/best_model", verbose=1) # + "/best_model"
    env = MovingTarget(visualization=visualization)
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

    
    with open('new_results_' + scenario + '.pickle', 'wb') as handle:
        pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if visualization:
        env.close()

if __name__ == '__main__':
    main()

