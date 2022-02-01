import numpy as np
import torch
import gym
import argparse
import os
import pickle

import utils
import TD3
import OurDDPG
import DDPG
from dubin import Dubin
from moving_target_env import MovingTarget
from gym.utils import seeding
gym.logger.set_level(40)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy1, eval_env1, seed, scenario, policy2=None, eval_episodes=15, locked_reward=60, 
                get_locked_reward=-40, Lock_Count=50, print_steps=False, enable_render = False):
    avg_reward1, avg_reward2 = 0., 0.
    np_random, _ = seeding.np_random(seed)
    eval_env1.seed(seed)
    N = 500
    locked_counts = 0
    getlocked_counts = 0


    for episode in range(eval_episodes):
        
        episode_reward1, episode_reward2 = 0., 0.
        counter = 0
        p_reward1_cnt, n_reward1_cnt = 0, 0
        
        if scenario == "moving_target":
            state1, done1 = eval_env1.reset(), False
        elif scenario == "dog_fight":
            states, done1, done2 = eval_env1.reset(), False, False
            state1, state2 = states

        for cnt in range(1, N+1):
            if enable_render:
                eval_env1.render()

            # Select policy
            if scenario == "dog_fight":
                action1 = policy1.select_action(np.array(state1))
                action1[1] = action1[1] * max_action[1] #the velocity will be between -max_vel and max_vel
                action1 = (action1 + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(min_action, max_action)

                action2 = policy1.select_action(np.array(state2))
                action2[1] = action2[1] * max_action[1] #the velocity will be between -max_vel and max_vel
                action2 = (action2 + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(min_action, max_action)

                next_states, rewards, dones, _ = eval_env1.step(action1=action1, action2=action2)
                next_state1, next_state2 = next_states
                reward1, reward2 = rewards
                done1, done2 = dones

                episode_reward2 += reward2

            elif scenario == "moving_target":
                action1 = policy1.select_action(np.array(state1[2:]))
                action1[1] = action1[1] * max_action[1] #the velocity will be between -max_vel and max_vel
                action1 = (action1 + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(min_action, max_action)

                next_state1, reward1, done1, _ = eval_env1.step(action1=action1)

            episode_reward1 += reward1

            if reward1 == locked_reward:
                p_reward1_cnt += 1
                n_reward1_cnt = 0
                reward1 = reward1 * p_reward1_cnt
            elif reward1 == get_locked_reward:
                p_reward1_cnt = 0
                n_reward1_cnt += 1
                reward1 = reward1 * n_reward1_cnt
            else:
                p_reward1_cnt, n_reward1_cnt = 0, 0

            
            if p_reward1_cnt >= Lock_Count:
                if print_steps:
                    print ("\n Plane B is destroyed!")
                locked_counts += 1
                done1, done2 = True, True
            elif n_reward1_cnt >= Lock_Count:
                if print_steps:
                    print ("\n Plane A is destroyed!")
                getlocked_counts += 1
                done1, done2 = True, True

            if done1 and done2:
                break

        episode_reward1 = episode_reward1 / (cnt*1000.0)
        episode_reward2 = episode_reward2 / (cnt*1000.0)
        avg_reward1 += episode_reward1
        avg_reward2 += episode_reward2

            # if scenario == "dog_fight":
            #     print(
            #         f"Current mean reward for Plane A: {avg_reward1/float(counter*1000):.3f}, for Plane B: {avg_reward2/float(counter*1000):.3f}")

        if print_steps:
            print("Evaluation episode: {0}/{1}".format(episode + 1, eval_episodes))
            if scenario == "dog_fight":
                print(
                    f"Evaluation reward for Plane A: {episode_reward1:.3f}, for Plane B: {episode_reward2:.3f}")
            else:
                print(f"Evaluation reward: {episode_reward1:.3f}")

    avg_reward1 /= (eval_episodes)
    avg_reward2 /= (eval_episodes)

    print("------------------------------------------------------------------------------")
    if scenario == "dog_fight":
        print(f"Over {eval_episodes} episodes average reward1: {avg_reward1:.3f}, average reward2: {avg_reward2:.3f} Succesfull deadlocks: {locked_counts+getlocked_counts}/{eval_episodes}")
        print("------------------------------------------------------------------------------")
        return (avg_reward1, avg_reward2), locked_counts + getlocked_counts

    elif scenario == "moving_target":
        print(f"Over {eval_episodes} episodes average reward: {avg_reward1:.3f}, Succesfull deadlocks: {locked_counts}/{eval_episodes}")
        print("------------------------------------------------------------------------------")
    
    return avg_reward1, locked_counts
    

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # OpenAI gym environment name
    parser.add_argument("--env", default="Dubin")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=500, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=100, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=2000, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_episodes", default=2000, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", default=True)
    # Switch on/off rendering 
    parser.add_argument("--render", default=True)
    # Model load file name, "" doesn't load, "TD3_dog_fight_200_30_Val"
    parser.add_argument("--load_model", default="TD3_moving_target_200_20")
    # train or validation mode
    parser.add_argument("--mode", default="validation")
    # still_target, moving_target or dog_fight
    parser.add_argument("--scenario", default="moving_target")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    prev_train_reward = -np.Inf
    prev_val_reward = -np.Inf

    t_prev = 0
    locked_reward=50.0
    get_locked_reward=-40.0
    Lock_Count = 50

    #env = gym.make(args.env)
    env1 = Dubin(scenario=args.scenario,  dog_fight_range=25.0, detection_angle=90.0,
                 opponent_range=25.0, opponent_angle=90.0, name="A", opponent="B", 
                 locked_reward=locked_reward, get_locked_reward=get_locked_reward)
    args.seed += 39
    env1.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 11  # env.observation_space.shape[0]
    action_dim = env1.action_space.shape[0]
    max_action = env1.action_space.high
    min_action = env1.action_space.low

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "min_action": min_action,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy1 = TD3.TD3(**kwargs)
        if args.scenario == "dog_fight":
            policy2 = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        folder = "Best_Models"
        policy_file = file_name if args.load_model == "default" else args.load_model

        if args.scenario == "dog_fight":
            policy1.load(f"./{folder}/{args.scenario}/{policy_file}_A")
            policy2.load(f"./{folder}/{args.scenario}/{policy_file}_B")
        else:
            policy1.load(f"./{folder}/{args.scenario}/{policy_file}")

    replay_buffer1 = utils.ReplayBuffer(state_dim, action_dim)
    if args.scenario == "dog_fight":
        replay_buffer2 = utils.ReplayBuffer(state_dim, action_dim)

    if args.mode == "train":
        # Evaluate untrained policy
        evaluations = []
        # evaluations = [eval_policy(policy, env, args.seed, args.scenario)]
        train1_rewards = []
        train2_rewards = []

        print("---------------------------------------")
        print(
            f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        for episode in range(1, args.max_episodes + 1):
            np_random, _ = seeding.np_random(args.seed)
            episode_reward1, episode_reward2 = 0, 0
            p_reward1_cnt, n_reward1_cnt = 0, 0 
            if args.scenario == "dog_fight":
                states, done1, done2 = env1.reset(), False, False
                state1, state2 = states
            elif args.scenario == "moving_target":
                state1, done1 = env1.reset(), False

            print("\n Episode: {0}/{1}".format(episode, args.max_episodes))

            for t in range(1, int(args.max_timesteps) + 1):
                if args.render:
                    env1.render()
                # Select action randomly or according to policy
                if t < args.start_timesteps:
                    action1 = env1.action_space.sample()
                    action1 = action1
                    if args.scenario == "dog_fight":
                        action2 = env1.action_space.sample()
                        action2 = action2
                else:
                    
                    if args.scenario == "dog_fight":
                        action1 = policy1.select_action(np.array(state1))
                        action1[1] = action1[1] * max_action[1] #the velocity will be between -max_vel and max_vel
                        action1 = (action1 + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(min_action, max_action)

                        action2 = policy2.select_action(np.array(state2))
                        action2[1] = action2[1] * max_action[1] #the velocity will be between -max_vel and max_vel
                        action2 = (action2 + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(min_action, max_action)
                    elif args.scenario == "moving_target":
                        action1 = policy1.select_action(np.array(state1[3:]))
                        action1[1] = action1[1] * max_action[1] #the velocity will be between -max_vel and max_vel
                        action1 = (action1 + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(min_action, max_action)
                    

                # Perform action
                if args.scenario == "dog_fight":
                    next_states, rewards, dones, _ = env1.step(action1=action1, action2=action2)
                    next_state1, next_state2 = next_states
                    reward1, reward2 = rewards
                    done1, done2 = dones
                    
                elif args.scenario == "moving_target":
                    next_state1, reward1, done1, _ = env1.step(action1=action1)

                if reward1 == locked_reward:
                    p_reward1_cnt += 1
                    n_reward1_cnt = 0
                    reward1 = reward1 * p_reward1_cnt
                    reward2 = reward2 * p_reward1_cnt
                elif reward1 == get_locked_reward:
                    p_reward1_cnt = 0
                    n_reward1_cnt += 1
                    reward1 = reward1 * n_reward1_cnt
                    reward2 = reward2 * n_reward1_cnt
                else:
                    p_reward1_cnt, n_reward1_cnt = 0, 0

                
                if p_reward1_cnt >= Lock_Count:
                    print ("\n Plane B is destroyed!")
                    done1, done2 = True, True
                elif n_reward1_cnt >= Lock_Count:
                    print ("\n Plane A is destroyed!")
                    done1, done2 = True, True

                # Store data in replay buffer
            
                if args.scenario == "dog_fight":
                    replay_buffer1.add(state1, action1, next_state1, reward1, float(done1))
                    replay_buffer1.add(state2, action2, next_state2, reward2, float(done2))
                    replay_buffer2.add(state2, action2, next_state2, reward2, float(done2))
                    replay_buffer2.add(state1, action1, next_state1, reward1, float(done1))
                elif args.scenario == "moving_target":
                    replay_buffer1.add(state1[3:], action1, next_state1[3:], reward1, float(done1))
                    
                state1 = next_state1
                episode_reward1 += reward1
                train_reward1 = episode_reward1 / float(100*t)
                if args.scenario == "dog_fight":
                    state2 = next_state2
                    episode_reward2 += reward2
                    train_reward2 = episode_reward2 / float(100*t)

                # Train agent after collecting sufficient data
                if t >= args.start_timesteps:
                    policy1.train(replay_buffer1, args.batch_size)
                    if args.scenario == "dog_fight":
                        policy2.train(replay_buffer2, args.batch_size)

                if t % 100 == 0:
                    if args.scenario == "dog_fight":
                        print(
                            f"Time Step: {t}/{args.max_timesteps} Mean Reward for A: {train_reward1:.3f}, Mean Reward for B: {train_reward2:.3f}")
                    else:
                        print(
                            f"Time Step: {t}/{args.max_timesteps} Mean Reward: {train_reward1:.3f}")

                if done1 and done2:
                    print(
                        f"Episode is completed in {t} time steps. Mean Reward1: {train_reward1:.3f}, Mean Reward2: {train_reward2:.3f}")
                    break

            
            train1_rewards.append(train_reward1)
            if args.scenario == "dog_fight":
                train2_rewards.append(episode_reward2 / float(t))

            # Evaluate episode
            if args.scenario == "moving_target":
                if train_reward1 > prev_train_reward and (t >= t_prev or t > args.start_timesteps):
                    prev_train_reward = train_reward1
                    t_prev = t
                    file_name = f"{args.policy}_{args.scenario}_{episode}_{args.seed}_Train"
                    print ("Better train results obtained and model saved")
                    if args.save_model:
                        policy1.save(f"./models/{file_name}")

                if episode % args.eval_freq == 0:
                    val_reward, locked_counts = eval_policy(policy1, env1, args.seed, args.scenario, enable_render=args.render)
                    #np.save(f"./results/{file_name}", evaluations)
                    file_name = f"{args.policy}_{args.scenario}_{episode}_{args.seed}_Val"
                    if args.save_model and (locked_counts >= 3 or val_reward > prev_val_reward):
                        print ("Better validation results obtained and model saved")
                        prev_val_reward = val_reward
                        policy1.save(f"./models/{file_name}")
                    file_name = f"Train_rewards_{args.policy}_{args.scenario}_{args.seed}.pkl"
                    with open(f"./results/{file_name}", 'wb') as f:
                        pickle.dump([train1_rewards, train2_rewards], f)
            elif args.scenario == "dog_fight":
                if (train_reward1 > prev_train_reward or train_reward2 > prev_train_reward) and (t >= t_prev or t > args.start_timesteps):
                    if train_reward1 > train_reward2:
                        prev_train_reward = train_reward1
                    else:
                        prev_train_reward = train_reward2

                    t_prev = t
                    file_name = f"{args.policy}_{args.scenario}_{episode}_{args.seed}_Train"
                    print ("Better train results obtained and model saved")
                    if args.save_model:
                        policy1.save(f"./models/{file_name}_A")
                        policy2.save(f"./models/{file_name}_B")

                if episode % args.eval_freq == 0:
                    val_rewards, total_locked_counts = eval_policy(policy1, env1, args.seed, args.scenario, policy2=policy2, enable_render=args.render)
                    val_reward1, val_reward2 = val_rewards
                    #np.save(f"./results/{file_name}", evaluations)
                    file_name = f"{args.policy}_{args.scenario}_{episode}_{args.seed}_Val"
                    if args.save_model and (total_locked_counts >= 3 or val_reward1 > prev_val_reward or val_reward2 > prev_val_reward):
                        if val_reward1 > val_reward2:
                            prev_val_reward = val_reward1
                        else:
                            prev_val_reward = val_reward2
                        print ("Better validation results obtained and model saved")
                        policy1.save(f"./models/{file_name}_A")
                        policy2.save(f"./models/{file_name}_B")
                    file_name = f"Train_rewards_{args.policy}_{args.scenario}_{args.seed}.pkl"
                    with open(f"./results/{file_name}", 'wb') as f:
                        pickle.dump([train1_rewards, train2_rewards], f)


    elif args.mode == "validation":
        seed = args.seed + 10
        print("---------------------------------------")
        print(
            f"Mode: {args.mode}, Policy: {args.policy}, Scenario: {args.scenario}, Env: {args.env}, Seed: {seed}")
        print("---------------------------------------")
        if args.scenario == "dog_fight":
            reward_avg = eval_policy(policy1, env1, seed, args.scenario, policy2=policy2, 
                                    locked_reward=locked_reward, get_locked_reward=get_locked_reward, 
                                    Lock_Count=Lock_Count, print_steps=True, enable_render=args.render)
        else:
            reward_avg = eval_policy(policy1, env1, seed, args.scenario, 
                                    locked_reward=locked_reward, get_locked_reward=get_locked_reward, 
                                    Lock_Count=Lock_Count, print_steps=True, enable_render=args.render)
