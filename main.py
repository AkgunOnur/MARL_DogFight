#=========================
# Import Libraries
#=========================
from nfz import NFZone

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

##################################
#Set mode and parameters
MODE = 'train'
timesteps = 1e6

# Save a checkpoint every 2000 steps
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./model/',
                                         name_prefix='rl_model')

#Wrap the custom environment and check it
env = NFZone(2)
check_env(env, warn=True)

# # Create log directory
# log_dir = "./tmp/sac/"
# os.makedirs(log_dir, exist_ok=True)

# # Logs will be saved in log_dir/monitor.csv
# env = Monitor(env, log_dir)

if(MODE == 'train'):
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="./tmp/sac/")
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)

    # Save Model
    model.save(str("LAST"))

elif(MODE == 'test'):
    # Load Model
    model = SAC.load(str("LAST"))

    # Evaluate
    for _ in range(15):
        obs = env.reset()
        ep_reward = 0
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            ep_reward += reward
            if done:
                break
        print("Episode Reward: ", ep_reward)
