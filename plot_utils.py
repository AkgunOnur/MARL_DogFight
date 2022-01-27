import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3 import A2C
import glob
from numpy.random import default_rng
import os

def limit_data(data_frame, N_interval=20):
    N_max = np.max(data_frame['index'].values)
    new_frame = []
    for i in range(0, N_max, N_interval):
        rows = data_frame[data_frame['index'] == i]
        for ind, rew in zip(rows['index'].values, rows['r'].values):
            new_frame.append([ind, rew])

    df = pd.DataFrame(new_frame, columns=['index', 'r'])
    return df


def show_both_approaches(level_list, scenario, curriculum_folder = None, nocurriculum_folder = None, index=0, N_interval=1, method_index=None):
    if curriculum_folder is not None:
        cur_directory = "saved_figs/" + curriculum_folder
    elif nocurriculum_folder is not None:
        cur_directory = "saved_figs/" + nocurriculum_folder

    color_list = ['b', 'c', 'm', 'y', 'g']
    if not os.path.exists(cur_directory):
        os.makedirs(cur_directory)
    

    if curriculum_folder is not None and nocurriculum_folder is not None:        
        fig = plt.figure(figsize=(15, 5))
        fig.add_subplot(1, 2, 1)
    else:
        fig = plt.figure(figsize=(10, 5))
        
    
    if curriculum_folder is not None:
        pd_frame_list = []
        for level in level_list:
            pd_frame = limit_data(load_results(curriculum_folder + "/model_outputs_" + level + str(index)), N_interval=N_interval)
            pd_frame_list.append(pd_frame)

        for ind in range(1, len(level_list)):
            pd_frame_list[ind]['index'] += np.max(pd_frame_list[ind - 1]['index'].values)

        
        for ind in range(len(level_list)):
            sns.lineplot(x='index', y='r', data=pd_frame_list[ind], color=color_list[ind])

        plt.legend(labels=level_list)
        if method_index is not None:
            plt.title('Curriculum Learning Curve for the Map ' + str(index) + ' - Method: ' + str(method_index) )
        else:
            plt.title('Curriculum Learning Curve for the Map ' + str(index))

        plt.xlabel('Episode')
        plt.ylabel('Reward')


    if curriculum_folder is not None and nocurriculum_folder is not None:
        fig.add_subplot(1, 2, 2)
        
    if nocurriculum_folder is not None:
        pd_frame = limit_data(load_results(nocurriculum_folder), N_interval=N_interval)
        sns.lineplot(x='index', y='r', data=pd_frame, color=color_list[-1])
        # plt.legend(labels=[level_list[-1]])
        if method_index is not None:
            plt.title('No-Curriculum Learning Curve for ' + scenario + ' Scenario - Method: ' + str(method_index))
        else:
            plt.title('No-Curriculum Learning Curve for ' + scenario + ' Scenario')
        

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        
    plt.savefig(cur_directory + '/map_' +str(index) + '.png')
    plt.show()
