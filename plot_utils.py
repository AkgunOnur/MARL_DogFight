import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3 import A2C
import glob
from numpy.random import default_rng
import os
from train import *




def limit_data(data_frame, N_interval=20):
    N_max = np.max(data_frame['index'].values)
    new_frame = []
    for i in range(0, N_max, N_interval):
        rows = data_frame[data_frame['index'] == i]
        for ind, rew in zip(rows['index'].values, rows['r'].values):
            new_frame.append([ind, rew])

    df = pd.DataFrame(new_frame, columns=['index', 'r'])
    return df

def show_new_curriculum(level_list, curriculum_folder = "output_cur_10", index=0, type = "train", N_interval=1):
    final_level = "level5"
    
    if curriculum_folder is not None:
        cur_directory = "saved_figs/" + curriculum_folder

    color_list = ['b', 'c', 'm', 'y', 'g']
    color_dict = {"level1": 'b', "level2": 'c', "level3": 'm', "level4": 'y', "level5": 'g'}

    if not os.path.exists(cur_directory):
        os.makedirs(cur_directory)
    
    fig = plt.figure(figsize=(10, 5))
        
    
    if curriculum_folder is not None:
        folder_list = [x[0].split('/')[1] for x in os.walk(curriculum_folder + "/") if x[0][-5:] == ("map_" + str(index)) and x[0].split('/')[1].split('_')[0] == type]
        order_list = [int(folder.split('_')[1]) for folder in folder_list]
        sorted_folders = [folder for order, folder in sorted(zip(order_list, folder_list))]
        sorted_levels = [folder.split('_')[2] for folder in sorted_folders]
        pd_frame_list = []

        empty_folders = []

        for ind, folder in enumerate(sorted_folders):
            read_data =  load_results(curriculum_folder + "/" + folder)
            
            if len(read_data) > 0:
                pd_frame = limit_data(read_data, N_interval=N_interval)
            else:
                empty_folders.append(ind)

            # pd_frame = limit_data(load_results(curriculum_folder + "/outputs_" + level + str(index)), N_interval=N_interval)
            
            pd_frame_list.append(pd_frame)

        for ind in range(1, len(sorted_folders)):
            prev_ind = np.copy(ind)
            while(prev_ind >= 0):
                prev_ind -= 1
                if len(pd_frame_list[prev_ind]['index'].values) > 0:
                    prev_max_index = np.max(pd_frame_list[prev_ind]['index'].values)
                    break
                

            pd_frame_list[ind]['index'] += prev_max_index

        
        for ind, level in enumerate(sorted_levels):
            if ind not in empty_folders:
                sns.lineplot(x='index', y='r', data=pd_frame_list[ind], color=color_dict[level])

        patch_list = []
        for color_index in color_dict.keys():
            patch_list.append(mpatches.Patch(color=color_dict[color_index], label=color_index))
        
        plt.legend(handles=patch_list)

        plt.title('Curriculum Learning Curve for the Map ' + str(index) + ' in ' + type + " Mode")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        # plt.xlim([0, 10000])
        
        
    plt.savefig(cur_directory + '/map_' +str(index) + '.png')
    plt.show()

def show_both_approaches(level_list, curriculum_folder = None, nocurriculum_folder = None, N_interval=1):
    final_level = level_list[-1]
    
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
            pd_frame = limit_data(load_results(curriculum_folder + "_" + level), N_interval=N_interval)
            # pd_frame = limit_data(load_results(curriculum_folder + "/outputs_" + level + str(index)), N_interval=N_interval)
            
            pd_frame_list.append(pd_frame)

        for ind in range(1, len(level_list)):
            pd_frame_list[ind]['index'] += np.max(pd_frame_list[ind - 1]['index'].values)

        
        for ind in range(len(level_list)):
            sns.lineplot(x='index', y='r', data=pd_frame_list[ind], color=color_list[ind])

        plt.legend(labels=level_list)
        
        plt.title('Curriculum Learning Curve for Dog Fight Scenario')

        plt.xlabel('Episode')
        plt.ylabel('Reward')


    if curriculum_folder is not None and nocurriculum_folder is not None:
        fig.add_subplot(1, 2, 2)
        
    if nocurriculum_folder is not None:
        pd_frame = limit_data(load_results(nocurriculum_folder + "_" + final_level), N_interval=N_interval)
        sns.lineplot(x='index', y='r', data=pd_frame, color=color_list[-1])
        plt.legend(labels=[level_list[-1]])
        plt.title('Curriculum Learning Curve for Dog Fight Scenario')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        
    plt.savefig(cur_directory + '/dog_fight.png')
    plt.show()

def show_curriculum(curriculum_folder = "output_cur_10",  index=0, N_interval=1):
    cur_directory = "saved_figs/" + curriculum_folder
    if not os.path.exists(cur_directory):
        os.makedirs(cur_directory)
    

    fig = plt.figure()
    fig = plt.figure(figsize=(15, 5))
    # fig.add_subplot(1, 2, 1)
    pd_frame0 = limit_data(load_results(curriculum_folder + "/model_outputs_" + "easy" + str(index)), N_interval=N_interval)
    pd_frame1 = limit_data(load_results(curriculum_folder + "/model_outputs_" + "medium" + str(index)), N_interval=N_interval)
    pd_frame2 = limit_data(load_results(curriculum_folder + "/model_outputs_" + "target" + str(index)), N_interval=N_interval)

    pd_frame1['index'] += np.max(pd_frame0['index'].values)
    pd_frame2['index'] += np.max(pd_frame1['index'].values)

    sns.lineplot(x='index', y='r', data=pd_frame0, color='b')
    sns.lineplot(x='index', y='r', data=pd_frame1, color='c')
    sns.lineplot(x='index', y='r', data=pd_frame2, color='g')
    plt.legend(labels=['Easy', 'Medium', 'Target'])
    plt.title('Curriculum Learning Curve for the Map ' + str(index))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    
    plt.savefig(cur_directory + '/map_' +str(index) + '.png')
    plt.show()


def show_figures(curriculum_folder = "output_cur_10", nocurriculum_folder = "output_nocur_10",  index=0, N_range = 20):
    cur_directory = "saved_figs/" + curriculum_folder
    if not os.path.exists(cur_directory):
        os.makedirs(cur_directory)

    pd_frame0_updated = []
    pd_frame1_updated = []
    pd_frame2_updated = []
    pd_frame_updated = []

    pd_frame0 = limit_data(load_results(curriculum_folder + "/model_outputs_" + "easy" + str(index)), N_interval=N_interval)
    pd_frame1 = limit_data(load_results(curriculum_folder + "/model_outputs_" + "medium" + str(index)), N_interval=N_interval)
    pd_frame2 = limit_data(load_results(curriculum_folder + "/model_outputs_" + "target" + str(index)), N_interval=N_interval)
    pd_frame1['index'] += np.max(pd_frame0['index'].values)
    pd_frame2['index'] += np.max(pd_frame1['index'].values)

    N_max0 = np.max(pd_frame0['index'].values)
    N_max1 = np.max(pd_frame1['index'].values)
    N_max2 = np.max(pd_frame2['index'].values)
    
    for i in range(0, N_max0, N_range):
        rows = pd_frame0[pd_frame0['index'] == i]
        for ind, rew in zip(rows['index'].values, rows['r'].values):
            pd_frame0_updated.append([ind, rew])

    for i in range(0, N_max1, N_range):
        rows = pd_frame1[pd_frame1['index'] == i]
        for ind, rew in zip(rows['index'].values, rows['r'].values):
            pd_frame1_updated.append([ind, rew])

    for i in range(0, N_max2, N_range):
        rows = pd_frame2[pd_frame2['index'] == i]
        for ind, rew in zip(rows['index'].values, rows['r'].values):
            pd_frame2_updated.append([ind, rew])

    fig = plt.figure()
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)
    
    pd_frame0 = pd.DataFrame(pd_frame0_updated, columns=['index', 'r'])
    pd_frame1 = pd.DataFrame(pd_frame1_updated, columns=['index', 'r'])
    pd_frame2 = pd.DataFrame(pd_frame2_updated, columns=['index', 'r'])

    sns.lineplot(x='index', y='r', data=pd_frame0, color='b')
    sns.lineplot(x='index', y='r', data=pd_frame1, color='c')
    sns.lineplot(x='index', y='r', data=pd_frame2, color='g')
    plt.legend(labels=['Easy', 'Medium', 'Target'])
    plt.title('Curriculum Learning Curve for the Map ' + str(index))
    plt.xlabel('Episode')
    plt.ylabel('Reward')


    fig.add_subplot(1, 2, 2)
    pd_frame = load_results(nocurriculum_folder + "/model_outputs_" + "target" + str(index))
    N_max = np.max(pd_frame['index'].values)
    
    for i in range(0, N_max, N_range):
        rows = pd_frame[pd_frame['index'] == i]
        for ind, rew in zip(rows['index'].values, rows['r'].values):
            pd_frame_updated.append([ind, rew])

    pd_frame = pd.DataFrame(pd_frame_updated, columns=['index', 'r'])

    sns.lineplot(x='index', y='r', data=pd_frame, color='g')
    plt.legend(labels=['Target'])
    plt.title('No-Curriculum Learning Curve for the Map ' + str(index))

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.savefig(cur_directory + '/map_' +str(index) + '.png')
    plt.show()


def create_maps(seed = 7, map_lim=30, N_maps = 10):
    np.random.seed(seed)
    rng = default_rng(seed)
    gen_list, easy_list, medium_list = [], [], []
    for i in range(N_maps):
        gen_map = generate_maps(map_lim=map_lim)
        easy_map = curriculum_design(gen_map, rng, level = "easy")
        medium_map = curriculum_design(gen_map, rng, level = "medium")
        gen_list.append(gen_map)
        easy_list.append(easy_map)
        medium_list.append(medium_map)
        
    with open('saved_maps_' + str(map_lim) + '.pickle', 'wb') as handle:
        pickle.dump([easy_list, medium_list, gen_list], handle, protocol=pickle.HIGHEST_PROTOCOL)