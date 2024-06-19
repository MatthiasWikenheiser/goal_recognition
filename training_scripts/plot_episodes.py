# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:37:12 2024

@author: Matthias
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_and_preprocess_df(path):
    df = pd.read_csv(path, sep=";")
    df.drop_duplicates(inplace = True)
    df.sort_values(by = "tmstmp", inplace = True)
    df.reset_index(inplace=True, drop =True)
    df.reset_index(inplace=True)
    df.rename(columns={"index":"episode"}, inplace=True)
    return df

def group_rewards_mode(df, bin_width = 1000):
    max_episode = df['episode'].max()
    df['episode_group'] = ((df['episode'] // bin_width) * bin_width).astype(str) + '-' + \
                             (((df['episode'] // bin_width) * bin_width) + bin_width).astype(str)
    episode_groups = [(i, f"{i}-{i+bin_width}") for i in range(0, int(np.ceil(max_episode / bin_width) * bin_width), bin_width)]
    ordered_groups = [label for _, label in episode_groups]
    df['episode_group'] = pd.Categorical(df['episode_group'], categories=ordered_groups, ordered=True)
    grouped_data = df.groupby(['episode_group', 'mode'])['episode_reward'].mean().unstack()
    return grouped_data

def group_count_mode(df, bin_width = 1000):
    max_episode = df['episode'].max()
    df['episode_group'] = ((df['episode'] // bin_width) * bin_width).astype(str) + '-' + \
                             (((df['episode'] // bin_width) * bin_width) + bin_width).astype(str)
    episode_groups = [(i, f"{i}-{i+bin_width}") for i in range(0, int(np.ceil(max_episode / bin_width) * bin_width), bin_width)]
    ordered_groups = [label for _, label in episode_groups]
    df['episode_group'] = pd.Categorical(df['episode_group'], categories=ordered_groups, ordered=True)
    grouped_data = df.groupby(['episode_group', 'mode'])['episode_reward'].count().unstack()
    grouped_data["total_count"] = grouped_data["directed"] + grouped_data["possible"] + grouped_data["random"]
    grouped_data["directed"] = grouped_data["directed"]/grouped_data["total_count"]
    grouped_data["possible"] = grouped_data["possible"]/grouped_data["total_count"]
    grouped_data["random"] = grouped_data["random"]/grouped_data["total_count"]
    grouped_data.drop(columns={"total_count"}, inplace=True)
    return grouped_data
    
def group_eps_mode(df, bin_width = 1000):
    max_episode = df['episode'].max()
    df['episode_group'] = ((df['episode'] // bin_width) * bin_width).astype(str) + '-' + \
                             (((df['episode'] // bin_width) * bin_width) + bin_width).astype(str)
    episode_groups = [(i, f"{i}-{i+bin_width}") for i in range(0, int(np.ceil(max_episode / bin_width) * bin_width), bin_width)]
    ordered_groups = [label for _, label in episode_groups]
    df['episode_group'] = pd.Categorical(df['episode_group'], categories=ordered_groups, ordered=True)
    grouped_data = df.groupby(['episode_group'], as_index = False)['epsilon'].mean()
    return grouped_data

def plot_rl(path, bin_width = 1000):
    color_map = {"directed": "orange", 
                 "possible": "blue",
                 "random" : "darkred"}
    
    
    df = load_and_preprocess_df(path)
    df_directed = df[df["mode"] == "directed"]
    df_random = df[df["mode"] == "random"]
    df_possible = df[df["mode"] == "possible"]
    grouped_rewards_mean = group_rewards_mode(df, bin_width)
    grouped_count_mode = group_count_mode(df, bin_width)
    grouped_eps = group_eps_mode(df, bin_width)
    
    fig, ax = plt.subplots(nrows=3, figsize=(10,20))
    ax[0].scatter(df_directed["episode"], df_directed["episode_reward"], c=color_map["directed"], label = "directed")
    ax[0].scatter(df_possible["episode"], df_possible["episode_reward"], c=color_map["possible"], label = "possible")
    ax[0].scatter(df_random["episode"], df_random["episode_reward"], c=color_map["random"], label ="random")
    ax[0].legend()
    ax[0].set_xlabel("episodes")    
    ax[0].set_ylabel("reward")  
    ax[0].grid(linestyle='-', linewidth=0.5)
    
    
    grouped_rewards_mean.plot(kind='bar', ax=ax[1], width=0.8, color = [color_map["directed"],
                                                      color_map["possible"],
                                                      color_map["random"]])
    ax[1].grid(linestyle='-', linewidth=0.5)
    ax[1].set_xlabel("episodes") 
    ax[1].set_ylabel("mean reward") 
    
    ax[2].plot(grouped_eps["episode_group"], grouped_eps["epsilon"], label = "epsilon", linewidth = 2, color = "green")
    grouped_count_mode.plot(kind='bar', ax=ax[2], width=0.8, color = [color_map["directed"],
                                                      color_map["possible"],
                                                      color_map["random"]])
    ax[2].grid(linestyle='-', linewidth=0.5)
    ax[2].set_xlabel("episodes") 
    ax[2].set_ylabel("mode count in percent") 
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()



plot_rl("E:/model_7_no_hl/goal_1/stats.csv", bin_width= 1000)
plot_rl("E:/model_7_ddqn_no_hl/goal_1/stats.csv")

df = load_and_preprocess_df("E:/model_7_no_hl/goal_1/stats.csv")

