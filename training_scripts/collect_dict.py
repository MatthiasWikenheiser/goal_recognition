# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:12:59 2024

@author: Matthias
"""
import pickle 
import gzip
import os



def collect_dict_by_key(*dicts):
    element = 1
    for dict_el in dicts:
        if element == 1:
            dict_collect = dict_el
            old_len = len(dict_collect)
        else:
            dict_element = dict_el
            new_keys = [key for key in dict_element.keys() if key not in dict_collect.keys()]
            for key in new_keys:
                dict_collect[key] = dict_element[key]
        element += 1
    print("old_len: ", old_len)
    print("new_len: ", len(dict_collect))
    print("new_keys: ", len(dict_collect) - old_len)
    return dict_collect

def collect_dict_by_values(*dicts):
    element = 1
    for dict_el in dicts:
        if element == 1:
            dict_collect = dict_el
        else:
            dict_element = dict_el  
            compare_key = list(dict_element.keys())[0]
            #print(dict_collect[compare_key])
            for idx in range(len(dict_element[compare_key])):
                if dict_element[compare_key][idx] not in dict_collect[compare_key]:
                    for key in dict_collect.keys():
                        dict_collect[key].append(dict_element[key][idx])   
        element += 1
    return dict_collect




if __name__ == "__main__":
    path_saved_actions = "E:/saved_actions.pkl.gz".replace("\\","/")
    
    with gzip.open(path_saved_actions, 'rb') as file:
        dict_dqn = pickle.load(file) 



    #collect_dict = collect_dict_by_key(dict_dqn, dict_ddqn)

    with gzip.open("E:\saved_actions.pkl.gz", 'wb') as file:
        print("log_saved_actions")
        pickle.dump(collect_dict, file)
        
    path_optimal = "E:/optimal_path_dict_goal_1.pkl.gz"
    with gzip.open(path_optimal, 'rb') as file:
        optimal_dict = pickle.load(file) 
        
    new_entry = {"optimal_paths": [[1,4],[10,9]], "start_points": ["at vba", "avdfvgfp"], 
                     "start_point_state": [optimal_dict["start_point_state"][0], optimal_dict["start_point_state"][1]]}
    
    test = collect_dict_by_values(optimal_dict, new_entry)
    
    
        
        
        
    