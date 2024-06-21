# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:15:27 2024

@author: Matthias
"""


#%% imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
import sys
sys.path.append(r"E:\goal_recognition")
sys.path.append(r"E:/")
import collect_dict
from pddl import *
from create_pddl_gym import GymCreator
from collections import deque
from datetime import datetime
import random
import hashlib
import pickle
import gzip
import time
import os 
from keras import backend as K
#%% Classes
# train NN settings
DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 50
MODEL_NAME = "model_7_no_hl"

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()        

class DQNAgent:    
    def __init__(self, environment, learning_rate=0.0001):
        self.environment = environment
        self.learning_rate = learning_rate
        # main model
        self.model = self.create_model(environment)
        
        # target model
        self.target_model = self.create_model(environment)
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.state_possible_actions = {}
    
    def create_model(self, environment):
        input_shape = len(environment.observation_dict)
        model = Sequential()
        model.add(Input(shape=(input_shape,)))
        #model.add(Dense(input_shape/2, activation='relu'))
        #model.add(Dense(input_shape / 3, activation='relu'))
        model.add(Dense(environment.action_space.n, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def get_qs(self, state):
        return self.model.predict(state[np.newaxis, :]) # abweichend
    
    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # hier eingreifen oder schon vorher beim befükken des replay_memory's?
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) 
        
        current_states = np.array([transition[0] for transition in minibatch])
        # Frage: ist das nullte Element immer der State?

        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) # warum drittes element, ist doch info?
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT*max_future_q
                
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            y.append(current_qs)
        
        if terminal_state:
            call_backs = None #self.tensorboard
        else:
            call_backs = None
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, 
                       verbose=0, shuffle=False, 
                       callbacks=call_backs)
        
        #updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

#%% Instantiate environment
model = 7
goal = 1
if model > 8:
    add_actions = [{'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-4b', 'loc-infirmary-kim']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-infirmary-kim', 'loc-outdoors-4b']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-infirmary-medicine', 'loc-infirmary-bathroom']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-infirmary-bathroom', 'loc-infirmary-medicine']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-laboratory-front', 'loc-outdoors-null-b']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-outdoors-null-b', 'loc-laboratory-front']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-laboratory-midright', 'loc-laboratory-library']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-laboratory-library', 'loc-laboratory-midright']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-dininghall-front', 'loc-outdoors-null-e']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-outdoors-null-e', 'loc-dininghall-front']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-dininghall-back-souptable', 'loc-outdoors-null-d']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-outdoors-null-d', 'loc-dininghall-back-souptable']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-dininghall-back-souptable', 'loc-outdoors-2b']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-outdoors-2b', 'loc-dininghall-back-souptable']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-livingquarters-hall', 'loc-outdoors-2a']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-outdoors-2a', 'loc-livingquarters-hall']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-livingquarters-hall', 'loc-outdoors-null-g']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-outdoors-null-g', 'loc-livingquarters-hall']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-brycesquarters-hall', 'loc-outdoors-null-f']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-outdoors-null-f', 'loc-brycesquarters-hall']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-brycesquarters-hall', 'loc-brycesquarters-bedroom']},
                   {'action_ungrounded': 'ACTION-MOVETOLOC',
                    'instances': ['loc-brycesquarters-bedroom', 'loc-brycesquarters-hall']}]
    cp = ["person_in_room", "neighboring"]
else:
    add_actions = None
    cp = ["person_in_room"]
#path = f"C:/Users/Matthias/OneDrive/goal_recognition/Domain and Logs/model_{model}/"
config = '495afca3d199dd8d66b44b1c5e414f225a19d42c9a540eabdcfec02e'   
    
path = "E:/best_domains/"
#domain = pddl_domain(path + f"{model}_crystal_island_domain.pddl")
domain = pddl_domain(path + f"model_{model}_{config}.pddl")
#problem = pddl_problem(path + f"model_{model}_goal_1_crystal_island_problem.pddl")
problem = pddl_problem(path + f"model_{model}_goal_{goal}_crystal_island_problem.pddl")
env_creator_ci = GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions)
env = env_creator_ci.make_env()

#%% instantiate optimal paths
saved_optimal_paths = True

redundant_actions = ["ACTION-RETRIEVEITEM", "ACTION-STOWITEM", 
                             "ACTION-DROP", "ACTION-PICKUP",
                             "ACTION-HAND-FINAL-WORKSHEET", 
                             "ACTION-CHOOSE-TESTCOMPUTER",
                             "ACTION-CHANGE-FINAL-REPORT-FINALINFECTIONTYPE",
                             "ACTION-UNSELECT-FINAL-REPORT-FINALINFECTIONTYPE",
                             "ACTION-UNSELECT-FINAL-REPORT-FINALTREATMENT",
                             "ACTION-UNSELECT-FINAL-REPORT-FINALDIAGNOSIS",
                             "ACTION-QUIZ"]


if not saved_optimal_paths:
    print("create new optimal paths")
    optimal_paths = [[3264,3266,3275,3236,3285,3247], 
                     [3244, 3407, 3323, 3297, 3293, 3236, 3285, 3247],
                     [3239, 3365, 3362, 3238, 3356, 3312, 3236, 3285, 3247],
                     [3401, 3242, 3398, 3297, 3293, 3236, 3285, 3247]]
    
    
    start_points = ["at loc-startgame", "at loc-brycesquarters-hall", "at loc-laboratory-library", "at loc-livingquarters-robertsroom"]
    
    start_point_state = []
    for s in start_points:
        print(s)
        s_state = env.reset()
        env.observation_dict["at loc-startgame"]["value"] = 0
        env.observation_dict[s]["value"] = 1
        print([x for x in env.get_current_fluents() if "at " in x])
        start_point_state.append(env._get_obs_vector())
    
    optimal_path_dict = {"optimal_paths": optimal_paths, "start_points": start_points, "start_point_state": start_point_state}
    with gzip.open(f"E:/optimal_path_dict_goal_{goal}.pkl.gz", 'wb') as file:
        pickle.dump(optimal_path_dict, file)   
        
else: 
    print("load optimal paths")
    if os.path.exists(f"E:/optimal_path_dict_goal_{goal}.pkl.gz"):
        with gzip.open(f"E:/optimal_path_dict_goal_{goal}.pkl.gz", 'rb') as file:
            optimal_path_dict = pickle.load(file)
        print("lenght of optimal_path_dict: ", len(optimal_path_dict["optimal_paths"]))
        optimal_paths = optimal_path_dict["optimal_paths"]
        start_points = optimal_path_dict["start_points"]
        start_point_state = optimal_path_dict["start_point_state"]
           
unique_actions = [] # only for checking 
for el in optimal_paths:
    for action in el:
        if env.action_dict[action]["action_ungrounded"] not in unique_actions:
            unique_actions.append(env.action_dict[action]["action_ungrounded"])

#%% set current goals for directed learning 
current_goals = [i for i in range(len(optimal_path_dict["start_points"])) 
               if "dininghall" in optimal_path_dict["start_points"][i]
               or "laboratory" in optimal_path_dict["start_points"][i] 
               or "brycesquarter" in optimal_path_dict["start_points"][i] 
               or "waterfall" in optimal_path_dict["start_points"]]

optimal_paths = [optimal_path_dict["optimal_paths"][i] for i in current_goals]
start_points = [optimal_path_dict["start_points"][i] for i in current_goals]
start_point_state = [optimal_path_dict["start_point_state"][i] for i in current_goals]

    
#%% Instantiate DQN-Agent
agent = DQNAgent(environment=env)

saved_action_path = f"E:/saved_actions.pkl.gz"

log_path = f"E:/{MODEL_NAME}/"
if not os.path.exists(log_path):
    os.mkdir(log_path)

log_path_goal = f"E:/{MODEL_NAME}/goal_{goal}/"

if not os.path.exists(log_path_goal):
    os.mkdir(log_path_goal)

if os.path.exists(saved_action_path):
    with gzip.open(saved_action_path, 'rb') as file:
        saved_actions = pickle.load(file)
    print("lenght of saved actions: ", len(saved_actions))
    agent.state_possible_actions = saved_actions
        
    
#if os.path.exists(f'{log_path_goal}replay_memory.pkl.gz'):           
    #with gzip.open(f'{log_path_goal}replay_memory.pkl.gz', 'rb') as file:
        #replay_mem = pickle.load(file)    
    #print("lenght of replay memory: ", len(replay_mem))
    #agent.replay_memory=replay_mem 
    
    
    
with gzip.open(r"E:\model_7_ddqn_no_hl\goal_1/replay_memory.pkl.gz", 'rb') as file:
    other_replay_mem = pickle.load(file)       
    
    
########CHANGE ABOVE############
    
    

    
    
    
    
#%% 
saved_models = [x for x in os.listdir(log_path_goal) if "keras" in x] 

def most_recent_filename(filenames):
    def extract_datetime(filename):
        # Split the filename to get the datetime part
        date_time_str = filename.split('__')[1].split('.keras')[0]
        # Convert the datetime part to a datetime object
        return datetime.strptime(date_time_str, '%y-%m-%d %H-%M-%S')
    
    # Find the most recent datetime
    most_recent = max(filenames, key=extract_datetime)
    return most_recent

print(most_recent_filename(saved_models))
loaded_model = tf.keras.models.load_model(log_path_goal + most_recent_filename(saved_models)) 
agent.model.set_weights(loaded_model.get_weights())
agent.target_model.set_weights(loaded_model.get_weights())

#%%weights adjustment?
d1 = agent.model.get_weights()[0]

env.action_dict[3247]

def test_func(observation_node):
    print(observation_node)
    for i in range(len(env.action_dict.keys())):
        preconditions = env.action_dict[i]["precondition"]
        j = 0
        while j < len(env.action_dict[i]["parameter_variable"]):
            var = env.action_dict[i]["parameter_variable"][j]
            inst = env.action_dict[i]["instances"][j]
            #print(var, inst)
            preconditions = preconditions.replace(var, inst)
            j+=1
        if observation_node in preconditions:
            print("-------")
            print(preconditions)
            print(env.action_dict[i]["action_grounded"])
        




test_func(env.observation_dict_key[0])










#%% 

learning_rate=0.0001

agent.model.optimizer.learning_rate.assign(learning_rate)

agent.model.optimizer.learning_rate.value
#%% Exploration settings
ep_mode = {"directed": 600, "random": 1000, "possible":1400}

epsilon = 0.25
EPSILON_DECAY = 0.999 
MIN_EPSILON = 0.05

# Stats settings
#AGGREGATE_STATS_EVERY = ep_mode["directed"] # wahrscheinlich abhängig von Lern-Phase
SAVE_MODEL_EVERY = 200

#possible actions search
mp = False
workers = 1

PRINT = True

ep_start = 1 #always start with 1
ep_mode_idx = 0
optimal_idx = 0

ep_rewards = []
q_action_list = []
q_min_list = []
q_max_list = []
q_mean_list = []
q_median_list = []
tmstmp_list = []
mode_list = []
st_point_list = []
epsilon_list = []

#q_prop_list = []


reward_corridor = {"min":-3,"max": 6}
#%% train 
EPISODES = 10_000

#ep_start = episode 

for episode in range(ep_start,EPISODES+1): #
    print(f"#################{episode}#################")
    step = 1
    agent.tensorboard.step = episode
    ep_mode_idx += 1
    if ep_mode_idx <= ep_mode["directed"]:
        mode = "directed"
        optimal_idx += 1
        if optimal_idx >= len(optimal_paths):
            optimal_idx = 0
    elif ep_mode_idx <= ep_mode["random"]:
        mode = "random"
    else:
        mode = "possible"
        if ep_mode_idx == ep_mode["possible"]+1:
            ep_mode_idx = 0
            

    print("mode: ", mode)
    
    episode_reward = 0
    
    if mode == "directed":
        st_point = start_points[optimal_idx]
        print("start_point: ", start_points[optimal_idx])
        current_state = env.reset(startpoint = False, state = start_point_state[optimal_idx])  
        print([x for x in env.get_current_fluents() if "at " in x])
        optimal_path = optimal_paths[optimal_idx]
        #print([env.action_dict[key]["action_grounded"] for key in optimal_path])
        
    else:
        
        
        
        
        

        
        #r = random.randint(0, len(agent.replay_memory))
        r = random.randint(0, len(other_replay_mem))
        #current_state = env.reset(startpoint = False, state=agent.replay_memory[r][0]) # change later !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        current_state = env.reset(startpoint = False, state=other_replay_mem[r][0]) # change later !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #start_current_state = env.reset(startpoint = False, state=agent.replay_memory[r][0])
        start_current_state = env.reset(startpoint = False, state=other_replay_mem[r][0])
        
        
        ########CHANGE ABOVE############
        
        
        
        
        
        
        st_point = [x for x in env.get_current_fluents() if "at " in x][0]
        print("start_point: ", st_point)
                
    taken_action = []  
        
    done = False
    
    l = 0
    
    len_replay_memory = len(agent.replay_memory)
    
    while not done:
        if mode == "directed":
            took_random=False
            if l < len(optimal_path):
                action = optimal_path[l]
                l+=1
                
        else:
            took_random = None
            if np.random.random() > epsilon:
                took_random = False
                
                if mode == "possible":
                    h = hashlib.new("sha224") 
                    h.update(f"{[x for x in env.state]}".encode())
                    hash_key = h.hexdigest()[:15]
                    if hash_key not in agent.state_possible_actions.keys():
                        print("update_possible_action")
                        s_time = datetime.now()
                        possible_actions = env.get_all_possible_actions(multiprocess=mp, workers=workers)
                        s_delta=datetime.now()-s_time
                        print(f"time: {s_delta.seconds}.{str(s_delta.microseconds)[:2]} seconds")
                        possible_actions = [a[0] for a in possible_actions]
                        agent.state_possible_actions[hash_key] = possible_actions
                        
                    p=agent.get_qs(current_state)
                    filtered_actions = [a for a in agent.state_possible_actions[hash_key]]
                                        #if env.action_dict[a]["action_grounded"].split("-")[1] not in ["CHANGE","UNSELECT"]]
                    
                    action = filtered_actions[np.argmax([p[0][i] for i in filtered_actions])] 
                
                if mode == "random":
                    action = np.argmax(agent.get_qs(current_state))
                    
            else:
                took_random = True
                
                if mode == "possible":
                    h = hashlib.new("sha224") 
                    h.update(f"{[x for x in env.state]}".encode())
                    hash_key = h.hexdigest()[:15]
                    if hash_key not in agent.state_possible_actions.keys():
                        print("update_possible_action")
                        s_time = datetime.now()
                        possible_actions = env.get_all_possible_actions(multiprocess=mp, workers=workers)
                        s_delta=datetime.now()-s_time
                        print(f"time: {s_delta.seconds}.{str(s_delta.microseconds)[:2]} seconds")
                        possible_actions = [a[0] for a in possible_actions]
                        agent.state_possible_actions[hash_key] = possible_actions
                    filtered_actions = [a for a in agent.state_possible_actions[hash_key]]
                                        #if env.action_dict[a]["action_grounded"].split("-")[1] not in ["CHANGE","UNSELECT"]]
                    action = random.choice(filtered_actions)
                
                if mode == "random":
                    action = random.choice(list(env.action_dict.keys()))
        
        taken_action.append(action)
        new_state, reward, done, _ = env.step(action)
       
        if PRINT:
            print("-----------episode: ", episode, " step: ", step)
            print(env.action_dict[action]["action_grounded"])
            print("took_random: ", took_random)
            print(reward)
        step+=1
        episode_reward += reward
    
        agent.update_replay_memory((current_state, action, reward, new_state, 
                                    done))
        agent.train(done)
        current_state = new_state
        if mode == "random" and step == 100:
            done = True
    
    # episode ends here
    
    #optimal_path_save
    if mode != "directed":
        if episode_reward > reward_corridor["min"] and episode_reward < reward_corridor["max"] and "infirimary" not in st_point:
            print("episode candidate as optimal")
            aleady_saved = False
            for key in range(len(optimal_path_dict["start_point_state"])):
                if (start_current_state == optimal_path_dict["start_point_state"][0]).all():
                    aleady_saved = True
            if not aleady_saved:
                print("added")
                new_row = []
                for el in taken_action:
                    action_ungrounded = env.action_dict[el]["action_ungrounded"]
                    if action_ungrounded not in redundant_actions:
                        new_row.append(el)
                optimal_path_dict["optimal_paths"].append(new_row)
                optimal_path_dict["start_points"].append(st_point)
                optimal_path_dict["start_point_state"].append(start_current_state)
            else:
                print("not added")
    
    # do stats
    q_values = np.round(agent.get_qs(current_state)[0],5)
    q_action = q_values[action]
    q_max = np.max(q_values)
    q_min = np.min(q_values)
    q_mean = np.round(np.mean(q_values),5)
    q_median = np.round(np.mean(q_values),5)
    tmstmp = datetime.now().strftime("%d-%m-%y %H-%M-%S")
    
    ep_rewards.append(episode_reward)
    q_action_list.append(q_action)
    q_min_list.append(q_min)
    q_max_list.append(q_max)
    q_mean_list.append(q_mean)
    q_median_list.append(q_median)
    tmstmp_list.append(tmstmp)
    mode_list.append(mode)
    st_point_list.append(st_point)
    epsilon_list.append(epsilon)
    
    
    
    
    #if q < 0:
        #q += 2*abs(q)
        #q_max += 2*abs(q)
    #q_prop = (q/q_max)*1000
    
    
    # collect stats
    if episode % 50 == 0 or episode == 1:
        print("episode_reward: ", episode_reward)
        print("q_action: ", q_action)
        print("q_min: ", q_min)
        print("q_max: ", q_max)
        print("q_mean: ", q_mean)
        print("q_median: ", q_median)
    
        stats_df = pd.DataFrame({"episode_reward": ep_rewards,
                                 "q_action": q_action_list,
                                 "q_min": q_min_list,
                                 "q_max": q_max_list,
                                 "q_mean": q_mean_list,
                                 "q_median": q_median_list,
                                 "tmstmp": tmstmp_list,
                                 "mode": mode_list,
                                 "start_point": st_point_list,
                                 "epsilon": epsilon_list})
        ep_rewards = []
        q_action_list = []
        q_min_list = []
        q_max_list = []
        q_mean_list = []
        q_median_list = []
        tmstmp_list = []
        mode_list = []
        st_point_list = []
        epsilon_list = []
        
        try_overwrite = True
        while try_overwrite:
            try:
                with gzip.open(f"E:/optimal_path_dict_goal_{goal}.pkl.gz", 'rb') as file:
                    old_optimal_path_dict = pickle.load(file)
                optimal_path_dict = collect_dict.collect_dict_by_values(old_optimal_path_dict, optimal_path_dict)
                with gzip.open(f"E:/optimal_path_dict_goal_{goal}.pkl.gz", 'wb') as file:
                    pickle.dump(optimal_path_dict, file)   
                try_overwrite = False
            except:
                print("error")
                time.sleep(1)

        
        if os.path.exists(f'{log_path_goal}stats.csv'):
            print("log_stats")
            old_stats = pd.read_csv(f'{log_path_goal}stats.csv', sep =";")
            stats_df = pd.concat([old_stats, stats_df])
        
        stats_df.to_csv(f'{log_path_goal}stats.csv',index=False, sep=";")
        
        try_overwrite = True
        while try_overwrite:
            try:
                with gzip.open(saved_action_path, 'rb') as file:
                    saved_actions = pickle.load(file)
                agent.state_possible_actions = collect_dict.collect_dict_by_key(agent.state_possible_actions, saved_actions)
                with gzip.open(saved_action_path, 'wb') as file:
                    print("log_saved_actions")
                    pickle.dump(agent.state_possible_actions, file)
                try_overwrite = False
            except:
                print("locked")
                time.sleep(1)

        with gzip.open(f'{log_path_goal}replay_memory.pkl.gz', 'wb') as file:
            print("log_replay_memory")
            pickle.dump(agent.replay_memory, file)
            
        
        #average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #agent.tensorboard.update_stats(reward_avg=average_reward,
                                       #reward_min=min_reward, 
                                       #reward_max=max_reward,
                                       #epsilon=epsilon)
        
    # Decay epsilon
    if epsilon > MIN_EPSILON and mode != "directed":
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        print(epsilon)
    
    if episode % SAVE_MODEL_EVERY == 0 or episode == 1:    
        agent.model.save(f'{log_path_goal}{MODEL_NAME}__{tmstmp}.keras')
        
        
        
        
        
        
        
        
        ##########CHANGE BELOW
        try: 
            with gzip.open(r"E:\model_7_ddqn_no_hl\goal_1/replay_memory.pkl.gz", 'rb') as file:
                other_replay_mem = pickle.load(file)
            print("read in other_replay_mem")
        except:
            pass
#agent.model.summary()


#%% test DQN-Agent
        
        
state = env.reset()
done = False

while not done:
    possible_actions = env.get_all_possible_actions(multiprocess=mp, workers=workers)
    idx_s = [a[0] for a in possible_actions]
    action = np.argmax(agent.get_qs(state))
    action = idx_s[np.argmax([agent.get_qs(state)[0][i] for i in idx_s])] 
    #action = 3247
    print(env.action_dict[action]["action_grounded"])
    state, reward, done, _ = env.step(action)
    
agent.get_qs(state)


optimal_path
env.action_dict[3247]["action_grounded"]

new_optimal_dict = {"optimal_paths":[], "start_points":[], "start_point_state":[]}
for row in range(len(optimal_path_dict["optimal_paths"])):
    if optimal_path_dict["optimal_paths"][row][-1] == 3247:
        new_optimal_dict["optimal_paths"].append(optimal_path_dict["optimal_paths"][row])
        new_optimal_dict["start_points"].append(optimal_path_dict["start_points"][row])
        new_optimal_dict["start_point_state"].append(optimal_path_dict["start_point_state"][row])
    
    
optimal_path_dict = new_optimal_dict


