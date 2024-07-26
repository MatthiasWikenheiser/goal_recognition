from pddl import pddl_domain, pddl_problem
from create_pddl_gym import GymCreator
import os
import numpy as np
import gzip
import pickle
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf



def load_model(path):
    return tf.keras.models.load_model(path)


class RlPlanner:
    def __init__(self, environment, rl_model):
        self.env = environment
        self.rl_model = rl_model
        self.plan = {}

    def set_state(self, state):
        self.env.reset(startpoint=False, state=state)
        self.plan = {}

    def reset(self):
        self.env.reset()
        self.plan = {}

    def solve(self, print_actions=True, timeout=10, redundant_actions=None):
        start_time = time.time()
        done = False
        state_collection = []
        old_state = self.env.state
        state_collection.append(old_state)
        first_step = True

        circular_states = []



        plan_step = 0
        while not done:
            print("achieved_goals", [f"achieved_goal_{goal}" for goal in range(1,8)
                                     if self.env.observation_dict[f"achieved_goal_{goal}"]["value"]== 1])
            # check for timeouts on multiple spots to increase performance
            time_passed = time.time()-start_time
            if time_passed >= timeout:
                self.plan = {}
                print("done here: ", round(time_passed, 2), "s")
                return
            else:
                if any([(self.env.state == old).all() for old in state_collection]) and not first_step:
                    print("calc possible actions")
                    calc_start_time = time.time()
                    if redundant_actions is None:
                        possible_actions = [a[0] for a in self.env.get_all_possible_actions(multiprocess=True)]
                    else:
                        possible_actions = [a[0] for a in self.env.get_all_possible_actions(multiprocess=True)
                                            if self.env.action_dict[a[0]]["action_ungrounded"] not in redundant_actions]
                    print("calc_time over , ", round(time.time() - calc_start_time, 2), "s")

                    if time_passed >= timeout:
                        print("timeout after possible_actions, ", round(time_passed, 2), "s")
                        self.plan = {}
                        return
                    if (old_state == self.env.state).all():
                        print("not possible action")
                        q_values_possible_actions = [q_values[0][i] for i in possible_actions]
                        action = possible_actions[np.argmax(q_values_possible_actions)]
                    else:
                        print("in circular state")
                        # check if state is already listed
                        if not any([(self.env.state == circ[0]).all() for circ in circular_states]):
                            circular_states.append([self.env.state, -2])
                            idx = -1
                        else:
                            # find the circular state
                            idx = np.argmax([1 if (self.env.state == circ[0]).all() else 0 for circ in circular_states])
                            print("idx: ", idx)
                            circular_states[idx][1] += -1
                        q_values = self.rl_model.predict(self.env._get_obs_vector()[np.newaxis, :])
                        q_values_possible_actions = [q_values[0][i] for i in possible_actions]
                        print("circular_states: ", circular_states)
                        print("circular_states[idx]: ", circular_states[idx])
                        print("possible_actions: ", possible_actions)
                        print("action_name: ", [self.env.action_dict[x]["action_grounded"] for x in possible_actions])
                        print("q_values_possible_actions: ", q_values_possible_actions)
                        print("np.argsort(q_values_possible_actions): ", np.argsort(q_values_possible_actions))
                        try:
                            action = possible_actions[np.argsort(q_values_possible_actions)[circular_states[idx][1]]]
                            print("np.argsort(q_values_possible_actions)[circular_states[idx][1]]: ",
                                  np.argsort(q_values_possible_actions)[circular_states[idx][1]])
                        except:
                            # it might happen that there is only one possible action in this state
                            # therefore, try to find another path in the next state
                            circular_states[idx][1] += 1
                            action = possible_actions[np.argsort(q_values_possible_actions)[circular_states[idx][1]]]
                            print("np.argsort(q_values_possible_actions)[circular_states[idx][1]]: ",
                                  np.argsort(q_values_possible_actions)[circular_states[idx][1]])



                        print("action: ", (action,self.env.action_dict[action]["action_grounded"] ))
                        self.plan[plan_step] = self.env.action_dict[action]["action_grounded"]

                else:
                    if not first_step:
                        self.plan[plan_step] = self.env.action_dict[action]["action_grounded"]
                        plan_step += 1
                    q_values = self.rl_model.predict(self.env._get_obs_vector()[np.newaxis, :])
                    action = np.argmax(q_values)
                first_step = False
                if print_actions:
                    print(self.env.action_dict[action]["action_grounded"])
                time_passed = time.time() - start_time
                if time_passed >= timeout:
                    print("timeout here, ", round(time_passed, 2), "s")
                    self.plan = {}
                    return
                old_state = self.env.state
                state_collection.append(old_state)
                _, reward, done, _ = self.env.step(action)

        # also memorize last action that lead to done
        self.plan[plan_step] = self.env.action_dict[action]["action_grounded"]
        print("solved, ",  round(time.time() - start_time, 2), "s")




if __name__ == '__main__':
    goal = 2
    keep_goal_1_reward = False
    random_samples = 10

    # instantiate domain
    model = 7
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
    config = '495afca3d199dd8d66b44b1c5e414f225a19d42c9a540eabdcfec02e'

    path_pddl = r"/home/mwiubuntu/best_domains/"
    domain = pddl_domain(path_pddl + f"model_{model}_{config}.pddl")
    problem = pddl_problem(path_pddl + f"model_{model}_goal_{goal}_crystal_island_problem.pddl")
    env_creator_ci = GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions)
    env = env_creator_ci.make_env()

    if keep_goal_1_reward:
        env.set_final_reward(20)
        env.set_additional_reward_fluents("(achieved_goal_1)", 10)

    # load rl_model
    rl_models_dict = {1: "model_7_no_hl__20-06-24 22-09-10.keras",
                      2: "model_7_no_hl__22-07-24 14-13-35.keras",
                      3: "model_7_no_hl__23-07-24 23-41-57.keras"}

    path_rl_model = "/home/mwiubuntu/finalised_rl_models/"
    rl_model = load_model(path_rl_model + f"goal_{goal}/" + rl_models_dict[goal])

    # load_sample_states
    remove_goal = [key for key in env.observation_dict_key
                   if env.observation_dict_key[key] == f"achieved_goal_{goal}"][0]


    with gzip.open(path_rl_model + 'sample_set_code-1.pkl.gz') as file:
        samples = pickle.load(file)
    result_dict = {}
    for key in samples.keys():
        samples[key] = np.where(samples[key] == -1, 0, samples[key])
        if samples[key][remove_goal] != 1:
            result_dict[key] = samples[key]
    samples = [result_dict[random.choice(list(result_dict.keys()))] for _ in range(random_samples)]

    # instantiate rl_planner
    rl_planner = RlPlanner(env, rl_model)

    #solve samples
    redundant_actions = ['ACTION-CHANGE-FINAL-REPORT-FINALINFECTIONTYPE',
                         'ACTION-UNSELECT-FINAL-REPORT-FINALINFECTIONTYPE',
                         'ACTION-CHANGE-FINAL-REPORT-FINALDIAGNOSIS',
                         'ACTION-UNSELECT-FINAL-REPORT-FINALDIAGNOSIS',
                         'ACTION-CHANGE-FINAL-REPORT-FINALSOURCE',
                         'ACTION-UNSELECT-FINAL-REPORT-FINALSOURCE',
                         'ACTION-CHANGE-FINAL-REPORT-FINALTREATMENT',
                         'ACTION-UNSELECT-FINAL-REPORT-FINALTREATMENT',
                         'ACTION-HAND-FINAL-WORKSHEET', 'ACTION-PICKUP',
                         'ACTION-DROP', 'ACTION-STOWITEM', 'ACTION-RETRIEVEITEM',
                         'ACTION-CHOOSE-TESTCOMPUTER', 'ACTION-QUIZ']

    print("--------")
    print("start_game")
    rl_planner.solve()

    for sample in samples:
        print("------")
        rl_planner.set_state(sample)
        print("start at : ", [x for x in rl_planner.env.get_current_fluents() if "(at " in x][0])
        rl_planner.solve(redundant_actions=redundant_actions)
        print(rl_planner.plan)


    #d = []
    #for key in env.action_dict.keys():
     #   if env.action_dict[key]["action_ungrounded"] not in d:
      #      d.append(env.action_dict[key]["action_ungrounded"])


    #possible_actions = [a[0] for a in env.get_all_possible_actions(multiprocess=True)
     #                                       if env.action_dict[a[0]]["action_ungrounded"] not in redundant_actions]

   # q_values = rl_planner.rl_model.predict(rl_planner.env._get_obs_vector()[np.newaxis, :])
    #q_values_possible_actions = [q_values[0][i] for i in possible_actions]