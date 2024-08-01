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
        self.plan_achieved = 0
        self.plan_cost = 0
        self.time = None

    def set_state(self, state):
        self.env.reset(startpoint=False, state=state)
        self.plan = {}
        self.plan_action = None
        self.plan_achieved = 0
        self.plan_cost = 0
        self.time = None

    def reset(self):
        self.env.reset()
        self.plan = {}
        self.plan_action = None
        self.plan_achieved = 0
        self.plan_cost = 0
        self.time = None

    @staticmethod
    def _remove_circular_states_indices(states):
        seen_states = {}
        clean_indices = []

        for index, state in enumerate(states):
            if state in seen_states:
                # If a circular state is detected, remove all elements up to the first occurrence of that state
                first_occurrence = seen_states[state]
                clean_indices = [i for i in clean_indices if i <= first_occurrence]
                # Update seen_states to reflect only the remaining indices
                seen_states = {states[i]: i for i in clean_indices}
            # Append the current index and update seen_states
            else:
                clean_indices.append(index)
                seen_states[state] = index

        return clean_indices

    def solve(self, print_actions=True, timeout=10, redundant_actions=None):
        start_time = time.time()
        done = False
        state_collection = []
        state_for_clean_collection = []
        state_for_clean_dict = {}
        state_for_clean_dict_key = 0
        old_state = self.env.state
        state_collection.append(old_state)
        first_step = True
        circular_states = []
        plan = []

        while not done:
            if print_actions:
                print("achieved_goals", [f"achieved_goal_{goal}" for goal in range(1,8)
                                     if self.env.observation_dict[f"achieved_goal_{goal}"]["value"]== 1])
            # check for timeouts on multiple spots to increase performance
            time_passed = time.time()-start_time
            if time_passed >= timeout:
                self.plan = {}
                print("timeout: ", round(time_passed, 2), "s")
                return
            else:
                if any([(self.env.state == old).all() for old in state_collection]) and not first_step:
                    if print_actions:
                        print("calc possible actions")
                    calc_start_time = time.time()
                    if redundant_actions is None:
                        possible_actions = [a[0] for a in self.env.get_all_possible_actions(multiprocess=True)]
                    else:
                        possible_actions = [a[0] for a in self.env.get_all_possible_actions(multiprocess=True)
                                            if self.env.action_dict[a[0]]["action_ungrounded"] not in redundant_actions]
                    if print_actions:
                        print("calc_time over , ", round(time.time() - calc_start_time, 2), "s")

                    if time_passed >= timeout:
                        print("timeout: ", round(time_passed, 2), "s")
                        self.plan = {}
                        return
                    if (old_state == self.env.state).all():
                        if print_actions:
                            print("not possible action") # can also be action that doesnt alter the state anymore
                        # maintain old q_values and filter for a new possible action
                        old_action = action
                        q_values_possible_actions = [q_values[0][i] for i in possible_actions]
                        j = -1
                        while old_action == action:
                            action = possible_actions[np.argsort(q_values_possible_actions)[j]]
                            j += -1

                    else:
                        if print_actions:
                            print("in circular state")
                        plan.append(action)
                        state_for_clean_collection.append([self.env.state, True])
                        state_for_clean_dict[state_for_clean_dict_key] = self.env.state
                        state_for_clean_dict_key += 1

                        # check if circular state is already listed
                        if not any([(self.env.state == circ[0]).all() for circ in circular_states]):
                            circular_states.append([self.env.state, -2])
                            idx = -1
                        else:
                            # find the circular state
                            idx = np.argmax([1 if (self.env.state == circ[0]).all() else 0 for circ in circular_states])
                            circular_states[idx][1] += -1
                        # state has changed from state before, therefore predict new q-values
                        q_values = self.rl_model.predict(self.env._get_obs_vector()[np.newaxis, :], verbose=0)
                        q_values_possible_actions = [q_values[0][i] for i in possible_actions]

                        try:
                            # it might happen that there is only one possible action in this state
                            # therefore, try to find another path in the next state
                            action = possible_actions[np.argsort(q_values_possible_actions)[circular_states[idx][1]]]
                        except:
                            # go back to last idx, since no other possible action is available
                            circular_states[idx][1] += 1
                            action = possible_actions[np.argsort(q_values_possible_actions)[circular_states[idx][1]]]

                else:
                    if not first_step:
                        plan.append(action)
                        state_for_clean_collection.append([self.env.state, False])
                    q_values = self.rl_model.predict(self.env._get_obs_vector()[np.newaxis, :], verbose=0)
                    action = np.argmax(q_values)
                first_step = False
                if print_actions:
                    print(self.env.action_dict[action]["action_grounded"])
                time_passed = time.time() - start_time
                if time_passed >= timeout:
                    print("timeout: ", round(time_passed, 2), "s")
                    self.plan = {}
                    return
                old_state = self.env.state
                state_collection.append(old_state)
                _, reward, done, _ = self.env.step(action)

        # also memorize last action that lead to done
        plan.append(action)
        state_for_clean_collection.append([self.env.state, False])

        if time_passed >= timeout:
            self.plan = {}
            print("timeout: ", round(time_passed, 2), "s")
            return

        # clean circular actions
        states_dirty = []
        if state_for_clean_dict_key > 0:
            for state in state_for_clean_collection:
                find_state_key = [key for key in state_for_clean_dict.keys()
                                  if (state_for_clean_dict[key] == state[0]).all()]
                if state[1]:
                    states_dirty.append(find_state_key[0])
                else:
                    if len(find_state_key) == 0:
                        state_for_clean_dict[state_for_clean_dict_key] = state[0]
                        states_dirty.append(state_for_clean_dict_key)
                        state_for_clean_dict_key += 1
                    else:
                        states_dirty.append(find_state_key[0])

            plan = [plan[idx] for idx in self._remove_circular_states_indices(states_dirty)]

        if time_passed >= timeout:
            self.plan = {}
            print("timeout: ", round(time_passed, 2), "s")
            return

        # clean redundant_actions
        if redundant_actions is not None:
            clean_plan_step = 0
            clean_plan = {}
            # remove redundant actions
            for step in self.plan.keys():
                if self.plan[step].split("_")[0] not in redundant_actions:
                    clean_plan[clean_plan_step] = self.plan[step]
                    clean_plan_step += 1
            self.plan = clean_plan

        if time_passed >= timeout:
            print("timeout: ", round(time_passed, 2), "s")
            self.plan = {}
            return
        self.plan_action = plan
        self.plan = {key: self.env.action_dict[value]['action_grounded'] for key, value in zip(range(len(plan)), plan)}
        self.plan_achieved = 1
        self.plan_cost = (sum([self.env.action_dict[p]['reward'] for p in self.plan_action])) * (-1)
        time_passed = round(time.time() - start_time, 2)
        if time_passed >= timeout:
            print("timeout: ", round(time_passed, 2), "s")
            self.plan = {}
            return
        self.time = time_passed
        print("solved, ",  self.time, "s")


if __name__ == '__main__':
    goal = 1
    keep_goal_1_reward = False
    random_samples = 10
    print_actions = False

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

    # solve samples
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
    rl_planner.solve(print_actions=print_actions)
    print(rl_planner.plan)
    sample = samples[1]

    for sample in samples:
        print("------")
        rl_planner.set_state(sample)
        print("start at : ", [x for x in rl_planner.env.get_current_fluents() if "(at " in x][0])
        rl_planner.solve(redundant_actions=redundant_actions, print_actions=print_actions)
        print(rl_planner.plan)

    env.action_dict