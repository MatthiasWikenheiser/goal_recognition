from pddl import pddl_domain, pddl_problem
from create_pddl_gym import GymCreator
import os
import gzip
import pickle
import rl_planner
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class MultiRLPlanner:
    def __init__(self, environment_list, rl_model_list, redundant_actions_dict=None):
        self.rl_planner_dict = self._create_environment_dict(environment_list, rl_model_list)
        self.redundant_actions_dict = self._redundant_actions(redundant_actions_dict)
        self.reset()
        self.plan = {}
        self.plan_achieved = {}
        self.plan_cost = {}
        self.time = {}
        self.plan_action = {}
        self.solved = 0  # 0:not tried yet, 1: success, 2: (at least one) timeout

    def _redundant_actions(self, redundant_actions_dict):
        if redundant_actions_dict is None:
            return {k: v for k, v in zip(self.rl_planner_dict.keys(), [None for _ in self.rl_planner_dict.keys()])}
        else:
            return redundant_actions_dict

    def _create_environment_dict(self, environment_list, rl_model_list):
        rl_planner_dict = {}
        if len(environment_list) != len(rl_model_list):
            print("length of environment_list != length of rl_model_list")
            return
        i = 0
        while i < len(environment_list):
            rl_planner_dict[environment_list[i].problem_name] = rl_planner.RlPlanner(environment=environment_list[i],
                                                                                      rl_model=rl_model_list[i])
            i+=1
        return rl_planner_dict

    def reset(self):
        for key in self.rl_planner_dict.keys():
            self.rl_planner_dict[key].reset()
        self.plan = {}
        self.plan_achieved = {}
        self.plan_cost = {}
        self.time = {}
        self.plan_action = {}
        self.solved = 0

    def set_state(self, state):
        self.reset()
        for key in self.rl_planner_dict.keys():
            self.rl_planner_dict[key].set_state(state)

    def solve(self, print_actions=False, timeout=10, goal_list=None):
        if goal_list is None:
            goal_list = self.rl_planner_dict.keys()
        with ThreadPoolExecutor() as executor:
            futures = []
            for key in goal_list:
                print("start: ", key)
                future = executor.submit(self.rl_planner_dict[key].solve,
                                         print_actions=print_actions,
                                         timeout=timeout,
                                         redundant_actions=self.redundant_actions_dict[key])
                futures.append(future)

            for future in futures:
                # Ensuring all futures are done
                future.result()
        plan_achieved = []
        for key in goal_list:
            self.plan[key] = self.rl_planner_dict[key].plan
            self.plan_achieved[key] = self.rl_planner_dict[key].plan_achieved
            plan_achieved.append(self.plan_achieved[key])
            self.plan_cost[key] = self.rl_planner_dict[key].plan_cost
            self.plan_action[key] = self.rl_planner_dict[key].plan_action
            self.time[key] = self.rl_planner_dict[key].time
        if all(plan_achieved):
            self.solved = 1
        else:
            self.solved = 2
if __name__ == "__main__":
    goals = {1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__20-06-24 22-09-10.keras"},
             2: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__22-07-24 14-13-35.keras"},
             3: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__23-07-24 23-41-57.keras"}}
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
    problem_list = [pddl_problem(path_pddl + f"model_{model}_goal_{goal}_crystal_island_problem.pddl")
                    for goal in goals.keys()]
    environment_list = [GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions).make_env()
                        for problem in problem_list]

    for goal in goals.keys():
        if goals[goal]["keep_goal_1_reward"]:
            environment_list[goal-1].set_final_reward(20)
            environment_list[goal-1].set_additional_reward_fluents("(achieved_goal_1)", 10)

    path_rl_model = "/home/mwiubuntu/finalised_rl_models/"
    rl_model_list = [rl_planner.load_model(path_rl_model + f"goal_{goal}/" + goals[goal]["rl_models_dict"])
                     for goal in goals.keys()]

    talk_to_redundant = ['ACTION-CHANGE-FINAL-REPORT-FINALINFECTIONTYPE',
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

    redundant_actions_dict = {"goal_1": talk_to_redundant,
                              "goal_2": talk_to_redundant,
                              "goal_3": talk_to_redundant}

    multi_rl_planner = MultiRLPlanner(environment_list, rl_model_list, redundant_actions_dict=redundant_actions_dict)
    print("startgame")
    multi_rl_planner.solve()

    remove_goals = [key for key in environment_list[0].observation_dict_key
                   if environment_list[0].observation_dict_key[key] in
                    [f"achieved_goal_{goal}" for goal in goals.keys()]]

    with gzip.open(path_rl_model + 'sample_set_code-1.pkl.gz') as file:
        samples = pickle.load(file)

    result_dict = {}
    for key in samples.keys():
        samples[key] = np.where(samples[key] == -1, 0, samples[key])
        not_any = []
        for remove_goal in remove_goals:
            not_any.append(samples[key][remove_goal] != 1)

        if all(not_any):
            result_dict[key] = samples[key]

    samples = [result_dict[random.choice(list(result_dict.keys()))] for _ in range(random_samples)]

    for sample in samples:
        print("------")
        multi_rl_planner.set_state(sample)
        print("start at : ", [x for x in environment_list[0].get_current_fluents() if "(at " in x][0])
        multi_rl_planner.solve()
        for goal in goals.keys():
            print(multi_rl_planner.plan[f"goal_{goal}"])