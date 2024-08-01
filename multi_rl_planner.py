from pddl import pddl_domain, pddl_problem
from create_pddl_gym import GymCreator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import rl_planner
from concurrent.futures import ThreadPoolExecutor

class MultiRLPlanner:
    def __init__(self, environment_list, rl_model_list, redundant_actions_dict=None):
        self.rl_planner_dict = self._create_environment_dict(environment_list, rl_model_list)
        self.redundant_actions_dict = self._redundant_actions(redundant_actions_dict)
        self.reset()
        self.processes = {}

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

    def set_state(self, state):
        for key in self.rl_planner_dict.keys():
            self.rl_planner_dict[key].set_state(state)

    def solve(self, print_actions=False, timeout=10):
        with ThreadPoolExecutor() as executor:
            futures = []
            for key in self.rl_planner_dict.keys():
                print("start: ", key)
                future = executor.submit(self.rl_planner_dict[key].solve,
                                         print_actions=print_actions,
                                         timeout=timeout,
                                         redundant_actions=self.redundant_actions_dict[key])
                futures.append(future)

            for future in futures:
                # Ensuring all futures are done
                future.result()


if __name__ == "__main__":
    goals = {1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__20-06-24 22-09-10.keras"},
             2: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__22-07-24 14-13-35.keras"},
             3: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__23-07-24 23-41-57.keras"}}


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
    multi_rl_planner.solve()