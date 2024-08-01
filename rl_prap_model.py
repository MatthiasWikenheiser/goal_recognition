from pddl import pddl_domain, pddl_problem
from create_pddl_gym import GymCreator
from pddl import pddl_observations
from multi_rl_planner import MultiRLPlanner
import rl_planner
import copy
import time


class PRAPAgent:
    class task: #just to copy the structure of prap (steps_optimal, steps_observed)
        def __init__(self, plan, plan_achieved, plan_cost, plan_action, solved, time):
            self.plan = plan
            self.plan_achieved = plan_achieved
            self.plan_cost = plan_cost
            self.plan_action = plan_action
            self.solved = solved
            self.time = time

    def __init__(self, multi_rl_planner, obs_action_sequence):
        self.multi_rl_planner = multi_rl_planner
        self.observation = obs_action_sequence
        self.env_obs = self._create_env_obs() #brauch man evtl gar nicht
        self.steps_optimal = None

    def _create_env_obs(self):
        key_0 = list(self.multi_rl_planner.rl_planner_dict.keys())[0]
        return copy.copy(self.multi_rl_planner.rl_planner_dict[key_0])

    def perform_solve_optimal(self, print_actions=False, timeout=10):
        start_time = time.time()
        self.multi_rl_planner.solve(print_actions=print_actions, timeout=timeout)
        self.steps_optimal = self.task(plan=self.multi_rl_planner.plan,
                                       plan_achieved=self.multi_rl_planner.plan_achieved,
                                       plan_cost=self.multi_rl_planner.plan_cost,
                                       plan_action=self.multi_rl_planner.plan_action,
                                       solved=self.multi_rl_planner.solved,
                                       time=self.multi_rl_planner.time)
        self.multi_rl_planner.reset()
        print("total time-elapsed: ", round(time.time() - start_time, 2), "s")







if __name__ == "__main__":
    goals = {1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__20-06-24 22-09-10.keras"},
             2: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__22-07-24 14-13-35.keras"},
             3: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__23-07-24 23-41-57.keras"}}
    obs_path = r'/home/mwiubuntu/Seminararbeit/Interaction Logs/model_7/Session1-StationA/'

    obs = pddl_observations(obs_path + '2_log_Salmonellosis.csv')

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
            environment_list[goal - 1].set_final_reward(20)
            environment_list[goal - 1].set_additional_reward_fluents("(achieved_goal_1)", 10)

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
    model = PRAPAgent(multi_rl_planner=multi_rl_planner, obs_action_sequence=obs)
    model.perform_solve_optimal()



