from pddl import pddl_domain, pddl_problem
from create_pddl_gym import GymCreator
from pddl import pddl_observations
from multi_rl_planner import MultiRLPlanner
import rl_planner
import copy
import time
import numpy as np
import pandas as pd


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
        self.steps_observed = []
        self.prob_nrmlsd_dict_list = []
        self.prob_dict_list = []
        self.predicted_step = {}
        self.cost_obs_cum_dict = {}
        self.summary_level_1 = None
        self.summary_level_2 = None
        self.summary_level_3 = None

    def _create_env_obs(self):
        key_0 = list(self.multi_rl_planner.rl_planner_dict.keys())[0]
        return copy.deepcopy(self.multi_rl_planner.rl_planner_dict[key_0].env)

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

    def _predict_step(self, step):
        #empty shell due to missing perform_solve_observed, just defined here for subclasses
        dict_proba = self.prob_nrmlsd_dict_list[step]
        most_likeli = 0
        key_most_likeli = []
        for key in list(dict_proba.keys()):
            if dict_proba[key] > most_likeli and dict_proba[key] > 0:
                key_most_likeli = [key]
                most_likeli = dict_proba[key]
            elif dict_proba[key] == most_likeli and dict_proba[key] > 0:
                key_most_likeli.append(key)
                most_likeli = dict_proba[key]
        return key_most_likeli

    def _create_summary(self):
        df_summary_agg = pd.DataFrame()
        i = 0
        for step in self.steps_observed:
            goals_name = []
            goals_achieved = []
            goals_costs = []
            goals_costs_cumulated = []
            goals_seconds = []
            goals_probs = []
            goals_probs_nrmlsd = []
            step_probs = self.prob_dict_list[i]
            step_probs_nrmlsd = self.prob_nrmlsd_dict_list[i]
            for goal in step.plan_achieved.keys():
                goals_name.append(goal)
                goals_probs.append(step_probs[goal])
                goals_probs_nrmlsd.append(step_probs_nrmlsd[goal])
                if goal in step.plan_achieved.keys():
                    if step.plan_achieved[goal] == 1:
                        goals_achieved.append(1)
                        goals_costs.append(step.plan_cost[goal] + self.cost_obs_cum_dict[i+1])
                        goals_costs_cumulated.append(self.cost_obs_cum_dict[i+1])
                        goals_seconds.append(step.time[goal])
                    else:
                        goals_achieved.append(0)
                        goals_costs.append(np.nan)
                        goals_costs_cumulated.append(np.nan)
                        goals_seconds.append(np.nan)
                else:
                    goals_achieved.append(0)
                    goals_costs.append(np.nan)
                    goals_costs_cumulated.append(np.nan)
                    goals_seconds.append(np.nan)
            df_action = pd.DataFrame({"observed_action_no": [i+1 for _ in range(len(goals_name))],
                                      "observed_action": [self.observation.obs_file.loc[i,"action"] \
                                                          for _ in range(len(goals_name))],
                                      "goal": goals_name,
                                      "goal_achieved": goals_achieved,
                                      "goal_cost": goals_costs,
                                      "goals_costs_cumulated": goals_costs_cumulated,
                                      "seconds": goals_seconds,
                                      "goal_prob": goals_probs,
                                      "goal_prob_nrmlsd": goals_probs_nrmlsd})
            df_summary_agg = pd.concat([df_summary_agg, df_action])
            i+=1
        df_summary_agg = df_summary_agg.reset_index().iloc[:,1:]
        df_summary_steps = pd.DataFrame()

        for i in range(len(self.steps_observed)):
            df_summary_agg_i = df_summary_agg[df_summary_agg["observed_action_no"] == i+1][["observed_action_no",
                                                                                          "observed_action",
                                                                                          "goal"]]
            df_summary_agg_i = df_summary_agg_i.reset_index().iloc[:,1:]
            action_dfs = pd.DataFrame()
            for goal in self.steps_observed[i].plan.keys():
                steps = [k for k in self.steps_observed[i].plan[goal].keys()]
                actions = [self.steps_observed[i].plan[goal][step] for step in steps]
                costs = [self.env_obs.action_domain_cost
                         [self.env_obs.action_dict[self.env_obs.inverse_action_dict[action]]["action_ungrounded"]]
                         for action in actions]

                action_df = pd.DataFrame({"step": [step+1 for step in steps], "action": actions, "action_cost":costs})
                action_df["goal"] = goal
                action_dfs = pd.concat([action_dfs,action_df])
            try:
                df_summary_agg_i = df_summary_agg_i.merge(action_dfs, on=["goal"], how="left")
                df_summary_agg_i = df_summary_agg_i.reset_index().iloc[:, 1:]
                df_summary_steps = pd.concat([df_summary_steps, df_summary_agg_i])
            except:
                pass
        if len(df_summary_steps) != 0:
            df_summary_steps = df_summary_steps[~(df_summary_steps["action"].isna())]
        df_summary_steps = df_summary_steps.reset_index().iloc[:, 1:]
        df_summary_top = self.observation.obs_file.copy()
        predicted_goals = []
        predicted_goals_no = []
        for i in self.predicted_step.keys():
            predicted_goals.append(str(self.predicted_step[i]).replace("'", ""))
            predicted_goals_no.append(len(self.predicted_step[i]))
        df_summary_top["predicted_goals"] = pd.Series(predicted_goals)
        df_summary_top["predicted_goals_no"] = pd.Series(predicted_goals_no)
        df_summary_top["correct_prediction"] = \
            df_summary_top.apply\
                (lambda x: 1 if str(x["label"]) != "nan" and \
                                str(x["label"]) in str(x["predicted_goals"]).replace("_", "") else 0,
                 axis = 1)
        df_summary_top["observed_action_no"] = self.observation.obs_file.index+1
        df_summary_top = df_summary_top[["observed_action_no"] +
                                        [c for c in df_summary_top.columns if c != "observed_action_no"]]
        total_goals_no = df_summary_agg.groupby("observed_action_no", as_index = False)["goal"].count()
        total_goals_no.rename(columns = {"goal": "total_goals_no"}, inplace = True)
        df_summary_top = df_summary_top.merge(total_goals_no, on="observed_action_no", how="left")
        goals_achieved = df_summary_agg[df_summary_agg["goal_achieved"] == 1]\
                                              .groupby("observed_action_no").\
                                              agg({"goal": lambda x: x.unique(),
                                                   "observed_action": "count",
                                                   "goal_prob_nrmlsd": "max",
                                                   "seconds": "max"})
        goals_achieved.rename(columns = {"goal": "goals_achieved", "observed_action": "goals_achieved_no",
                                         "goal_prob_nrmlsd": "prob"},inplace = True)
        goals_achieved["goals_achieved"] = goals_achieved["goals_achieved"].astype(str).str.replace("'", "")
        df_summary_top = df_summary_top.merge(goals_achieved, on = "observed_action_no", how = "left")
        df_summary_top["time_left"] = np.where(df_summary_top["total_goals_no"] == df_summary_top["goals_achieved_no"],
                                               df_summary_top["diff_t"] - df_summary_top["seconds"], 0)
        df_summary_top["time_left"] = np.where(df_summary_top["time_left"] < 0, 0, df_summary_top["time_left"])
        df_summary_top = df_summary_top[['observed_action_no', 'action', 't', 'goals_remaining','total_goals_no',
                                         'goals_achieved', 'goals_achieved_no','label', 'predicted_goals',
                                         'predicted_goals_no', 'correct_prediction', 'prob', 'diff_t','seconds',
                                         'time_left']]
        df_summary_top.rename(columns = {"action": "observed_action"}, inplace = True)
        return df_summary_top, df_summary_agg, df_summary_steps

    def perform_solve_observed(self, step=-1, print_actions=False, priors=None, beta=1):
        self.env_obs.reset()
        start_time = time.time()
        suffix_cost = 0
        if step == -1:
            step = self.observation.obs_len
        i = 0
        while i < step:
            time_step = self.observation.obs_file.loc[i, "diff_t"]
            step_time = time.time()
            print("-----------")
            print("step:", i, ",time elapsed:", round(step_time - start_time, 2), "s")
            performed_action = self.observation.obs_file.loc[i, "action"].replace(" ", "_")
            print(performed_action + ", " + str(time_step) + " seconds to solve")
            goals_remaining = [goal for goal in self.multi_rl_planner.rl_planner_dict.keys()
                               if goal in self.observation.obs_file.loc[i, "goals_remaining"]]
            print("goals_left: ", goals_remaining)
            action = self.env_obs.inverse_action_dict[performed_action]
            action_ungrounded = self.env_obs.action_dict[action]["action_ungrounded"]
            suffix_cost += self.env_obs.action_domain_cost[action_ungrounded]
            self.cost_obs_cum_dict[i + 1] = suffix_cost
            print("suffix_cost: ", suffix_cost)
            new_state, _, _, _ = self.env_obs.step(action)
            self.multi_rl_planner.set_state(new_state)
            self.multi_rl_planner.solve(print_actions=print_actions, timeout=time_step, goal_list=goals_remaining)
            for goal in self.multi_rl_planner.plan.keys():
                print(self.multi_rl_planner.plan[f"{goal}"])
            self.steps_observed.append(self.task(plan=self.multi_rl_planner.plan,
                                                 plan_achieved=self.multi_rl_planner.plan_achieved,
                                                 plan_cost=self.multi_rl_planner.plan_cost,
                                                 plan_action=self.multi_rl_planner.plan_action,
                                                 solved=self.multi_rl_planner.solved,
                                                 time=self.multi_rl_planner.time))

            result_probs = self._calc_prob(goals_remaining=goals_remaining, step=i + 1, suffix_cost=suffix_cost,
                                           priors=priors, beta=beta)
            for g in goals_remaining:
                if g not in result_probs[0].keys():
                    result_probs[0][g] = 0.00
                    result_probs[1][g] = 0.00
            print(result_probs)
            self.prob_dict_list.append(result_probs[0])
            self.prob_nrmlsd_dict_list.append(result_probs[1])
            self.predicted_step[i+1] = self._predict_step(step=i)

            i+=1
        self.summary_level_1, self.summary_level_2, self.summary_level_3 = self._create_summary()
        print("total time-elapsed: ", round(time.time() - start_time, 2), "s")


    def _calc_prob(self, goals_remaining, suffix_cost, step = 1, priors= None, beta = 1):
        if step == 0:
            print("step must be > 0 ")
            return None
        if priors == None:
            priors_dict = {}
            for key in self.steps_observed[step - 1].plan_achieved.keys():
                priors_dict[key] = 1 / len(goals_remaining)
        else:
            priors_dict = {}
            for key in self.steps_observed[step - 1].plan_achieved.keys():
                priors_dict[key] = priors[key]
        p_observed = {}
        p_optimal = {}
        achieved_goals = [key for key in self.steps_observed[step - 1].plan_achieved.keys()
                          if self.steps_observed[step - 1].plan_achieved[key] == 1]

        for key in achieved_goals:
            optimal_costs = self.steps_optimal.plan_cost[key]
            p_optimal_costs_likeli = np.exp(-beta * optimal_costs)
            p_optimal[key] = priors_dict[key] * p_optimal_costs_likeli
            observed_costs = self.steps_observed[step - 1].plan_cost[key] + suffix_cost
            p_observed_costs_likeli = np.exp(-beta * observed_costs)
            p_observed[key] = priors_dict[key] * p_observed_costs_likeli
        prob = []
        prob_dict = {}
        for key in achieved_goals:
            prob.append(p_observed[key] / (p_observed[key] + p_optimal[key]))
            prob_dict[key] = p_observed[key] / (p_observed[key] + p_optimal[key])
        prob_normalised_dict = {}
        for i in range(len(prob)):
            key = achieved_goals[i]
            prob_normalised_dict[key] = (prob[i] / (sum(prob)))
            prob_normalised_dict[key] = np.round(prob_normalised_dict[key], 4)
        return prob_dict, prob_normalised_dict


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
    model.perform_solve_observed(step=8)

