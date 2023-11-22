import os
from pddl import pddl_domain, pddl_problem, pddl_observations
from metric_ff_solver import metric_ff_solver
import hashlib
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import copy
#import prap_model
#import gm_model

def save_model(model, filename):
    path = model.domain_root.domain_path.replace(model.domain_root.domain_path.split("/")[-1], "")
    with open(path + filename, "wb") as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
def load_model(file):
    return pickle.load(open(file, "rb"))
def _clean_literal(literal):
    left_bracket_clean = re.sub("\s*\(\s*", "(", literal)
    right_bracket_clean = re.sub("\s*\)\s*", ")", left_bracket_clean)
    inner_whitespace_clean = re.sub("\s+", " ", right_bracket_clean)
    return inner_whitespace_clean
def _is_action_possible(pddl_action, start_fluents, observation_title):
    start_fluents = [_clean_literal(sf) for sf in start_fluents]
    if len(observation_title.split(" ")) > 1:
        obs_params = [(observation_title.split(" ")[i+1].lower(),
                       pddl_action.action_parameters[i].parameter) for i in range(len(pddl_action.action_parameters))]
        new_action = copy.deepcopy(pddl_action)
        for param in obs_params:
            new_action.action_preconditions = new_action.action_preconditions.replace(param[1], param[0])
        pddl_action = new_action
    precondition_given = _recursive_check_precondition(pddl_action, start_fluents, start_point=True)
    return precondition_given
def _recursive_check_precondition(pddl_action, start_fluents, inside_when = False, start_point = False, key_word = None):
    if start_point:
        parse_string = pddl_action.action_preconditions
        parse_string = "(" + parse_string + ")"
    else:
        parse_string = pddl_action
    string_cleaned_blanks = parse_string.replace("\t", "").replace(" ", "").replace("\n","")
    if string_cleaned_blanks.startswith("(and"):
        key_word = "and"
    elif string_cleaned_blanks.startswith("(or"):
        key_word = "or"
    if key_word in ["and", "or"]:
        is_true_list = []
        split_list = _split_recursive_and_or(parse_string, key_word)
        for split_element in split_list:
            is_true = _recursive_check_precondition(split_element, start_fluents, inside_when = inside_when)
            is_true_list.append(is_true)
        if key_word == "and":
            if all(is_true_list):
                return True
            else:
                return False
        elif key_word == "or":
            if any(is_true_list):
                return True
            else:
                return False
    if key_word is None:
        parse_string = _clean_literal(parse_string)
        if "=" in parse_string and len(_clean_literal(parse_string).split(" ")) > 1:
            parse_str_split = parse_string.split(" ")
            var = parse_str_split[1].replace(" ", "")
            obj = parse_str_split[2].replace(" ", "").replace(")", "")
            if "(not(" in parse_string:
                if var != obj:
                    return True
                else:
                    return False
            else:
                if var == obj:
                    return True
                else:
                    return False
        elif len([op for op in ["=", ">", "<"] if op in parse_string]) == 1 and len(_clean_literal(parse_string).split(" ")) == 1:
            operator = parse_string[1]
            reference_point_action = float(re.findall('\d+', parse_string)[0])
            function_name = re.findall('\(\w+-*_*\w*\)', parse_string)[0]
            problem_state = [_clean_literal(fluent) for fluent in start_fluents if function_name in _clean_literal(fluent)][0]
            problem_number = float(re.findall('\d+', problem_state)[0])
            if operator == "=":
                if problem_number == reference_point_action:
                    return True
                else:
                    return False
            if operator == ">":
                if problem_number > reference_point_action:
                    return True
                else:
                    return False
            if operator == "<":
                if problem_number < reference_point_action:
                    return True
                else:
                    return False
        else:
            if "(not(" in parse_string:
                rm_not = re.findall('\(\w+-*_*\w*[\s*\w+\-*_*\w+\-*_*\w+\-*_*\w+\-*_*]*\)', _clean_literal(parse_string))[0]
                if rm_not not in start_fluents:
                    return True
                else:
                    return False
            else:
                if parse_string in start_fluents:
                    return True
                else:
                    return False

def _split_recursive_and_or(parse_string, key_word):
    split_list = []
    strt_idx = parse_string.find(key_word) + len(key_word)
    new_string = parse_string[strt_idx:]
    strt_idx += new_string.find("(")
    end_idx = len(parse_string) - 1
    parse_back = True
    while end_idx > 0 and parse_back:
        if parse_string[end_idx] == ")":
            parse_string = parse_string[:end_idx]
            parse_back = False
        end_idx -= 1
    c = strt_idx + 1
    parse_bracket = 1
    while c < len(parse_string):
        if parse_string[c] == "(":
            parse_bracket += 1
        if parse_string[c] == ")":
            parse_bracket -= 1
        if parse_bracket == 0:
            split_list.append(parse_string[strt_idx:c + 1])
            strt_idx = c + parse_string[c:].find("(")
            c = strt_idx + 1
            parse_bracket = 1
        c += 1
    return split_list
class gr_model:
    """superclass that solves a goal recognition problem.
    """
    def __init__(self, domain_root, goal_list, obs_action_sequence, planner = "ff_2_1"):
        """
        :param domain_root: pddl_domain on which the goal recognition problem is solved.
        :param goal_list: list of pddl_problems, which represent the goals to assign probabilites to.
        :param obs_action_sequence: agents observations of type _pddl_observations.
        :param planner: name of executable planner, here default ff_2_1 (MetricFF Planner version 2.1)
        """
        self.domain_root = domain_root
        self.changed_domain_root = None
        self.crystal_island = domain_root.name == "crystal_island"
        self.observation = obs_action_sequence
        self.crystal_island_solution = self._crystal_island_solution()
        self.goal_list = [goal_list]
        self.planner = planner
        self.steps_observed = []
        self.prob_dict_list = []
        self.prob_nrmlsd_dict_list = []
        self.steps_optimal = metric_ff_solver(planner = self.planner)
        self.mp_seconds = None
        self.predicted_step = {}
        self.hash_code = self._create_hash_code()
        self.path_error_env = "/home/mwiubuntu/error_write/"
        self.error_write_files = os.listdir(self.path_error_env)
    def _crystal_island_solution(self):
        if self.crystal_island:
            self._crystal_island_default_path = self.domain_root.domain_path
            self._crystal_island_ecoli_path = self.domain_root.domain_path.replace(".pddl", "_ecoli.pddl")
            self._crystal_island_salmonellosis_path = self.domain_root.domain_path.replace(".pddl", "_salmonellosis.pddl")
            file_name_obs = self.observation.observation_path.split("/")[-1]
            change_domain = ""
            for split_element in self.domain_root.domain_path.split("/")[:-1]:
                change_domain += split_element + "/"
            if "_E.coli" in file_name_obs:
                solution = "ecoli"
                new_domain = self.domain_root.domain_path.split("/")[-1].replace(".pddl", "_ecoli.pddl")
                change_domain += new_domain
                self.changed_domain_root = self.domain_root
                self.domain_root = pddl_domain(change_domain)
            elif "_Salmonellosis" in file_name_obs:
                solution =  "salmonellosis"
                new_domain = self.domain_root.domain_path.split("/")[-1].replace(".pddl", "_salmonellosis.pddl")
                change_domain += new_domain
                self.changed_domain_root = self.domain_root
                self.domain_root = pddl_domain(change_domain)
            else:
                solution =  None #domain remains on default (salmonellosis)
        else:
            solution =  None
        return solution
    def _create_hash_code(self):
        action_list = list(self.domain_root.action_dict.keys())
        action_list.sort()
        input_string = ""
        for item in action_list:
            input_string += item
        input_string = input_string.encode()
        h = hashlib.new("sha224")
        h.update(input_string)
        return h.hexdigest()
    def perform_solve_optimal(self, multiprocess=True, type_solver='3', weight='1', timeout=90):
        """
        RUN before perform_solve_observed.
        Solves the optimal plan for each goal in goal_list.
        :param multiprocess: if True, all problems (goals) are solved in parallel
        :param type_solver: option for type solver in Metricc-FF Planner, however only type_solver = '3' ("weighted A*) is
         considered
        :param weight: weight for type_solver = '3' ("weighted A*); weight = '1' resolves to unweighted A*
        :param timeout: after specified timeout is reached, all process are getting killed.
        """
        self.chosen_optimal_timeout = timeout
        start_time = time.time()
        self.steps_optimal.solve(self.domain_root, self.goal_list[0], multiprocess=multiprocess,
                                 type_solver=type_solver, weight=weight, timeout=timeout)
        print("total time-elapsed: ", round(time.time() - start_time, 2), "s")
        if multiprocess:
            self.mp_seconds = round(time.time() - start_time, 2)
    def _predict_step(self, step):
        #empty shell due to missing perform_solve_observed, just defined here for subclasses
        dict_proba = self.prob_nrmlsd_dict_list[step]
        most_likeli = 0
        key_most_likeli = []
        for key in list(dict_proba.keys()):
            if dict_proba[key] > most_likeli:
                key_most_likeli = [key]
                most_likeli = dict_proba[key]
            elif dict_proba[key] == most_likeli:
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
            goals_seconds = []
            goals_probs = []
            goals_probs_nrmlsd = []
            step_probs = self.prob_dict_list[i]
            step_probs_nrmlsd = self.prob_nrmlsd_dict_list[i]
            for goal in step.problem:
                goals_name.append(goal.name)
                goals_probs.append(step_probs[goal.name])
                goals_probs_nrmlsd.append(step_probs_nrmlsd[goal.name])
                if goal.name in step.plan_achieved.keys():
                    if step.plan_achieved[goal.name] == 1:
                        goals_achieved.append(1)
                        goals_costs.append(step.plan_cost[goal.name])
                        goals_seconds.append(step.time[goal.name])
                    else:
                        goals_achieved.append(0)
                        goals_costs.append(np.nan)
                        goals_seconds.append(np.nan)
                else:
                    goals_achieved.append(0)
                    goals_costs.append(np.nan)
                    goals_seconds.append(np.nan)
            df_action = pd.DataFrame({"observed_action_no": [i+1 for _ in range(len(goals_name))],
                                      "observed_action": [self.observation.obs_file.loc[i,"action"] \
                                                          for _ in range(len(goals_name))],
                                      "goal": goals_name,
                                      "goal_achieved": goals_achieved,
                                      "goal_cost": goals_costs,
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
                costs = [self.steps_observed[i].domain.action_dict[action.split(" ")[0]].action_cost\
                         for action in actions]
                action_df = pd.DataFrame({"step": [step+1 for step in steps], "action": actions, "action_cost":costs})
                action_df["goal"] = goal
                action_dfs = pd.concat([action_dfs,action_df])
            try:
                df_summary_agg_i = df_summary_agg_i.merge(action_dfs, on = ["goal"], how = "left")
                df_summary_agg_i = df_summary_agg_i.reset_index().iloc[:, 1:]
                df_summary_steps = pd.concat([df_summary_steps,df_summary_agg_i])
            except:
                pass
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
                (lambda x: 1 if str(x["label"]) in str(x["predicted_goals"]).replace("_", "") else 0,
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
        df_summary_top = df_summary_top[['observed_action_no', 't', 'action', 'goals_remaining','total_goals_no',
                                         'goals_achieved', 'goals_achieved_no','label', 'predicted_goals',
                                         'predicted_goals_no', 'correct_prediction', 'prob', 'diff_t','seconds',
                                         'time_left']]
        return df_summary_top, df_summary_agg, df_summary_steps
    def plot_prob_goals(self, adapt_y_axis, figsize_x=8, figsize_y=5):
        """
        RUN perform_solve_observed BEFORE.
        plots probability  for each goal to each step (specified perform_solve_observed) in of obs_action_sequence
        :param figsize_x: sets size of x-axis (steps)
        :param figsize_y: sets size of y-axis (probability)
        :param adapt_y_axis: if True plot zooms in into necessary range of [0,0.25,0.5,0.75,1]. Default is False.
        """
        goal_name = [self.goal_list[0][i].name for i in range(len(self.goal_list[0]))]
        df = pd.DataFrame()
        for step in range(len(self.prob_nrmlsd_dict_list)):
            cur_df = {}
            cur_df["step"] = [step + 1]
            for key in self.prob_nrmlsd_dict_list[step].keys():
                cur_df[key] = [self.prob_nrmlsd_dict_list[step][key]]
            df_step = pd.DataFrame(cur_df)
            df = pd.concat([df, df_step])
        df = df.reset_index().iloc[:, 1:]
        plt.figure(figsize=(figsize_x, figsize_y))
        for goal in goal_name:
            plt.plot(df["step"], df[goal], label=goal)
        plt.legend()
        plt.xticks(range(1, len(self.steps_observed) + 1))
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.xlim(1, len(self.steps_observed))
        if adapt_y_axis:
            max_prob = 0
            for step_dict in self.prob_nrmlsd_dict_list:
                #max_prob_step = max([step_dict[key] for key in list(step_dict.keys())])
                max_prob_step = max(df[[x for x in df.columns if x != "step"]].max())
            if max_prob_step > max_prob:
                max_prob = max_prob_step
            ticks = np.array([0, 0.25, 0.5, 0.75, 1])
            plt.ylim(np.min(ticks[max_prob > ticks]), np.min(ticks[max_prob < ticks]))
        else:
            plt.ylim(0, 1)
        plt.grid()
        plt.show()
if __name__ == '__main__':
    toy_example_domain = pddl_domain('domain.pddl')
    problem_a = pddl_problem('problem_A.pddl')
    problem_b = pddl_problem('problem_B.pddl')
    problem_c = pddl_problem('problem_C.pddl')
    problem_d = pddl_problem('problem_D.pddl')
    problem_e = pddl_problem('problem_E.pddl')
    problem_f = pddl_problem('problem_F.pddl')
    toy_example_problem_list = [problem_a, problem_b, problem_c, problem_d, problem_e, problem_f]
    obs_toy_example = pddl_observations('Observations.csv')
    model = gr_model(toy_example_domain, toy_example_problem_list, obs_toy_example)
    print(model.hash_code)
    model.perform_solve_optimal()
    print(model.steps_optimal.plan)
    print(model.path_error_env)
    print(model.error_write_files)