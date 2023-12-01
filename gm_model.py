from pddl import pddl_domain, pddl_problem, pddl_observations
from metric_ff_solver import metric_ff_solver
import threading
import gr_model
import re
import os
import shutil
import time
import numpy as np
import psutil
import logging
import matplotlib.pyplot as plt
import hashlib
class gm_model(gr_model.gr_model):
    """class that solves a goal recognition problem according to the vanilla plain approach Goal Mirroring (GM)
     by Vered et al., 2016.
    """
    def __init__(self, domain_root, goal_list, obs_action_sequence, planner = "ff_2_1"):
        """
        :param domain_root: pddl_domain on which the goal recognition problem is solved.
        :param goal_list: list of pddl_problems, which represent the goals to assign probabilites to.
        :param obs_action_sequence: agents observations of type _pddl_observations.
        :param planner: name of executable planner, here default ff_2_1 (MetricFF Planner version 2.1)
        """
        super().__init__(domain_root, goal_list, obs_action_sequence, planner)
        self.at_goal = None
        self.cost_obs_cum_dict = {}
        self.model_type = "gm_model"
    def _recursive_effect_check(self, parse_string, zipped_parameters, start_fluents,
                                inside_when=False, key_word=None, is_consequence = False):
        effects = []
        string_cleaned_blanks = parse_string.replace("\t", "").replace(" ", "").replace("\n", "")
        if string_cleaned_blanks.startswith("(when"):
            key_word = "when"
        elif string_cleaned_blanks.startswith("(and"):
            key_word = "and"
        elif string_cleaned_blanks.startswith("(or"):
            key_word = "or"
        if key_word in ["and", "or"]:
            is_true_list = []
            split_list = gr_model._split_recursive_and_or(parse_string, key_word)
            if inside_when:
                for split_element in split_list:
                    is_true, effect = self._recursive_effect_check(split_element, zipped_parameters, start_fluents,
                                                                   inside_when=inside_when,
                                                                   is_consequence = is_consequence)
                    is_true_list.append(is_true)
                    [effects.append(e) for e in effect if e not in effects]
                if key_word == "and":
                    if all(is_true_list):
                        return True, effects
                    else:
                        return False, "_"
                elif key_word == "or":
                    if any(is_true_list):
                        return True, effects
                    else:
                        return False, "_"
            else:
                for split_element in split_list:
                    is_true, effect = self._recursive_effect_check(split_element, zipped_parameters,
                                                                   start_fluents,
                                                                   inside_when=inside_when,
                                                                   is_consequence= is_consequence)
                    if is_true:
                        [effects.append(e) for e in effect if e not in effects]
                return True, effects
        if key_word == "when":
            new_string = parse_string[parse_string.find("when"):]
            new_string = new_string[new_string.find("("):]
            parse_bracket = 1
            c = 1
            parse = True
            while (c < len(new_string) and parse):
                if new_string[c] == "(":
                    parse_bracket += 1
                if new_string[c] == ")":
                    parse_bracket -= 1
                if parse_bracket == 0:
                    consequence = new_string[c + 1:]
                    cons_idx = 0
                    parse_con = True
                    while cons_idx < len(consequence) and parse_con:
                        if consequence[cons_idx] == "(":
                            consequence = consequence[cons_idx:]
                            parse_con = False
                        cons_idx += 1
                    cons_idx = len(consequence) - 1
                    parse_con = True
                    while cons_idx > 0 and parse_con:
                        if consequence[cons_idx] == ")":
                            consequence = consequence[:cons_idx]
                            parse_con = False
                        cons_idx -= 1
                    new_string = new_string[:c + 1]
                    parse = False
                c += 1
            is_true, effect = self._recursive_effect_check(new_string, zipped_parameters,
                                                           start_fluents, inside_when=True,
                                                           is_consequence = is_consequence)
            if is_true:
                [effects.append(e) for e in effect if e not in effects]
                is_true, effect = self._recursive_effect_check(consequence, zipped_parameters, start_fluents,
                                                               inside_when= inside_when, is_consequence = True)
                [effects.append(e) for e in effect if e not in effects]
                return True, effects
            else:
                return False, "_"
        if key_word is None:
            if "?" in parse_string and "=" in parse_string:
                parse_str_split = parse_string.split(" ")
                var = parse_str_split[1].replace(" ", "")
                obj = parse_str_split[2].replace(" ", "").replace(")", "")
                tuple_param = [zip_param for zip_param in zipped_parameters if zip_param[1] == var][0]
                if tuple_param[0] == obj and tuple_param[1] == var:
                    return True, "_"
                else:
                    return False, "_"
            elif "?" in parse_string and len([op for op in ["=", ">", "<"] if op in parse_string]) == 0:
                parse_str_split = parse_string.split(" ")
                var_s = [var.replace(" ", "").replace("\t", "").replace("\n", "").replace("(", "").replace(")", "")
                         for var in parse_str_split if "?" in var]
                for var in var_s:
                    tuple_param = [zip_param for zip_param in zipped_parameters if zip_param[1] == var][0]
                    parse_string = parse_string.replace(var, tuple_param[0])
                return True, [parse_string] + effects
            elif "?" not in parse_string and len([op for op in ["=", ">", "<"] if op in parse_string]) > 0:
                cleaned_comp_str = gr_model._clean_literal(parse_string)
                comp_number = float(re.findall('\d+\d*\.*\d*', cleaned_comp_str)[0])
                op = cleaned_comp_str[1]
                func_var = re.findall('\(\w*-*_*\w*-*_*\w*-*_*\w*\)', cleaned_comp_str)[0]
                state_fl = gr_model._clean_literal([fl for fl in start_fluents if func_var in fl][0])
                state_number = float(re.findall('\d+\d*\.*\d*', state_fl)[0])
                #the following makes this method not general applicable
                if func_var == "(tests-remaining)":
                    state_number -= 1
                if op == "=":
                    if state_number == comp_number:
                        return True, "_"
                    else:
                        return False, "_"
                if op == ">":
                    if state_number > comp_number:
                        return True, "_"
                    else:
                        return False, "_"
                if op == "<":
                    if state_number < comp_number:
                        return True, "_"
                    else:
                        return False, "_"
            else:
                if inside_when and not is_consequence:
                    cleaned_parse_str = gr_model._clean_literal(parse_string)
                    if "(not" not in cleaned_parse_str:
                        if cleaned_parse_str in [gr_model._clean_literal(fl) for fl in start_fluents]:
                            return True, "_"
                        else:
                            return False, "_"
                    else:
                        rm_not = re.findall('\(\w+-*_*\w*[\s*\w+\-*_*\w+\-*_*\w+\-*_*\w+\-*_*]*\)',
                                            cleaned_parse_str)[0]
                        if rm_not in [gr_model._clean_literal(fl) for fl in start_fluents]:
                            return False, "_"
                        else:
                            return True, "_"
                else:
                    idx = 0
                    parse = True
                    while idx < len(parse_string) and parse:
                        if parse_string[idx] == "(":
                            parse_string = parse_string[idx:]
                            parse = False
                        idx += 1
                    idx = len(parse_string) - 1
                    parse = True
                    while idx > 0 and parse:
                        if parse_string[idx] == ")":
                            parse_string = parse_string[:idx + 1]
                            parse = False
                        idx -= 1
                    return True, [parse_string] + effects
    def _call_effect_check(self, parse_string,zipped_parameters, start_fluents):
        _, effects = self._recursive_effect_check(parse_string, zipped_parameters, start_fluents)
        return [effect for effect in effects if effect != "_"]
    def _create_obs_goal(self, goal_idx = 0, step = 1):
        goal = self.goal_list[-1][goal_idx]
        new_goal = f"(define (problem {goal.name})\n"
        new_goal = new_goal + f"(:domain {self.domain_root.name})"
        new_goal = new_goal + "\n(:objects)"
        new_goal = new_goal + "\n(:init "
        #print("step from _create_obs_goal: ", step)
        start_fluents = self._create_new_start_fluents(goal_idx, step)
        for start_fluent in start_fluents:
            new_goal = new_goal + "\n" + start_fluent
        new_goal = new_goal + "\n)"
        if len(goal.goal_fluents) > 1:
            new_goal = new_goal + "\n(:goal (and "
            for goal_fluent in goal.goal_fluents:
                new_goal = new_goal + "\n" + goal_fluent
            new_goal = new_goal + ")\n)"
        else:
            new_goal = new_goal + "\n(:goal " + goal.goal_fluents[0] + ")"
        new_goal = new_goal +f"\n(:metric minimize ({goal.metric_min_func}))\n)"
        return new_goal
    def _create_new_start_fluents(self, goal_idx, step = 1):
        #print(step)
        action_step = self.observation.obs_action_sequence.loc[step-1]
        action_title = action_step.split(" ")[0]
        goal = self.goal_list[step-1][goal_idx] #some unkown bug
        #goal = pddl_problem(self.goal_list[step-1][goal_idx].problem_path)
        if len(action_step.split(" ")) > 1:
            action_objects = action_step.split(" ")[1:]
        else:
            action_objects = []
        action_objects = [obj.lower() for  obj in action_objects]
        domain = self.domain_temp
        pddl_action = domain.action_dict[action_title]
        action_parameters = [param.parameter for param in pddl_action.action_parameters]
        zipped_parameters = list(zip(action_objects, action_parameters))
        effects = self._call_effect_check(pddl_action.action_effects, zipped_parameters, goal.start_fluents)
        #if "TALK" in action_title:
        #print(step, action_step)
        #print("effects: ", effects)
        functions = [function[1:-1] for function in domain.functions]
        new_start_fluents = [x for x in goal.start_fluents] # = would lead to pointer identity
        for effect in effects:
            effect_is_func = len([function for function in functions if function in effect]) > 0
            if effect_is_func:
                identified_func = [function for function in functions if function in effect][0]
                idx_func_start_fluents = [i for i in range(len(new_start_fluents )) if identified_func in new_start_fluents[i]][0]
                replace_func = new_start_fluents[idx_func_start_fluents]
                curr_number = re.findall(r'\d+\.*\d*', replace_func)[0]
                effect_change_number = re.findall(r'\d+\.*\d*', effect)[0]
                if "increase" in effect:
                    new_start_fluents[idx_func_start_fluents] = replace_func.replace(curr_number,str(float(curr_number) +
                                                                                float(effect_change_number)))
                elif "decrease" in effect:
                    new_start_fluents[idx_func_start_fluents] = replace_func.replace(curr_number,str(float(curr_number) -
                                                                              float(effect_change_number)))
                if identified_func == goal.metric_min_func:
                    self.cost_obs_cum = float(re.findall(r'\d+\.*\d*', new_start_fluents[idx_func_start_fluents])[0])
                    self.cost_obs_cum_dict[step] = self.cost_obs_cum
            else:
                effect = gr_model._clean_literal(effect)
                if "(not(" in effect:
                    opposite = re.findall("\([\s*\w*\-*\s*]*\)", effect)[0]
                else:
                    opposite = "(not" + effect + ")"
                remember_index = -1
                for i in range(len(new_start_fluents)):
                    if opposite == gr_model._clean_literal(new_start_fluents[i]):
                        remember_index = i
                if remember_index != -1:
                    new_start_fluents[remember_index] = effect
                else:
                    new_start_fluents.append(effect)
            remove_not_fluents = [fl for fl in new_start_fluents if "(not(" not in fl]
            result_fluents = []
            for fl in remove_not_fluents:
                if fl not in result_fluents:
                    result_fluents.append(fl)
        return result_fluents
    def _add_step(self, step= 1):
        last_step = self.observation.obs_len == step
        path = self.domain_root.domain_path.replace(self.domain_root.domain_path.split("/")[-1],"") + "temp"
        if not os.path.exists(path):
            os.mkdir(path)
        if step == 1:
            domain_string = self.domain_root.domain
            with open(path + "/domain_gm_model.pddl", "w") as new_domain:
               new_domain.write(domain_string)
            self.domain_temp = pddl_domain(path + "/domain_gm_model.pddl")
        new_goal_list = []

        for goal in range(len(self.goal_list[-1])):
            add_problem = True
            goal_string = self._create_obs_goal(goal, step)
            with open(path + f"/goal_{goal}_obs_step_{step}.pddl", "w") as new_goal:
                new_goal.write(goal_string)
            #if last_step:
            check = pddl_problem(path + f"/goal_{goal}_obs_step_{step}.pddl")
            if len([f for f in check.goal_fluents if f in check.start_fluents]) == len(check.goal_fluents):
                self.at_goal = check
                print(f"-----------at---------{self.at_goal.name}-----------")
                os.remove(check.problem_path)
                add_problem = False
            if add_problem:
                new_goal_list.append(pddl_problem(path + f"/goal_{goal}_obs_step_{step}.pddl"))
        self.goal_list.append(new_goal_list)
        if step == 1:
            shutil.copy(f'{self.planner}', path + f'/{self.planner}')
    def _remove_step(self, step = 1):
        path = self.domain_temp.domain_path.replace(self.domain_temp.domain_path.split("/")[-1],"")
        if os.path.exists(path):
            path_goal = [x.problem_path for x in self.goal_list[step]]
            self.goal_list = self.goal_list[0:step]
            for goal in path_goal:
                os.remove(goal)
            if step == 1:
                path_domain = self.domain_temp.domain_path
                os.remove(path_domain)
                os.remove(path + f"/{self.planner}")
                [os.remove(f) for f in os.listdir(path)]
                os.rmdir(path)
    def _calc_prob(self, step=1):
        if step == 0:
            print("step must be > 0 ")
            return None
        keys = list(self.steps_observed[step - 1].plan_cost.keys())
        optimal_costs = [self.steps_optimal.plan_cost[key] for key in keys]
        optimal_costs = np.array(optimal_costs)
        suffix_costs = [self.steps_observed[step - 1].plan_cost[key] for key in keys]
        suffix_costs = np.array(suffix_costs)
        p_observed = optimal_costs / (self.cost_obs_cum_dict[step] + suffix_costs)
        p_observed = np.round(p_observed, 4)
        prob_dict = {}
        sum_probs = 0
        for i in range(len(p_observed)):
            key = keys[i]
            prob_dict[key] = p_observed[i]
            sum_probs += p_observed[i]
        prob_normalised_dict = {}
        for i in range(len(p_observed)):
            key = keys[i]
            prob_normalised_dict[key] = (prob_dict[key] / sum_probs)
            prob_normalised_dict[key] = np.round(prob_normalised_dict[key], 4)
        return prob_dict, prob_normalised_dict
    def _thread_solve(self, i, multiprocess, time_step):
        #print([goal.problem_path for goal in self.goal_list[i + 1]])
        self.task_thread_solve = metric_ff_solver(planner=self.planner)
        if len(self.goal_list[i + 1]) > 0:
            if len(self.domain_root.domain_path.split("/")) == 1:
                base_domain = self.domain_root.domain_path.replace(".pddl","")
            else:
                base_domain = self.domain_root.domain_path.split("/")[-1].replace(".pddl","")
            try:
                if self.crystal_island:
                    self.task_thread_solve.solve(self.domain_temp, self.goal_list[i + 1], multiprocess=multiprocess,
                                                 timeout=time_step,
                                                 # base_domain= self.domain_root.domain_path.replace(".pddl",""),
                                                 base_domain= base_domain,
                                                 observation_name= self.observation.observation_path.split("/")[-2]+ "-" +
                                                                   self.observation.name)
                else:
                    self.task_thread_solve.solve(self.domain_temp, self.goal_list[i + 1], multiprocess=multiprocess,
                                             timeout=time_step,
                                             #base_domain= self.domain_root.domain_path.replace(".pddl",""),
                                             base_domain = base_domain,
                                             observation_name = self.observation.name)
                                             #, observation_name= self.observation.name)
            except:
                error_message = f"""------------------------------------------------------
                                    model_type {self.model_type},
                                    file {self.observation.observation_path}, 
                                    domain: {self.domain_temp.domain_path}, 
                                    step: {i+1}"""
                logging.exception(error_message)
    def perform_solve_observed(self, step = -1, multiprocess = True):
        """
        BEFORE running this, RUN perform_solve_optimal!
        Solves the transformed pddL_domain and list of pddl_problems (goal_list) for specified steps
        from given obs_action_sequence.
        :param step: specifies how many observations in observation sequence get solved.
                     If set to -1 (default) entire observation sequence is solved
        :param multiprocess: if True, all transformed problems (goals) of one step are solved in parallel

        UNDER CONSTRUCTION - set timeout to time in obs_action_sequence

        """
        start_time = time.time()
        if step == -1:
            step = self.observation.obs_len
        i = 0
        while i < step:
            logging.info(f"step: {i + 1}")
            time_step = self.observation.obs_file.loc[i,"diff_t"]
            step_time = time.time()
            print("\nstep:", i+1, ",time elapsed:", round(step_time - start_time,2), "s")
            self._add_step(i+1)
            print(self.observation.obs_file.loc[i,"action"] + ", " + str(time_step) + " seconds to solve")
            try:
                #time_step = 3
                t = threading.Thread(target=self._thread_solve,
                                     args=[i, multiprocess, time_step])
                t.start()
            except:
                pass
            check_failure_t = time.time()
            failure = False
            background_loop = True
            s = 3
            while background_loop:
                time.sleep(0.7)
                #print("task.solved: ",self.task_thread_solve.solved )
                if not (self.task_thread_solve.solved == 0 and (time.time() - check_failure_t <= time_step + 10) and len(
                    self.goal_list[i + 1]) > 0):
                    background_loop = False
                if (time.time() - check_failure_t >= time_step + 10):
                    print("timeout reached")
                    while s > 0:
                        print("continue in ", s)
                        s -= 1
                        time.sleep(1)
                        [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
                    failure = True
            if not failure:
                self.steps_observed.append(self.task_thread_solve)
            else:
                [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
                print("failure, read in files ")
                failure_task = metric_ff_solver()
                failure_task.problem = self.goal_list[i + 1]
                failure_task.domain = self.domain_temp
                failure_task.domain_path = failure_task.domain.domain_path
                print("failure_task.domain_path, ", failure_task.domain_path)
                path = ""
                for path_pc in failure_task.domain_path.split("/")[:-1]:
                    path = path + path_pc + "/"
                print(path)
                for goal in failure_task.problem:
                    key = goal.name
                    print(key)
                    file_path = path + f"output_goal_{key}.txt"
                    print(file_path)
                    if os.path.exists(file_path):
                        print(file_path, " exists")
                        f = open(file_path, "r")
                        failure_task.summary[key] = f.read()
                        failure_task.plan[key] = failure_task._legal_plan(failure_task.summary[key], file_path)
                        failure_task.plan_cost[key] = failure_task._cost(failure_task.summary[key], file_path)
                        failure_task.plan_achieved[key] = 1
                        failure_task.time[key] = failure_task._time_2_solve(failure_task.summary[key], file_path)
                        os.remove(file_path)
                self.steps_observed.append(failure_task)
            i += 1
        print("total time-elapsed: ", round(time.time() - start_time,2), "s")
        for i in range(step):
            result_probs = self._calc_prob(i + 1)
            for g in self.goal_list[i + 1]:
                if g.name not in result_probs[0].keys():
                    result_probs[0][g.name] = 0.00
                    result_probs[1][g.name] = 0.00
            self.prob_dict_list.append(result_probs[0])
            self.prob_nrmlsd_dict_list.append(result_probs[1])
            self.predicted_step[i + 1] = self._predict_step(step=i)
        for i in range(step,0,-1):
            self._remove_step(i)
        self.summary_level_1,self.summary_level_2, self.summary_level_3 = self._create_summary()
    def plot_prob_goals(self, figsize_x=8, figsize_y=5, adapt_y_axis=True):
        return super().plot_prob_goals(figsize_x=figsize_x, figsize_y=figsize_y, adapt_y_axis=adapt_y_axis)
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
    model = gm_model(toy_example_domain, toy_example_problem_list, obs_toy_example)
    print(model.hash_code)
    model.perform_solve_optimal()
    model.perform_solve_observed()
    print(model.predicted_step)
    print(model.prob_nrmlsd_dict_list)
    print(model.observation.name)
