from pddl import pddl_domain, pddl_problem, pddl_observations
from metric_ff_solver import metric_ff_solver
import gr_model
import re
import os
import shutil
import time
import numpy as np
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
    def _split_recursive_and_or(self, parse_string, key_word):
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
    def _recursive_effect_check(self, parse_string, zipped_parameters, inside_when=False, key_word=None):
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
            split_list = self._split_recursive_and_or(parse_string, key_word)
            if inside_when:
                for split_element in split_list:
                    is_true, effect = self._recursive_effect_check(split_element, zipped_parameters, inside_when=inside_when)
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
                    is_true, effect = self._recursive_effect_check(split_element, zipped_parameters, inside_when=inside_when)
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
            is_true, effect = self._recursive_effect_check(new_string, zipped_parameters, inside_when=True)
            if is_true:
                [effects.append(e) for e in effect if e not in effects]
                is_true, effect = self._recursive_effect_check(consequence, zipped_parameters)
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
    def _call_effect_check(self, parse_string,zipped_parameters):
        _, effects = self._recursive_effect_check(parse_string, zipped_parameters)
        return [effect for effect in effects if effect != "_"]
    def _create_obs_goal(self, goal_idx = 0, step = 1):
        goal = self.goal_list[0][goal_idx]
        new_goal = f"(define (problem {goal.name})\n"
        new_goal = new_goal + f"(:domain {self.domain_root.name})"
        new_goal = new_goal + "\n(:objects)"
        new_goal = new_goal + "\n(:init "
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
        action_step = self.observation.obs_action_sequence.loc[step-1]
        action_title = action_step.split(" ")[0]
        goal = self.goal_list[step-1][goal_idx]
        if len(action_step.split(" ")) > 1:
            action_objects = action_step.split(" ")[1:]
        else:
            action_objects = []
        action_objects = [obj.lower() for  obj in action_objects]
        domain = self.domain_temp
        pddl_action = domain.action_dict[action_title]
        action_parameters = [param.parameter for param in pddl_action.action_parameters]
        zipped_parameters = list(zip(action_objects, action_parameters))
        effects = self._call_effect_check(pddl_action.action_effects, zipped_parameters)
        functions = [function[1:-1] for function in domain.functions]
        new_start_fluents = goal.start_fluents
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
            else:
                effect = self._clean_effect(effect)
                if "(not(" in effect:
                    opposite = re.findall("\([\s*\w*\-*\s*]*\)", effect)[0]
                else:
                    opposite = "(not" + effect + ")"
                remember_index = -1
                for i in range(len(new_start_fluents)):
                    if opposite == self._clean_effect(new_start_fluents[i]):
                        remember_index = i
                if remember_index != -1:
                    new_start_fluents[remember_index] = effect
                else:
                    new_start_fluents.append(effect)
        return [fl for fl in new_start_fluents if "(not(" not in fl]
    def _clean_effect(self, effect):
        left_bracket_clean = re.sub("\s*\(\s*", "(", effect)
        right_bracket_clean = re.sub("\s*\)\s*", ")", left_bracket_clean)
        inner_whitespace_clean = re.sub("\s+", " ", right_bracket_clean)
        return inner_whitespace_clean
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
        for goal in range(len(self.goal_list[0])):
            add_problem = True
            goal_string = self._create_obs_goal(goal, step)
            with open(path + f"/goal_{goal}_obs_step_{step}.pddl", "w") as new_goal:
                new_goal.write(goal_string)
            if last_step:
                check = pddl_problem(path + f"/goal_{goal}_obs_step_{step}.pddl")
                if len([f for f in check.goal_fluents if f in check.start_fluents]) == len(check.goal_fluents):
                    self.at_goal = check
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
        if self.at_goal is None:
            optimal_costs = [self.steps_optimal.plan_cost[key] for key in keys]
            optimal_costs = np.array(optimal_costs)
            suffix_costs = [self.steps_observed[step - 1].plan_cost[key] for key in keys]
            suffix_costs = np.array(suffix_costs)
            p_observed = optimal_costs / (self.cost_obs_cum + suffix_costs)
        else:
            print("here ", self.at_goal.name)
            dict_last_step = {}
            keys_not_at_goal = keys
            optimal_costs = [self.steps_observed[-1].plan_cost[key] for key in keys_not_at_goal]
            optimal_costs = np.array(optimal_costs)
            suffix_costs = [self.steps_observed[step - 1].plan_cost[key] for key in keys]
            suffix_costs = np.array(suffix_costs)
            p_observed_help = optimal_costs / (self.cost_obs_cum + suffix_costs)
            for key in range(len(keys_not_at_goal)):
                dict_last_step[keys_not_at_goal[key]] = p_observed_help[key]
            dict_last_step[self.at_goal.name] = self.steps_optimal.plan_cost[self.at_goal.name] / self.cost_obs_cum
            print(dict_last_step)
            p_observed = []
            keys.append(self.at_goal.name)
            for key in keys:
                print(key)
                p_observed.append(dict_last_step[key])
            p_observed = np.array(p_observed)
            print(p_observed)
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
        #for key in [key for key in list(self.steps_optimal.plan_cost.keys()) if key not in prob_normalised_dict.keys()]:
         #   prob_dict[key] = 0
          #  prob_normalised_dict[key] = 0
        return prob_dict, prob_normalised_dict

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
        for i in range(step):
            time_step = self.observation.obs_file.loc[i,"diff_t"]
            step_time = time.time()
            print("\nstep:", i+1, ",time elapsed:", round(step_time - start_time,2), "s")
            self._add_step(i+1)
            print(self.observation.obs_file.loc[i,"action"] + ", " + str(time_step) + " seconds to solve")
            try:
                task = metric_ff_solver(planner = self.planner)
                task.solve(self.domain_temp,self.goal_list[i+1], multiprocess = multiprocess,timeout=time_step)
            except:
                print("timeout")
            self.steps_observed.append(task)
            result_probs = self._calc_prob(i + 1)
            self.prob_dict_list.append(result_probs[0])
            self.prob_nrmlsd_dict_list.append(result_probs[1])
            self.predicted_step[i+1] = self._predict_step(step= i)
        print("total time-elapsed: ", round(time.time() - start_time,2), "s")
        for i in range(step,0,-1):
            self._remove_step(i)
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
