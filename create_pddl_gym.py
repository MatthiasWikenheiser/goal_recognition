from pddl import *
import os
import gym
import sys
from itertools import product
import time


def _recursive_goal_check(goal_string, fluents, start_point = False):
    string_cleaned_blanks = goal_string.replace("\t", "").replace(" ", "").replace("\n", "")
    if start_point:
        fluents = [_clean_literal(fl) for fl in fluents]
    if string_cleaned_blanks.startswith("(and"):
        key_word = "and"
    elif string_cleaned_blanks.startswith("(or"):
        key_word = "or"
    else:
        key_word = None
    if key_word in ["and", "or"]:
        is_true_list = []
        split_list = _split_recursive_and_or(goal_string, key_word)
        for split_element in split_list:
            is_true = _recursive_goal_check(split_element, fluents)
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
        goal_string = _clean_literal(goal_string)
        if "(not(" in goal_string:
            check_str = goal_string.replace("(not", "")[:-1]
            if check_str in fluents:
                return False
            else:
                return True
        elif "(=(" in goal_string or "(>(" in goal_string or "(<(" in goal_string:
            if "(=(" in goal_string:
                op = 0
                operator_string = "(=("
            elif "(>(" in goal_string:
                op = 1
                operator_string = "(>("
            elif "(<(" in goal_string:
                op = -1
                operator_string = "(<("
            func = goal_string.replace(operator_string,"")
            func = "("+func[:func.find(")")]+")"
            goal_value_str = re.findall(r'\d.*\d*', goal_string)[0].replace("(", "").replace(")", "")
            goal_value_fl = float(goal_value_str)
            current_value_str = [f for f in fluents if func in f][0]
            current_value_str = re.findall(r'\d.*\d*', current_value_str)[0].replace("(", "").replace(")", "")
            current_value = float(current_value_str)

            if op == 0:
                if current_value == goal_value_fl:
                    return True
                else:
                    return False
            if op == 1:
                if current_value > goal_value_fl:
                    return True
                else:
                    return False
            if op == -1:
                if current_value < goal_value_fl:
                    return True
                else:
                    return False
        else:
            if goal_string in fluents:
                return True
            else:
                return False



def _clean_literal(literal):
    left_bracket_clean = re.sub("\s*\(\s*", "(", literal)
    right_bracket_clean = re.sub("\s*\)\s*", ")", left_bracket_clean)
    inner_whitespace_clean = re.sub("\s+", " ", right_bracket_clean)
    return inner_whitespace_clean

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

def _recursive_check_precondition(precondition, fluents,inside_when=False, start_point = False,
                                  key_word=None):
    parse_string = precondition
    if start_point:
        parse_string = "(" + parse_string + ")"
    string_cleaned_blanks = parse_string.replace("\t", "").replace(" ", "").replace("\n", "")
    if string_cleaned_blanks.startswith("(and"):
        key_word = "and"
    elif string_cleaned_blanks.startswith("(or"):
        key_word = "or"
    if key_word in ["and", "or"]:
        # is_true_list = []

        split_list = _split_recursive_and_or(parse_string, key_word)

        if key_word == "and":
            for split_element in split_list:
                if not _recursive_check_precondition(split_element, fluents, inside_when=inside_when):
                    return False
            return True

        elif key_word == "or":
            for split_element in split_list:
                if _recursive_check_precondition(split_element, fluents, inside_when=inside_when):
                    return True
            return False

        # for split_element in split_list:
        # is_true = _recursive_check_precondition(split_element, fluents, inside_when=inside_when)
        # is_true_list.append(is_true)
    #  if key_word == "and":
    #  if all(is_true_list):
    #  return True
    # else:
    # return False
    # elif key_word == "or":
    # if any(is_true_list):
    # return True
    # else:
    # return False
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
        elif len([op for op in ["=", ">", "<"] if op in parse_string]) == 1 and len(
                _clean_literal(parse_string).split(" ")) == 1:
            operator = parse_string[1]
            reference_point_action = float(re.findall('\d+', parse_string)[0])
            function_name = re.findall('\(\w+-*_*\w*\)', parse_string)[0]
            problem_state = \
            [_clean_literal(fluent) for fluent in fluents if function_name in _clean_literal(fluent)][0]
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
                rm_not = \
                re.findall('\(\w+-*_*\w*[\s*\w+\-*_*\w+\-*_*\w+\-*_*\w+\-*_*]*\)', _clean_literal(parse_string))[0]
                if rm_not not in fluents:
                    return True
                else:
                    return False
            else:
                if parse_string in fluents:
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

def _recursive_effect_check(parse_string, zipped_parameters, start_fluents,
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
        split_list = _split_recursive_and_or(parse_string, key_word)
        if inside_when:
            for split_element in split_list:
                is_true, effect = _recursive_effect_check(split_element, zipped_parameters, start_fluents,
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
                is_true, effect = _recursive_effect_check(split_element, zipped_parameters,
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
        is_true, effect = _recursive_effect_check(new_string, zipped_parameters,
                                                       start_fluents, inside_when=True,
                                                       is_consequence = is_consequence)
        if is_true:
            [effects.append(e) for e in effect if e not in effects]
            is_true, effect = _recursive_effect_check(consequence, zipped_parameters, start_fluents,
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
            cleaned_comp_str = _clean_literal(parse_string)
            comp_number = float(re.findall('\d+\d*\.*\d*', cleaned_comp_str)[0])
            op = cleaned_comp_str[1]
            func_var = re.findall('\(\w*-*_*\w*-*_*\w*-*_*\w*\)', cleaned_comp_str)[0]
            state_fl = _clean_literal([fl for fl in start_fluents if func_var in fl][0])
            state_number = float(re.findall('\d+\d*\.*\d*', state_fl)[0])
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
                cleaned_parse_str = _clean_literal(parse_string)
                if "(not" not in cleaned_parse_str:
                    if cleaned_parse_str in [_clean_literal(fl) for fl in start_fluents]:
                        return True, "_"
                    else:
                        return False, "_"
                else:
                    rm_not = re.findall('\(\w+-*_*\w*[\s*\w+\-*_*\w+\-*_*\w+\-*_*\w+\-*_*]*\)',
                                        cleaned_parse_str)[0]
                    if rm_not in [_clean_literal(fl) for fl in start_fluents]:
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


class GymCreator:
    """Class that transforms a pddl domain and problem file to an OpenAI Gym instance.
        The specified reward_functions is always interpreted as costs and therefore has a negative reward effect.
        The name of the reward_function can be changed or set to None, then every action that does not achieve the goal
        has a cost of -1.
    """


    def __init__(self, domain, problem, constant_predicates=None, reward_function="costs", add_actions = None,
                 negative_booleans_coding=0):
        self.domain = domain
        self.problem = problem
        self.env_name = self.domain.name
        self.py_path = self._create_py_path()
        self.constant_predicates = constant_predicates if constant_predicates is not None else []
        self.rwrd_func = reward_function
        self.add_actions = add_actions
        self.obs_space = self._create_obs_space()
        self.action_params = self._create_action_space()
        self.start_fluents = self._start_fluents()
        self.neg_bool_code = negative_booleans_coding

    def _create_obs_space(self):
        obs_space = {}
        obs_space["booleans"] = []
        obs_space["true"] = []
        for predicate in self.domain.predicates_dict.keys():
            if len(self.domain.predicates_dict[predicate]) == 0:
                obs_space["booleans"].append(predicate)
            elif predicate not in self.constant_predicates:
                combination_lists = []
                for param in self.domain.predicates_dict[predicate]:
                    object_type = param[1]
                    combination_lists.append(self.domain.constants[object_type])
                if len(combination_lists) == 1:
                    for el in combination_lists[0]:
                        obs_space["booleans"].append(f"{predicate} {el}")
                else:
                    combinations = self._generate_combinations(combination_lists)
                    for el in combinations:
                        obs_space["booleans"].append(f"{predicate} {el}")
            else:
                for fl in self.problem.start_fluents:
                    if predicate == re.findall("\s*\(\s*\w*" ,fl)[0].replace("(","").replace(" ",""):
                        p = ""
                        for s in fl.replace("(","").replace(")","").split(" "):
                            if s != "":
                                p += f"{s} "
                        p=p[:-1]
                        obs_space["true"].append(p)
        obs_space["numeric"] = []
        for f in self.domain.functions:
            func = f.replace("(","").replace(")","").replace(" ","")
            if func != self.rwrd_func:
                obs_space["numeric"].append(func)
        return obs_space

    @staticmethod
    def _generate_combinations(lists):
        combinations = product(*lists)
        result = [' '.join(combination) for combination in combinations]
        return result

    @staticmethod
    def generate_combinations_2(a):
        combinations = list(product(*a))
        return combinations

    def _create_py_path(self):
        path = f"{os.getcwd()}/py_path/"
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def _str_observation_space(self):
        str_os = ""
        booleans = False
        numerics = False
        if len(self.obs_space["booleans"]) > 0:
            booleans = True
        if len(self.obs_space["numeric"]) > 0:
            numerics = True
        if booleans and numerics:
            str_os += f"\n        self.binary_space = MultiBinary({len(self.obs_space['booleans'])})"
            str_os += f"\n        self.numeric_space = Box(low=-float('inf'), high=float('inf'), shape=({len(self.obs_space['numeric'])},), dtype=float)"
            str_os += f"\n        self.observation_space = Tuple((self.binary_space, self.numeric_space))"
        elif(booleans):
            str_os += f"\n        self.observation_space = MultiBinary({len(self.obs_space['booleans'])})"
        elif(numerics):
            str_os += f"\n        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=({len(self.obs_space['numeric'])},), dtype=float)"
        str_os += f"\n        self.observation_dict = {self._str_obs_dict()}"
        str_os += f"\n        self.observation_dict_key = {self._str_enumerate_obs_dict()}"
        str_os += f"\n        self.always_true = {self._str_true()}"
        return str_os

    def _start_fluents(self):
        start_fluents = {}
        for sfl in self.problem.start_fluents:
            if "=" not in sfl:
                fl = ""
                for f in re.findall("[\w*\-*]*", sfl):
                    if f != '':
                        fl += f"{f} "
                fl = fl[:-1]
                if fl not in self.obs_space["true"]:
                    start_fluents[fl] = 1
            else:
                p = [x for x in re.findall("[\w*\-*]*", sfl) if x!= "" and not x.isdigit()][0]
                val = [float(x) for x in re.findall("[\d]*", sfl) if x != ''][0]
                if p != self.rwrd_func:
                    start_fluents[p] = val
        return start_fluents


    def _str_import_statements(self):
        str_imports = f"""
import subprocess
def install(module):
    subprocess.check_call(['pip', 'install', module])
    print("The module", module, "was installed")
try:
    import gym
except:
    install('gym')
    import gym
try:
    import sys
except:
    install('sys')
    import sys
try:
    import re
except:
    install('re')
    import re
try:
    import numpy as np
except:
    install('numpy')
    import numpy as np
try: 
    from multiprocessing import Pool
except:
    install("multiprocessing")
    multiprocessing


import os
from gym import Env
from gym.spaces import Discrete, MultiBinary, Box, Tuple
sys.path.append(r'E:/goal_recognition/create_pddl_gym.py')
from create_pddl_gym import _clean_literal, _split_recursive_and_or, _recursive_check_precondition, _recursive_effect_check, _recursive_goal_check
from all_actions_mp import get_all_possible_actions          
        
        """
        return str_imports

    def _str_enumerate_obs_dict(self):
        self.obs_dict_key = {}
        i = 0
        for key in self.obs_dict.keys():
            self.obs_dict_key[i] = key
            i += 1
        return self.obs_dict_key

    def _str_obs_dict(self):
        obs_dict = {}
        if len(self.obs_space["booleans"]) > 0:
            for el in self.obs_space["booleans"]:
                obs_el = {}
                obs_el["type"] = "boolean"
                obs_el["value"] = self.neg_bool_code
                obs_dict[el] = obs_el
        if len(self.obs_space["numeric"]) > 0:
            for el in self.obs_space["numeric"]:
                obs_el = {}
                obs_el["type"] = "numeric"
                obs_el["value"] = 0.0
                obs_dict[el] = obs_el
        self.obs_dict = obs_dict
        return obs_dict

    def _str_true(self):
        true_dict = {}
        if len(self.obs_space["true"]) > 0:
            for el in self.obs_space["true"]:
                p = el.split(" ")[0]
                if p not in true_dict.keys():
                    true_dict[p] = []
                true_dict[p].append(el)
        return true_dict

    def _ungrounded_actions(self):
        str_ungrounded_actions = "["
        for action in self.domain.action_dict.keys():
            str_ungrounded_actions += f"'{action}', "
        str_ungrounded_actions = str_ungrounded_actions[:-2]
        str_ungrounded_actions += "]"
        return str_ungrounded_actions

    def _create_action_space(self):
        actions = []
        action_params = []
        for action in self.domain.action_dict.keys():
            len_params = len(self.domain.action_dict[action].action_parameters)
            if len_params == 0:
                actions_dict_params = {}
                actions_dict_params["action_grounded"] = action
                actions_dict_params["action_ungrounded"] = action
                actions_dict_params["parameter"] = []
                actions_dict_params["parameter_variable"] = []
                actions_dict_params["instances"] = []
                actions_dict_params["precondition"] = self.domain.action_dict[action].action_preconditions
                actions_dict_params["effects"] = self.domain.action_dict[action].action_effects
                actions.append(action)
                action_params.append(actions_dict_params)
            else:
                precon = self.domain.action_dict[action].action_preconditions
                action_prune = False
                keep_constant_predicates = []
                for constant_predicate in self.constant_predicates:
                    if constant_predicate in precon:
                        action_prune = True
                        keep_constant_predicates.append(constant_predicate)
                if not action_prune:
                    if len_params == 1:
                        param_type = self.domain.action_dict[action].action_parameters[0].parameter_type
                        for c in self.domain.constants[param_type]:
                            a = f"{action}_{c}"
                            actions.append(a)
                            actions_dict_params = {}
                            actions_dict_params["action_grounded"] = a.upper()
                            actions_dict_params["action_ungrounded"] = action
                            actions_dict_params["parameter"] = [param_type]
                            actions_dict_params["parameter_variable"] = \
                                [self.domain.action_dict[action].action_parameters[0].parameter]
                            actions_dict_params["instances"] = [c]
                            actions_dict_params["precondition"] = \
                                self.domain.action_dict[action].action_preconditions
                            actions_dict_params["effects"] = \
                                self.domain.action_dict[action].action_effects
                            action_params.append(actions_dict_params)
                    else:
                        constants = []
                        param_types = []
                        param_variables = []
                        for param in self.domain.action_dict[action].action_parameters:
                            param_type = param.parameter_type
                            param_types.append(param_type)
                            param_variables.append(param.parameter)
                            constants.append([c for c in self.domain.constants[param_type]])
                        combinations = self._generate_combinations(constants)
                        combination_tuples = self.generate_combinations_2(constants)
                        combinations = [f"{action}_{c}" for c in combinations]
                        c = 0
                        for comb in combinations:
                            actions.append(comb)
                            actions_dict_params = {}
                            actions_dict_params["action_grounded"] = comb.upper()
                            actions_dict_params["action_ungrounded"] = action
                            actions_dict_params["parameter"] = param_types
                            actions_dict_params["parameter_variable"] = param_variables
                            actions_dict_params["instances"] = combination_tuples[c]
                            actions_dict_params["precondition"] = \
                                self.domain.action_dict[action].action_preconditions
                            actions_dict_params["effects"] = \
                                self.domain.action_dict[action].action_effects
                            action_params.append(actions_dict_params)
                            c+=1
                else:
                    dict_fix_constants = {}
                    for k in keep_constant_predicates:
                        dict_k = {}
                        object_types = [t[1] for t in self.domain.predicates_dict[k]]
                        dict_k["object_types"] = object_types
                        dict_k["tuples"] = []
                        for fl in self.problem.start_fluents:
                            if k in fl:
                                fl_list = [c for c in
                                           re.findall("\w*-*\w*-*\w*-*\w*-*\w*-*", fl)
                                           if c != "" and c != k]
                                dict_k["tuples"].append(fl_list)
                        dict_fix_constants[k] = dict_k
                    param_list = []
                    param_variables = []
                    for param in self.domain.action_dict[action].action_parameters:
                        param_list.append(param.parameter_type)
                        param_variables.append(param.parameter)
                    tuples = []
                    for key in dict_fix_constants.keys():
                        j = 0
                        while j < len(dict_fix_constants[key]["tuples"]):
                            param_obs = []
                            unique_param_list = []
                            for param in param_list:
                                if param not in unique_param_list:
                                    unique_param_list.append(param)
                            for param in unique_param_list:
                                if param in dict_fix_constants[key]["object_types"]:
                                    i = 0
                                    while i < len(dict_fix_constants[key]["object_types"]):
                                        if dict_fix_constants[key]["object_types"][i] == param:
                                            param_obs.append(dict_fix_constants[key]["tuples"][j][i])
                                        i+=1
                                else:
                                    param_obs.append(None)
                            tuples.append(param_obs)
                            j+=1
                    new_tuples = []
                    for t in tuples:
                        i = 0
                        create_new_tuples = False
                        dict_new_constants = {}
                        dict_index = {}
                        while i<len(t):
                            if t[i] is None:
                                create_new_tuples = True
                                constant_type = param_list[i]
                                dict_new_constants[constant_type] = []
                                dict_index[constant_type] = i
                                for c in self.domain.constants[constant_type]:
                                    dict_new_constants[constant_type].append(c)
                            i+=1
                        if not create_new_tuples:
                            new_tuples.append(t)
                        else:
                            c_list = []
                            for key in dict_new_constants.keys():
                                c_list.append(dict_new_constants[key])
                            combos = self.generate_combinations_2(c_list)
                            for comb in combos:
                                new_t = [el for el in t]
                                i = 0
                                for key in dict_index.keys():
                                    index = dict_index[key]
                                    new_t[index] = comb[i]
                                    i = i+1
                                new_tuples.append(new_t)
                    for t in new_tuples:
                        a = action
                        for c in t:
                            a += f"_{c}"
                        actions.append(a)
                        actions_dict_params = {}
                        actions_dict_params["action_grounded"] = a.upper()
                        actions_dict_params["action_ungrounded"] = action
                        actions_dict_params["parameter"] = param_list
                        actions_dict_params["parameter_variable"] = param_variables
                        actions_dict_params["instances"] = t
                        actions_dict_params["precondition"] = \
                            self.domain.action_dict[action].action_preconditions
                        actions_dict_params["effects"] = \
                            self.domain.action_dict[action].action_effects
                        action_params.append(actions_dict_params)
        if not self.add_actions is None:
            for action in self.add_actions:
                actions_dict_params = {}
                action_grounded = action["action_ungrounded"].upper()
                action_ungrounded = action["action_ungrounded"].upper()
                for instance in action["instances"]:
                    action_grounded+= f"_{instance.upper()}"
                actions_dict_params["action_grounded"] = action_grounded
                actions_dict_params["action_ungrounded"] = action_ungrounded
                param_list = [p.parameter_type for p
                              in self.domain.action_dict[action_ungrounded].action_parameters]
                actions_dict_params["parameter"] = param_list
                param_variables = [p.parameter for p
                              in self.domain.action_dict[action_ungrounded].action_parameters]
                actions_dict_params["parameter_variable"] = param_variables
                actions_dict_params["instances"] = action["instances"]
                actions_dict_params["precondition"] = \
                    self.domain.action_dict[action_ungrounded].action_preconditions
                actions_dict_params["effects"] = \
                    self.domain.action_dict[action_ungrounded].action_effects
                action_params.append(actions_dict_params)

        action_dict = {}
        i = 0
        for action in action_params:
            action_dict[i] = action
            i += 1
        action_dict
        return action_dict

    def _str_env_class(self):
        str_class=f"""
class PDDLENV(Env):
    def __init__(self):
        self.name = "{self.env_name}"
        self.action_space = Discrete({len(self.action_params.keys())})
        self.action_dict= {self.action_params}
        self.ungrounded_actions = {self._ungrounded_actions()}
        self.start_fluents = {self._start_fluents()}
        self.goal_fluents = '{self.problem.goal_fluents[0]}'
        self.final_reward = 10
        self.additional_reward_fluents = dict()
        {self._str_observation_space()}
        _ = self.reset()
        """
        if self.rwrd_func is None:
            str_class+="""
        self.rwrd_func is None
        
        """
        else:
            str_class += f"""
        self.rwrd_func = '{self.rwrd_func}'
        """
        return str_class

    def make_env(self):
        py_code = f"""{self._str_import_statements()}

{self._str_env_class()}
"""
        py_code += f"""
    def set_final_reward(self, reward):
        self.final_reward = reward
    
    def set_additional_reward_fluents(self, fluent, reward):
        self.additional_reward_fluents[fluent] = reward
    
    def reset(self, startpoint=True, state=None):
        if startpoint:
            for key in self.observation_dict.keys():
                self.observation_dict[key]["value"] = {self.neg_bool_code}
            for key in self.start_fluents.keys():
                self.observation_dict[key]["value"] = self.start_fluents[key]
            self.state = self._get_obs_vector()
        else:
            self.state = state
            self._set_observation_dict(state)
        return self.state

    def _get_obs_vector(self):
        i = 0
        vector_list = []
        while i < len(self.observation_dict_key.keys()):
            key = self.observation_dict_key[i]
            vector_list.append(self.observation_dict[key]["value"])
            i+=1
        return np.array(vector_list)

    def _set_observation_dict(self, array_state):
        i = 0
        while i < len(self.observation_dict_key.keys()):
            key = self.observation_dict_key[i]
            self.observation_dict[key]["value"] = array_state[i]
            i+=1

    def get_current_fluents(self):
        current_fluents = []
        for key in self.observation_dict.keys():
            obs = self.observation_dict[key]
            if obs['type'] == 'boolean' and obs["value"] == 1:
                current_fluents.append("(" + key +")")
            elif obs['type'] == 'numeric':
                current_fluents.append("(=(" + key + ") " + str(obs["value"]) + ")")        
        for key in self.always_true:
            for el in self.always_true[key]:
                current_fluents.append("(" + el +")")
        return current_fluents
    
    def _action_possible(self, action_idx):
        action = self.action_dict[action_idx]
        fluents = [_clean_literal(sf) for sf in self.get_current_fluents()]
        precondition = action["precondition"]
        if len(action["parameter_variable"]) > 0:
            i = 0
            while i < len(action["parameter_variable"]):
                param = action["parameter_variable"][i]
                inst = action["instances"][i]
                precondition = precondition.replace(param,inst)
                i+=1
        precondition_given = _recursive_check_precondition(precondition, fluents, start_point=True)
        return precondition_given
    
    def get_all_possible_actions(self, multiprocess = False, workers = 3):
        if not multiprocess:
            list_actions = []
            for idx in range(len(self.action_dict.keys())):
                if self._action_possible(idx):
                    list_actions.append((idx, self.action_dict[idx]["action_grounded"]))
            return list_actions
        else:
            return get_all_possible_actions(self.action_dict, self.get_current_fluents(), workers=workers)


    def _action_effects(self, action_idx):
        action = self.action_dict[action_idx]
        action_effects = action["effects"]
        fluents = [_clean_literal(sf) for sf in self.get_current_fluents()]
        zipped_parameters = list(zip(action["instances"], action["parameter_variable"]))
        _, effects = _recursive_effect_check(action_effects, zipped_parameters, fluents)
        return [_clean_literal(effect) for effect in effects if effect != "_"]
    
    def render(self):
        pass
    
    def step(self, action):
        old_fluents = self.get_current_fluents()
        if self._action_possible(action):
            effects = self._action_effects(action)
            for effect in effects:
                # numeric effects
                if "(increase(" in effect or "(decrease(" in effect:
                    change_value_str = re.findall(r'\d.*\d*', effect)[0].replace("(","").replace(")","")
                    change_value_fl = float(change_value_str)
                    func = effect.replace(change_value_str,"").replace("(increase(","").replace("(decrease(","").replace(")","")
                    # rwrd_func? 
                    if self.rwrd_func is None:
                        reward = -1
                    else: 
                        if func == self.rwrd_func:
                            reward = -abs(change_value_fl)
                        elif "(increase(" in effect:
                            self.observation_dict[func]["value"] += change_value_fl
                        else:
                            self.observation_dict[func]["value"] -= change_value_fl
                # predicate effect
                elif "(not(" in effect: 
                    effect = effect.replace("(not(","").replace(")","")
                    self.observation_dict[effect]["value"] = {self.neg_bool_code}
                else:
                    effect = effect.replace("(","").replace(")","")
                    self.observation_dict[effect]["value"] = 1
        else:
            reward = -10

        new_fluents = self.get_current_fluents()
        for new_fluent in new_fluents:
            if new_fluent in self.additional_reward_fluents.keys() and new_fluent not in old_fluents:
                reward += self.additional_reward_fluents[new_fluent] 
        
        done = _recursive_goal_check(self.goal_fluents, new_fluents, start_point = True)
        if done:
            reward += self.final_reward  
        
        info = "action: " + self.action_dict[action]['action_grounded']
        info += ", deleted: "
        for el in old_fluents:
            if el not in new_fluents:
                info += el+ " "
        info += ", added: "
        for el in new_fluents:
            if el not in old_fluents:
                info += el+ " "

        self.state = self._get_obs_vector()
        info = dict()
        return self.state, reward, done, info


"""
        py_file = self.py_path + "env_pddl.py"
        with open(py_file, "w") as py_env:
            py_env.write(py_code)
        sys.path.append(self.py_path)
        from env_pddl import PDDLENV
        #os.remove(py_file)
        del sys.modules['env_pddl']
        return PDDLENV()


if __name__ == '__main__':
    model = 11
    if model> 8:
        add_actions = [{'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-4b', 'loc-infirmary-kim']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-infirmary-kim', 'loc-outdoors-4b']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-infirmary-medicine', 'loc-infirmary-bathroom']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-infirmary-bathroom', 'loc-infirmary-medicine']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-laboratory-front', 'loc-outdoors-null-b']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-null-b', 'loc-laboratory-front']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-laboratory-midright', 'loc-laboratory-library']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-laboratory-library', 'loc-laboratory-midright']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-dininghall-front', 'loc-outdoors-null-e']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-null-e', 'loc-dininghall-front']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-dininghall-back-souptable', 'loc-outdoors-null-d']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-null-d', 'loc-dininghall-back-souptable']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-dininghall-back-souptable', 'loc-outdoors-2b']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-2b', 'loc-dininghall-back-souptable']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-livingquarters-hall', 'loc-outdoors-2a']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-2a', 'loc-livingquarters-hall']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-livingquarters-hall', 'loc-outdoors-null-g']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-null-g', 'loc-livingquarters-hall']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-brycesquarters-hall', 'loc-outdoors-null-f']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-null-f', 'loc-brycesquarters-hall']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-brycesquarters-hall', 'loc-brycesquarters-bedroom']},
                       {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-brycesquarters-bedroom', 'loc-brycesquarters-hall']}]
        cp = ["person_in_room", "neighboring"]
    else:
        add_actions = None
        cp = ["person_in_room"]
    path = f"/home/mwiubuntu/Seminararbeit/domain/environment in domain file/For Grid-Search/"
    domain = pddl_domain(path + f"{model}_crystal_island_domain.pddl")
    problem = pddl_problem(path + f"model_{model}_goal_1_crystal_island_problem.pddl")
    env_creator_ci = GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions,
                                negative_booleans_coding=-1)
    env_ci = env_creator_ci.make_env()

    #z = [env_creator_ci.action_params[key] for key in env_creator_ci.action_params.keys() if "MOVE" in env_creator_ci.action_params[key]["action_grounded"]]

    env_creator_toy = GymCreator(pddl_domain("domain.pddl"), pddl_problem("problem_B.pddl"))
    env_toy = env_creator_toy.make_env()
    d = env_ci.get_all_possible_actions()
    #steps = ["ACTION-MOVETOLOC-STARTGAME-OUTDOORS_6B", "ACTION-MOVETOLOC-OUTDOORS_6B-OUTDOORS_5B",
             #"ACTION-MOVETOLOC-OUTDOORS_5B-OUTDOORS_4B", "ACTION-MOVETOLOC-OUTDOORS_4B-INFIRMARY_KIM",
             #"ACTION-TALK-TO_TERESA_LOC-INFIRMARY-KIM"]
    steps = ["ACTION-MOVETOLOC_LOC-STARTGAME_LOC-OUTDOORS-6B", "ACTION-MOVETOLOC_LOC-OUTDOORS-6B_LOC-OUTDOORS-5B",
    "ACTION-MOVETOLOC_LOC-OUTDOORS-5B_LOC-OUTDOORS-4B", "ACTION-OPEN_DOOR-INFIRMARY-FRONT",
             "ACTION-MOVETOLOC_LOC-OUTDOORS-4B_LOC-INFIRMARY-KIM", "ACTION-TALK-TO_TERESA_LOC-INFIRMARY-KIM",
             "ACTION-TALK-TO_KIM_LOC-INFIRMARY-KIM"]
    st = []
    for step in steps:
        st.append([(key, env_ci.action_dict[key]["action_grounded"]) for key in env_ci.action_dict.keys()
               if step in env_ci.action_dict[key]["action_grounded"]][0])
    print(st)
    #print([(key, env_ci.action_dict[key]["action_grounded"]) for key in env_ci.action_dict.keys()
     #      if "OPEN" in  env_ci.action_dict[key]["action_grounded"]])



    #print(env_ci._action_possible(480))
    #collect = []
    #for key in range(len(env_ci.action_dict.keys())):
        #effects = env_ci._action_effects(key)
        #print(env_ci.action_dict[key]["action_grounded"])
        #print(effects)
        #if len(effects) == 0:
            #collect.append(env_ci.action_dict[key]["action_grounded"])
    print("---------")
    print(env_ci.step(3265))
    print(env_ci.step(3264))
    print("---------")
    print(env_ci.step(4))
    print(env_ci.step(3265))
    print("---------")
    print(env_ci.step(3272))
    print(env_ci.step(17))
    print("---------")
    print(env_ci.step(3327))
    print(env_ci.step(3236))
    print("---------")
    print(env_ci.step(3327))
    print(env_ci.step(1781))
    print("---------")
    print(env_ci.step(3250))
    print(env_ci.step(3247))

    len(env_ci.action_dict)
    #c = _recursive_goal_check(env_ci.goal_fluents, env_ci.get_current_fluents(), start_point = True)
    # print("c: ", c)
    #_recursive_goal_check(goal_string, fluents, inside_when, start_point=False, key_word=None)
    env_ci.get_current_fluents()
    env_ci.reset()

    start = time.time()
    env_ci.get_all_possible_actions()
    print(time.time() - start)
    start = time.time()
    env_ci.get_all_possible_actions(multiprocess=True)
    print(time.time() - start)



