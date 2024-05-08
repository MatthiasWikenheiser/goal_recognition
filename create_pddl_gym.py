from pddl import *
import os
import gym
import sys
from itertools import product

class action:
    def __init__(self, action_dict):
        self.action_grounded = action_dict["action_grounded"]
        self.action_ungrounded = action_dict["action_ungrounded"]
        self.parameter = action_dict["parameter"]
        self.parameter_variable = action_dict["parameter_variable"]
        self.instances = action_dict["instances"]

    def precondition(self):
        pass

    def effects(self):
        pass

class GymCreator:
    def __init__(self, domain, problem, constant_predicates=None, reward_function="costs"):
        self.domain = domain
        self.problem = problem
        self.env_name = self.domain.name
        self.py_path = self._create_py_path()
        self.constant_predicates = constant_predicates if constant_predicates is not None else []
        self.rwrd_func = reward_function
        self.obs_space = self._create_obs_space()
        self.action_params = self._create_action_space()
        self.start_fluents = self._start_fluents()

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
    import numpy as np
except:
    install('numpy')
    import numpy as np

from gym import Env
from gym.spaces import Discrete, MultiBinary, Box, Tuple
sys.path.append(r'E:/goal_recognition/create_pddl_gym.py')
from create_pddl_gym import action
        
        
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
                obs_el["value"] = 0
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

 #   def _create_action_keys(self):
        #action_dict_del = {}
        #i = 0
        #for action in self.action_params:
            #action_dict_del[i] = action
            #i += 1
        #self.action_params = action_dict_del
        #return self.action_params

    def _precon_index(self, index):
        str_func = f"""
def precon_func_{index}():
    print("precon_func_{index}")
        """
        return str_func

    def _precon_all(self):
        result = ""
        for key in self.action_params.keys():
            result += self._precon_index(key)
        return result


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
                actions_dict_params["action_grounded"] = None
                actions_dict_params["action_ungrounded"] = action
                actions_dict_params["parameter"] = [None]
                actions_dict_params["parameter_variable"] = [None]
                actions_dict_params["instances"] = [None]
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
        self.action_dict_del= {self.action_params}
        self.ungrounded_actions = {self._ungrounded_actions()}
        self.start_fluents = {self._start_fluents()}
        {self._str_observation_space()}
        _ = self.reset()
        self._init_action_dict()
"""
        return str_class

    def make_env(self):
        py_code = f"""{self._str_import_statements()}

{self._precon_all()}

{self._str_env_class()}

    def _init_action_dict(self):  
        self.action_dict = dict()
        for key in self.action_dict_del.keys():
            self.action_dict[key] = action(self.action_dict_del[key])
        del self.action_dict_del"""


        for action_idx in self.action_params.keys():
            py_code += f"""
        self.action_dict[{action_idx}].precondition = precon_func_{action_idx}
        self.action_dict[{action_idx}].effects = self.hello_world"""
        

        py_code += f"""
    def reset(self):
        for key in self.start_fluents.keys():
            self.observation_dict[key]["value"] = self.start_fluents[key]
        return self._get_obs_vector()

    def _get_obs_vector(self):
        i = 0
        vetor_list = []
        while i < len(self.observation_dict_key.keys()):
            key = self.observation_dict_key[i]
            vetor_list.append(self.observation_dict[key]["value"])
            i+=1
        return np.array(vetor_list)

    def get_current_fluents(self):
        current_fluents = []
        for key in self.observation_dict.keys():
            obs = self.observation_dict[key]
            if obs['type'] == 'boolean' and obs["value"] == 1:
                current_fluents.append(key)
            elif obs['type'] == 'numeric':
                current_fluents.append(key + '=' + str(obs["value"]))
        return current_fluents
    
    def hello_world(self):
        print("hello_world")
    
    
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
        cp = ["person_in_room", "neighboring"]
    else:
        cp = ["person_in_room"]
    path = f"C:/Users/Matthias/OneDrive/goal_recognition/Domain and Logs/model_{model}/"
    domain = pddl_domain(path + f"{model}_crystal_island_domain.pddl")
    problem = pddl_problem(path + f"model_{model}_goal_1_crystal_island_problem.pddl")
    env_creator_ci = GymCreator(domain, problem, constant_predicates= cp)
    env_ci = env_creator_ci.make_env()


    env_creator_toy = GymCreator(pddl_domain("domain.pddl"), pddl_problem("problem_B.pddl"))
    env_toy = env_creator_toy.make_env()

    #d = [x for x in env_creator_ci.action_params if "TALK" in x["action_grounded"]]
    #e = [x for x in env_creator_ci.action_params if "MOVE" in x["action_grounded"]]
    #print(env_creator_ci._ungrounded_actions())

    #d = [env_ci.action_dict[key] for key in env_ci.action_dict.keys() if "HAND" not in env_ci.action_dict[key]["action_grounded"]]

    env_ci.action_dict[0].effects()
    funcs = env_creator_ci._precon_all()
    env_ci.action_dict[0].effects()
    env_ci.action_dict[2587].precondition()
    env_toy.action_dict[7].precondition()

