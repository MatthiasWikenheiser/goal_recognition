import re
import pandas as pd
import os
class pddl_domain:
    """PDDL Domain data structure read from a .pddl-domain file"""
    def __init__(self, domain_path):
        """:param domain_path: Path of a .pddl-domain file"""
        self.domain_path = self._domain_path(domain_path)
        self.domain = self._clean_domain_comments()
        self.name = self._get_name_domain()
        self.action_dict = self._get_actions()
        self.action_count = len(self.action_dict)
        self.requirements = self._get_requirements()
        self.types = self._get_types()
        self.constants = self._get_constants()
        self.predicates = self._get_predicates()
        self.functions = self._get_functions()
    class pddl_action:
        """PDDL action data structure for actions within within a pddl_domain"""
        def __init__(self, action):
            self.action = action
            self.name = self._get_name_action()
            self.action_parameters = self._get_action_parameters()
            self.action_preconditions = self._get_action_preconditions()
            self.action_effects = self._get_action_effects()
            self.action_cost = self._get_action_cost()
        def _get_name_action(self):
            return self.action.split()[1]
        def _get_action_parameters(self):
            action = self.action
            idx_strt = action.find(":parameters")
            idx_end = idx_strt + action[idx_strt:].find(":",2)
            string_parameters = action[idx_strt:idx_end]
            string_parameters = string_parameters.replace("\n", "").replace("\t", "").replace("(", "").replace(")","")
            list_string = [x for x in string_parameters.split(" ") if x != ""][1:]
            list_action_parameters = []
            i = 0
            while(i < len(list_string)):
                variables = []
                variable_type = ""
                j = i
                parse = True
                while(parse and j < len(list_string)):
                    if list_string[j] == "-":
                        [variables.append(x) for x in list_string[i:j]]
                        variable_type = list_string[j+1]
                        parse = False
                        i = j+1
                    j = j+1
                for variable in variables:
                    list_action_parameters.append(self.pddl_action_parameter(variable, variable_type))
                i = i+1
            return(list_action_parameters)
        def _get_action_cost(self):
            action = self.action
            idx_strt = action.find("increase (costs)")
            parse = True
            c = idx_strt + 1
            count_bracket = 0
            while parse:
                if action[c] == ")":
                    count_bracket = count_bracket + 1
                if count_bracket == 2:
                    idx_end = c
                    parse = False
                c = c+1
            string_cost = ""
            for c in action[idx_strt:idx_end]:
                if c.isdigit() or c == ".":
                    string_cost = string_cost+c
            return float(string_cost)
        def set_action_cost(self, cost):
            """change action cost for a action of a pddl_domain
            :param cost: new cost for action"""
            idx_strt = self.action.find("increase (costs)")
            parse = True
            c = idx_strt + 1
            count_bracket = 0
            while parse:
                if self.action[c] == ")":
                    count_bracket = count_bracket + 1
                if count_bracket == 2:
                    idx_end = c
                    parse = False
                c = c+1
            new_cost_string = self.action[idx_strt:idx_end]
            new_cost_string = new_cost_string[:new_cost_string.find(")")+1] + " " + str(cost)
            new_action_string = self.action[:idx_strt] + new_cost_string + self.action[idx_end:]
            self.action = new_action_string
            self.action_cost = cost
        def _get_action_preconditions(self):
            action = self.action
            idx_strt = action.find(":precondition")
            idx_end = idx_strt + action[idx_strt:].find(":",2)
            string_preconditions = action[idx_strt:idx_end]
            string_preconditions = string_preconditions.replace("\n", "").replace("\t", "").replace(":precondition", "")
            #remove precondition_brackets
            parse = True
            c = 0
            while parse:
                if string_preconditions[c]== "(":
                    string_preconditions = string_preconditions[c+1:]
                    parse = False
                c = c+1
            parse = True
            c = -1
            while parse:
                if string_preconditions[c]== ")":
                    string_preconditions = string_preconditions[:c]
                    parse = False
                c = c-1
            #check if any precondition exists
            if string_preconditions.replace(" ", "") == "":
                return string_preconditions.replace(" ", "")
            else:
                return(string_preconditions)
        def _get_action_effects(self):
            action = self.action
            idx_strt = action.find(":effect")
            string_effects = action[idx_strt:]
            string_effects = string_effects.replace("\n", "").replace("\t", "").replace(":effect", "")
            #remove effect_brackets
            parse = True
            c = 0
            while parse:
                if string_effects[c]== "(":
                    string_effects = string_effects[c:]
                    parse = False
                c = c+1
            parse = True
            c = -1
            while parse:
                if string_effects[c]== ")":
                    string_effects = string_effects[:c]
                    parse = False
                c = c-1
            parse = True
            c = -1
            while parse:
                if string_effects[c]== ")":
                    string_effects = string_effects[:c+1]
                    parse = False
                c = c-1
            #check if any precondition exists
            if string_effects.replace(" ", "") == "":
                return string_effects.replace(" ", "")
            else:
                return string_effects
        class pddl_action_parameter:
            """action parameter of a pddl_action within a ppdl_domain"""
            def __init__(self, parameter, parameter_type):
                self.parameter = parameter
                self.parameter_type = parameter_type
    def _domain_path(self, domain_path):
        if len(domain_path.split("/")) == 1:
            return os.getcwd() + "/" + domain_path
        else:
            return domain_path
    def _clean_domain_comments(self):
        domain_list = open(self.domain_path, "r").readlines()
        for i in range(len(domain_list)):
            if ";" in domain_list[i]:
                #find position of ; in line
                idx = domain_list[i].find(";")
                new_line = domain_list[i][:idx]
                domain_list[i] = new_line
        cleaned_domain = ""
        for line in domain_list:
            cleaned_domain = cleaned_domain + "\n" + line
        return cleaned_domain
    def _get_name_domain(self):
        scan_domain_name = True
        i = 0
        while scan_domain_name:
            if self.domain.split()[i] == "(domain":
                domain_name = (self.domain.split()[i+1]).replace(" ", "").replace(")", "")
                scan_domain_name = False
            i = i+1
        return domain_name
    def _get_types(self):
        idx_strt = self.domain.find("(:types")
        idx_end = idx_strt + self.domain[idx_strt:].find("(:",1)
        return self.domain[idx_strt:idx_end].replace("\n", "")
    def _get_requirements(self):
        idx_strt = self.domain.find("(:requirements")
        idx_end = idx_strt + self.domain[idx_strt:].find("(:",1)
        return self.domain[idx_strt:idx_end].replace("\n", "")
    def _get_constants(self):
        idx_strt = self.domain.find("(:constants")
        idx_end = idx_strt + self.domain[idx_strt:].find("(:",1)
        return self.domain[idx_strt:idx_end].replace("\n", "")
    def _get_predicates(self):
        predicates_list = []
        idx_strt = self.domain.find("(:predicates")
        idx_end = idx_strt + self.domain[idx_strt:].find("(:",1)
        string_predicates = self.domain[idx_strt:idx_end].replace("(:predicates", "").replace("\n", "")
        i = 0
        while i < len(string_predicates):
            if string_predicates[i] =="(":
                j = i
                parse = True
                while parse and j < len(string_predicates):
                    if string_predicates[j] == ")":
                        predicate = string_predicates[i:j+1]
                        i = j
                        predicates_list.append(predicate)
                        parse = False
                    j = j+1
            i=i+1
        return predicates_list
    def _get_functions(self):
        idx_strt = self.domain.find("(:functions")
        idx_end = idx_strt + self.domain[idx_strt:].find("(:",1)
        parse_string = self.domain[idx_strt:idx_end].replace("\n", "").replace("\t", "").replace(" ", "")
        parse_string = parse_string[11:]
        parse = True
        idx = len(parse_string) - 1
        while idx > 0 and parse:
            if parse_string[idx] == ")":
                parse_string = parse_string[:idx]
                parse = False
            idx -= 1
        functions = []
        idx = 0
        while idx < len(parse_string):
            if parse_string[idx] == "(":
                bracket_start = idx
            if parse_string[idx] == ")":
                functions.append(parse_string[bracket_start:idx+1])
            idx += 1
        return functions
    def _get_actions(self):
        parsed_domain = re.split(" |\n",self.domain)
        actions = []
        parse_action = True
        i = 0
        while (parse_action and i< len(parsed_domain)):
            if parsed_domain[i] == "(:action":
                in_action = True
                j = i+1
                while (in_action and j< len(parsed_domain)):
                    if (parsed_domain[j].startswith("(:") or j == len(parsed_domain)-1):
                        action_list = parsed_domain[i:j-1]
                        action = ""
                        for el in action_list:
                            action = action + " " + el
                        actions.append(self.pddl_action(action))
                        in_action = False
                        i = j-1
                    j = j+1
            i = i +1
        actions_name = [(x.name).upper() for x in actions]
        dict_actions = {}
        for i in range(len(actions_name)):
            dict_actions[actions_name[i]] = actions[i]
        return dict_actions
class pddl_problem:
    """PDDL Problem data structure read from a .pddl-problem file"""
    def __init__(self, problem_path):
        """:param problem_path: Path of a .pddl-problem file"""
        self.problem_path = self._problem_path(problem_path)
        self.problem = self._clean_problem_comments()
        self.name = self._get_name_problem()
        self.start_fluents = self._get_start_fluents()
        self.goal_fluents = self._get_goal_fluents()
        self.metric_min_func = self._get_metric_min_func()
    def _get_metric_min_func(self):
        metric_min_str = re.findall("\(:\s*metric\s+minimize\s+\(\s*\w+\s*\)\s*\)", self.problem)[0]
        min_func = re.findall("\(\s*\w+\s*\)", metric_min_str)[0]
        min_func = re.findall("\w+" , min_func)[0]
        return min_func
    def _problem_path(self, problem_path):
        if len(problem_path.split("/")) == 1:
            return os.getcwd() + "/" + problem_path
        else:
            return problem_path
    def _clean_problem_comments(self):
        problem_list = open(self.problem_path, "r").readlines()
        for i in range(len(problem_list)):
            if ";" in problem_list[i]:
                #find position of ; in line
                idx = problem_list[i].find(";")
                new_line = problem_list[i][:idx]
                problem_list[i] = new_line
        cleaned_problem = ""
        for line in problem_list:
            cleaned_problem = cleaned_problem + "\n" + line
        return cleaned_problem
    def _get_name_problem(self):
        scan_problem_name = True
        i = 0
        while scan_problem_name:
            if self.problem.split()[i] == "(define":
                problem_name = (self.problem.split()[i+2]).replace(" ", "").replace("(", "").replace(")", "")
                scan_problem_name = False
            i = i+1
        return problem_name
    def _get_start_fluents(self):
        #parsed_problem = self.problem.split(" ")
        parsed_problem = re.split(" |\n",self.problem)
        start_fluents = []
        parse_problem = True
        i = 0
        while (parse_problem and i< len(parsed_problem)):
            if parsed_problem[i] == "(:init":
                in_init = True
                j = i+1
                while (in_init and j < len(parsed_problem)):
                    if (parsed_problem[j].startswith("(:")):
                        in_init = False
                    j = j+1
                init_string = ""
                for el in parsed_problem[i+1:j-1]:
                    init_string = init_string + " " + el
            i = i+1
        start_fluent = 0
        record = ""
        for c in init_string:
            if c == "(":
               start_fluent = start_fluent + 1
            if start_fluent > 0:
                record = record + c
            if c == ")":
               start_fluent = start_fluent - 1
               if start_fluent  == 0:
                   start_fluents.append(record)
                   record = ""
        return start_fluents
    def _get_goal_fluents(self):
        parsed_problem = re.split(" |\n",self.problem)
        goal_fluents = []
        parse_problem = True
        i = 0
        while (parse_problem and i < len(parsed_problem)):
            if parsed_problem[i] == "(:goal":
                in_goal = True
                j = i+1
                while (in_goal and j < len(parsed_problem)):
                    if(parsed_problem[j].startswith("(:")):
                        in_goal = False
                    j = j + 1
                goal_string = ""
                for el in parsed_problem[i+1: j-1]:
                    goal_string= goal_string  + " " + el
            i = i+1
        goal_fluent = 0
        record = ""
        for c in goal_string:
            if c == "(":
               goal_fluent = goal_fluent + 1
            if goal_fluent > 0:
                record = record + c
            if c == ")":
               goal_fluent = goal_fluent - 1
               if goal_fluent  == 0:
                   goal_fluents.append(record)
                   record = ""
        return goal_fluents

class pddl_observations:
    """"data structure for observations of an agent which aligns to a pddl_domain
    UNDER construction"""
    def __init__(self, csv_file):
        self.obs_file = pd.read_csv(csv_file)
        self.obs_len = len(self.obs_file)
        self.obs_action_sequence = self.obs_file["action"]
    def partial_action_sequence(self, sequence_length = None):
        if sequence_length  == None:
            return self.obs_action_sequence
        else:
            return self.obs_action_sequence.iloc[:sequence_length ]

if __name__ == '__main__':
    toy_example_domain = pddl_domain('domain.pddl')
    print(toy_example_domain.action_dict["MOVE_LEFT_FROM"].action_parameters[0].parameter)
    problem_a = pddl_problem('problem_A.pddl')
    print(problem_a.metric_min_func)