from pddl import pddl_domain, pddl_problem, pddl_observations
from metric_ff_solver import metric_ff_solver
import re
class gm_model:
    """class that solves a goal recognition problem according to the vanilla plain approach Goal Mirroring (GM)
     by Vered et al., 2016.
    """
    def __init__(self, domain_root, goal_list, obs_action_sequence, planner = "ff_2_1"):
        """
        :param domain_root: pddl_domain on which the prap_model evolves in order to assign a probability to each goal.
        :param goal_list: list of pddl_problems, which represent the goals to assign probabilites to.
        :param obs_action_sequence: agents observations of type _pddl_observations.
        :param planner: name of executable planner, here default ff_2_1 (MetricFF Planner version 2.1)
        """
        self.domain_list = [domain_root]
        self.goal_list = [goal_list]
        self.planner = planner
        self.observation = obs_action_sequence
        self.steps_observed = []
        self.prob_dict_list = []
        self.prob_nrmlsd_dict_list = []
        self.steps_optimal = metric_ff_solver(planner = self.planner)
        self.mp_seconds = None
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
        new_goal = new_goal + f"(:domain {self.domain_list[0].name})"
        new_goal = new_goal + "\n(:objects)"
        new_goal = new_goal + "\n(:init "
        for start_fluent in goal.start_fluents:
            new_goal = new_goal + "\n" + start_fluent
        new_goal = new_goal + "\n)"
        """
        new_goal = new_goal + "\n(:goal (and "
        for goal_fluent in goal.goal_fluents:
            new_goal = new_goal + "\n" + goal_fluent
        #new_goal = new_goal + f"\n(obs_precondition_{step}))\n)"
        for i in range(step):
            new_goal = new_goal + f"\n(obs_precondition_{i+1})"
        new_goal = new_goal + ")\n)"
        new_goal = new_goal +"\n(:metric minimize (costs))\n)"
        """
        return new_goal
    def _create_new_start_fluents(self, step = 1):
        action_step = self.observation.obs_action_sequence.loc[step-1]
        action_title = action_step.split(" ")[0]
        goal = self.goal_list[step-1][0] #start fluents remain equal for all files
        if len(action_step.split(" ")) > 1:
            action_objects = action_step.split(" ")[1:]
        else:
            action_objects = []
        action_objects = [obj.lower() for  obj in action_objects]
        domain = self.domain_list[0]
        pddl_action = domain.action_dict[action_title]
        action_parameters = [param.parameter for param in pddl_action.action_parameters]
        zipped_parameters = list(zip(action_objects, action_parameters))
        effects = self._call_effect_check(pddl_action.action_effects, zipped_parameters)
        functions = [function[1:-1] for function in domain.functions]
        print(goal.start_fluents)
        new_start_fluents = goal.start_fluents
        for effect in effects:
            effect_is_func = len([function for function in functions if function in effect]) > 0
            if effect_is_func:
                identified_func = [function for function in functions if function in effect][0]
                idx_func_start_fluents = [i for i in range(len(new_start_fluents )) if identified_func in new_start_fluents[i]][0]
                replace_func = new_start_fluents[idx_func_start_fluents]
                curr_number = re.findall(r'\d+', replace_func)[0]
                effect_change_number = re.findall(r'\d+', effect)[0]
                if "increase" in effect:
                    new_start_fluents[idx_func_start_fluents] = replace_func.replace(curr_number,str(float(curr_number) +
                                                                                float(effect_change_number)))
                elif "decrease" in effect:
                    new_start_fluents[idx_func_start_fluents] = replace_func.replace(curr_number,str(float(curr_number) -
                                                                              float(effect_change_number)))
            else:
                effect = self._clean_effect(effect)
                if "(not(" in effect:
                    opposite = re.findall("\([\s*\w*\s*]*\)", effect)[0]
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

if __name__ == '__main__':
    toy_example_domain = pddl_domain('domain.pddl')
    problem_a = pddl_problem('problem_A.pddl')
    problem_b = pddl_problem('problem_B.pddl')
    problem_c = pddl_problem('problem_C.pddl')
    problem_d = pddl_problem('problem_D.pddl')
    problem_e = pddl_problem('problem_E.pddl')
    problem_f = pddl_problem('problem_F.pddl')
    toy_example_problem_list= [problem_a, problem_b, problem_c, problem_d, problem_e, problem_f]
    obs_toy_example = pddl_observations('Observations.csv')
    model = gm_model(toy_example_domain, toy_example_problem_list, obs_toy_example)
    print(model._create_new_start_fluents())




