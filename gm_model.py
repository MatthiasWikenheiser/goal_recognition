from pddl import pddl_domain, pddl_problem, pddl_observations
from metric_ff_solver import metric_ff_solver

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
    def _split_recursive_and_or(string, zipped_parameters, key_word):
        split_list = []
        strt_idx = string.find(key_word) + len(key_word)
        new_string = string[strt_idx:]
        strt_idx += new_string.find("(")
        end_idx = len(string) - 1
        parse_back = True
        while end_idx > 0 and parse_back:
            if string[end_idx] == ")":
                string = string[:end_idx]
                parse_back = False
            end_idx -= -1
        c = strt_idx + 1
        parse_bracket = 1
        while c < len(string):
            if string[c] == "(":
                parse_bracket += 1
            if string[c] == ")":
                parse_bracket -= 1
            if parse_bracket == 0:
                split_list.append(string[strt_idx:c + 1])
                strt_idx = c + string[c:].find("(")
                c = strt_idx + 1
                parse_bracket = 1
            c += 1
        return split_list
    def recursive_effect_check(self, string, zipped_parameters, key_word=None):
        print("------------")
        print(string)
        string_cleaned_blanks = string.replace("\t", "").replace(" ", "").replace("\n", "")
        if string_cleaned_blanks.startswith("(when"):
            key_word = "when"
        elif string_cleaned_blanks.startswith("(and"):
            key_word = "and"
        elif string_cleaned_blanks.startswith("(or"):
            key_word = "or"
        if key_word in ["and", "or"]:
            c = len(string) - 1
            parse = True
            while c > 0 and parse:
                if string[c] == ")":
                    new_string = string[string.find(key_word) + len(key_word):c]
                    parse = False
                c -= -1
        elif key_word == "when":
            new_string = string[string.find("when"):]
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
                    new_string = new_string[:c + 1]
                    parse = False
                c += 1
        if key_word is None:
            return string
        else:
            self.recursive_effect_check(new_string, zipped_parameters, key_word)


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
    def _get_effects_from_obs_action_sequence(self, step = 1):
        action_step = self.observation.obs_action_sequence.loc[step-1]
        action_title = action_step.split(" ")[0]
        if len(action_step.split(" ")) > 1:
            action_objects = action_step.split(" ")[1:]
        else:
            action_objects = []
        domain = self.domain_list[0]
        pddl_action = domain.action_dict[action_title]
        action_effects = pddl_action.action_effects
        action_parameters = [param.parameter for param in pddl_action.action_parameters]
        zipped_parameters = list(zip(action_objects, action_parameters))
        print(zipped_parameters)
        if action_effects.replace(" ", "").replace("\n", "").replace("\t", "").startswith("and"):
            effect_elements = []
            print("begin parse")
            print(action_effects)
            idx_strt = action_effects.find("(")
            parse_bracket = 0
            c = 0
            parse_string = action_effects[idx_strt:]
            idx_strt = 0
            print(parse_string[idx_strt:])

            while c < len(parse_string):
                if parse_string[c] == "(":
                    parse_bracket += 1
                if parse_string[c] == ")":
                    parse_bracket -= 1
                if parse_bracket == 0:
                    effect_elements.append(parse_string[idx_strt:c+1])
                    #print(parse_string[idx_strt:c+1])
                    idx_strt = c + parse_string[c:].find("(")
                    c = idx_strt - 1
                c += 1
            return(effect_elements)

        else:
            return [action_effects]




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
    #print(model._create_obs_goal())
    #print(obs_toy_example.obs_action_sequence)
    print(model._get_effects_from_obs_action_sequence())
