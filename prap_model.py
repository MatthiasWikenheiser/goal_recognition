from pddl import pddl_domain, pddl_problem, pddl_observations
from metric_ff_solver import metric_ff_solver
import time
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pickle
def save_prap_model(model, filename):
    path = model.domain_list[0].domain_path.replace(model.domain_list[0].domain_path.split("/")[-1], "")
    with open(path + filename, "wb") as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
def load_prap_model(file):
    return pickle.load(open(file, "rb"))
class prap_model:
    """class that solves a goal recognition problem according to the vanilla plain approach Plan recognition as Planning
    (PRAP) by Ramirez and Geffner, 2010.
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
        self.predicted_step = {}
    def _create_obs_domain(self, step = 1):
        domain = self.domain_list[step-1]
        new_domain = f"(define (domain {domain.name})\n"
        new_domain = new_domain + domain.requirements + "\n"
        new_domain = new_domain + domain.types + "\n"
        new_domain = new_domain + domain.constants + "\n"
        new_domain = new_domain + self._create_obs_predicates(step) + "\n"
        new_domain = new_domain + "(:functions "
        for function in domain.functions:
            new_domain = new_domain + function + "\n"
        new_domain = new_domain + ")\n"
        for action in domain.action_dict.keys():
            new_domain = new_domain + domain.action_dict[action].action +"\n"
        new_domain = new_domain + self._create_obs_action(step) + "\n)"
        return new_domain
    def _create_obs_action(self, step):
        domain = self.domain_list[step-1]
        if step == 1:
            ob = [self.observation.obs_action_sequence[0]]
            action_key_curr = ob[0].split()[0]
        else:
            ob = self.observation.obs_action_sequence[:step]
            action_key_curr= ob[step-1].split()[0]
        cur_action = domain.action_dict[action_key_curr]
        new_action = f"(:action {cur_action.name}_obs_precondition_{step}"
        idx_parameter_strt = cur_action.action.find(":parameters")
        idx_parameter_end = idx_parameter_strt + cur_action.action[idx_parameter_strt:].find(")")
        new_action =  new_action + "\n" +  cur_action.action[idx_parameter_strt:idx_parameter_end + 1] +"\n"
        if step == 1:
            new_action = new_action  + ":precondition(" +  cur_action.action_preconditions + ")"
        else:
            before_pre = f"obs_precondition_{step-1}"
            new_action = new_action  + f":precondition(and({before_pre})(" + cur_action.action_preconditions + "))"
        cur_pre = f"obs_precondition_{step}"
        #check if parameters exist
        new_action = new_action + " :effect"
        idx_and = cur_action.action_effects.find("and")+3
        if len(cur_action.action_parameters) == 0:
            new_action = new_action + cur_action.action_effects[:idx_and] + f" ({cur_pre}) " + cur_action.action_effects[idx_and:] + ")"
        elif len(cur_action.action_parameters) == 1:
            new_action = new_action + cur_action.action_effects[:idx_and] + f" (when "
            new_action = new_action  + f"(= {cur_action.action_parameters[0].parameter} {ob[len(ob)-1].split()[0+1]}) "
            new_action = new_action + f" ({cur_pre})) " + cur_action.action_effects[idx_and:]  + ")"
        elif len(cur_action.action_parameters) > 1:
            new_action = new_action + cur_action.action_effects[:idx_and] + f" (when (and"
            for i in range(len(cur_action.action_parameters)):
                new_action = new_action + f"(= {cur_action.action_parameters[i].parameter} {ob[len(ob)-1].split()[i+1]}) "
            new_action = new_action + f") ({cur_pre})) " + cur_action.action_effects[idx_and:]  + ")"
        return new_action
    def _create_obs_predicates(self, step):
        domain = self.domain_list[step-1]
        predicates_string = "(:predicates"
        for predicate in domain.predicates:
            predicates_string = predicates_string + " " + predicate
        predicates_string = predicates_string + " " + f"(obs_precondition_{step}" + "))"
        return predicates_string
    def _create_obs_goal(self, goal_idx = 0, step = 1):
        goal = self.goal_list[0][goal_idx]
        new_goal = f"(define (problem {goal.name})\n"
        new_goal = new_goal + f"(:domain {self.domain_list[step-1].name})"
        new_goal = new_goal + "\n(:objects)"
        new_goal = new_goal + "\n(:init "
        for start_fluent in goal.start_fluents:
            new_goal = new_goal + "\n" + start_fluent
        new_goal = new_goal + "\n)"
        new_goal = new_goal + "\n(:goal (and "
        for goal_fluent in goal.goal_fluents:
            new_goal = new_goal + "\n" + goal_fluent
        #new_goal = new_goal + f"\n(obs_precondition_{step}))\n)"
        for i in range(step):
            new_goal = new_goal + f"\n(obs_precondition_{i+1})"
        new_goal = new_goal + ")\n)"
        new_goal = new_goal + f"\n(:metric minimize ({goal.metric_min_func}))\n)"
        return new_goal
    def _add_step(self, step= 1):
        path = self.domain_list[0].domain_path.replace(self.domain_list[0].domain_path.split("/")[-1],"") + "temp"
        if not os.path.exists(path):
            os.mkdir(path)
        domain_string = self._create_obs_domain(step)
        with open(path + f"/domain_obs_step_{step}.pddl", "w") as new_domain:
            new_domain.write(domain_string)
        self.domain_list.append(pddl_domain(path + f"/domain_obs_step_{step}.pddl"))
        new_goal_list = []
        for goal in range(len(self.goal_list[0])):
            goal_string = self._create_obs_goal(goal, step)
            with open(path + f"/goal_{goal}_obs_step_{step}.pddl", "w") as new_goal:
                new_goal.write(goal_string)
            new_goal_list.append(pddl_problem(path + f"/goal_{goal}_obs_step_{step}.pddl"))
        self.goal_list.append(new_goal_list)
        if step == 1:
            shutil.copy(f'{self.planner}', path + f'/{self.planner}')
    def _remove_step(self, step = 1):
        path = self.domain_list[0].domain_path.replace(self.domain_list[0].domain_path.split("/")[-1],"") + "temp"
        if os.path.exists(path):
            path_domain = self.domain_list[step].domain_path
            path_goal = [x.problem_path for x in self.goal_list[step]]
            self.domain_list = self.domain_list[0:step]
            self.goal_list = self.goal_list[0:step]
            os.remove(path_domain)
            for goal in path_goal:
                os.remove(goal)
            if step == 1:
                os.remove(path + f"/{self.planner}")
                os.rmdir(path)
    def perform_solve_observed(self, step = -1, priors = None, beta = 1, multiprocess = True):
        """
        BEFORE running this, RUN perform_solve_optimal!
        Solves the transformed pddL_domain and list of pddl_problems (goal_list) for specified steps
        from given obs_action_sequence.
        :param step: specifies how many observations in observation sequence get solved.
                     default (-1) results steps = length of observation sequence
        :param priors: priors of goal_list, default assigns equal probabilites to each goal
        :param beta: beta in P(O|G) and P(!O|G)
        :param multiprocess: if True, all transformed problems (goals) of one step are solved in parallel

        UNDER CONSTRUCTION - set timeout to time in obs_action_sequence

        """
        if step == -1:
            step =  self.observation.obs_len
        start_time = time.time()
        for i in range(step):
            step_time = time.time()
            print("step:", i+1, ",time elapsed:", round(step_time - start_time,2), "s")
            self._add_step(i+1)
            task = metric_ff_solver(planner = self.planner)
            task.solve(self.domain_list[i+1],self.goal_list[i+1], multiprocess = multiprocess)
            self.steps_observed.append(task)
            self.prob_dict_list.append(self._calc_prob(i+1, priors, beta)[0])
            self.prob_nrmlsd_dict_list.append(self._calc_prob(i+1, priors, beta)[1])
            self.predicted_step[i + 1] = self._predict_step(step=i)
        print("total time-elapsed: ", round(time.time() - start_time,2), "s")
        for i in range(step,0,-1):
            self._remove_step(i)
    def perform_solve_optimal(self, multiprocess = True, type_solver = '3', weight = '1', timeout = 90):
        """
        RUN before perform_solve_observed.
        Solves the optimal plan for each goal in goal_list.
        :param multiprocess: if True, all problems (goals) are solved in parallel
        :param type_solver: option for type solver in Metricc-FF Planner, however only type_solver = '3' ("weighted A*) is
         considered
        :param weight: weight for type_solver = '3' ("weighted A*); weight = '1' resolves to unweighted A*
        :param timeout: after specified timeout is reached, all process are getting killed.
        """
        start_time = time.time()
        self.steps_optimal.solve(self.domain_list[0],self.goal_list[0], multiprocess = multiprocess,
                   type_solver= type_solver, weight = weight, timeout = timeout)
        print("total time-elapsed: ", round(time.time() - start_time,2), "s")
        if multiprocess:
            self.mp_seconds = round(time.time() - start_time,2)
    def _calc_prob(self, step = 1, priors= None, beta = 1):
        if step == 0:
            print("step must be > 0 ")
            return None
        if priors == None:
            priors = np.array([1/len(self.goal_list[0]) for _ in range(len(self.goal_list[0]))])
        else:
            priors = np.array(priors)
        optimal_costs = [self.steps_optimal.plan_cost[key] for key in list(self.steps_optimal.plan_cost.keys())]
        optimal_costs = np.array(optimal_costs)
        p_optimal_costs_likeli = np.exp(-beta * optimal_costs)
        p_optimal = priors * p_optimal_costs_likeli
        observed_costs = [self.steps_observed[step-1].plan_cost[key] for key in list(self.steps_observed[step-1].plan_cost.keys())]
        observed_costs = np.array(observed_costs)
        p_observed_costs_likeli = np.exp(-beta * observed_costs)
        p_observed = priors * p_observed_costs_likeli
        prob = []
        prob_dict = {}
        for i in range(len(optimal_costs)):
            prob.append(p_observed[i]/(p_observed[i] + p_optimal[i]))
            key = list(self.steps_optimal.plan_cost.keys())[i]
            prob_dict[key] = p_observed[i]/(p_observed[i] + p_optimal[i])
            prob_dict[key]  = np.round(prob_dict[key], 4)
        prob_normalised_dict = {}
        for i in range(len(prob)):
            key = list(self.steps_optimal.plan_cost.keys())[i]
            prob_normalised_dict[key] = (prob[i]/(sum(prob)))
            prob_normalised_dict[key] = np.round(prob_normalised_dict[key], 4)
        return prob_dict, prob_normalised_dict

    def plot_prob_goals(self, figsize_x=8, figsize_y=5, adapt_y_axis=False):
        """
        RUN perform_solve_observed BEFORE.
        plots probability  for each goal to each step (specified perform_solve_observed) in of obs_action_sequence
        :param figsize_x: sets size of x-axis (steps)
        :param figsize_y: sets size of y-axis (probability)
        :return:
        """
        goal_name = [self.goal_list[0][i].name for i in range(len(self.goal_list[0]))]
        probs_nrmlsd = []
        for goal in goal_name:
            probs_nrmlsd.append([self.prob_nrmlsd_dict_list[step][goal] for step in range(len(self.steps_observed))])
        x = [step for step in range(1, len(self.steps_observed) + 1)]
        plt.figure(figsize=(figsize_x, figsize_y))
        for i in range(len(probs_nrmlsd)):
            plt.plot(x, probs_nrmlsd[i], label=goal_name[i])
        plt.legend()
        plt.xticks(range(1, len(self.steps_observed) + 1))
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.xlim(1, len(self.steps_observed))
        if adapt_y_axis:
            max_prob = 0
            for step_dict in self.prob_nrmlsd_dict_list:
                max_prob_step = max([step_dict[key] for key in list(step_dict.keys())])
            if max_prob_step > max_prob:
                max_prob = max_prob_step
            ticks = np.array([0, 0.25, 0.5, 0.75, 1])
            plt.ylim(np.min(ticks[max_prob > ticks]), np.min(ticks[max_prob < ticks]))
        else:
            plt.ylim(0, 1)
        plt.grid()
        plt.show()
    def _predict_step(self, step):
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
    model = prap_model(toy_example_domain, toy_example_problem_list, obs_toy_example)
    model.perform_solve_optimal(multiprocess=True)
    model.perform_solve_observed(multiprocess=True)
    print(model.predicted_step)
    print(model.prob_nrmlsd_dict_list)

