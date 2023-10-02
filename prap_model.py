from pddl import pddl_domain, pddl_problem, pddl_observations
from metric_ff_solver import metric_ff_solver
import gr_model
import time
import os
import shutil
import psutil
import numpy as np
import threading
import gm_model
class prap_model(gr_model.gr_model):
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
        super().__init__(domain_root, goal_list, obs_action_sequence, planner)
        self.domain_list = [self.domain_root]
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
    def _create_obs_goal(self, goal, step = 1):
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
        if "goals_remaining" not in self.observation.obs_file.columns:
            for goal_idx in range(len(self.goal_list[0])):
                goal = self.goal_list[0][goal_idx]
                goal_string = self._create_obs_goal(goal, step)
                with open(path + f"/goal_{goal_idx}_obs_step_{step}.pddl", "w") as new_goal:
                    new_goal.write(goal_string)
                new_goal_list.append(pddl_problem(path + f"/goal_{goal_idx}_obs_step_{step}.pddl"))
            #self.goal_list.append(new_goal_list)
        else:
            for goal in self.goal_list[0]:
                if goal.name in self.observation.obs_file.loc[step - 1, "goals_remaining"]:
                    goal_string = self._create_obs_goal(goal, step)
                    goal_idx = goal.name.split("_")[-1]
                    with open(path + f"/goal_{goal_idx}_obs_step_{step}.pddl", "w") as new_goal:
                        new_goal.write(goal_string)
                    new_goal_list.append(pddl_problem(path + f"/goal_{goal_idx}_obs_step_{step}.pddl"))
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
    def _thread_solve(self, i, multiprocess, time_step):
        #print([goal.problem_path for goal in self.goal_list[i + 1]])

        ##            task = metric_ff_solver(planner = self.planner)
            ##task.solve(self.domain_list[i+1],self.goal_list[i+1], multiprocess = multiprocess, timeout=time_step)


        self.task_thread_solve = metric_ff_solver(planner=self.planner)
        if len(self.goal_list[i + 1]) > 0:
            if len(self.domain_root.domain_path.split("/")) == 1:
                base_domain = self.domain_root.domain_path.replace(".pddl","")
            else:
                base_domain = self.domain_root.domain_path.split("/")[-1].replace(".pddl","")
            if self.crystal_island:
                self.task_thread_solve.solve(self.domain_list[i+1], self.goal_list[i + 1], multiprocess=multiprocess,
                                             timeout=time_step,
                                             # base_domain= self.domain_root.domain_path.replace(".pddl",""),
                                             base_domain= base_domain,
                                             observation_name= self.observation.observation_path.split("/")[-2]+ "-" +
                                                               self.observation.name)
            else:
                self.task_thread_solve.solve(self.domain_list[i+1], self.goal_list[i + 1], multiprocess=multiprocess,
                                         timeout=time_step,
                                         #base_domain= self.domain_root.domain_path.replace(".pddl",""),
                                         base_domain = base_domain,
                                         observation_name = self.observation.name)
                                         #, observation_name= self.observation.name)

    def perform_solve_observed(self, step = -1, priors = None, beta = 1, multiprocess = True, gm_support = False):
        """
        BEFORE running this, RUN perform_solve_optimal!
        Solves the transformed pddL_domain and list of pddl_problems (goal_list) for specified steps
        from given obs_action_sequence.
        :param step: specifies how many observations in observation sequence get solved.
                     If set to -1 (default) entire observation sequence is solved
        :param priors: expects dictionary with priors of goal_list, default assigns equal probabilites to each goal.
        :param beta: beta in P(O|G) and P(!O|G)
        :param multiprocess: if True, all transformed problems (goals) of one step are solved in parallel

        UNDER CONSTRUCTION - set timeout to time in obs_action_sequence

        """
        if step == -1:
            step =  self.observation.obs_len
        start_time = time.time()
        if gm_support:
            path_support = self.domain_list[0].domain_path.replace(self.domain_list[0].domain_path.split("/")[-1],
                                                                   "") + "temp"
            if self.changed_domain_root is None:
                gm_support_model = gm_model.gm_model(self.domain_root,self.goal_list,
                                                 self.observation, self.planner)
            else:
                gm_support_model = gm_model.gm_model(self.changed_domain_root, self.goal_list[0],
                                                  self.observation, self.planner)
            gm_support_model.domain_temp = self.domain_root
            if not os.path.exists(path_support):
                os.mkdir(path_support)
        for i in range(step):
            time_step = self.observation.obs_file.loc[i, "diff_t"]
            step_time = time.time()
            print("step:", i+1, ",time elapsed:", round(step_time - start_time,2), "s")
            if gm_support:
                new_goal_support_list = []
                for goal in range(len(self.goal_list[-1])):
                    if self.goal_list[-1][goal].name in self.observation.obs_file.loc[i, "goals_remaining"]:
                        goal_string_support = gm_support_model._create_obs_goal(goal_idx = goal, step = i+1)
                        path_support_file = path_support + f"/gm_support_goal_{goal+1}_obs_step_{i}.pddl"
                        with open(path_support_file, "w") as new_goal_support:
                            new_goal_support.write(goal_string_support)
                        new_goal_support_list.append(pddl_problem(path_support_file))
                gm_support_model.goal_list.append(new_goal_support_list)
            print(self.observation.obs_file.loc[i, "action"] + ", " + str(time_step) + " seconds to solve")
            self._add_step(i + 1)
            try:
                time_step = 7
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
                # print("task.solved: ",self.task_thread_solve.solved )
                if not (self.task_thread_solve.solved == 0 and (
                        time.time() - check_failure_t <= time_step + 10) and len(
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
                result_probs = self._calc_prob(i + 1)
                for g in self.goal_list[i+1]:
                    if g.name not in result_probs[0].keys():
                        result_probs[0][g.name] = 0.00
                        result_probs[1][g.name] = 0.00
                self.prob_dict_list.append(result_probs[0])
                self.prob_nrmlsd_dict_list.append(result_probs[1])
                self.predicted_step[i+1] = self._predict_step(step= i)
            else:
                [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
                print("failure, read in files ")
                failure_task = metric_ff_solver()
                failure_task.problem = self.goal_list[i + 1]
                failure_task.domain = self.domain_list[i + 1]
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
                result_probs = self._calc_prob(i + 1)
                for g in self.goal_list[i + 1]:
                    if g.name not in result_probs[0].keys():
                        result_probs[0][g.name] = 0.00
                        result_probs[1][g.name] = 0.00
                self.prob_dict_list.append(result_probs[0])
                self.prob_nrmlsd_dict_list.append(result_probs[1])
                self.predicted_step[i + 1] = self._predict_step(step=i)
        print("total time-elapsed: ", round(time.time() - start_time,2), "s")
        if gm_support:
            for gl in gm_support_model.goal_list[1:]:
                for g in gl:
                    os.remove(g.problem_path)
        for i in range(step,0,-1):
            self._remove_step(i)
    def _calc_prob(self, step = 1, priors= None, beta = 1):
        if step == 0:
            print("step must be > 0 ")
            return None
        if priors == None:
            priors_dict = {}
            for key in self.steps_observed[step - 1].plan_achieved.keys():
                priors_dict[key] = 1 / len(self.goal_list[step])
        else:
            priors_dict = {}
            for key in self.steps_observed[step - 1].plan_achieved.keys():
                priors_dict[key] = priors[key]
        p_observed = {}
        p_optimal = {}
        for key in self.steps_observed[step - 1].plan_achieved.keys():
            optimal_costs = self.steps_optimal.plan_cost[key]
            p_optimal_costs_likeli = np.exp(-beta * optimal_costs)
            p_optimal[key] = priors_dict[key] * p_optimal_costs_likeli
            observed_costs = self.steps_observed[step - 1].plan_cost[key]
            p_observed_costs_likeli = np.exp(-beta * observed_costs)
            p_observed[key] = priors_dict[key] * p_observed_costs_likeli
        prob = []
        prob_dict = {}
        for i in range(len(self.steps_observed[step - 1].plan_achieved.keys())):
            key = list(self.steps_observed[step - 1].plan_achieved.keys())[i]
            prob.append(p_observed[key] / (p_observed[key] + p_optimal[key]))
            prob_dict[key] = p_observed[key] / (p_observed[key] + p_optimal[key])
            prob_dict[key] = np.round(prob_dict[key], 4)
        prob_normalised_dict = {}
        for i in range(len(prob)):
            key = list(self.steps_observed[step - 1].plan_achieved.keys())[i]
            prob_normalised_dict[key] = (prob[i] / (sum(prob)))
            prob_normalised_dict[key] = np.round(prob_normalised_dict[key], 4)
        return prob_dict, prob_normalised_dict
    def plot_prob_goals(self, figsize_x=8, figsize_y=5, adapt_y_axis=False):
        return super().plot_prob_goals(figsize_x=figsize_x, figsize_y=figsize_y, adapt_y_axis=adapt_y_axis)
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
    print(model.hash_code)
    model.perform_solve_optimal(multiprocess=True)
    print(model.steps_optimal.plan)

    model.perform_solve_observed(multiprocess=True)
    print(model.predicted_step)
    print(model.prob_nrmlsd_dict_list)

