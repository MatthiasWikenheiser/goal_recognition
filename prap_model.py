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
import pandas as pd
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
        new_domain = new_domain + '(:constants '
        for constant_type in domain.constants.keys():
            for constant in domain.constants[constant_type]:
                new_domain += constant + " "
            new_domain += f"- {constant_type} "
        new_domain += ")\n"
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
    def _create_obs_con_goal(self, goal, step = 1):
        new_goal = f"(define (problem {goal.name})\n"
        new_goal = new_goal + f"(:domain {self.domain_list[step - 1].name})"
        new_goal = new_goal + "\n(:objects)"
        new_goal = new_goal + "\n(:init "
        for start_fluent in goal.start_fluents:
            new_goal = new_goal + "\n" + start_fluent
        for i in range(step):
            new_goal = new_goal + f"\n(obs_precondition_{i + 1})"
        new_goal = new_goal + "\n)"
        new_goal = new_goal + "\n(:goal (and "
        for goal_fluent in goal.goal_fluents:
            new_goal = new_goal + "\n" + goal_fluent
        new_goal = new_goal + ")\n)"
        new_goal = new_goal + f"\n(:metric minimize ({goal.metric_min_func}))\n)"
        return new_goal
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
            self.observation.obs_file["goals_remaining"] = self.observation.obs_file["goals_remaining"].astype(str)
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
        self._remove_current_step = step
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

    #def __reduce__(self):
     #   return (self.__class__, (self.found_errors,))
    def test_observations(self, test_i = 0):
        """tests whether the fiven observation file is valid"""
        step = self.observation.obs_len
        self.found_errors = []
        self.test_path = ""
        for el in self.domain_root.domain_path.split("/")[:-1]:
            self.test_path += el + "/"
        self.test_path += f"test_{test_i}/"
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)
        if self.changed_domain_root is None:
            self.gm_test_model = gm_model.gm_model(self.domain_root, self.goal_list[0],
                                                      self.observation, self.planner)
        else:
            self.gm_test_model = gm_model.gm_model(self.changed_domain_root, self.goal_list[0],
                                                      self.observation, self.planner)
        self.gm_test_model.domain_temp = self.domain_root
        list_problem_obs_con = []
        i = 0
        while i < step:
            #if i % 50 == 0:
             #   print(f"checked {i}/{step} steps")
            cur_ob = self.observation.obs_file.loc[i, "action"]
            new_goal_test_list = []
            for goal in range(len(self.goal_list[-1])):
                if self.goal_list[-1][goal].name in self.observation.obs_file.loc[i, "goals_remaining"]:
                    goal_string_test = self.gm_test_model._create_obs_goal(goal_idx=goal, step=i + 1)
                    path_test_file = self.test_path + f"/gm_test_goal_{goal + 1}_obs_step_{i}.pddl"
                    with open(path_test_file, "w") as new_goal_test:
                        new_goal_test.write(goal_string_test)
                    new_problem = pddl_problem(path_test_file)
                    new_goal_test_list.append(new_problem)
            self.goal_list.append(new_goal_test_list)
            if i == 0:
                keep_new_problem = new_problem
                problem_obs_con_test = self.goal_list[0][-1]
            if i > 0:
                obs_con_test_str = self._create_obs_con_goal(keep_new_problem, i)
                keep_new_problem = new_problem
                path_obs_con_test = self.test_path + f"gm_obs_con_test_{goal + 1}_obs_step_{i}.pddl"
                with open(path_obs_con_test, "w") as new_goal_test:
                    new_goal_test.write(obs_con_test_str)
                problem_obs_con_test = pddl_problem(path_obs_con_test)
                list_problem_obs_con.append(problem_obs_con_test)
            self.gm_test_model.goal_list.append(new_goal_test_list)
            domain_test_string = self._create_obs_domain(i+1)
            path_obs_con_domain = self.test_path + f"domain_obs_step_{i+1}.pddl"
            with open(path_obs_con_domain, "w") as new_domain_test:
                new_domain_test.write(domain_test_string)
            self.domain_list.append(pddl_domain(path_obs_con_domain))
            new_action_key = \
            [action for action in self.domain_list[i + 1].action_dict.keys() if f"OBS_PRECONDITION_{i + 1}" in action][
                0]
            action = self.domain_list[i + 1].action_dict[new_action_key]
            action_possible = gr_model._is_action_possible(action,
                                                           problem_obs_con_test.start_fluents,
                                                           cur_ob)
            params_in_constants = True
            error_constants = []
            for fluent in problem_obs_con_test.start_fluents:
                fluent = fluent.replace("(", "").replace(")", "")
                if len(fluent.split(" ")) > 1 and "=" not in fluent:
                    key_predicate = fluent.split(" ")[0]
                    for p in range(len(self.domain_list[i + 1].predicates_dict[key_predicate])):
                        var_type =  self.domain_list[i + 1].predicates_dict[key_predicate][p][1]
                        param = fluent.split(" ")[p+1]
                        rel_constants = self.domain_list[i + 1].constants[var_type]
                        if param not in rel_constants:
                            error_constants.append(param)
                            params_in_constants = False
            dict_error = {"step": i, "obs": cur_ob}
            if not action_possible or not params_in_constants:
                print(f'step {i + 1}: {cur_ob}')
            if not action_possible:
                dict_error["action"] = action
                cur_state = ""
                for fl in problem_obs_con_test.start_fluents:
                    cur_state += fl+"\n"
                dict_error["cur_state"] = cur_state
                print(f"action_possible: {action_possible}")
            if not params_in_constants:
                dict_error["error_constants"] = error_constants
                print("params_in_constants: ", params_in_constants)
            if not action_possible or not params_in_constants:
                self.found_errors.append(dict_error)
            i+=1
        for gl in self.gm_test_model.goal_list[1:]:
            for g in gl:
                os.remove(g.problem_path)
        self.domain_list = self.domain_list[:1]
        self.goal_list = self.goal_list[:1]
        for i in range(step):
            os.remove(self.test_path + f"domain_obs_step_{i+1}.pddl")
        for o in list_problem_obs_con:
            os.remove(o.problem_path)
        os.rmdir(self.test_path)
    def perform_solve_observed(self, step = -1, priors = None, beta = 1, multiprocess = True, gm_support = False, _i = 1):
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
            list_problem_obs_con = []
            path_support = self.domain_list[0].domain_path.replace(self.domain_list[0].domain_path.split("/")[-1],
                                                                   "") + "temp"
            if self.changed_domain_root is None:
                self.gm_support_model = gm_model.gm_model(self.domain_root,self.goal_list[0],
                                                 self.observation, self.planner)
            else:
                self.gm_support_model = gm_model.gm_model(self.changed_domain_root, self.goal_list[0],
                                                  self.observation, self.planner)
            self.gm_support_model.domain_temp = self.domain_root
            if not os.path.exists(path_support):
                os.mkdir(path_support)
        i = 0
        domain_bug = False
        while i < step and not domain_bug:
            time_step = self.observation.obs_file.loc[i, "diff_t"]
            step_time = time.time()
            print("step:", _i, ",time elapsed:", round(step_time - start_time,2), "s")
            if gm_support:
                #print(self._create_obs_action(i+1))
                new_goal_support_list = []
                for goal in range(len(self.goal_list[-1])):
                    if self.goal_list[-1][goal].name in self.observation.obs_file.loc[i, "goals_remaining"]:
                        goal_string_support = self.gm_support_model._create_obs_goal(goal_idx = goal, step = i+1)
                        path_support_file = path_support + f"/gm_support_goal_{goal+1}_obs_step_{i}.pddl"
                        with open(path_support_file, "w") as new_goal_support:
                            new_goal_support.write(goal_string_support)
                        new_problem = pddl_problem(path_support_file)
                        new_goal_support_list.append(new_problem)
                if i == 0:
                    keep_new_problem = new_problem
                    problem_obs_con_support = self.goal_list[0][-1]
                if i > 0:
                    obs_con_support_str = self._create_obs_con_goal(keep_new_problem, i)
                    keep_new_problem = new_problem
                    path_obs_con_support = path_support + f"/gm_obs_con_support_{goal + 1}_obs_step_{i}.pddl"
                    with open(path_obs_con_support, "w") as new_goal_support:
                        new_goal_support.write(obs_con_support_str)
                    problem_obs_con_support = pddl_problem(path_obs_con_support)
                    list_problem_obs_con.append(problem_obs_con_support)
                self.gm_support_model.goal_list.append(new_goal_support_list)
            print(self.observation.obs_file.loc[i, "action"] + ", " + str(time_step) + " seconds to solve")
            self._add_step(i + 1)
            if gm_support:
                new_action_key = [action for action in self.domain_list[i + 1].action_dict.keys() if f"OBS_PRECONDITION_{i + 1}" in action][0]
                action = self.domain_list[i + 1].action_dict[new_action_key]
                action_possible = gr_model._is_action_possible(action,
                                           problem_obs_con_support.start_fluents,
                                           self.observation.obs_file.loc[i, "action"])
                print(action_possible)
                if not action_possible:
                    error_action_possible = f"error in observation: {self.observation.observation_path}\n"
                    error_action_possible += f"error caused by observation: {self.observation.obs_file.loc[i, 'action']}\n"
                    error_action_possible += "action not possible:\n"
                    error_action_possible += action.action + "\n\n\n"
                    error_action_possible += "state at moment of error: \n"
                    for fluent in problem_obs_con_support.start_fluents:
                        error_action_possible += f"{fluent}\n"
                    corrected_observation_name = self.observation.observation_path.split("/")[-3] + "_" + self.observation.observation_path.split("/")[-2]
                    path_error_action_possible = (self.path_error_env + "error_action_possible_ " + corrected_observation_name + "_" + self.observation.name +
                                                  "_step_" + str(i+1) + ".txt")
                    with open(path_error_action_possible, "w") as new_goal_support:
                        new_goal_support.write(error_action_possible)
            try:
                time_step = 5
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
            if i> 0 and (i+1) % 50 == 0:
                print("reached 50 steps, set failure and mff_bug to true")
                failure = True
                time.sleep(5)
            if not failure:
                self.steps_observed.append(self.task_thread_solve)
            else:
                [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
                print("failure, read in files ")
                mff_bug = False
                if len(os.listdir(self.path_error_env )) != len(self.error_write_files) and gm_support:
                    for error_file in [x for x in os.listdir(self.path_error_env) if x not in self.error_write_files]:
                        read_error = open(self.path_error_env + error_file, "r").read()
                        if "unknown optimization method" in read_error:
                            mff_bug = True
                            domain_bug = True
                            os.remove(self.path_error_env + error_file)
                        else:
                            domain_bug = True
                if i> 0 and (i+1) % 50 == 0:
                    mff_bug = True
                    domain_bug = True
                if mff_bug:
                    print("-------bug reached---- wait 40 s")
                    time.sleep(40)
                    [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
                    print("instantiate new prap_model")
                    time.sleep(5)
                    bug_path_split = self.domain_list[i].domain_path.split("/")[:-1]
                    temp_path = ""
                    for slice in bug_path_split:
                        temp_path += slice +"/"
                    bug_path = temp_path + "bug_path/"
                    print("bug_path :", bug_path)
                    os.mkdir(bug_path)
                    shutil.copy(f'{self.planner}', bug_path + f'/{self.planner}')
                    shutil.copy(self.domain_root.domain_path, bug_path + "domain_root.pddl")
                    gm_support_step_goals = [x for x in os.listdir(temp_path) if "gm_support_goal" in x
                                             and f"obs_step_{i-1}.pddl" in x]
                    bug_goal_list = []
                    for gm_support_goal in gm_support_step_goals:
                        shutil.copy(temp_path + gm_support_goal, bug_path + gm_support_goal)
                        bug_goal_list.append(pddl_problem(bug_path + gm_support_goal))
                    bug_observation_left = pd.read_csv(self.observation.observation_path)
                    bug_observation_left = bug_observation_left.iloc[i:, :]
                    bug_observation_left = bug_observation_left.reset_index().iloc[:,1:]
                    bug_observation_left.to_csv(bug_path + "bug_observation_left.csv", index = False)
                    self.bug_prap_model = prap_model(pddl_domain(bug_path + "domain_root.pddl"), bug_goal_list,
                                                pddl_observations(bug_path + "bug_observation_left.csv"))
                    #print("check optimal_feasible")
                    #bug_prap_model.perform_solve_optimal(multiprocess=True, type_solver='3', weight='1',
                                                         #timeout=self.chosen_optimal_timeout)
                    self.bug_prap_model.steps_optimal = self.steps_optimal
                    print("continue check")
                    time.sleep(10)
                    self.bug_prap_model.perform_solve_observed(step = -1, priors = priors, beta= beta,
                                                          multiprocess= multiprocess, gm_support= gm_support, _i = _i)
                    os.remove(bug_path + f'/{self.planner}')
                    os.remove(bug_path + "domain_root.pddl")
                    for gm_support_goal in gm_support_step_goals:
                        os.remove(bug_path + gm_support_goal)
                    os.remove(bug_path + "bug_observation_left.csv")
                    os.rmdir(bug_path)
                if not mff_bug:
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
            i += 1
            _i += 1
        print("total time-elapsed: ", round(time.time() - start_time,2), "s")
        if gm_support:
            for gl in self.gm_support_model.goal_list[1:]:
                for g in gl:
                    os.remove(g.problem_path)
            for obs_con in list_problem_obs_con:
                os.remove(obs_con.problem_path)
        for i in range(step):
            result_probs = self._calc_prob(i + 1)
            for g in self.goal_list[i + 1]:
                if g.name not in result_probs[0].keys():
                    result_probs[0][g.name] = 0.00
                    result_probs[1][g.name] = 0.00
            self.prob_dict_list.append(result_probs[0])
            self.prob_nrmlsd_dict_list.append(result_probs[1])
            self.predicted_step[i + 1] = self._predict_step(step=i)
        self.summary_level_1, self.summary_level_2, self.summary_level_3 = self._create_summary()
        for j in range(i+1,0,-1):
            self._remove_step(j)
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
    print(model.path_error_env)
    print(model.error_write_files)
    model.perform_solve_optimal(multiprocess=True)
    print(model.steps_optimal.plan)

    model.perform_solve_observed(multiprocess=True)
    print(model.predicted_step)
    print(model.prob_nrmlsd_dict_list)

