import os
import time
from time import sleep
import datetime as dt
import threading
import psutil
from multiprocessing import Process
import subprocess
from pddl import pddl_domain
from pddl import pddl_problem
path_error = "/home/mwiubuntu/error_write/"
class metric_ff_solver:
    """Call of MetricFF Planner with specified pddl_domain and at least one pddl_problem.
    ATTENTION: MetricFF Planner must be compiled and available in pddl_domain.domain_path
    """
    def __init__(self, planner = "ff_2_1"):
        """:param planner: name of executable planner, here default ff_2_1 (MetricFF Planner version 2.1)"""
        self.summary = {}
        self.plan = {}
        self.plan_cost = {}
        self.plan_achieved = {}
        self.time = {}
        self.problem = []
        self.solved = 0 #0:not tried yet, 1: success, 2:timeout
        self.planner = planner
    def _legal_plan(self, summary):
        dict_plan = {}
        idx_str = summary.find("step")
        idx_end = summary.find("plan cost") -1
        plan_string = summary[idx_str + 4 :idx_end].split("\n")
        for step in plan_string:
            idx = step.find(":")
            step_number = int(step[:idx].replace(" ", ""))
            action = step[idx+2:]
            dict_plan[step_number] = action
        return(dict_plan)
    def _cost(self, summary, file_path = None):
        idx_strt = summary.find("plan cost")
        return(float(summary[idx_strt:].split()[2]))
    def _time_2_solve(self, summary, file_path = None):
        idx_strt = summary.find("plan cost")
        idx_strt = summary.find("max depth")
        idx_end = summary.find("seconds total time")
        time_string = summary[idx_strt:idx_end].split("\n")
        t = time_string[-1]
        return(float(t))
    def _path(self):
        path = self.domain.domain_path.split("/")[:-1]
        path_result = path[0]
        for el in path[1:]:
            path_result = path_result + "/" + el
        path_result += "/"
        return path_result
    def _read_in_output(self):
        path = self.path
        achieved_goals = [g for g in os.listdir(path) if "output_goal_" in g and ".txt" in g]
        #print(achieved_goals)
        for g in achieved_goals:
            key = g.replace(".txt", "").replace("output_goal_", "")
            with open(path + g) as read_output:
                self.summary[key] = read_output.read()
            if self.summary[key] != "":
                self.plan[key] = self._legal_plan(self.summary[key])
                self.plan_cost[key] = self._cost(self.summary[key])
                self.plan_achieved[key] = 1
                self.time[key] = self._time_2_solve(self.summary[key])
            os.remove(path + g)
    def solve(self, domain, problem, multiprocess = True, type_solver = '3', weight = '1', timeout = 90.0,
              base_domain = None, observation_name = None):
        """solves one or many pddl_problem(s), given a pddl_domain and provides a data structure for result outputs
        :param domain: pddl_domain object
        :param problem: one or many (list) pddl_problem objects
        :param multiprocess: if True, list of pddl_problem objects are solved with multiprocessing (parallelized)
        :param type_solver: option for type solver in Metricc-FF Planner, however only type_solver = '3' ("weighted A*) is
         considered
        :param weight: weight for type_solver = '3' ("weighted A*); weight = '1' resolves to unweighted A*
        :param timeout: after specified timeout is reached, all process are getting killed
        """
        all_solved = 1
        if type(problem) != list:
            problem = [problem]
        self.problem = problem
        self.domain = domain
        self.domain_path = self.domain.domain_path.split("/")[-1]
        self.path = self._path()
        self.type_solver= type_solver
        self.weight = weight
        if self.path != "":
            os.chdir(self.path)
        self.problem_path = [self.problem[i].problem_path.split("/")[-1] for i in range(len(self.problem))]
        if multiprocess == False:
            i = 0
            for p in self.problem:
                start_time = time.time()
                if len(self.problem) > 1: print(p.name)
                t = threading.Thread(target = self._run_metric_ff_loop, args =[self.domain_path,self.problem_path[i], i])
                t.start()
                while t.is_alive():
                    is_time_left = time.time() - start_time < timeout
                    if not is_time_left:
                        all_solved = 0
                        print("time_out finished")
                        [x.kill() for x in psutil.process_iter() if self.planner in x.name()]
                    time.sleep(0.1)
                i = i+1
            if all_solved == 1:
                self.solved = 1
            else:
                self.solved = 2
        else:
            self.processes = {}
            if self.path != "":
                if os.path.basename(os.getcwd()) != os.path.basename(self.path):
                  os.chdir(self.path)
            for i in range(len(self.problem)):
                key = self.problem[i].name
                self.processes[key] = Process(target=self._run_metric_ff_mp, args=[self.domain_path,
                                                                           self.problem_path[i],i,key, base_domain,
                                                                                   observation_name])
            for i in range(len(self.problem)):
                key = self.problem[i].name
                print("start_", self.processes[key])
                self.processes[key].start()
            start_time = time.time()
            read_in = False
            while(len([x for x in psutil.process_iter() if self.planner in x.name()]) != 0):
                sleep(0.1)
                if time.time() - start_time >= timeout:
                    self.solved = 2
                    print("time_out finished")
                    self._read_in_output()
                    read_in = True
                    [x.kill() for x in psutil.process_iter() if self.planner in x.name()]
            if not read_in:
                self._read_in_output()
            self.solved = 1
            self.processes = {}
            #if self.solved != 0:
                #[os.remove(fp) for fp in files]
    def _run_metric_ff_mp(self, domain, problem, i, key, base_domain, observation_name):
        command_string = f"./{self.planner} -o {domain} -f {problem} -s {self.type_solver} -w {self.weight}"
        #print(command_string)
        path = ""
        for path_pc in domain.split("/")[:-1]:
            path = path + path_pc +"/"
        try:
            output = subprocess.check_output(command_string, shell = True)
            with open(path + f"output_goal_{key}.txt", "w") as output_write:
                output_write.write(output.decode("ascii"))
                print(key)
        except subprocess.CalledProcessError as e:
            error = e.output.decode("ascii")
            if ("advancing to goal distance:" not in error and error != ""):
                translate_neg_cond_error = False
                if error.splitlines()[-1].startswith("translating negated cond") \
                        or error.splitlines()[-2].startswith("translating negated cond") \
                        or error.splitlines()[-3].startswith("translating negated cond"):
                    translate_neg_cond_error = True
                if not translate_neg_cond_error:
                    print(f"-------------ERROR-------------")
                    #print(error)
                    now = dt.datetime.now()
                    year = str(now.year)
                    month = str(now.month) if now.month > 10 else '0' + str(now.month)
                    day = str(now.day) if now.day > 10 else '0' + str(now.day)
                    hour = str(now.hour) if now.hour > 10 else '0' + str(now.hour)
                    minute = str(now.minute) if now.minute > 10 else '0' + str(now.minute)
                    seconds = str(now.second) if now.second > 10 else '0' + str(now.second)
                    tmstmp = f"{year}{month}{day}-{hour}{minute}{seconds}"
                    d = domain.replace(".pddl", "")
                    p = problem.replace(".pddl", "")
                    if base_domain is None or observation_name is None:
                        with open(path_error + f"error_{tmstmp}_{d}_{p}.txt", "w") as error_write:
                            error_write.write(error)
                    else:
                        with open(path_error + f"error_{tmstmp}_{base_domain}_{observation_name}_{p}.txt", "w") as error_write:
                            error_write.write(error)
                    print("-------------ERROR-------------")
    def _run_metric_ff_loop(self, domain, problem, i):
        command_string = f"./{self.planner} -o {domain} -f {problem} -s {self.type_solver} -w {self.weight}"
        output = subprocess.check_output(command_string, shell = True)
        key = self.problem[i].name
        self.summary[key] = output.decode('ascii')
        self.plan[key] = self._legal_plan(self.summary[key])
        self.plan_cost[key] = self._cost(self.summary[key])
        self.time[key] = self._time_2_solve(self.summary[key])

if __name__ == '__main__':
    toy_example_domain = pddl_domain('domain.pddl')
    problem_a = pddl_problem('problem_A.pddl')
    problem_b = pddl_problem('problem_B.pddl')
    problem_c = pddl_problem('problem_C.pddl')
    task = metric_ff_solver()
    task.solve(toy_example_domain, [problem_a, problem_b])
    print(task.plan)
