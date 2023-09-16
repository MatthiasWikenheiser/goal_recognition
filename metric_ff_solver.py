import os
import time
from time import sleep
import threading
import psutil
from multiprocessing import Process
import multiprocessing as mp
import subprocess
from pddl import pddl_domain
from pddl import pddl_problem
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

        pass
    def _legal_plan(self, summary, file_path=None):
        if summary =="":
            print("summary empty")
            try_read = True
            while try_read and self.solved != 2:
                try:
                    f = open(file_path, "r")
                    try_read = False
                    time.sleep(0.5)
                except:
                    pass
            summary = f.read()
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
        if summary =="":
            f = open(file_path, "r")
            summary = f.read()
        idx_strt = summary.find("plan cost")
        return(float(summary[idx_strt:].split()[2]))
    def _time_2_solve(self, summary, file_path = None):
        if summary =="":
            f = open(file_path, "r")
            summary = f.read()
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
        return path_result
    def solve(self, domain, problem, multiprocess = True, type_solver = '3', weight = '1', timeout = 90.0):
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
            self.mp_output_goals = {}
            for i in range(len(self.problem)):
                key = self.problem[i].name
                self.mp_output_goals[key] = mp.Array("c", 10**4)
            self.mp_goal_computed = {}
            for i in range(len(self.problem)):
                key = self.problem[i].name
                self.mp_goal_computed[key] = mp.Value("i", 0)
            #build processes
            self.processes = {}
            if self.path != "":
                if os.path.basename(os.getcwd()) != os.path.basename(self.path):
                  os.chdir(self.path)
            for i in range(len(self.problem)):
                key = self.problem[i].name
                self.processes[key] = Process(target=self._run_metric_ff_mp, args=[self.domain_path,
                                                                           self.problem_path[i],i,key] )
            for i in range(len(self.problem)):
                key = self.problem[i].name
                print("start_", self.processes[key])
                self.processes[key].start()
            #j = 0
            start_time = time.time()
            while(len([x for x in psutil.process_iter() if self.planner in x.name()]) != 0):
                sleep(0.1)
                if time.time() - start_time >= timeout:
                    self.solved = 2
                    print("time_out finished")
                    path = ""
                    for path_pc in self.domain_path.split("/")[:-1]:
                        path = path + path_pc + "/"
                    files = []
                    achieved_goals = [key for key in list(self.mp_goal_computed.keys()) if self.mp_goal_computed[key].value == 1]
                    for key in achieved_goals:
                        self.summary[key] = self.mp_output_goals[key].value.decode('ascii')
                        file_path = path + f"output_goal_{key}.txt"
                        files.append(file_path)
                        if self.summary[key] == "":
                            try_read = True
                            while try_read:
                                try:
                                    f = open(file_path, "r")
                                    try_read = False
                                    time.sleep(0.5)
                                except:
                                    pass
                            self.summary[key] = f.read()
                        self.plan[key] = self._legal_plan(self.summary[key], file_path)
                        self.plan_cost[key] = self._cost(self.summary[key], file_path)
                        self.plan_achieved[key] = self.mp_goal_computed[key].value
                        self.time[key] = self._time_2_solve(self.summary[key], file_path)
                        os.remove(file_path)
                    [x.kill() for x in psutil.process_iter() if self.planner in x.name()]
            path = ""
            for path_pc in self.domain_path.split("/")[:-1]:
                path = path + path_pc +"/"
            files = []
            for i in range(len(self.problem)):
                key = self.problem[i].name
                self.summary[key] = self.mp_output_goals[key].value.decode('ascii')
                file_path = path + f"output_goal_{key}.txt"
                files.append(file_path)
                self.plan[key] = self._legal_plan(self.summary[key], file_path)
                self.plan_cost[key] = self._cost(self.summary[key], file_path)
                self.plan_achieved[key] = self.mp_goal_computed[key].value
                self.time[key] = self._time_2_solve(self.summary[key], file_path)
            #in order to use pickle, destroy mp.arrays
            self.mp_output_goals = {} # set empty
            self.mp_goal_computed = {}
            self.processes = {}
            self.solved = 1
            if self.solved != 0:
                [os.remove(fp) for fp in files]
    def _run_metric_ff_mp(self, domain, problem, i, key):
        command_string = f"./{self.planner} -o {domain} -f {problem} -s {self.type_solver} -w {self.weight}"
        #print(command_string)
        path = ""
        for path_pc in domain.split("/")[:-1]:
            path = path + path_pc +"/"
        output = subprocess.check_output(command_string, shell = True)
        with open(path + f"output_goal_{key}.txt", "w") as output_write:
            output_write.write(output.decode("ascii") )
        self.mp_output_goals[key].value = output
        self.mp_goal_computed[key].value = 1
        print(key)
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