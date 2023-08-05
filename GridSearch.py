import pickle
import pandas as pd
import numpy as np
import datetime as dt
import itertools
import prap_model
from pddl import *
from prap_model import *
from gm_model import *
import subprocess
import threading
import psutil
import multiprocessing as mp
from multiprocessing import Process
import random
from copy import copy
from multiprocess_df import _multiprocess_df
import sqlite3 as db
def save_gridsearch(gs):
    """
    Pickle GridSearch-object in gs.path
    :param gs: expects object of type GridSearch.GridSearch
    """
    with open (gs.path + gs.name + ".pickle", "wb") as outp:
        pickle.dump(gs, outp, pickle.HIGHEST_PROTOCOL)
def load_gridsearch(file):
    """
    Load GridSearch.GridSearch-object
    :param file: path + file of GridSearch.GridSearch-object
    """
    return pickle.load(open(file, "rb"))
class GridSearch:
    """
    Class that performs a GridSearch to a goal recognition model.
    A goal recognition model can be of type prap_model or gm_model.
    """
    def __init__(self, model_root, name, planner = "ff_2_1"):
        """
        :param model_root: model for goal_recognition.
        :param name: name for Gridsearch-object, created directories and files contain this name.
        :param planner: name of executable planner, here default ff_2_1 (MetricFF Planner version 2.1).
        """
        self.model_root = model_root
        self.model_type = type(self.model_root)
        self.model_list = []
        self.name = name
        self.planner = planner
        self.grid = self._create_df_action_cost()
        self.grid_item = []
        self.path = ""
        self.goal_list_path = []
        self.temperature_control = False
        self.temperature_mean_cur = 40.0  # just for init
        self.temperature_array = np.repeat(self.temperature_mean_cur, 10)
        self.hash_code = self.model_root.hash_code

        # warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    def _update_db_grid_type(self, grid_type, row=None):
        if grid_type == 1:
            grid = self.grid[self.grid["optimal_feasible"] == 1]
            if "reduced" not in grid.columns:
                grid["reduced"] = 0
        elif grid_type == 2:
            try:
                grid = self.grid_expanded[self.grid_expanded["optimal_feasible"] == 1]
            except:
                print("grid.expanded not yet created")
                return None
        if type(self.model_root) == prap_model:
            domain = copy(self.model_root.domain_list[0])
        elif type(self.model_root) == gm_model:
            domain = copy(self.model_root.domain_root)
        action_list = list(domain.action_dict.keys())
        action_list.sort()
        for action in action_list:
            grid.loc[:, action] = grid.loc[:, action].astype(float)
        now = dt.datetime.now()
        time_stamp = now.strftime("%Y-%m-%d %H:%M:%S")
        if len([c for c in grid.columns if c.startswith("ACTION")]) != len(action_list):
            print([c for c in grid.columns if not c.startswith("ACTION")])
            return "length of action in grid is not equivalent to actions in domain"
        else:
            all_cols_equivalent = True
            missing_equivalent = []
            for col in [c for c in grid.columns if c.startswith("ACTION")]:
                if col not in action_list:
                    all_cols_equivalent = False
                    missing_equivalent.append(col)
            if not all_cols_equivalent:
                return f"Following columns from grid could not be found in domain actions {missing_equivalent}"
        if row == None:
            upload_grid = pd.DataFrame()
            for idx in range(len(grid)):
                action_str, hash_str_action = self._hash_action(grid, idx, action_list)
                idx_df = pd.DataFrame({"hash_code_model": [self.hash_code], "hash_code_action": [hash_str_action],
                                       "action_conf": [action_str], "time_stamp": [time_stamp],
                                       "config": [grid.iloc[idx]["config"]],
                                       "optimal_feasible": [grid.iloc[idx]["optimal_feasible"]],
                                       "seconds": [grid.iloc[idx]["seconds"]],
                                       "reduced": [grid.iloc[idx]["reduced"]]})
                upload_grid = pd.concat([upload_grid, idx_df])
            upload_grid = upload_grid.reset_index().iloc[:, 1:]
        else:
            action_str, hash_str_action = self._hash_action(grid, row, action_list)
            upload_grid = pd.DataFrame({"hash_code_model": [self.hash_code], "hash_code_action": [hash_str_action],
                                        "action_conf": [action_str], "time_stamp": [time_stamp],
                                        "config": [grid.iloc[row]["config"]],
                                        "optimal_feasible": [grid.iloc[row]["optimal_feasible"]],
                                        "seconds": [grid.iloc[row]["seconds"]],
                                        "reduced": [grid.iloc[row]["reduced"]]})
        query = f"SELECT DISTINCT(hash_code_action) FROM model_grid WHERE hash_code_model = '{self.hash_code}'"
        db_gr = db.connect("/home/mwiubuntu/Seminararbeit/db_results/goal_recognition.db")
        existing_actions = pd.read_sql_query(query, db_gr)
        db_gr.close()
        #upload_grid = upload_grid[upload_grid["hash_code_action"].isin(existing_actions)]


        upload_grid.loc[:, "config"] = ("x_" + upload_grid.loc[:, "config"].str.split("_").str[-2] + "_" +
                                            upload_grid.loc[:, "config"].str.split("_").str[-1])
                                        

        return upload_grid
    










    def _hash_action(self, grid, row, action_list):
        action_str = ""
        for action in action_list:
            action_str += str(grid.iloc[row, :][action]) + "-"
        action_str = action_str[:-1]
        h = hashlib.new("sha224")
        h.update(action_str.encode())
        hash_str_action = h.hexdigest()
        return action_str, hash_str_action
    def reset_grid_expanded(self):
        """resets grid_expanded in order to find new configurations"""
        self.model_list_expanded = []
        self.grid_expanded = self.grid_expanded[self.grid_expanded["reduced"] == 1]
    def _create_df_action_cost(self):
        df_action_costs = pd.DataFrame()
        df_action_costs["config"] = [self.name + "_baseline"]
        df_action_costs["optimal_feasible"] = np.nan
        df_action_costs["seconds"] = np.nan
        if type(self.model_root) == prap_model:
            domain_root = self.model_root.domain_list[0]
        elif type(self.model_root) == gm_model:
            domain_root = self.model_root.domain_root
        for key in domain_root.action_dict.keys():
            df_action_costs[key] = [domain_root.action_dict[key].action_cost]
        return df_action_costs
    def add_grid_item(self, grid_tuple):
        """RUN this BEFORE create_grid()
        Adds item or list of items in shape of tuple to GridSearch items
        :param grid_tuple: either tuple or list of tuples that will be grid-searched
        """
        if type(grid_tuple) == list:
            for element in grid_tuple:
                self.grid_item.append(element)
        elif type(grid_tuple) == tuple:
            self.grid_item.append(grid_tuple)
        else:
            print("please pass list of tuples or tuple to function")
    def create_grid(self, random=False, size=None):
        """
        Creates grid from added grid items. If random = False all combinations are created.
        If random == True combinations of specified tuples are sampled, in this case parameter size has to be specified
        :param random: boolean that specifies if a random grid will be sampled or whether all combinations
        are considered
        :param size: only necessary if random == True
        """
        if not random:
            a = [x[1] for x in self.grid_item]
            a = list(itertools.product(*a))
            result_df = self.grid.copy()
            for comb in a:
                c = 0
                d = []
                for el in comb:
                    action = [x[0] for x in self.grid_item][c]
                    d.append((el, action))
                    c = c + 1
                new_df = self.grid.copy()
                for x in d:
                    new_df.loc[0, x[1]] = x[0]
                result_df = pd.concat([result_df, new_df])
            result_df = result_df.drop_duplicates()
            result_df = result_df.reset_index().iloc[:, 1:]
            for i in range(1, len(result_df)):
                result_df.loc[i, "config"] = self.name + "_config_" + str(i)
            result_df["optimal_feasible"] = np.nan
            result_df["seconds"] = np.nan
            self.grid = result_df
        else:
            max_combs = abs(np.prod([len(x[1]) for x in self.grid_item]))
            if max_combs == 0: # number too big
                max_combs = 10**10
            if size >= max_combs:
                return "size is equal or larger than maximal combinations "
            else:
                rgs = self._gs_random_generator(self.grid_item, self.grid, size)
                f = rgs._create_random_grid()
                f = f.reset_index().iloc[:, 1:]
                f["config"] = self.name + "_config_" + f.index.astype(str)
                self.grid = pd.concat([self.grid, f])
                self.grid = self.grid.reset_index().iloc[:, 1:]
    def _create_domain_config(self, idx, model_list_type = 1):
        #model_list_type == 1: model_list, == 2: model_reduce_cur, model_list_type == 3: model_list_expanded
        if type(self.model_root) == prap_model:
            domain = copy(self.model_root.domain_list[0])
        elif type(self.model_root) == gm_model:
            domain = copy(self.model_root.domain_root)
        if model_list_type == 1:
            config_idx = self.grid.iloc[idx, :]
        if model_list_type == 2 or model_list_type == 3:
            config_idx = self.grid_expanded.iloc[idx, :]
        new_domain = f"(define (domain {domain.name})\n"
        new_domain = new_domain + domain.requirements + "\n"
        new_domain = new_domain + domain.types + "\n"
        new_domain = new_domain + domain.constants + "\n"
        new_domain = new_domain + "(:predicates"
        for predicate in domain.predicates:
            new_domain = new_domain + " " + predicate
        new_domain = new_domain + ")\n"
        new_domain = new_domain + "(:functions "
        for function in domain.functions:
            new_domain = new_domain + function + "\n"
        new_domain = new_domain + ")\n"
        for action in domain.action_dict.keys():
            new_cost = config_idx[action]
            domain.action_dict[action].set_action_cost(new_cost)
            new_domain = new_domain + domain.action_dict[action].action + "\n"
        new_domain = new_domain + "\n)"
        if model_list_type == 1:
            with open(self.path + config_idx["config"] + ".pddl", "w") as domain_config:
                domain_config.write(new_domain)
            self.model_list.append(self.model_type(pddl_domain(self.path + config_idx["config"] + ".pddl"),
                                          self.goal_list_path, self.model_root.observation, planner = self.planner))
        elif model_list_type == 2:
            with open(self.path_reduce + config_idx["config"] + ".pddl", "w") as domain_config:
                domain_config.write(new_domain)
            self.model_reduce_cur = self.model_type(pddl_domain(self.path_reduce + config_idx["config"] + ".pddl"),
                                                    self.goal_list_path, self.model_root.observation,
                                                    planner=self.planner)
        if model_list_type == 3:
            with open(self.path + config_idx["config"] + ".pddl", "w") as domain_config:
                domain_config.write(new_domain)
            self.model_list_expanded.append(self.model_type(pddl_domain(self.path + config_idx["config"] + ".pddl"),
                                                   self.goal_list_path, self.model_root.observation,
                                                   planner=self.planner))
    def _remove_model_domain_config(self, i=0, type_grid=1):
        if type_grid == 1: #1:check_optimal_feasible 2:expand grid 3:expand grid in check_optimal_feasible
            model_remove = self.model_list[i]
        elif type_grid == 2:
            model_remove = self.model_reduce_cur
        elif type_grid == 3:
            model_remove = self.model_list_expanded[i]
        if type(model_remove) == prap_model:
            file_remove = model_remove.domain_list[0].domain_path
        elif type(model_remove) == gm_model:
            file_remove = model_remove.domain_root.domain_path
        os.remove(file_remove)
        if type_grid == 1:
            self.model_list.remove(model_remove)
        if type_grid == 3:
            self.model_list_expanded.remove(model_remove)
    def _monitor_temperature_mean(self, celsius_stop, cool_down_time, update_time):
        self.temperature_control = True
        temperature_str = str(subprocess.check_output("sensors", shell=True))
        tctl_start = temperature_str.find("Tctl")
        temperature = float(
            temperature_str[tctl_start + 6:temperature_str[tctl_start:].find("xc2") + tctl_start - 1].replace("+",
                                                                                                              "").replace(
                " ", ""))
        temperatures = np.repeat(temperature, 15)
        while self.temperature_control:
            temperature_str = str(subprocess.check_output("sensors", shell=True))
            tctl_start = temperature_str.find("Tctl")
            temperature = float(
                temperature_str[tctl_start + 6:temperature_str[tctl_start:].find("xc2") + tctl_start - 1].replace("+",
                                                                                                                  "").replace(
                    " ", ""))
            temperatures = np.append(temperatures[1:], temperature)
            self.temperature_mean_cur = np.mean(temperatures)
            # print("current, ", self.temperature_mean_cur )
            self.temperature_array = np.append(self.temperature_array[1:], self.temperature_mean_cur)
            # print("array, ", self.temperature_array )
            time.sleep(update_time)
    def reduce_feasible_configs(self, multiprocess=True, keep_files=True, type_solver='3', weight='1',
                              timeout=90, pickle=False, celsius_stop=72, cool_down_time=40, update_time=2):
        """
        Reduces feasible grid-items found from hundreds to 1 or in steps of 10ths. Might still be buggy.
        :param multiprocess: if True, all problems (goals) are solved in parallel.
        :param keep_files: if True, feasible_domains are kept in directory.
        :param type_solver: option for type solver in Metricc-FF Planner, however only type_solver = '3' ("weighted A*)
         is considered
        :param weight: weight for type_solver = '3' ("weighted A*); weight = '1' resolves to unweighted A*
        :param timeout: after specified timeout is reached, all process for one grid configuration are getting killed,
        next grid configuration is then executed.
        :param pickle: if True, GridSearch object gets pickled everytime a feasible grid configuration is found.
        :param celsius_stop: for a less noisy apartment and for increasing the durability of hardware, cool down break
        between grid configurations if core temperature (AMD cpu) averages specified temperature.
        :param cool_down_time: cool down break between grid configurations if core temperature (AMD cpu) averages
        specified temperature in celsius_stop.
        :param update_time: time interval for checking average temperature.
        """
        self.grid_expanded = self.grid[self.grid["optimal_feasible"] == 1]
        self.grid_expanded = self.grid_expanded.reset_index().iloc[:,1:]
        self.grid_expanded["config"] = self.grid_expanded["config"].str.replace("config", "reduce")
        t = threading.Thread(target=self._monitor_temperature_mean, args=[celsius_stop, cool_down_time, update_time])
        t.start()
        if type(self.model_root) == prap_model:
            domain_root = self.model_root.domain_list[0]
        elif type(self.model_root) == gm_model:
            domain_root = self.model_root.domain_root
        domain = copy(domain_root)
        path_pcs = domain.domain_path.split("/")
        path = ""
        for path_pc in path_pcs[:-1]:
            path = path + path_pc + "/"
        path += self.name + "/reduce"
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)
        self.path_reduce = path + "/"
        if not os.path.exists(self.path_reduce + f"{self.planner}"):
            shutil.copy(f"{self.planner}", self.path_reduce + f"{self.planner}")
        for goal in self.model_root.goal_list[0]:
            if not os.path.exists(self.path_reduce + goal.problem_path.split("/")[-1]):
                shutil.copy(goal.problem_path, self.path_reduce + goal.problem_path.split("/")[-1])
        idx = 0
        while idx < len(self.grid_expanded):
            print("-------------------")
            necessary_actions = []
            self._create_domain_config(idx, model_list_type = 2)
            if type(self.model_reduce_cur) == prap_model:
                cur_idx_config = self.model_reduce_cur.domain_list[0].domain_path.split("/")[-1]
            elif type(self.model_reduce_cur) == gm_model:
                cur_idx_config = self.model_reduce_cur.domain_root.domain_path.split("/")[-1]
            cols_grid_exp = [col for col in self.grid_expanded.columns if col not in ["config", "optimal_feasible",
                                                                                      "seconds"]]
            rel_cols_grid_exp = []
            for col in cols_grid_exp:
                if self.grid_expanded.loc[idx, col] > 1:
                    rel_cols_grid_exp.append(col)
            cols_grid_exp = rel_cols_grid_exp
            i = 0
            while i  < len(cols_grid_exp):
                if self.temperature_mean_cur >= celsius_stop:
                    if np.sum(self.temperature_array >= celsius_stop) == len(self.temperature_array):
                        print("cooldown")
                        for sec in range(0, cool_down_time):
                            if sec % 10 == 0:
                                print("temperature_mean_cur: ", self.temperature_mean_cur)
                            time.sleep(1)
                print(f"\n{cur_idx_config}")
                print(f"{i+1}/{len(cols_grid_exp)}")
                old_val = self.grid_expanded.loc[idx,cols_grid_exp[i]]
                print(cols_grid_exp[i], ": ", old_val)
                self.grid_expanded.loc[idx, cols_grid_exp[i]] = 1
                self._create_domain_config(idx, model_list_type = 2)
                time.sleep(1)
                while (max(psutil.cpu_percent(percpu=True)) > 30):
                    time.sleep(1)
                try:
                    self.model_reduce_cur.perform_solve_optimal(multiprocess=multiprocess, type_solver=type_solver,
                                                             weight=weight, timeout=timeout)
                except:
                    pass
                start_time = time.time()
                restart = False
                s = 15
                while (self.model_reduce_cur.steps_optimal.solved == 0 and (time.time() - start_time <= timeout + 15)):
                    if (time.time() - start_time > timeout):
                        print(time.time() - start_time, "s" )
                        print("timeout reached")
                        print("continue in ", s)
                        s -= 1
                        restart = True
                    time.sleep(1)
                if not restart:
                    if (self.model_reduce_cur.steps_optimal.solved == 1):
                        self.grid_expanded.loc[idx, "optimal_feasible"] = 1
                        if multiprocess:
                            self.grid_expanded.loc[idx, "seconds"] = self.model_reduce_cur.mp_seconds
                        # else:
                        # keys = self.model_list[i+1].prap_steps_optimal.time.keys()
                        # self.model_list[i+1].prap_steps_optimal.time= max([self.model_list[i+1].prap_steps_optimal.time[key]
                        #     for key in keys])
                        if pickle:
                           save_gridsearch(self)
                    else:
                        self.grid_expanded.loc[idx, "optimal_feasible"] = 0
                        self.grid_expanded.loc[idx, cols_grid_exp[i]] = old_val
                        print(f"keep old value for {cols_grid_exp[i]}\n")
                        necessary_actions.append(cols_grid_exp[i])
                    if keep_files:
                        if self.grid_expanded.loc[idx, "optimal_feasible"] == 0:
                           self._remove_model_domain_config(type_grid = 2)
                    else:
                        self._remove_model_domain_config(type_grid = 2)
                else:
                    [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
                    self.grid_expanded.loc[idx, cols_grid_exp[i]] = old_val
                    i -=1
                i += 1
            print("reduce_columns")
            for action in necessary_actions:
                print(action)
                solvable = False
                c_action = 10
                while not solvable:
                    print("\n",c_action)
                    old_val = self.grid_expanded.loc[idx, action]
                    if c_action == old_val:
                        solvable = True
                    self.grid_expanded.loc[idx, action] = c_action
                    self._create_domain_config(idx, model_list_type=2)
                    time.sleep(1)
                    while (max(psutil.cpu_percent(percpu=True)) > 30):
                        time.sleep(1)
                    try:
                        self.model_reduce_cur.perform_solve_optimal(multiprocess=multiprocess, type_solver=type_solver,
                                                                    weight=weight, timeout=timeout)
                    except:
                        pass
                    start_time = time.time()
                    restart = False
                    s = 15
                    while (self.model_reduce_cur.steps_optimal.solved == 0 and (time.time() - start_time <= timeout + 15)):
                        if (time.time() - start_time > timeout):
                            print(time.time() - start_time, "s" )
                            print("timeout reached")
                            print("continue in ", s)
                            s -= 1
                            restart = True
                        time.sleep(1)
                    if not restart:
                        if (self.model_reduce_cur.steps_optimal.solved == 1):
                            self.grid_expanded.loc[idx, "optimal_feasible"] = 1
                            solvable = True
                            if multiprocess:
                                self.grid_expanded.loc[idx, "seconds"] = self.model_reduce_cur.mp_seconds
                            # else:
                            # keys = self.model_list[i+1].prap_steps_optimal.time.keys()
                            # self.model_list[i+1].prap_steps_optimal.time= max([self.model_list[i+1].prap_steps_optimal.time[key]
                            #     for key in keys])
                            if pickle:
                                save_gridsearch(self)
                        else:
                            self.grid_expanded.loc[idx, "optimal_feasible"] = 0
                            self.grid_expanded.loc[idx, action] = old_val
                            c_action += 10
                        if keep_files:
                            if self.grid_expanded.loc[idx, "optimal_feasible"] == 0:
                                self._remove_model_domain_config(type_grid=2)
                        else:
                            self._remove_model_domain_config(type_grid=2)
                    else:
                        [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
            idx += 1
        self.temperature_control = False
    def expand_grid(self, size):
        """
        Expands grid_expanded from reduced solvable configurations by size specified.
        RUN METHOD reduce_feasible_configs BEFORE! Keeps reduced grid_items from before.
        :param size: specifies the amount of configurations added to grid_expanded
        """
        if type(self.model_root) == prap_model:
            domain_root = self.model_root.domain_list[0]
        elif type(self.model_root) == gm_model:
            domain_root = self.model_root.domain_root
        domain = copy(domain_root)
        name_config = domain.domain_path.split("/")[-1]
        name_config +=  "_reduce_config_"
        new_expand_grid = self.grid_expanded.copy()
        if not "reduced" in new_expand_grid.columns:
            new_expand_grid["reduced"] = 1
        new_expand_grid = new_expand_grid[[c for c in new_expand_grid.columns
                                           if c in ['config', 'optimal_feasible', 'seconds']] + ["reduced"] +
              [c for c in new_expand_grid.columns if c not in ['config', 'optimal_feasible', 'seconds', 'reduced']]]
        if size < len(new_expand_grid) :
            len_list_rows = size
        else: len_list_rows = len(new_expand_grid)
        list_rows = []
        for i in range(len_list_rows):
            row_df = new_expand_grid[new_expand_grid["config"] == new_expand_grid.loc[i,"config"]]
            row_df = row_df.reset_index().iloc[:,1:]
            list_rows.append(row_df)
        size_per_row = size // len_list_rows
        rows_rgs = []
        for row in list_rows:
            print(row.loc[0,"config"])
            new_list_grid_item = []
            for item in self.grid_item:
                if row[item[0]].iloc[0] == 1:
                    new_list_grid_item.append((item[0], list(range(1,4))))
                else:
                    new_list_grid_item.append((item[0], list(range(row[item[0]].iloc[0],row[item[0]].iloc[0] + 5))))
            rgs = self._gs_random_generator(new_list_grid_item, row, size=size_per_row)
            row_rgs = rgs._create_random_grid()
            rows_rgs.append(row_rgs)
            row_rgs["reduced"] = 0
            time.sleep(1)
        new_expand_grid = pd.concat([new_expand_grid] + rows_rgs)
        new_expand_grid = new_expand_grid.reset_index().iloc[:,1:]
        new_expand_grid["config"] = name_config + new_expand_grid.index.astype(str)
        self.grid_expanded = new_expand_grid
    def check_feasible_domain(self, grid_type = 1, multiprocess=True, keep_files=True, type_solver='3', weight='1',
                              timeout=90, pickle=False, celsius_stop=72, cool_down_time=40, update_time=2):
        """
        Checks whether optimal plan for all grid-items (not considering observations) can be achieved within timeout.
        :param grid_type: if grid_type = 1 self.grid is checked, with grid_type = 2 self.grid_expanded instead.
        :param multiprocess: if True, all problems (goals) are solved in parallel.
        :param keep_files: if True, feasible_domains are kept in directory.
        :param type_solver: option for type solver in Metricc-FF Planner, however only type_solver = '3' ("weighted A*)
         is considered
        :param weight: weight for type_solver = '3' ("weighted A*); weight = '1' resolves to unweighted A*
        :param timeout: after specified timeout is reached, all process for one grid configuration are getting killed,
        next grid configuration is then executed.
        :param pickle: if True, GridSearch object gets pickled everytime a feasible grid configuration is found.
        :param celsius_stop: for a less noisy apartment and for increasing the durability of hardware, cool down break
        between grid configurations if core temperature (AMD cpu) averages specified temperature.
        :param cool_down_time: cool down break between grid configurations if core temperature (AMD cpu) averages
        specified temperature in celsius_stop.
        :param update_time: time interval for checking average temperature.
        """
        if grid_type == 1:
            grid = self.grid
            model_list = self.model_list
        elif grid_type == 2:
            grid = self.grid_expanded
            self.model_list_expanded = []
            model_list = self.model_list_expanded
        t = threading.Thread(target=self._monitor_temperature_mean, args=[celsius_stop, cool_down_time, update_time])
        t.start()
        if type(self.model_root) == prap_model:
            domain_root = self.model_root.domain_list[0]
        elif type(self.model_root) == gm_model:
            domain_root = self.model_root.domain_root
        domain = copy(domain_root)
        path_pcs = domain.domain_path.split("/")
        path = ""
        for path_pc in path_pcs[:-1]:
            path = path + path_pc + "/"
        path += self.name
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path + "/"
        if not os.path.exists(self.path + f"{self.planner}"):
            shutil.copy(f"{self.planner}", self.path + f"{self.planner}")
        for goal in self.model_root.goal_list[0]:
            if not os.path.exists(self.path + goal.problem_path.split("/")[-1]):
                shutil.copy(goal.problem_path, self.path + goal.problem_path.split("/")[-1])
            if grid_type == 1:
                self.goal_list_path.append(pddl_problem(self.path + goal.problem_path.split("/")[-1]))
        i = 0
        idx = 0
        """if grid_type == 1:
            i = 0
            idx = 0
        if grid_type == 2:
            i = min(self.grid_expanded[self.grid_expanded["reduced"] == 0].index)
            idx = i"""
        while idx < len(grid):
            if self.temperature_mean_cur >= celsius_stop:
                if np.sum(self.temperature_array >= celsius_stop) == len(self.temperature_array):
                    print("cooldown")
                    for sec in range(0, cool_down_time):
                        if sec % 10 == 0:
                            print("temperature_mean_cur: ", self.temperature_mean_cur)
                        time.sleep(1)
            print("-------------------")
            if grid_type == 1:
                self._create_domain_config(idx, model_list_type = 1)
            elif grid_type == 2:
                self._create_domain_config(idx, model_list_type=3)
            if type(self.model_root) == prap_model:
                print(model_list[i].domain_list[0].domain_path.split("/")[-1])
            elif type(self.model_root) == gm_model:
                print(model_list[i].domain_root.domain_path.split("/")[-1])
            # print([self.model_list[i].domain_list[0].action_dict[key].action_cost for key in self.model_list[-1].domain_list[0].action_dict.keys()])
            time.sleep(1)  # remove pending tasks from cpu
            while (max(psutil.cpu_percent(percpu=True)) > 30):
                time.sleep(1)
            try:
                model_list[i].perform_solve_optimal(multiprocess=multiprocess, type_solver=type_solver,
                                                         weight=weight, timeout=timeout)
            except:
                pass
            start_time = time.time()
            restart = False
            s = 15
            while (model_list[i].steps_optimal.solved == 0 and (time.time() - start_time <= timeout + 15)):
                if (time.time() - start_time > timeout):
                    print("timeout reached")
                    print("continue in ", s)
                    s -= 1
                    restart = True
                time.sleep(1)
            if not restart:
                if (model_list[i].steps_optimal.solved == 1):
                    grid.loc[idx, "optimal_feasible"] = 1
                    if multiprocess:
                        grid.loc[idx, "seconds"] = model_list[i].mp_seconds
                    # else:
                    # keys = model_list[i+1].prap_steps_optimal.time.keys()
                    # model_list[i+1].prap_steps_optimal.time= max([model_list[i+1].prap_steps_optimal.time[key]
                    #     for key in keys])
                    if pickle:
                        save_gridsearch(self)
                else:
                    grid.loc[idx, "optimal_feasible"] = 0
                if keep_files:
                    if grid.loc[idx, "optimal_feasible"] == 0:
                        if grid_type == 1:
                            self._remove_model_domain_config(i)
                        elif grid_type == 2:
                            self._remove_model_domain_config(i, type_grid=3)
                        i -= 1
                else:
                    if grid_type == 1:
                        self._remove_model_domain_config(i)
                    elif grid_type == 2:
                        self._remove_model_domain_config(i, type_grid=3)
                    i -= 1
                i += 1
                idx += 1
            else:
                [x.kill() for x in psutil.process_iter() if f"{self.planner}" in x.name()]
                if grid_type == 1:
                    self._remove_model_domain_config(i)
                elif grid_type == 2:
                    self._remove_model_domain_config(i, type_grid=3)
        if type(self.model_root) == prap_model:
            for action in self.model_root.domain_list[0].action_dict.keys():
                new_cost = grid.iloc[0, :][action]
                self.model_root.domain_list[0].action_dict[action].set_action_cost(new_cost)
        elif type(self.model_root) == gm_model:
            for action in self.model_root.domain_root.action_dict.keys():
                new_cost = grid.iloc[0, :][action]
                self.model_root.domain_root.action_dict[action].set_action_cost(new_cost)
        self.temperature_control = False
        if pickle:
            save_gridsearch(self)
    class _gs_random_generator:
        def __init__(self, list_grid_items, df_baseline, size):
            self.list_grid_items = list_grid_items
            self.size = size
            self.max_combs = self._max_combs()
            self.df_baseline = df_baseline.copy()
            self.grid_argmax = np.argmax([len(x[1]) for x in self.list_grid_items])
            self.split_item = self.list_grid_items[self.grid_argmax]
            self.max_parallel = int(min(psutil.cpu_count() / 2, len(self.split_item[1]) // 2))
            self.grid_result = [[] for _ in range(self.max_parallel)]
            self.grid_result_not_unique = [[] for _ in range(self.max_parallel)]
            self.collect_i = [True for _ in range(self.max_parallel)]
            self.size_i = self._assign_size_i()
            self.dict_actions = self._assign_dict_actions()
            self.process = []
            self.result_shared_array = [mp.Array("i", self.size_i[i] * len(list_grid_items)) for i in
                                        range(self.max_parallel)]
        def _max_combs(self):
            max_comb = 1
            for grid_item in self.list_grid_items:
                max_comb = max_comb * len(grid_item[1])
            print(self.size / max_comb, "% of possible combinations requested")
            return max_comb
        def _assign_size_i(self):
            reg_size = self.size // self.max_parallel
            last_size = self.size - (self.max_parallel - 1) * reg_size
            distribution = []
            for i in range(self.max_parallel - 1):
                distribution.append(reg_size)
            distribution.append(last_size)
            return distribution
        def _assign_dict_actions(self):
            dict_actions = {}
            dict_actions[0] = self.split_item
            idx = 1
            for el in self.list_grid_items:
                if el[0] != self.split_item[0]:
                    dict_actions[idx] = el
                    idx += 1
            return dict_actions
        def _create_random_grid(self):
            for i in range(self.max_parallel):
                self.process.append(Process(target=self._generate_position_i, args=[i]))
                self.process[i].start()
            for j in range(self.max_parallel):
                self.process[i].join()
            d = pd.DataFrame()
            i = 0
            tuples_array = [(np.array(self.result_shared_array[idx]),
                             self.list_grid_items,
                             self.df_baseline,
                             self.dict_actions) for idx in range(len(self.result_shared_array))]
            with mp.Pool() as p:
                y = p.map(_multiprocess_df, tuples_array)
            d = pd.concat(y)
            d = d.sort_values(by=list(d.columns))
            return d
        def _generate_position_i(self, i):
            reg_range_len = len(self.split_item[1]) // self.max_parallel
            if (i != self.max_parallel - 1):
                idx_range_strt = i * reg_range_len
                idx_range_end = (i + 1) * reg_range_len
            else:
                idx_range_strt = i * reg_range_len
                idx_range_end = len(self.split_item[1]) - 1
            N = self.size_i[i]
            collect_i = True
            t_i = threading.Thread(target=self._collect_data, args=[idx_range_strt, idx_range_end, i, N])
            t_i.start()
            time.sleep(0.5)
            s = 0
            while len(self.grid_result[i]) < N:
                while s > len(self.grid_result_not_unique[i]):
                    print("wait")
                    time.sleep(0.5)
                try:
                    if self.grid_result_not_unique[i][s] not in self.grid_result[i]:
                        self.grid_result[i].append(self.grid_result_not_unique[i][s])
                except:
                    print("error")
                    time.sleep(0.5)
                    s -= 1
                s += 1
            self.collect_i[i] = False
            r = 0
            for obs in self.grid_result[i]:
                for el in obs:
                    self.result_shared_array[i][r] = el
                    r += 1
        def _collect_data(self, idx_range_strt, idx_range_end, i, size):
            j = 0
            while self.collect_i[i] and j < size * 10:
                data = []
                data.append(random.sample(self.split_item[1][idx_range_strt:idx_range_end], 1)[0])
                for grid_item in list(self.dict_actions.keys())[1:]:
                    data.append(random.sample(self.dict_actions[grid_item][1], 1)[0])
                self.grid_result_not_unique[i].append(data)
                j += 1
if __name__ == '__main__':
    """
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
    #model = gm_model(toy_example_domain, toy_example_problem_list, obs_toy_example)
    gs = GridSearch(model, "toy_example_gs")
    gs.add_grid_item([("MOVE_LEFT_FROM", range(5, 10))])
    gs.add_grid_item(("MOVE_RIGHT_FROM", range(100, 200)))
    gs.add_grid_item(("MOVE_DOWN_FROM", range(200, 3000)))
    gs.add_grid_item(("MOVE_LOWER_RIGHT_FROM", range(50, 60)))
    gs.add_grid_item(("MOVE_LOWER_LEFT_FROM", range(50, 60)))
    gs.create_grid(random=True, size=4)
    gs.check_feasible_domain(multiprocess=True, timeout= 5, keep_files = True, pickle = False)
    print(gs.grid)#"""

    """
    #print(gs.hash_code)
    #gs.reset_grid_expanded()
    #gs.expand_grid(size = 1000)
    #gs.check_feasible_domain(grid_type = 2, keep_files=True, timeout=40, pickle=True)
    #save_gridsearch(gs)
    """
    gs = load_gridsearch("/home/mwiubuntu/Seminararbeit/domain/environment in domain file/For Grid-Search/model_7_mod_wo_books_label/model_7_mod_wo_books_label.pickle")
    f = gs._update_db_grid_type(grid_type=2)
