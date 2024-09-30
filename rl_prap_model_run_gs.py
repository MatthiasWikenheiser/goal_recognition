from pddl import pddl_domain, pddl_problem
from create_pddl_gym import GymCreator
from pddl import pddl_observations
from multi_rl_planner import MultiRLPlanner
from rl_prap_model import PRAPAgent
import rl_planner
import numpy as np
import os
import datetime as dt
import sqlite3 as db
import time

model_type = 'rl_prap_model'

def create_db_tables(model, hash_code_model, hash_code_action, station, log_file):
    now = dt.datetime.now()
    time_stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    rl_type = 0
    iterations = np.nan

    model_grid_obs = model.summary_level_1.copy()
    model_grid_obs["model_type"] = model_type
    model_grid_obs["hash_code_model"] = hash_code_model
    model_grid_obs["hash_code_action"] = hash_code_action
    model_grid_obs["rl_type"] = rl_type
    model_grid_obs["iterations"] = iterations
    model_grid_obs["station"] = station
    model_grid_obs["log_file"] = log_file
    model_grid_obs = model_grid_obs[["model_type", "hash_code_model", "hash_code_action", "rl_type", "iterations",
                                     "station", "log_file"]
                                    + [c for c in model.summary_level_1.columns]]
    model_grid_obs["time_stamp"] = time_stamp
    model_grid_obs_costs = model.summary_level_2.copy()
    model_grid_obs_costs["model_type"] = model_type
    model_grid_obs_costs["hash_code_model"] = hash_code_model
    model_grid_obs_costs["hash_code_action"] = hash_code_action
    model_grid_obs_costs["rl_type"] = rl_type
    model_grid_obs_costs["iterations"] = iterations
    model_grid_obs_costs["station"] = station
    model_grid_obs_costs["log_file"] = log_file
    model_grid_obs_costs = model_grid_obs_costs[
        ["model_type", "hash_code_model", "hash_code_action", "rl_type", "iterations",
         "station", "log_file"]
        + [c for c in model.summary_level_2.columns]]
    model_grid_obs_costs["time_stamp"] = time_stamp
    model_grid_obs_steps = model.summary_level_3.copy()
    model_grid_obs_steps["model_type"] = model_type
    model_grid_obs_steps["hash_code_model"] = hash_code_model
    model_grid_obs_steps["hash_code_action"] = hash_code_action
    model_grid_obs_steps["rl_type"] = rl_type
    model_grid_obs_steps["iterations"] = iterations
    model_grid_obs_steps["station"] = station
    model_grid_obs_steps["log_file"] = log_file
    model_grid_obs_steps = model_grid_obs_steps[
        ["model_type", "hash_code_model", "hash_code_action", "rl_type", "iterations",
         "station", "log_file"]
        + [c for c in model.summary_level_3.columns]]
    model_grid_obs_steps["time_stamp"] = time_stamp
    return model_grid_obs, model_grid_obs_costs, model_grid_obs_steps

def upload_observed_tables(model_grid_observed, model_grid_observed_costs, model_grid_observed_steps):
    hash_code_model = model_grid_observed["hash_code_model"].unique()[0]
    hash_code_action = model_grid_observed["hash_code_action"].unique()[0]
    rl_type = model_grid_observed["rl_type"].unique()[0]
    station = model_grid_observed["station"].unique()[0]
    log_file = model_grid_observed["log_file"].unique()[0]
    db_gr = db.connect("/home/mwiubuntu/Seminararbeit/db_results/goal_recognition.db")
    for table in ["model_grid_observed", "model_grid_observed_costs", "model_grid_observed_steps"]:
        query = f"""DELETE FROM {table}
                    WHERE model_type = '{model_type}'
                        AND hash_code_model = '{hash_code_model}'
                        AND hash_code_action = '{hash_code_action}'
                        AND rl_type = {rl_type}
                        AND iterations IS NULL
                        AND station = '{station}'
                        AND log_file = '{log_file}'
        """
        db_gr.execute(query)
    db_gr.commit()
    db_gr.close()
    db_gr = db.connect("/home/mwiubuntu/Seminararbeit/db_results/goal_recognition.db")
    model_grid_observed.to_sql("model_grid_observed", db_gr, if_exists='append',
                                          index=False)
    model_grid_observed_costs.to_sql("model_grid_observed_costs", db_gr, if_exists='append',
                          index=False)
    model_grid_observed_steps.to_sql("model_grid_observed_steps", db_gr, if_exists='append',
                                index=False)
    db_gr.close()

if __name__ == "__main__":
    model_no = 7
    path_logs = f'/home/mwiubuntu/Seminararbeit/Interaction Logs/model_{model_no}/'
    log_folders = os.listdir(path_logs)
    list_files_obs = []
    for folder in log_folders:
        path_folder = path_logs + folder + "/"
        for file in os.listdir(path_folder):
            file_path = path_folder + file
            list_files_obs.append(file_path)
    list_files_obs.sort()
    observations = [pddl_observations(file) for file in list_files_obs]

    for observation in observations[142:]:
        station = observation.observation_path.split("/")[-2]
        log_file = observation.observation_path.split("/")[-1]
        print("-----------------")
        print(station, log_file)
        print("-----------------")
        goals = {1: {"keep_goal_1_reward": False,
                     "rl_models_dict": "from_goal_5_model_7_no_hl__04-08-24 23-08-06.keras"},
                 # 1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__04-08-24 23-07-13.keras"},
                 # 1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__04-08-24 23-07-13.keras"}
                 2: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__22-07-24 14-13-35.keras"},
                 3: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__06-08-24 09-29-21.keras"},
                 4: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__29-07-24 07-51-27.keras"},
                 5: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__04-08-24 23-08-06.keras"},
                 # 6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 07-23-38.keras"},
                 6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 08-09-00.keras"},
                 # 6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 09-36-56.keras"},
                 }
        if "Salmonellosis" in observation.name:
            #goals[7] = {"keep_goal_1_reward": True, "rl_models_dict": "salmonellosis_model_7_no_hl__14-08-24 22-29-26.keras"}
            #goals[7] = {"keep_goal_1_reward": True, "rl_models_dict": "salmonellosis_model_7_no_hl__15-08-24 14-46-54.keras"}
            goals[7] = {"keep_goal_1_reward": True, "rl_models_dict": "salmonellosis_model_7_no_hl__17-08-24 16-01-31.keras"}
        elif "E.coli" in observation.name:
            goals[7] = {"keep_goal_1_reward": True,
                        "rl_models_dict": "ecoli_model_7_no_hl__17-08-24 16-01-31.keras"}
        else:
            print("error", observation.observation_path)

        # instantiate domain
        model = 7
        if model > 8:
            add_actions = [{'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-outdoors-4b', 'loc-infirmary-kim']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC', 'instances': ['loc-infirmary-kim', 'loc-outdoors-4b']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-infirmary-medicine', 'loc-infirmary-bathroom']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-infirmary-bathroom', 'loc-infirmary-medicine']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-laboratory-front', 'loc-outdoors-null-b']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-outdoors-null-b', 'loc-laboratory-front']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-laboratory-midright', 'loc-laboratory-library']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-laboratory-library', 'loc-laboratory-midright']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-dininghall-front', 'loc-outdoors-null-e']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-outdoors-null-e', 'loc-dininghall-front']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-dininghall-back-souptable', 'loc-outdoors-null-d']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-outdoors-null-d', 'loc-dininghall-back-souptable']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-dininghall-back-souptable', 'loc-outdoors-2b']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-outdoors-2b', 'loc-dininghall-back-souptable']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-livingquarters-hall', 'loc-outdoors-2a']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-outdoors-2a', 'loc-livingquarters-hall']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-livingquarters-hall', 'loc-outdoors-null-g']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-outdoors-null-g', 'loc-livingquarters-hall']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-brycesquarters-hall', 'loc-outdoors-null-f']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-outdoors-null-f', 'loc-brycesquarters-hall']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-brycesquarters-hall', 'loc-brycesquarters-bedroom']},
                           {'action_ungrounded': 'ACTION-MOVETOLOC',
                            'instances': ['loc-brycesquarters-bedroom', 'loc-brycesquarters-hall']}]
            cp = ["person_in_room", "neighboring"]
        else:
            add_actions = None
            cp = ["person_in_room"]

        hash_code_model = '0af2422953491e235481a3b7e0a10b74afc640356989b94d627359d1'
        config = '495afca3d199dd8d66b44b1c5e414f225a19d42c9a540eabdcfec02e'

        path_pddl = r"/home/mwiubuntu/best_domains/"


        if "Salmonellosis" in observation.name:
            domain = pddl_domain(path_pddl + f"model_{model}_{config}_salmonellosis.pddl")
        elif "E.coli" in observation.name:
            domain = pddl_domain(path_pddl + f"model_{model}_{config}_ecoli.pddl")
        else:
            print("error", observation.observation_path)
        print(domain.domain_path)
        problem_list = [pddl_problem(path_pddl + f"model_{model}_goal_{goal}_crystal_island_problem.pddl")
                        for goal in goals.keys()]
        environment_list = [GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions).make_env()
                            for problem in problem_list]

        g=0
        for goal in goals.keys():
            if goals[goal]["keep_goal_1_reward"]:
                environment_list[g].set_final_reward(20)
                environment_list[g].set_additional_reward_fluents("(achieved_goal_1)", 10)
            g+=1

        path_rl_model = "/home/mwiubuntu/finalised_rl_models/"
        rl_model_list = [rl_planner.load_model(path_rl_model + f"goal_{goal}/" + goals[goal]["rl_models_dict"])
                         for goal in goals.keys()]

        talk_to_redundant = ['ACTION-CHANGE-FINAL-REPORT-FINALINFECTIONTYPE',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALINFECTIONTYPE',
                             'ACTION-CHANGE-FINAL-REPORT-FINALDIAGNOSIS',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALDIAGNOSIS',
                             'ACTION-CHANGE-FINAL-REPORT-FINALSOURCE',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALSOURCE',
                             'ACTION-CHANGE-FINAL-REPORT-FINALTREATMENT',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALTREATMENT',
                             'ACTION-HAND-FINAL-WORKSHEET', 'ACTION-PICKUP',
                             'ACTION-DROP', 'ACTION-STOWITEM', 'ACTION-RETRIEVEITEM',
                             'ACTION-CHOOSE-TESTCOMPUTER', 'ACTION-QUIZ']

        goal_six_redundant = ['ACTION-CHANGE-FINAL-REPORT-FINALINFECTIONTYPE',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALINFECTIONTYPE',
                             'ACTION-CHANGE-FINAL-REPORT-FINALDIAGNOSIS',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALDIAGNOSIS',
                             'ACTION-CHANGE-FINAL-REPORT-FINALSOURCE',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALSOURCE',
                             'ACTION-CHANGE-FINAL-REPORT-FINALTREATMENT',
                             'ACTION-UNSELECT-FINAL-REPORT-FINALTREATMENT',
                             'ACTION-HAND-FINAL-WORKSHEET', "ACTION-TALK-TO"]

        goal_seven_redundant = ['ACTION-PICKUP','ACTION-DROP', 'ACTION-STOWITEM', 'ACTION-RETRIEVEITEM',
                                'ACTION-CHOOSE-TESTCOMPUTER', 'ACTION-QUIZ']

        redundant_actions_dict = {"goal_1": talk_to_redundant,
                                  "goal_2": talk_to_redundant,
                                  "goal_3": talk_to_redundant,
                                  "goal_4": talk_to_redundant,
                                  "goal_5": talk_to_redundant,
                                  "goal_6": goal_six_redundant,
                                  "goal_7": goal_seven_redundant}

        multi_rl_planner = MultiRLPlanner(environment_list, rl_model_list, redundant_actions_dict=redundant_actions_dict)
        model = PRAPAgent(multi_rl_planner=multi_rl_planner, obs_action_sequence=observation)
        model.perform_solve_optimal()
        model.perform_solve_observed()

        model_grid_observed, model_grid_observed_costs, model_grid_observed_steps = create_db_tables(
            model=model, hash_code_model=hash_code_model,hash_code_action=config, station=station, log_file=log_file)
        t = time.time()
        upload_observed_tables(model_grid_observed, model_grid_observed_costs, model_grid_observed_steps)
        print("upload took:", time.time() - t, "seconds")
