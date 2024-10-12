from pddl import pddl_domain, pddl_problem, pddl_observations
from create_pddl_gym import GymCreator
import time
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rl_planner import RlPlanner, load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
from scipy.optimize import fsolve

class GRAQL:
    def __init__(self, env_list, rl_model_list, observation_sequence, goal_keys=[f"goal_{i}" for i in range(1,8)],
                 discount_factor=0.95, additional_reward_fluents=None, scale_to=9, standard_neg_reward=-1):
        self.observation = observation_sequence
        self.additional_reward_fluents = additional_reward_fluents
        self.rl_model_dict, self.env_dict = self._init_dict(rl_model_list, goal_keys, env_list)
        self.performed_step = {}
        self.env_obs = env_list[0]
        self.current_state = self.env_obs.reset()
        self.q_table_startgame_predicted = {} # used to linearize q-values later
        self.discount_factor=discount_factor
        self.scale_to = scale_to
        self.standard_neg_reward = standard_neg_reward

    def _additional_fluent_true(self, key):
        if key in self.additional_reward_fluents.keys():
            env = self.env_dict[key]
            additional_fluent_true = env.observation_dict[self.additional_reward_fluents[key]["fluent"]]["value"] == 1
            return additional_fluent_true

    def goal_fluent_true(self,key):
        if key in self.additional_reward_fluents.keys():
            env = self.env_dict[key]
            goal_fluent_true = env.observation_dict[env.goal_fluents.replace("(", "").replace(")", "")]["value"] == 1
            return goal_fluent_true


    def _init_dict(self, rl_model_list, goal_keys, env_list):
        rl_model_dict = {}
        env_dict = {}
        i = 0
        while i < len(rl_model_list):
            rl_model_dict[goal_keys[i]] = rl_model_list[i]
            env_dict[goal_keys[i]] = env_list[i]
            #env_dict[goal_keys[i]].reset()
            i+=1
        return rl_model_dict, env_dict

    def _create_env_obs(self):
        key_0 = list(self.rl_model_dict.keys())[0]
        return copy.deepcopy(self.rl_model_dict[key_0]["env"])

    def _get_q_values(self, rl_model_key, state, action_key):
        #print(action_key)
        start_time = time.time()
        all_q_values = self.rl_model_dict[rl_model_key].predict(state[np.newaxis, :], verbose=0)
        #predicted_action = np.argmax(all_q_values)
        #print(rl_model_key, predicted_action, self.env_obs.action_dict[predicted_action])
        action_q_value = all_q_values[0][action_key]

        shift_all_q_values = np.abs(np.min(all_q_values)) + all_q_values
        nrmlsd_all_q_values = shift_all_q_values / np.sum(shift_all_q_values)

        #nrmlsd_all_q_values = (np.abs(np.min(all_q_values)) +  all_q_values)/ np.sum(all_q_values)
        nrmlsd_action_q_value = nrmlsd_all_q_values[0][action_key]

        dict_goal = {
            rl_model_key:
                {"all_q_values": all_q_values,
                 "action_q_value": action_q_value,
                 "nrmlsd_all_q_values": nrmlsd_all_q_values,
                 "nrmlsd_action_q_value": nrmlsd_action_q_value,
                 "time": time.time() - start_time
                 }
            }
        return dict_goal

    def _adjust_q(self, key, q_value, t_final, additional_fluent, t_between=None):
        if key == "goal_1":
            return (q_value - (self.q_table_startgame_predicted["goal_5"]["theoretical_q_values"][5] - self.scale_to) *
                    (self.discount_factor ** t_final))

        elif key in self.additional_reward_fluents.keys():
            if additional_fluent:
                return (q_value - (self.additional_reward_fluents[key]["target"] - self.scale_to) *
                        (self.discount_factor ** t_final))
            else:
                reward_corrected = self.additional_reward_fluents[key]["additional"] - self.standard_neg_reward
                #print("reward_corrected: ", reward_corrected)
                #print("q_value: ", q_value)
                #print("t_between: ", t_between)
                #print("t_final: ", t_final)
                return (q_value - (reward_corrected * (self.discount_factor ** t_between)) -
                 (self.additional_reward_fluents[key]["target"] - self.scale_to) * (self.discount_factor ** t_final))
        else:
            return q_value

    def _calc_distance(self, q_value, reward_value, minus_rewards=-1, inital_guess=15):
        def q_value_equation(N):
            return (minus_rewards * (1 - self.discount_factor ** N) /
                    (1 - self.discount_factor)) + (self.discount_factor ** N * reward_value) - q_value
        distance = fsolve(q_value_equation, inital_guess)
        return distance[0]


    def perform_solve_optimal(self, print_actions=False, timeout=10):
        for key in self.env_dict.keys():
            print("%%%%%%%%%%%%%%")
            print(key)
            action_title_list = []
            action_list = []
            q_value_list = []
            #q_value_adjusted_list = []
            reward_list = []
            done_list = []
            step_list = []
            additional_fluent_list = []
            env = self.env_dict[key]
            rl_model = self.rl_model_dict[key]
            rl_planner = RlPlanner(env, rl_model)
            #print(rl_planner)
            rl_planner.solve(print_actions=False)
            #print(rl_planner.plan)
            current_state = env.reset()
            for step in rl_planner.plan.keys():
                #print("--------------------")
                step_list.append(step+1)
                action_title = rl_planner.plan[step]
                action = env.inverse_action_dict[action_title]
                q_value = rl_model.predict(current_state[np.newaxis, :], verbose=0)[0][action]
                #print(action_title, action)
                #print("Q-value:", q_value)
                additional_fluent = self._additional_fluent_true(key)
                current_state, reward, done, _ = env.step(action)
                goal_fluent_true = self.goal_fluent_true(key)
                additional_fluent_list.append(additional_fluent or goal_fluent_true)
                #print("reward ", reward)
                #print("done ", done)
                action_title_list.append(action_title)
                action_list.append(action)
                q_value_list.append(q_value)
                #q_value_adjusted_list.append(q_value_adjusted)
                reward_list.append(reward)
                done_list.append(done)


            #linearize = reward_list[-1] != self.reward_scale
            #two_rewards = len(analyze_rewards) > 1




            self.q_table_startgame_predicted[key] = {"step": step_list,
                                            "action":  action_title_list,
                                            "action_key": action_list,
                                            "q_value": q_value_list,
                                            #"q_value_adjusted": q_value_adjusted_list,
                                            "reward": reward_list,
                                            "additional_fluent": additional_fluent_list,
                                            "done": done_list
                                            #"linearize": linearize,
                                            #"two_rewards": two_rewards
                                            }
            #if key == "goal_1":
                #self.q_table_startgame_predicted[key]["linearize"] = True #rl_model from goal5 used

            self.q_table_startgame_predicted[key]["theoretical_q_values"]= self._theoretical_q_values(key)
            analyze_rewards = [x for x in reward_list if x > 0]
            if len(analyze_rewards) == 2:
                #print(analyze_rewards)
                #print(analyze_rewards[0])
                #print(np.argwhere(np.array(reward_list) == analyze_rewards[0]))
                idx = np.argwhere(np.array(reward_list) == analyze_rewards[0])[0][0]
                q_val_between = self.q_table_startgame_predicted[key]["theoretical_q_values"][idx]
                self.q_table_startgame_predicted[key]["q_val_between"] = q_val_between




        #print("/////calc adjusted q_values")
        for key in self.q_table_startgame_predicted.keys():
            #print(key)
            theoretical_q_value_adjusted_list = []
            step = 1
            theoretical_q_values = self.q_table_startgame_predicted[key]["theoretical_q_values"]
            additional_fluents = self.q_table_startgame_predicted[key]["additional_fluent"]
            for theoretical_q_value in theoretical_q_values:
                if not additional_fluents[step-1] is None:
                    if not additional_fluents[step-1]:
                        t_between = self._calc_distance(theoretical_q_value,
                                                        self.q_table_startgame_predicted[key]["q_val_between"])
                else:
                    t_between = None
                #print("t_between: ",t_between )

                theoretical_q_value_adjusted = self._adjust_q(q_value=theoretical_q_value, # for goal 1 it will be wrong
                                                              key=key,
                                                              t_final=len(theoretical_q_values) - step,
                                                              additional_fluent=additional_fluents[step-1],
                                                              t_between=t_between)
                theoretical_q_value_adjusted_list.append(theoretical_q_value_adjusted)
                step += 1
            self.q_table_startgame_predicted[key]["theoretical_q_value_adjusted"] = theoretical_q_value_adjusted_list

    def _theoretical_q_values(self, key):
        theoretical_q_values = []
        i = len(self.q_table_startgame_predicted[key]["reward"]) - 1
        current_q = self.q_table_startgame_predicted[key]["reward"][i]
        i-=1
        theoretical_q_values.append(current_q)

        while i >= 0:
            current_q = self.q_table_startgame_predicted[key]["reward"][i] + self.discount_factor*current_q
            theoretical_q_values.append(current_q)
            i-=1

        return theoretical_q_values[::-1]







    def perform_solve_observed(self, metric_func=None):
        start_time = time.time()
        i = 0
        while i < 30:#self.observation.obs_len and str(self.observation.obs_file.loc[i, "label"]) != "nan":
            step_time = time.time()
            action = self.observation.obs_file.loc[i, "action"].replace(" ", "_")
            time_step = self.observation.obs_file.loc[i, "diff_t"]
            goals_remaining = [goal for goal in self.rl_model_dict.keys() if goal in
                                                          self.observation.obs_file.loc[i, "goals_remaining"]]
            print("-----------")
            print("step:", i, ",time elapsed:", round(step_time - start_time, 2), "s")
            print(action, ",", time_step, "seconds to solve")
            print("goals_left: ", goals_remaining)

            action_key = self.env_obs.inverse_action_dict[action]
            result_dict = {}
            with ThreadPoolExecutor() as executor:
                futures = []
                for key in goals_remaining:
                    print("start: ", key)
                    future = executor.submit(self._get_q_values,
                                             rl_model_key=key,
                                             state=self.current_state,
                                             action_key=action_key
                                             )
                    futures.append(future)

                for future in futures:
                    # Ensuring all futures are done
                    result_dict.update(future.result())

            self.performed_step[i] = {"action": action,
                                      "time_step": time_step,
                                      "goals_remaining": goals_remaining,
                                      "action_key": action_key,
                                      }
            max_q_value = (None, 0)
            max_nrmlds_q_value = (None, 0)

            for result in result_dict.keys():
                self.performed_step[i][result] = result_dict[result]
                print(result, self.performed_step[i][result]["action_q_value"],
                      self.performed_step[i][result]["nrmlsd_action_q_value"])
                if self.performed_step[i][result]["action_q_value"] > max_q_value[1]:
                    max_q_value = (result, self.performed_step[i][result]["action_q_value"])
                if self.performed_step[i][result]["nrmlsd_action_q_value"] > max_nrmlds_q_value[1]:
                    max_nrmlds_q_value = (result, self.performed_step[i][result]["nrmlsd_action_q_value"])

            print("predicted by q-value", max_q_value)
            print("predicted by normalised q-value", max_nrmlds_q_value)

            self.current_state, _, _, _ = self.env_obs.step(action_key)
            i+=1


if __name__ == "__main__":
    model_no = 7
    path_logs = f'E:/Interaction logs/model_{model_no}/logs_model_{model_no}/'
    log_folders = os.listdir(path_logs)
    list_files_obs = []
    for folder in log_folders:
        path_folder = path_logs + folder + "/"
        for file in os.listdir(path_folder):
            file_path = path_folder + file
            list_files_obs.append(file_path)
    list_files_obs.sort()
    observations = [pddl_observations(file) for file in list_files_obs]

    goals = {1: {"keep_goal_1_reward": False, "rl_models_dict": "from_goal_5_model_7_no_hl__04-08-24 23-08-06.keras"},
             # 1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__04-08-24 23-07-13.keras"},
             # 1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__04-08-24 23-07-13.keras"}
             2: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__22-07-24 14-13-35.keras"},
             3: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__06-08-24 09-29-21.keras"},
             4: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__29-07-24 07-51-27.keras"},
             5: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__04-08-24 23-08-06.keras"},
             # 6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 07-23-38.keras"},
             6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 08-09-00.keras"},
             # 6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 09-36-56.keras"},
             # 7: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__14-08-24 22-29-26.keras"}
             # 7: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__15-08-24 14-46-54.keras"}
             7: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__17-08-24 16-01-31.keras"}
             }

    obs = observations[81]
    # obs = random.choice(observations)
    print(obs.observation_path)

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
    config = '495afca3d199dd8d66b44b1c5e414f225a19d42c9a540eabdcfec02e'

    path_pddl = r"E:/best_domains/"
    domain = pddl_domain(path_pddl + f"model_{model}_{config}.pddl")
    #problem = pddl_problem(path_pddl + f"model_{model}_goal_1_crystal_island_problem.pddl")
    problem_list = [pddl_problem(path_pddl + f"model_{model}_goal_{goal}_crystal_island_problem.pddl")
                    for goal in goals.keys()]

    environment_list = [GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions).make_env()
                        for problem in problem_list]

    #env = GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions).make_env()

    g = 0
    for goal in goals.keys():
        if goals[goal]["keep_goal_1_reward"]:
            environment_list[g].set_final_reward(20)
            environment_list[g].set_additional_reward_fluents("(achieved_goal_1)", 10)
        g += 1

    environment_list[-2].set_final_reward(20)
    environment_list[-2].set_additional_reward_fluents("(wearable_picked food-milk)", 10)
    drop_keys = [k for k in environment_list[-2].action_dict.keys()
                 if "ACTION-DROP_FOOD-MILK" in environment_list[-2].action_dict[k]["action_grounded"]
                 and "LOC-LABORATORY-FRONT" not in environment_list[-2].action_dict[k]["action_grounded"]]
    for drop_key in drop_keys:
        environment_list[-2].action_dict[drop_key]["effects"] = \
            environment_list[-2].action_dict[drop_key]["effects"].replace("(increase (costs) 1.0)",
                                                                          "(increase (costs) 100.0)")

    path_rl_model = "E:/finalised_rl_models/"
    rl_model_list = [load_model(path_rl_model + f"goal_{goal}/" + goals[goal]["rl_models_dict"])
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

    goal_seven_redundant = ['ACTION-PICKUP', 'ACTION-DROP', 'ACTION-STOWITEM', 'ACTION-RETRIEVEITEM',
                            'ACTION-CHOOSE-TESTCOMPUTER', 'ACTION-QUIZ']

    redundant_actions_dict = {"goal_1": talk_to_redundant,
                              "goal_2": talk_to_redundant,
                              "goal_3": talk_to_redundant,
                              "goal_4": talk_to_redundant,
                              "goal_5": talk_to_redundant,
                              "goal_6": goal_six_redundant,
                              "goal_7": goal_seven_redundant}

    additional_reward_fluents = {"goal_3": {"fluent" : "achieved_goal_1", "additional": 9, "target": 19},
                                 "goal_4": {"fluent": "achieved_goal_1", "additional": 9, "target": 19},
                                 "goal_5": {"fluent": "achieved_goal_1", "additional": 9, "target": 19},
                                 "goal_6": {"fluent": "wearable_picked food-milk", "additional": 9, "target": 19},
                                 "goal_7": {"fluent": "achieved_goal_1", "additional": 9, "target": 19}}


    model = GRAQL(env_list=environment_list, rl_model_list=rl_model_list,
                   observation_sequence=obs, additional_reward_fluents=additional_reward_fluents)
    model.perform_solve_optimal()

    model._calc_distance(-2.87,9)



   # model.perform_solve_observed()





