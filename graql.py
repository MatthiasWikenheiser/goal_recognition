from pddl import pddl_domain, pddl_problem, pddl_observations
from create_pddl_gym import GymCreator
import time
import copy
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rl_planner import RlPlanner, load_model
import os
import datetime as dt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
from scipy.optimize import fsolve
from dotenv import load_dotenv

load_dotenv('.env.graql')

class GRAQL:
    def __init__(self, env_list, rl_model_list, observation_sequence, hash_code_model, hash_code_action,
                 goal_keys=[f"goal_{i}" for i in range(1,8)],
                 discount_factor=0.95, additional_reward_fluents=None, scale_to=9, standard_neg_reward=-1):
        self.observation = observation_sequence
        self.station = self.observation.observation_path.split("/")[-2]
        self.log_file = self.observation.observation_path.split("/")[-1]
        self.additional_reward_fluents = additional_reward_fluents
        self.rl_model_dict, self.env_dict = self._init_dict(rl_model_list, goal_keys, env_list)
        self.hash_code_model = hash_code_model
        self.hash_code_action = hash_code_action
        self.performed_step = {}
        self.env_obs = env_list[0]
        self.current_state = self.env_obs.reset()
        self.q_table_startgame_predicted = {} # used to linearize q-values later
        self.discount_factor=discount_factor
        self.scale_to = scale_to
        self.standard_neg_reward = standard_neg_reward
        self.test_q_adjustment = {}
        self.metric = None
        self.q_summary = None
        self.summary = None
        self.threshold_control_table = None
        self.threshold = None
        self.metric_help_iterative = {}

    def _additional_fluent_true(self, key):
        if key in self.additional_reward_fluents.keys():
            env = self.env_dict[key]
            additional_fluent_true = env.observation_dict[self.additional_reward_fluents[key]["fluent"]]["value"] == 1
            return additional_fluent_true

    def goal_fluent_true(self, key):
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

    def _get_q_values(self, key, state, action_key):
        additional_fluent_true = self._additional_fluent_true(key)
        start_time = time.time()
        all_q_values = self.rl_model_dict[key].predict(state[np.newaxis, :], verbose=0)
        action_q_value = all_q_values[0][action_key]
        self.env_dict[key].step(action_key)
        goal_fluent_true = self.goal_fluent_true(key)
        after_between_reward = additional_fluent_true or goal_fluent_true
        if key in self.additional_reward_fluents.keys():
            if after_between_reward:
                t_between = None
                t_final = self._calc_distance(action_q_value,
                                              self.additional_reward_fluents[key]["target"])
            else:
                t_between = self._calc_distance(action_q_value,
                                                self.q_table_startgame_predicted[key]["q_val_between"])
                t_final = self.q_table_startgame_predicted[key]["steps_after_first_reward"] + t_between
        else:
            t_between = None
            if key == "goal_1":
                target_q = self.q_table_startgame_predicted["goal_5"]["theoretical_q_values"][5]
            else:
                target_q = self.scale_to  # actually only goal 2
            t_final = self._calc_distance(action_q_value, target_q)
        q_adjusted = self._adjust_q(q_value=action_q_value, key=key,
                                    t_final=max(t_final, 0),
                                    additional_fluent=after_between_reward,
                                    t_between=t_between)

        #shift_all_q_values = np.abs(np.min(all_q_values)) + all_q_values
        #nrmlsd_all_q_values = shift_all_q_values / np.sum(shift_all_q_values)
        #nrmlsd_all_q_values = (np.abs(np.min(all_q_values)) +  all_q_values)/ np.sum(all_q_values)
        #nrmlsd_action_q_value = nrmlsd_all_q_values[0][action_key]

        dict_goal = {
            key:
                {"all_q_values": all_q_values,
                 "action_q_value": action_q_value,
                 "q_adjusted": q_adjusted,
                 "t_final": t_final,
                 "t_between": t_between,
                 #"nrmlsd_all_q_values": nrmlsd_all_q_values,
                 #"nrmlsd_action_q_value": nrmlsd_action_q_value,
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

    def perform_solve_optimal(self, test_theoretical_adjustment=False):
        print("Compute theoretical q-values for linearization")
        for key in self.env_dict.keys():
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
            rl_planner.solve(print_actions=False)
            current_state = env.reset()
            for step in rl_planner.plan.keys():
                step_list.append(step+1)
                action_title = rl_planner.plan[step]
                action = env.inverse_action_dict[action_title]
                q_value = rl_model.predict(current_state[np.newaxis, :], verbose=0)[0][action]
                additional_fluent = self._additional_fluent_true(key)
                current_state, reward, done, _ = env.step(action)
                goal_fluent_true = self.goal_fluent_true(key)
                additional_fluent_list.append(additional_fluent or goal_fluent_true)
                action_title_list.append(action_title)
                action_list.append(action)
                q_value_list.append(q_value)
                reward_list.append(reward)
                done_list.append(done)

            self.q_table_startgame_predicted[key] = {"step": step_list,
                                            "action":  action_title_list,
                                            "action_key": action_list,
                                            "q_value": q_value_list,
                                            "reward": reward_list,
                                            "additional_fluent": additional_fluent_list,
                                            "done": done_list
                                            }


            self.q_table_startgame_predicted[key]["theoretical_q_values"]= self._theoretical_q_values(key)
            analyze_rewards = [x for x in reward_list if x > 0]
            if len(analyze_rewards) == 2:
                idx = np.argwhere(np.array(reward_list) == analyze_rewards[0])[0][0]
                self.q_table_startgame_predicted[key]["steps_after_first_reward"] = len(reward_list[idx+1:])
                q_val_between = self.q_table_startgame_predicted[key]["theoretical_q_values"][idx]
                self.q_table_startgame_predicted[key]["q_val_between"] = q_val_between

        if test_theoretical_adjustment:
            print("calc adjusted theoretical q_values")
            for key in self.q_table_startgame_predicted.keys():
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

    def test_q_adjustment_optimal_plans(self):
        self.test_q_adjustment = {}

        print("test q_adjustment on optimalplans")
        for key in self.env_dict.keys():
            env = self.env_dict[key]
            state = env.reset()
            done = False
            after_between_reward_list = []
            action_list = []
            action_q_list = []
            t_between_list = []
            t_final_list = []
            q_adjusted_list = []
            distance_list = []
            while not done:
                q_values = self.rl_model_dict[key].predict(state[np.newaxis, :], verbose=0)[0]
                action = np.argmax(q_values)
                action_q = q_values[action]
                action_grounded = env.action_dict[action]["action_grounded"]
                action_list.append(action_grounded)
                action_q_list.append(action_q)

                additional_fluent_true = self._additional_fluent_true(key)
                state, _, done, _ = env.step(action)
                goal_fluent_true = self.goal_fluent_true(key)

                after_between_reward = additional_fluent_true or goal_fluent_true
                after_between_reward_list.append(after_between_reward)
                if key in self.additional_reward_fluents.keys():
                    if after_between_reward:
                        t_between = None
                        t_final = self._calc_distance(action_q,
                                                        self.additional_reward_fluents[key]["target"])
                    else:
                        t_between = self._calc_distance(action_q,
                                                          self.q_table_startgame_predicted[key]["q_val_between"])
                        t_final = self.q_table_startgame_predicted[key]["steps_after_first_reward"] + t_between
                else:
                    t_between = None
                    if key == "goal_1":
                        target_q = self.q_table_startgame_predicted["goal_5"]["theoretical_q_values"][5]
                    else:
                        target_q = self.scale_to # actually only goal 2
                    t_final = self._calc_distance(action_q, target_q)

                t_between_list.append(t_between)
                t_final_list.append(t_final)
                q_adjusted = self._adjust_q(q_value=action_q, key=key,
                                                              t_final=max(t_final,0),
                                                              additional_fluent=(additional_fluent_true
                                                                                 or goal_fluent_true),
                                                              t_between=t_between)
                q_adjusted_list.append(q_adjusted)
                distance = self._calc_distance(q_adjusted, self.scale_to)
                distance_list.append(distance)

            self.test_q_adjustment[key] = {"after_between_reward": after_between_reward_list,
                                           "action": action_list,
                                           "q_values": action_q_list,
                                           "t_between": t_between_list,
                                           "t_final": t_final_list,
                                           "distance": distance_list,
                                           "q_adjusted": q_adjusted_list}

    def _reset_all_envs(self):
        for key in self.env_dict.keys():
            self.env_dict[key].reset()

    def _metric_kl_divergence(self, step, threshold):
        self.metric = "kl_divergence"
        self.threshold = threshold
        self.metric_help_iterative[step] = {"threshold_tolerance": None,
                                            "tolerance": None,
                                            "best_metric": None}

        # threshold as prob, f.e. every softmax q-value with prob higher as e.g. 0.5 (threshold-parameter) as tolerance
        # threshold is then: prob * log(prob)
        # threshold is cumulated over steps
        # if no goal is above that threshold just take the closest one
        t = np.abs(threshold)
        t = t * np.log(t)

        goals_remaining = self.performed_step[step]["goals_remaining"]
        max_closeness = -np.inf

        if step == 0:
            changed_goal_set = True
        else:
            goals_remaining_before = self.performed_step[step-1]["goals_remaining"]
            if len(goals_remaining_before) != len(goals_remaining):
                changed_goal_set = True
            else:
                changed_goal_set = False

        if changed_goal_set:
            self.metric_help_iterative[step]["threshold_tolerance"] = t
        else:
            self.metric_help_iterative[step]["threshold_tolerance"] = (t + self.metric_help_iterative
                                                                       [step - 1]["threshold_tolerance"])

        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10

        for goal in goals_remaining:
            # due to overflow, assumption: no action can cause a state where it needs 300 steps to reach the goal
            # in addition cast to float64
            qs = np.where(self.performed_step[step][goal]["all_q_values"][0] < -300, -300,
                          self.performed_step[step][goal]["all_q_values"][0]).astype(np.float64)
            # shift all values to start from zero
            min_q = np.min(qs)  # np.array([[q1, q1]]) -> index [0]
            shifted_qs = abs(min_q) + qs

            # deviating from amado et al, implement real softmax policy
            exp_Q = np.exp(shifted_qs)
            sum_q_values = np.sum(exp_Q)

            policy = exp_Q[self.performed_step[step]["action_key"]]
            q_transformed = policy / sum_q_values
            if q_transformed == 0:
                q_transformed += epsilon
            q_transformed * np.log(q_transformed)

            if changed_goal_set:
                sum_div = q_transformed * np.log(q_transformed)
            else:
                sum_div = ((q_transformed * np.log(q_transformed)) +
                           self.metric_help_iterative[step - 1][goal]["cumulated_sum"])
            self.metric_help_iterative[step][goal] = {"cumulated_sum": sum_div}

            self.performed_step[step][goal]["metric"] = sum_div  # measure of closeness
            if sum_div > max_closeness:
                max_closeness = sum_div

        self.metric_help_iterative[step]["best_metric"] = max_closeness
        # for relative application of threshold
        #lower_tolerance = max_closeness + self.metric_help_iterative[step]["threshold_tolerance"]
        lower_tolerance = self.metric_help_iterative[step]["threshold_tolerance"]
        self.metric_help_iterative[step]["tolerance"] = lower_tolerance

        # for relative application of threshold
        # self.performed_step[step]["prediction"] = [goal for goal in goals_remaining
                                                   # if self.performed_step[step][goal]["metric"] >= lower_tolerance]

        self.performed_step[step]["prediction"] = [goal for goal in goals_remaining
                                                   if self.performed_step[step][goal]["metric"] >= lower_tolerance
                                                   or self.performed_step[step][goal]["metric"] == max_closeness]

        goal_prob_nrmlsd = self.distances_to_probabilities_dict(step)
        for key in goal_prob_nrmlsd:
            self.performed_step[step][key]["goal_prob_nrmlsd"] = goal_prob_nrmlsd[key]

    def _metric_max_util(self, step, threshold):
        self.metric = "max_util"
        self.threshold = threshold
        self.metric_help_iterative[step] = {"threshold_tolerance": None,
                                            "tolerance": None,
                                            "best_metric": None}

        goals_remaining = self.performed_step[step]["goals_remaining"]
        max_closeness = -np.inf

        if step == 0:
            changed_goal_set = True
        else:
            goals_remaining_before = self.performed_step[step-1]["goals_remaining"]
            if len(goals_remaining_before) != len(goals_remaining):
                changed_goal_set = True
            else:
                changed_goal_set = False

        if changed_goal_set:
            self.metric_help_iterative[step]["threshold_tolerance"] = threshold
        else:
            self.metric_help_iterative[step]["threshold_tolerance"] = (threshold +
                                                                       self.metric_help_iterative[step - 1]
                                                                       ["threshold_tolerance"])
        for goal in goals_remaining:
            if changed_goal_set:
                sum_qs = self.performed_step[step][goal]["q_adjusted"]
            else:
                sum_qs = (self.performed_step[step][goal]["q_adjusted"] +
                          self.metric_help_iterative[step-1][goal]["cumulated_q"] )
            self.metric_help_iterative[step][goal] = {"cumulated_q": sum_qs}

            #for i in range(step+1):
                #sum_qs += self.performed_step[i][goal]["q_adjusted"]
                #threshold_tolerance += threshold

            self.performed_step[step][goal]["metric"] = sum_qs # measure of closeness
            if sum_qs > max_closeness:
                max_closeness = sum_qs

        lower_tolerance = max_closeness - self.metric_help_iterative[step]["threshold_tolerance"]
        self.metric_help_iterative[step]["best_metric"] = max_closeness
        self.metric_help_iterative[step]["tolerance"] = lower_tolerance
        self.performed_step[step]["prediction"] = [goal for goal in goals_remaining
                                                   if self.performed_step[step][goal]["metric"] >= lower_tolerance]

        goal_prob_nrmlsd = self.distances_to_probabilities_dict(step)
        for key in goal_prob_nrmlsd:
            self.performed_step[step][key]["goal_prob_nrmlsd"] = goal_prob_nrmlsd[key]

    def _metric_distance_to_goal(self, step, threshold):
        self.metric = "distance_to_goal"
        self.threshold = threshold

        goals_remaining = self.performed_step[step]["goals_remaining"]
        lowest_distance = np.inf

        for goal in goals_remaining:
            distance = self._calc_distance(self.performed_step[step][goal]["q_adjusted"],
                                           self.scale_to)
            if distance < lowest_distance:
                lowest_distance=distance
            self.performed_step[step][goal]["metric"] = distance

        upper_tolerance = lowest_distance + threshold
        self.performed_step[step]["prediction"] = [goal for goal in goals_remaining
                                                   if self.performed_step[step][goal]["metric"]<=upper_tolerance]

        self.metric_help_iterative[step] = {"threshold_tolerance": threshold,
                                            "tolerance": upper_tolerance,
                                            "best_metric": lowest_distance}

        goal_prob_nrmlsd = self.distances_to_probabilities_dict(step)
        for key in goal_prob_nrmlsd:
            self.performed_step[step][key]["goal_prob_nrmlsd"] = goal_prob_nrmlsd[key]

    def distances_to_probabilities_dict(self, step):
        # Extract the keys and values (distances) from the input dictionary
        #distances_dict = {k:v for k,v in zip(self.performed_step[step]["goals_remaining"],
                                             #[self.performed_step[step][goal]["metric"]
                                              #for goal in self.performed_step[step]["goals_remaining"]]
                                             #)}

        #keys = list(distances_dict.keys())
        #distances = np.array(list(distances_dict.values()))

        # Subtract the maximum distance to avoid numerical overflow
        #max_distance = np.max(distances)
        #adjusted_distances = distances - max_distance

        # Apply the softmax function to the adjusted distances
        #exp_values = np.exp(-adjusted_distances)  # Negative to favor smaller distances
        #probabilities = np.round(exp_values / np.sum(exp_values), 4)

        # Create a new dictionary with the same keys and the computed probabilities
        #probabilities_dict = dict(zip(keys, probabilities))
        probabilities_dict = {k: v for k, v in zip(self.performed_step[step]["goals_remaining"],
                                                 [np.nan
                                                  for goal in self.performed_step[step]["goals_remaining"]])}
        return probabilities_dict

    def _create_threshold_control_table(self):
        control_table = self.summary[["model_type", "hash_code_model", "hash_code_action", "rl_type", "iterations",
                                      "station", "log_file", "observed_action_no", "observed_action", "t",
                                      "goals_remaining"]].copy()
        for goal in self.performed_step[0]["goals_remaining"]:
            goal_metric = []
            for step in self.performed_step.keys():
                if goal in self.performed_step[step].keys():
                    goal_metric.append(self.performed_step[step][goal]["metric"])
                else:
                    goal_metric.append(np.nan)
            control_table.loc[:, f"{goal}_metric"] = pd.Series(goal_metric)

        threshold_tolerance_list = []
        tolerance_list = []
        best_metric_list = []
        for step in self.metric_help_iterative.keys():
            threshold_tolerance_list.append(self.metric_help_iterative[step]["threshold_tolerance"])
            tolerance_list.append(self.metric_help_iterative[step]["tolerance"])
            best_metric_list.append(self.metric_help_iterative[step]["best_metric"])
        control_table.loc[:, "best_metric"] = pd.Series(best_metric_list)
        control_table.loc[:, "threshold_tolerance"] = pd.Series(threshold_tolerance_list)
        control_table.loc[:, "tolerance"] = pd.Series(tolerance_list)
        control_table["prediction"] = self.summary["predicted_goals"]
        control_table["time_stamp"] = self.summary["time_stamp"]
        return control_table

    def _create_summary(self):
        model_type = []
        hash_code_model = []
        hash_code_action = []
        rl_type = []
        iterations = []
        station = []
        log_file = []
        observed_action_no = []
        observed_action = []
        t = []
        goals_remaining = []
        total_goals_no = []
        goals_achieved = []
        goals_achieved_no = []
        label = []
        predicted_goals = []
        predicted_goals_no = []
        prob = []
        diff_t = []
        seconds = []

        for step_key in self.performed_step.keys():
            step_dict = self.performed_step[step_key]
            model_type.append(f"graql_model_{self.metric}")
            hash_code_model.append(self.hash_code_model)
            hash_code_action.append(self.hash_code_action)
            rl_type.append("0")
            iterations.append(f"threshold: {self.threshold}")
            station.append(self.station)
            log_file.append(self.log_file)
            observed_action_no.append(step_key+1)
            observed_action.append(step_dict["action"])
            t.append(self.observation.obs_file.loc[step_key, "t"])
            goals_remaining.append(str(step_dict["goals_remaining"]).replace("'",""))
            total_goals_no.append(len(step_dict["goals_remaining"]))
            goals_achieved.append(str(step_dict["goals_remaining"]).replace("'","")
                                  if len(step_dict["prediction"]) > 0 else "[]")
            goals_achieved_no.append(len(step_dict["goals_remaining"]) if len(step_dict["prediction"]) > 0 else "[]")
            l = self.observation.obs_file.loc[step_key, "label"]
            label.append(l)
            predicted_goals.append(str(step_dict["prediction"]).replace("'",""))
            predicted_goals_no.append(len(step_dict["prediction"]))
            try:
                prob.append(max([(step_dict[g]["goal_prob_nrmlsd"], g) for g in step_dict["prediction"]])[0])
            except:
                prob.append(np.nan)
            diff_t.append(step_dict["time_step"])
            seconds.append(step_dict["solved_time"])

        summary = pd.DataFrame({"model_type": model_type,
                                "hash_code_model": hash_code_model,
                                "hash_code_action": hash_code_action,
                                "rl_type": rl_type,
                                "iterations": iterations,
                                "station": station,
                                "log_file": log_file,
                                "observed_action_no": observed_action_no,
                                "observed_action": observed_action,
                                "t": t,
                                "goals_remaining": goals_remaining,
                                "total_goals_no": total_goals_no,
                                "goals_achieved": goals_achieved,
                                "goals_achieved_no": goals_achieved_no,
                                "label": label,
                                "predicted_goals": predicted_goals,
                                "predicted_goals_no": predicted_goals_no
                  })


        summary["correct_prediction"] = \
            summary.apply \
                (lambda x: 1 if str(x["label"]) != "nan" and \
                                str(x["label"]) in str(x["predicted_goals"]).replace("_", "") else 0,
                 axis=1)

        summary["prob"] = prob
        summary["diff_t"] = diff_t
        summary["seconds"] = seconds
        summary["time_left"] = np.where(summary["total_goals_no"] == summary["goals_achieved_no"],
                                               summary["diff_t"] - summary["seconds"], 0)
        summary["time_stamp"] = self.time_stamp_summary
        return summary

    def _create_q_summary(self):
        self.time_stamp_summary = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        list_step_dfs = []
        for step_key in self.performed_step.keys():
            step = self.performed_step[step_key]
            step_df = pd.DataFrame({
                "model_type":[f"graql_model_{self.metric}" for _ in step["goals_remaining"]],
                "hash_code_model": [self.hash_code_model for _ in step["goals_remaining"]],
                "hash_code_action": [self.hash_code_action for _ in step["goals_remaining"]],
                "station":[self.station for _ in step["goals_remaining"]],
                "log_file": [self.log_file for _ in step["goals_remaining"]],
                "observed_action_no": [step_key+1 for _ in step["goals_remaining"]],
                "observed_action": [step["action"] for _ in step["goals_remaining"]],
                "goal": step["goals_remaining"],
                # for goal_achieved dont distinguish between goals
                "goal_achieved": [1 if len(step["prediction"]) > 0 else 0 for _ in step["goals_remaining"]],
                "q_value": [step[goal]["action_q_value"] for goal in step["goals_remaining"]],
                "q_adjusted": [step[goal]["q_adjusted"] for goal in step["goals_remaining"]],
                "t_final": [step[goal]["t_final"] for goal in step["goals_remaining"]],
                "t_between": [step[goal]["t_between"] for goal in step["goals_remaining"]],
                "seconds": [step[goal]["time"] for goal in step["goals_remaining"]],
                "metric": [step[goal]["metric"] for goal in step["goals_remaining"]],
                "threshold": [self.threshold for _ in step["goals_remaining"]],
                "goal_prob_nrmlsd": [step[goal]["goal_prob_nrmlsd"] for goal in step["goals_remaining"]],
                "time_stamp": [self.time_stamp_summary for _ in step["goals_remaining"]]})
            list_step_dfs.append(step_df)
        q_summary = pd.concat(list_step_dfs)
        return q_summary

    def perform_solve_observed(self, metric, threshold):
        self.current_state = self.env_obs.reset()
        self._reset_all_envs()
        start_time = time.time()
        i = 0
        while i < self.observation.obs_len and str(self.observation.obs_file.loc[i, "label"]) != "nan":
            step_time = time.time()
            action = self.observation.obs_file.loc[i, "action"].replace(" ", "_")
            time_step = self.observation.obs_file.loc[i, "diff_t"]
            goals_remaining = [goal for goal in self.rl_model_dict.keys() if goal in
                                                          self.observation.obs_file.loc[i, "goals_remaining"]]
            print("-----------")
            print("step:", i+1, ",time elapsed:", round(step_time - start_time, 2), "s")
            print(action, ",", time_step, "seconds to solve")
            print("goals_left: ", goals_remaining)

            action_key = self.env_obs.inverse_action_dict[action]
            result_dict = {}
            with ThreadPoolExecutor() as executor:
                futures = []
                for key in goals_remaining:
                    print("start: ", key)
                    future = executor.submit(self._get_q_values,
                                             key=key,
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

            for result in result_dict.keys():
                self.performed_step[i][result] = result_dict[result]

            if metric=="distance_to_goal":
                self._metric_distance_to_goal(step=i, threshold=threshold)
            elif metric=="max_util":
                self._metric_max_util(step=i, threshold=threshold)
            elif metric =="kl_divergence":
                self._metric_kl_divergence(step=i, threshold=threshold)

            solved_time = time.time() - step_time
            print("solved in ", round(solved_time, 2), "s")

            if solved_time > time_step:
                print("timeout")
                self.performed_step[i]["prediction"] = [] # erase


            self.performed_step[i]["solved_time"] = solved_time

            self.current_state, _, _, _ = self.env_obs.step(action_key)
            i+=1
        self.q_summary = self._create_q_summary()
        self.summary = self._create_summary()
        self.threshold_control_table = self._create_threshold_control_table()


if __name__ == "__main__":
    hash_code_model = os.getenv('HASH_CODE_MODEL')
    config = os.getenv('CONFIG')
    path_logs = os.getenv('PATH_LOGS').replace("number", os.getenv("MODEL_NO"))
    log_folders = os.listdir(path_logs)
    list_files_obs = []
    for folder in log_folders:
        path_folder = path_logs + folder + "/"
        for file in os.listdir(path_folder):
            file_path = path_folder + file
            list_files_obs.append(file_path)
    list_files_obs.sort()
    observations = [pddl_observations(file) for file in list_files_obs]

    goals = {1: {"keep_goal_1_reward": False, "rl_models_dict": os.getenv("RL_GOAL_1")},
             # 1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__04-08-24 23-07-13.keras"},
             # 1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__04-08-24 23-07-13.keras"}
             2: {"keep_goal_1_reward": False, "rl_models_dict": os.getenv("RL_GOAL_2")},
             3: {"keep_goal_1_reward": True, "rl_models_dict": os.getenv("RL_GOAL_3")},
             4: {"keep_goal_1_reward": True, "rl_models_dict": os.getenv("RL_GOAL_4")},
             5: {"keep_goal_1_reward": True, "rl_models_dict": os.getenv("RL_GOAL_5")},
             # 6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 07-23-38.keras"},
             6: {"keep_goal_1_reward": False, "rl_models_dict": os.getenv("RL_GOAL_6")},
             # 6: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__13-09-24 09-36-56.keras"},
             # 7: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__14-08-24 22-29-26.keras"}
             # 7: {"keep_goal_1_reward": True, "rl_models_dict": "model_7_no_hl__15-08-24 14-46-54.keras"}
             7: {"keep_goal_1_reward": True, "rl_models_dict": os.getenv("RL_GOAL_7")}
             }

    obs = observations[81]
    # obs = random.choice(observations)
    print(obs.observation_path)

    #----------
    #obs = pddl_observations(r"E:\Interaction logs\Test-Session/1_log_Salmonellosis.csv")
    #----------

    # instantiate domain
    model = int(os.getenv("MODEL_NO"))
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


    path_pddl = os.getenv("PATH_PDDL")
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

    path_rl_model = os.getenv("PATH_RL_MODEL")
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


    model = GRAQL(env_list=environment_list, rl_model_list=rl_model_list, hash_code_model=hash_code_model,
                  hash_code_action=config, observation_sequence=obs,
                  additional_reward_fluents=additional_reward_fluents)
    model.perform_solve_optimal(test_theoretical_adjustment=False)
    # model.test_q_adjustment_optimal_plans()

    model.perform_solve_observed(metric=os.getenv("METRIC"), threshold=float(os.getenv("THRESHOLD")))
    # x = model.q_summary[model.q_summary["goal"] == 'goal_6']
