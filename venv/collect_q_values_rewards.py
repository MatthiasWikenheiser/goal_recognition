from pddl import *
from create_pddl_gym import GymCreator
from rl_planner import RlPlanner, load_model
import numpy as np
import pandas as pd

if __name__ == "__main__":
    model_no = 7

    goals = {1: {"keep_goal_1_reward": False, "rl_models_dict": "from_goal_5_model_7_no_hl__04-08-24 23-08-06.keras"},
             #1: {"keep_goal_1_reward": False, "rl_models_dict": "model_7_no_hl__04-08-24 23-07-13.keras"},
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
    problem_list = [pddl_problem(path_pddl + f"model_{model}_goal_{i}_crystal_island_problem.pddl") for i in range(1,8)]

    environment_list = [GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions).make_env()
                        for problem in problem_list]

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

    dict_results = {"goal":[],"action_title": [],"action": [],"q_value": [],"reward": [], "done": []}

    for i in range(len(environment_list)):
        print("%%%%%%%%%%%%%%")
        print(f"goal_{i+1}")
        action_title_list = []
        action_list = []
        q_value_list = []
        reward_list = []
        done_list = []
        env = environment_list[i]
        rl_model = rl_model_list[i]
        rl_planner = RlPlanner(env, rl_model)
        rl_planner.solve(print_actions=False)
        print(rl_planner.plan)
        current_state = env.reset()
        for step in rl_planner.plan.keys():
            print("--------------------")
            action_title = rl_planner.plan[step]
            action = env.inverse_action_dict[action_title]
            q_value = rl_model.predict(current_state[np.newaxis, :], verbose=0)[0][action]
            print(action_title, action)
            print("Q-value:", q_value)
            current_state,reward, done, _ = env.step(action)
            print("reward ", reward)
            print("done ", done)
            action_title_list.append(action_title)
            action_list.append(action)
            q_value_list.append(q_value)
            reward_list.append(reward)
            done_list.append(done)

        dict_results["goal"] += [f"goal_{i+1}" for _ in range(len(action_title_list))]
        dict_results["action_title"] += action_title_list
        dict_results["action"] += action_list
        dict_results["q_value"] += q_value_list
        dict_results["reward"] += reward_list
        dict_results["done"] += done_list
        result = pd.DataFrame(dict_results)
        #result.to_csv(r"E:\q_values.csv", sep=";", decimal=",")


