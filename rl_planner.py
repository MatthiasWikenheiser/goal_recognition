from pddl import pddl_domain, pddl_problem
from create_pddl_gym import GymCreator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np


def load_model(path):
    return tf.keras.models.load_model(path)


class RlPlanner:
    def __init__(self, environment, rl_model):
        self.env = environment
        self.rl_model = rl_model

    def set_state(self, state):
        self.env.reset(startpoint=False, state=self.start_state)

    def reset(self):
        self.env.reset()

    def solve(self, print_actions=True):
        done = False
        while not done:
            q_values = self.rl_model.predict(self.env._get_obs_vector()[np.newaxis,:])
            action = np.argmax(q_values)
            if print_actions:
                print(self.env.action_dict[action]["action_grounded"])
            _, reward, done, _ = env.step(action)




if __name__ == '__main__':
    goal = 1
    keep_goal_1_reward = False

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

    path_pddl = r"/home/mwiubuntu/best_domains/"
    domain = pddl_domain(path_pddl + f"model_{model}_{config}.pddl")
    problem = pddl_problem(path_pddl + f"model_{model}_goal_{goal}_crystal_island_problem.pddl")
    env_creator_ci = GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions)
    env = env_creator_ci.make_env()

    if keep_goal_1_reward:
        env.set_final_reward(20)
        env.set_additional_reward_fluents("(achieved_goal_1)", 10)

    # load rl_model
    rl_models_dict = {1: "model_7_no_hl__20-06-24 22-09-10.keras",
                      2: "model_7_no_hl__22-07-24 14-13-35.keras",
                      3: "model_7_no_hl__23-07-24 23-41-57.keras"}

    path_rl_model = f"/home/mwiubuntu/finalised_rl_models/goal_{goal}/"
    rl_model = load_model(path_rl_model + rl_models_dict[goal])

    #instantiate rl_planner
    rl_planner = RlPlanner(env, rl_model)

    #load_sample_states
    

    rl_planner.solve()
