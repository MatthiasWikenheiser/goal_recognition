from pddl import *
import create_pddl_gym
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import time

def _check_action(action_dict, action_idx, current_fluents):
    if _action_possible(action_dict, action_idx, current_fluents):
        return (action_idx, action_dict[action_idx]["action_grounded"])
    return None

def get_all_possible_actions(action_dict, fluents, workers = 1):
    indices = range(len(action_dict.keys()))
    check_action_partial = partial(_check_action, action_dict, current_fluents=fluents)
    with Pool(workers) as pool:
    #with ThreadPoolExecutor(max_workers=workers) as executor:
        # Create a partial function with action_dict and fluents already set
        results = pool.map(check_action_partial, indices)
        #results = list(executor.map(check_action_partial, indices))

    # Filter out None values from the results
    list_actions = [action for action in results if action is not None]
    return list_actions


def _action_possible(action_dict, action_idx, current_fluents):
    action = action_dict[action_idx]
    fluents = [create_pddl_gym._clean_literal(sf) for sf in current_fluents]
    precondition = action["precondition"]
    if len(action["parameter_variable"]) > 0:
        i = 0
        while i < len(action["parameter_variable"]):
            param = action["parameter_variable"][i]
            inst = action["instances"][i]
            precondition = precondition.replace(param, inst)
            i += 1
    precondition_given = create_pddl_gym._recursive_check_precondition(precondition, fluents, start_point=True)
    return precondition_given


if __name__ == '__main__':
    model = 11
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
    path = f"C:/Users/Matthias/OneDrive/goal_recognition/Domain and Logs/model_{model}/"
    domain = pddl_domain(path + f"{model}_crystal_island_domain.pddl")
    problem = pddl_problem(path + f"model_{model}_goal_1_crystal_island_problem.pddl")
    env_creator_ci = create_pddl_gym.GymCreator(domain, problem, constant_predicates=cp, add_actions=add_actions)
    env_ci = env_creator_ci.make_env()

    #_action_possible(env_ci.action_dict, 1, env_ci.get_current_fluents())

    start = time.time()
    get_all_possible_actions(env_ci.action_dict, env_ci.get_current_fluents(), workers=3)
    print(time.time() - start)
    start = time.time()
    env_ci.get_all_possible_actions()
    print(time.time() - start)

    #print(_check_action(env_ci.action_dict, 1, env_ci.get_current_fluents()))

