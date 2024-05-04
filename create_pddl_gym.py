from pddl import *
import pickle
import os


class GymCreator:
    def __init__(self, domain, problem):
        self.domain = domain
        self.problem = problem
        self.env_name = self.domain.name
        self.pickle_path = self._create_pickle_path()
        self.pickled_env = f"{self.pickle_path}env_{self.env_name}.pickle"

    def _str_pickle(self):
        str_pickle = f"""
with open (r"{self.pickled_env}", "wb") as outp:
    pickle.dump(env, outp, pickle.HIGHEST_PROTOCOL)
"""
        return str_pickle

    def _create_pickle_path(self):
        path = f"{os.getcwd()}/pickle_env/"
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def _read_pickled_env(self):
        return pickle.load(open(self.pickled_env, "rb"))

    def make_env(self):
        py_code = f"""
import pickle
import sys
sys.path.append("{os.getcwd()}")
from pddl import *

domain_path = "{self.domain.domain_path}"
env = pddl_domain(domain_path)

print(env.domain_path)

{self._str_pickle()}
        
        
"""
        py_file = self.pickle_path + f"env_{self.env_name}.py"
        with open(py_file, "w") as py_env:
            py_env.write(py_code)
        exec(open(py_file).read())
        env = self._read_pickled_env()
        os.remove(py_file)
        os.remove(self.pickled_env)
        return env



if __name__ == '__main__':
    env_creator = GymCreator(pddl_domain('domain.pddl'), pddl_problem('problem_A.pddl'))
    print(env_creator.env_name)
    env_creator._create_pickle_path()
    domain = env_creator.make_env()
    print(domain.action_dict)
