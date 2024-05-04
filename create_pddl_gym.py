from pddl import *
import os
import gym
import sys


class GymCreator:
    def __init__(self, domain, problem):
        self.domain = domain
        self.problem = problem
        self.env_name = self.domain.name
        self.py_path = self._create_py_path()

    def _create_py_path(self):
        path = f"{os.getcwd()}/py_path/"
        if not os.path.exists(path):
            os.mkdir(path)
        return path


    def _str_import_statements(self):
        str_imports = f"""
import subprocess
def install(module):
    subprocess.check_call(['pip', 'install', module])
    print("The module", module, "was installed")
try:
    import gym
except:
    install('gym')
    import gym

from gym import Env
from gym.spaces import Discrete
        
        
        """
        return str_imports

    def _str_env_class(self):
        str_class=f"""
class PDDLENV(Env):
    def __init__(self):
        self.name = "{self.env_name}"
        self.action_space = Discrete(3)
        
        
        
"""
        return str_class

    def make_env(self):
        py_code = f"""{self._str_import_statements()}
{self._str_env_class()}"""


        py_file = self.py_path + "env_pddl.py"
        with open(py_file, "w") as py_env:
            py_env.write(py_code)
        sys.path.append(self.py_path)
        from env_pddl import PDDLENV
        return PDDLENV()



if __name__ == '__main__':
    env_creator = GymCreator(pddl_domain('domain.pddl'), pddl_problem('problem_A.pddl'))
    env = env_creator.make_env()
    print(env.name)
