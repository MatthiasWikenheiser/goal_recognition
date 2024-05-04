
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
        
        
        

class PDDLENV(Env):
    def __init__(self):
        self.name = "toy_prap"
        self.action_space = Discrete(3)
        
        
        
