
from .greedy import Greedy
from .random import Random
from .hybrid import Hybrid
#from .rl import RL
from .strategy import Strategy

strategies = {
    'greedy': Greedy,
    'random': Random,
    'hybrid': Hybrid,
#    'rl': RL
}
