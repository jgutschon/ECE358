from numpy import random
from math import log

# Section 4.4.1
def exponential_random(lambda: float) -> float:
    x = - (1 / lambda) * log(1 - random.uniform(0, 1))

# Section 4.5.1

