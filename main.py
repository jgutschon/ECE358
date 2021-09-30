from numpy import random
from math import log
from typing import Tuple, List

# TODO: Function docstrings?
# TODO: Packet lengths? Looking at 4.5.1.1b


# Total Simulation Time (s)
# TODO: 4.5 outlines a method of finding an appropriate T. This may need to be
# adjusted accordingly
T = 1000

# Average number of packets generated/arrived (packets/second)
# Section 4.2, "queue is expected to receive 10 packets/sec on average"
LAMBDA = 10

# Average length of packet (bits)
# Section 5, Question 3
L = 2000

# Transmission rate of the output link (bits/second)
# Section 5, Question 3, "assume C = 1Mbps"
C = 1e6
""" Section 4.4 """

# TODO: Unsure if this is unnecessary. In 4.4.1 "Generating Exponential Random
# Variables", they say "We will use the inverse method", but is that the exponential
# method?
def inverse_random(min: float = 0, max: float = 1) -> float:
    return 1 / (random.uniform(min, max))

""" Section 4.4.1 """

def exponential_random(lam: float, min: float = 0, max: float = 1) -> float:
    return - (1 / lam) * log(1 - random.uniform(0, 1))

""" Section 4.5.1 """

# 4.5.1.1a
def packet_arrival_times() -> Tuple[List, List]:

    # List containing the arrival times for each packet, in order, and the
    # corresponding packet length
    arrival_times = list()
    packet_lengths = list()

    # Total time elapsed
    total_time_elapsed = 0

    while total_time_elapsed < T:
        # TODO: 4.5.1.1a says to use the "inverse method" while also saying the
        # "exponential distribution". Which is it?
        arrival_time = exponential_random(LAMBDA)
        # TODO: Unsure if passing L is correct
        packet_length = exponential_random(L)

        # Update variables. Update the arrival_times list before the
        # total_time_elapsed variable, since this needs to start with 0 seconds. Then
        # we will have:
        #
        # arrival_times packet_lengths
        # arrival_0 = 0 length_0
        # arrival_1     length_1
        # arrival_2     length_2
        arrival_times.append(total_time_elapsed)
        packet_lengths.append(packet_length)
        total_time_elapsed += arrival_time

    return arrival_times, packet_lengths

# 4.5.1.1b
def calculate_packet_departures():
    pass
