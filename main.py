import numpy as np
from math import log
from typing import List, Tuple
from dataclasses import dataclass

# TODO: Function docstrings?

# Total Simulation Time (s)
# TODO: 4.5 outlines a method of finding an appropriate T. This may need to be
# adjusted accordingly
T = 1000

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
    return 1 / (np.random.uniform(min, max))


""" Section 4.4.1 """


def exponential_random(lam: float, min: float = 0, max: float = 1) -> float:
    return -(1 / lam) * log(1 - np.random.uniform(min, max))


""" Section 4.5.1 """


@dataclass
class QueueData:
    arrival_time: float
    packet_length: float
    departure_time: float = 0.0


# 4.5.1.1a
def packet_arrival_times(avg_time: float) -> List[QueueData]:

    # List containing the arrival times for each packet, in order, and the
    # corresponding packet length
    queue_data = list()

    # Total time elapsed
    total_time_elapsed = 0

    while total_time_elapsed < T:
        # TODO: 4.5.1.1a says to use the "inverse method" while also saying the
        # "exponential distribution". Which is it?
        arrival_time = exponential_random(1 / avg_time)
        packet_length = exponential_random(1 / L)

        # Update variables. Update the arrival_times list before the
        # total_time_elapsed variable, since this needs to start with 0 seconds. Then
        # we will have:
        #
        # arrival_times packet_lengths
        # arrival_0 = 0 length_0
        # arrival_1     length_1
        # arrival_2     length_2
        packet_info = QueueData(total_time_elapsed, packet_length)
        queue_data.append(packet_info)
        total_time_elapsed += arrival_time

    return queue_data


# 4.5.1.1b
def calculate_MM1_departures(
    queue_data: List[QueueData],
) -> Tuple[List[QueueData], float]:
    # M/M/1 Scenario. -1 Indicates infinite buffer size
    previous_departure_time = 0
    time_idle = 0
    for i in range(len(queue_data)):
        # The best departure time is caculated by the arrival time
        best_begin_processing_time = queue_data[i].arrival_time

        # If the previous departure time is greater than the best departure time,
        # we have to begin processing after that has ended
        begin_processing_time = max(previous_departure_time, best_begin_processing_time)
        time_idle += begin_processing_time - previous_departure_time

        # The transmission time needs to be added, which is the packet length
        # divided by the transmission rate
        queue_data[i].departure_time = (
            begin_processing_time + queue_data[i].packet_length / C
        )
        previous_departure_time = queue_data[i].departure_time
        # print(
        #     queue_data[i].arrival_time,
        #     queue_data[i].packet_length / C,
        #     queue_data[i].departure_time,
        #     queue_data[i].departure_time - queue_data[i].arrival_time,
        #     True if begin_processing_time is best_begin_processing_time else False,
        # )

    # queue_data, time_idle
    return queue_data, time_idle


if __name__ == "__main__":

    # Q1
    lam = 75
    number_sample = list()
    for _ in range(1000):
        number_sample.append(exponential_random(lam))

    # Expected values found from https://en.wikipedia.org/wiki/Exponential_distribution
    print("Q1 - Exponential Random (1000 samples):")
    print(
        f"\tAverage(actual): {np.average(number_sample)} Average(expected): {1 / lam}"
    )
    print(
        f"\tVariance(actual): {np.var(number_sample)} Variance(expected):"
        f"{1 / (lam ** 2)}"
    )

    # Q2 - Done in Report
    # Q3

    # TODO: Part 1 asks for what the average number of packets in the queue, as a
    # function of rho
    # TODO: Part 2 asks for a function. It's just P_idle = 1 - rho
    # TODO: Part 2 asks for how we got this function

    print("Q3 - Queue Implementation:")
    for rho in np.arange(0.25, 0.95, 0.1):
        # Rearranging rho (utilization rate) generates the following
        lam = rho * C / L
        average_val = 1 / lam
        queue_data = packet_arrival_times(average_val)
        queue_data, time_idle = calculate_MM1_departures(queue_data)
        print(
            f"\tRho: {rho:.2f}, Number of Packets: {len(queue_data)}, "
            f"Proportion Time Idle (s/s): {time_idle / T:.4f}"
        )

    # Q4
    print("Q4 - Rho = 1.2:")
    rho = 1.2
    lam = rho * C / L
    average_val = 1 / lam
    queue_data = packet_arrival_times(average_val)
    queue_data, time_idle = calculate_MM1_departures(queue_data)
    print(
        f"\tRho: {rho:.2f}, Number of Packets: {len(queue_data)}, "
        f"Proportion Time Idle (s/s): {time_idle / T:.4f}"
    )
