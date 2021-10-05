#!/usr/bin/python3
import numpy as np
from math import log
from typing import Tuple
from enum import Enum
import datetime


# Total Simulation Time (s)
T = 1000

# Average length of packet (bits)
L = 2000

# Transmission rate of the output link (bits/second)
C = 1e6

# 5, since lab manual specifies: "Generate a set of random observation times according
# to the packet arrival distribution with rate at least 5 times the rate of the packet
# arrival"
RATIO_OBSERVATIONS_TO_ARRIVALS = 5

""" Section 4.4.1 """


def exponential_random(lam: float, min: float = 0, max: float = 1) -> float:
    """Computes an exponential random value. It is computed using the following:
    -(1 / lam) * log(1 - np.random.uniform(min, max)).

    Args:
        lam (float): Inverse of the average exponential random value.
        min (float, optional): Minimum value for the random value. Defaults to 0.
        max (float, optional): Maximum value for the random value.  Defaults to 1.

    Returns:
        float: The randomly generated exponential random value.
    """
    return -(1 / lam) * log(1 - np.random.uniform(min, max))


""" Section 4.5.1 """


class Queue:
    class EventType(Enum):
        ARRIVAL = 0
        DEPARTURE = 1
        OBSERVATION = 2

    def __init__(self,) -> None:
        self._reset_lists()

    def simulate(
        self, avg_arrival_time: float, K: int = None
    ) -> Tuple[int, int, int, int]:
        """Simulates the M/M/1 or M/M/1/K queue, as specified by the K parameter. It
        begins by resetting the current queue information, then populates the packet
        arrival times. The packet arrival times are dependent on avg_arrival_time. The
        observer events are then created, at 5 * avg_arrival_time, as specified to do
        so from the lab manual. The departure times are then computed for each
        corresponding packet, based on the transmission rate. If a packet is dropped,
        the arrival time and departure time in their corresponding lists are set to
        None. Then, the observer, departure, and arrival lists are combined, and the
        queue statistics are computed.

        Args:
            avg_arrival_time (float): Average arrival packet arrival time.
            K (int, optional): Queue packet limit. If it is not specified, a M/M/1
            queue with infinite buffer length will be used instead of a M/M/1/K.
            Defaults to None.

        Returns:
            Tuple[int, int, int, int]: Returns a tuple of the:
            1. computed number of arrival packets,
            2. average number of packets in the queue
            3. p_idle, proportion of time the server is idle
            4. p_loss, probability of packet loss. Ratio of the total number of packets
            lost due to the buffer being full, to the total number of generated packets.
        """
        self._K = K

        # Reset the list
        self._reset_lists()
        self._generate_packet_arrival_times(avg_arrival_time)
        self._generate_observer_events(avg_arrival_time)
        self._generate_departure_times()
        self._queue_simulation()

        num_arrival_packets = len(self._arrival_times)

        # E[N]
        avg_packets_in_queue = self._queue_packet_count / len(self._observation_times)
        p_idle = self._queue_idle_count / len(self._observation_times)
        p_loss = self._queue_dropped_packets / num_arrival_packets

        return num_arrival_packets, avg_packets_in_queue, p_idle, p_loss

    def _reset_lists(self) -> None:
        """Resets the various lists of the queue, that are required for its operation.
        This should be called if the queue has been previous used, and needs to clear
        its state for a different dataset
        """
        self._arrival_times = list()
        self._packet_lengths = list()
        self._departure_times = list()
        self._observation_times = list()

        self._queue_dropped_packets = 0
        self._queue_idle_count = 0
        self._queue_packet_count = 0

    """ Section 4.5.1.1a """

    def _generate_packet_arrival_times(self, avg_arrival_time: float) -> None:
        """Generates a list of packets with their arrival times. Their arrival times
        are equal to the previous arrival time + the exponential random value. The
        final time in the list will just be greater than the total simulation time.
        """

        total_time_elapsed = 0

        while total_time_elapsed < T:
            arrival_time = exponential_random(1 / avg_arrival_time)
            # The packet length is in bits, which must be an integer value
            packet_length = int(exponential_random(1 / L))

            # Update variables. Update the arrival_times list before the
            # total_time_elapsed variable, since this needs to start with 0 seconds. Then
            # we will have:
            #
            # arrival_times packet_lengths
            # arrival_0 = 0 length_0
            # arrival_1     length_1
            # arrival_2     length_2
            self._arrival_times.append(total_time_elapsed)
            self._packet_lengths.append(packet_length)
            total_time_elapsed += arrival_time

        self._departure_times = [None] * len(self._arrival_times)

    """ Section 4.5.1.1b """

    def _generate_departure_times(self) -> None:
        """Takes the self._arrival_times list, and computes the corresponding departure
        time for the packet. This is determined either by the departure time of the
        previous packet plus the transmission time for the packet
        (packet length / transmission rate (C)), or the addition of the packets arrival
        time + it's transmission time. The greater value is used. For the former, it is
        assuming that the packet has not been processed yet, while the latter assumes
        that the previous packet has been completed before the next arrival and that
        the queue is empty.

        For the M/M/1/K case, the number of packets currently in the queue are counted.
        If the number of packets in the queue is greater than K, the arrival time and
        departure time will be set to None. This allows the sorting step to recognize
        which packets were dropped, and simply drop the packets. This allows the
        determining of packet drops to be controlled within this function, and not
        later.
        """

        previous_departure_time = 0
        time_idle = 0
        for i in range(len(self._arrival_times)):

            # M/M/1/K Scenario
            if self._K is not None:
                # Count the number of items currently in the queue

                # Work backwards in the departure list, starting from the previous queue
                # item. back_count represents how far in the list to search back, while
                # queue_count increments when self._departure_times[i - back_count]
                # > self._arrival_times[i].

                back_count = 1
                queue_count = 0
                while True:

                    # If back_count > i, then we're at the beginning of the departure list
                    if back_count > i:
                        break

                    # This packet was dropped. Continue searching backwards so increment
                    # the back_count, but DO NOT increment the queue_count since this
                    # packet never entered the queue
                    if self._departure_times[i - back_count] is None:
                        back_count += 1
                        continue

                    # Once we reach a departure time that is less than the current arrival
                    # time, then we know we've found the last possible queue item, since
                    # the departure times are sorted
                    if self._departure_times[i - back_count] < self._arrival_times[i]:
                        break

                    # At this point, we've found an item that is in the queue. Increment
                    # the queue_count and continue reverse searching the departure list
                    back_count += 1
                    queue_count += 1

                # If there are currently too many items in the queue, drop the packet.
                if self._K is queue_count:
                    # departure_time and arrival_time = None signifies that this packet was
                    # dropped.

                    # When the lists are merged and sorted, these will be ignored. This
                    # results in a sorted DES list where there will be no dropped packets.
                    self._departure_times[i] = None
                    self._arrival_times[i] = None
                    self._queue_dropped_packets += 1

                    # Skip this iteration, since none of the below steps applies to a
                    # dropped packet
                    continue

            # The best departure time is caculated by the arrival time
            best_begin_processing_time = self._arrival_times[i]

            # If the previous departure time is greater than the best departure time,
            # we have to begin processing after that has ended
            begin_processing_time = max(
                previous_departure_time, best_begin_processing_time
            )

            if best_begin_processing_time == begin_processing_time:
                time_idle += begin_processing_time - previous_departure_time

            # The transmission time needs to be added, which is the packet length
            # divided by the transmission rate
            self._departure_times[i] = (
                begin_processing_time + self._packet_lengths[i] / C
            )
            previous_departure_time = self._departure_times[i]

    """ Section 4.5.1.1c """

    def _generate_observer_events(self, avg_arrival_time: float) -> None:
        """Generates observer times at a an avg time of avg_arrival_time / 5. Their
        arrival times are equal to the previous arrival time + the exponential random
        value. The final time in the list will just be greater than the total
        simulation time.

        Args:
            avg_arrival_time (float): Generates observer times at a average rate of 
            avg_arrival_time / 5
        """
        # Reset the list
        self._observation_times = list()

        # First instance
        total_time_elapsed = exponential_random(
            1 / (avg_arrival_time / RATIO_OBSERVATIONS_TO_ARRIVALS)
        )

        while total_time_elapsed < T:
            observation_time = exponential_random(
                1 / (avg_arrival_time / RATIO_OBSERVATIONS_TO_ARRIVALS)
            )

            # Update variables
            self._observation_times.append(total_time_elapsed)
            total_time_elapsed += observation_time

    # 4.5.2 and 4.5.3
    def _queue_simulation(self) -> None:
        """Simulates the Queue. Creates a DES, where the list of packet events are
        sorted in simulation order. The combined list includes the arrival,
        observation, and departure times. Any None values in the arrival and departure
        lists are removed, since these were determined to be dropped. This function
        will update the self._queue_idle_count and self._queue_packet_count variables
        """

        """ Section 4.5.2 """
        # Combine all 3 event lists and their time stamps, and sort the list in
        # ascending order of time. Keep track of which

        # These were set to None when the buffer dropped this packet. These values are
        # removed before sorting
        sorted_events = list()
        arr_i = dep_i = obs_i = 0

        while (
            arr_i != len(self._arrival_times)
            or dep_i != len(self._departure_times)
            or obs_i != len(self._observation_times)
        ):
            if arr_i != len(self._arrival_times) and self._arrival_times[arr_i] is None:
                arr_i += 1
                continue

            if (
                dep_i != len(self._departure_times)
                and self._departure_times[dep_i] is None
            ):
                dep_i += 1
                continue

            lowest_time = float("inf")

            if (
                arr_i != len(self._arrival_times)
                and lowest_time > self._arrival_times[arr_i]
            ):
                event = Queue.EventType.ARRIVAL
                lowest_time = self._arrival_times[arr_i]

            if (
                dep_i != len(self._departure_times)
                and lowest_time > self._departure_times[dep_i]
            ):
                event = Queue.EventType.DEPARTURE
                lowest_time = self._departure_times[dep_i]

            if (
                obs_i != len(self._observation_times)
                and lowest_time > self._observation_times[obs_i]
            ):
                event = Queue.EventType.OBSERVATION
                lowest_time = self._observation_times[obs_i]

            if event is Queue.EventType.ARRIVAL:
                arr_i += 1
            elif event is Queue.EventType.DEPARTURE:
                dep_i += 1
            elif event is Queue.EventType.OBSERVATION:
                obs_i += 1

            sorted_events.append(event)

        """ Section 4.5.3 """

        # Process the DES events in order. Keep a counter of the number of arrivals,
        # departures, and observer events.
        arrival_count = 0
        departure_count = 0
        observer_events_count = 0

        # On an observer event, count the number of times the buffer is idle, and a
        # increment the counter of the number of packets in the buffer
        self._queue_idle_count = 0
        self._queue_packet_count = 0

        for event in sorted_events:
            if event == Queue.EventType.ARRIVAL:
                arrival_count += 1
                continue
            elif event == Queue.EventType.DEPARTURE:
                departure_count += 1
                continue

            # This is an Queue.EventType.OBSERVATION
            if arrival_count == departure_count:
                self._queue_idle_count += 1

            self._queue_packet_count += arrival_count - departure_count
            observer_events_count += 1


if __name__ == "__main__":

    # Q1
    lam = 75
    number_sample = list()
    for _ in range(1000):
        number_sample.append(exponential_random(lam))

    # Expected values found from https://en.wikipedia.org/wiki/Exponential_distribution
    print("Q1 - Exponential Random (1000 samples):")
    print(
        f"\tAverage(actual): {np.average(number_sample):.4f} Average(expected): {1 / lam:.4f}"
    )
    print(
        f"\tVariance(actual): {np.var(number_sample):.4f} Variance(expected):"
        f"{1 / (lam ** 2):.4f}"
    )

    # Q2 - Done in Report
    # Q3
    print("Q3 - Queue Implementation:")
    queue = Queue()
    for rho in np.arange(0.25, 1.05, 0.1):
        # Rearranging rho (utilization rate) generates the following
        lam = rho * C / L
        average_time = 1 / lam
        num_packets, _, p_idle, _ = queue.simulate(average_time)
        print(
            f"\tRho: {rho:.2f}, Number of Packets: {num_packets}, "
            f"P_Idle (s/s): {p_idle:.4f}"
        )

    # Q4
    print("Q4 - Rho = 1.2:")
    rho = 1.2
    lam = rho * C / L
    average_time = 1 / lam
    num_packets, _, p_idle, _ = queue.simulate(average_time)
    print(
        f"\tRho: {rho:.2f}, Number of Packets: {num_packets}, "
        f"Proportion Time Idle (s/s): {p_idle:.4f}"
    )

    # Q5 - Done in Report
    # Q6
    print("Q6 - M/M/1/K:")
    for K_val in [10, 25, 50]:
        for rho in np.arange(0.5, 1.6, 0.1):
            lam = rho * C / L
            average_time = 1 / lam
            num_packets, e_n, p_idle, p_loss = queue.simulate(average_time, K_val)

            print(
                f"\tM/M/1/{K_val} - Rho: {rho:.2f}, "
                f"Average Packets in Queue (E[N]): {e_n}, "
                f"Proportion Time Idle (s/s): {p_idle:.4f}: "
                f"Proportion Packets Lost (packet/packet): {p_loss:.4f}"
            )
