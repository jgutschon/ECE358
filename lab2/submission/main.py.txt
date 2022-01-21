#!/usr/bin/python3
"""ECE358 - Computer Networks, Lab2. Group 95.
Code written solely by Jonathan Gutschon and Daniel Robson.
To run, run the command: 
python3 ./main.py
"""
import argparse
import math
import random
from collections import deque
from typing import List, Tuple

# Distance Between Neighbouring Nodes [m]
DISTANCE = 10

# Speed of Light [m / s]
LIGHT_SPEED = 3e8

# Propagation Speed [m / s]
PROPAGATION_SPEED = 2 / 3 * LIGHT_SPEED

# Length of packet [bits]
PACKET_LENGTH = 1500

# Transmission rate of the output link [bits / s]
TRANSMISSION_RATE = 1e6

# Propagation Delay [s]
T_PROPAGATION = DISTANCE / PROPAGATION_SPEED

# Transmission Delay [s]
T_TRANSMISSION = PACKET_LENGTH / TRANSMISSION_RATE


def exponential_random(rate: float, min: float = 0, max: float = 1) -> float:
    """Computes an exponential random value.

    Args:
        rate (float): Inverse of the average exponential random value.
        min (float, optional): Minimum value for the random value. Defaults to 0.
        max (float, optional): Maximum value for the random value.  Defaults to 1.

    Returns:
        float: The randomly generated exponential random value.
    """
    return -(1 / rate) * math.log(1 - random.random())


def exponential_backoff(collision_counter: int) -> float:
    """The exponential backoff time to delay when a collision occurs. It is given by
    the formula: rand(0, 2**i - 1) * 512 bit times.

    Args:
        collision_counter (int): collision counter

    Returns:
        float: the amount of time to wait
    """
    return_val = random.randint(0, 2 ** collision_counter - 1) * (
        512 / TRANSMISSION_RATE
    )
    return return_val


class Packet:
    def __init__(self, arrival_time: float):
        """Class defining a packet. A packet stores when it will arrive and its
        collision counters.
        Args:
            arrival_time (float): arrival time for a packet
        """
        self.arrival_time = arrival_time
        self.collision_counter = 0
        self.busy_collision_counter = 0


class Node:
    def __init__(self, arrival_rate: float, simulation_time: float):
        """Node class. Defines a connection to a bus, and holds a queue of packets
        that need to be transmitted.

        Args:
            arrival_rate (float): The average arrival rate for a packet in packets/s
            simulation_time (float): Simulation time to generate packets for.
        """
        self._simulation_time = simulation_time
        self.packet_queue = deque()
        self._generate_packet_arrival_times(arrival_rate)

    def _generate_packet_arrival_times(self, arrival_rate: float) -> None:
        """Generates a list of packets with their arrival times. Their arrival times
        are equal to the previous arrival time + the exponential random value. The
        final time in the list will just be greater than the total simulation time.

        Args:
            arrival_rate (float): Average time for a packet to arrive at a node.
        """

        next_arrival_time = exponential_random(arrival_rate)

        while next_arrival_time < self._simulation_time:
            self.packet_queue.append(Packet(next_arrival_time))
            next_arrival_time += exponential_random(arrival_rate)

    def remove_front_packet(self) -> None:
        """Removes the front packet in the queue. Updates the next packet to be the
        popped packet's arrival time if it's greater."""

        prev_packet = self.packet_queue.popleft()

        if self.is_queue_empty():
            return

        current_packet = self.packet_queue[0]

        current_packet.arrival_time = max(
            prev_packet.arrival_time, current_packet.arrival_time
        )

    def is_queue_empty(self) -> bool:
        """Returns whether the packet queue is empty.

        Returns:
            bool: Returns True if the packet queue is empty, or if the first packet's
            value is greater than the simulation time, False otherwise.
        """
        if not self.packet_queue:
            return True

        # If the arrival time is greater than the simulation time, return true, meaning
        # that the packet is empty
        return self.packet_queue[0].arrival_time > self._simulation_time

    def increment_collision_counter(self) -> bool:
        """Increments the collision counter for the front packet. If the packet's
        collision counter is greater than 10, the packet is dropped. Returns whether
        the packet was dropped.

        Returns:
            bool: Returns True if packet was dropped. False otherwise.
        """
        current_packet = self.packet_queue[0]
        current_packet.collision_counter += 1

        if current_packet.collision_counter > 10:
            self.remove_front_packet()
            return True

        return False

    def increment_busy_collision_counter(self) -> bool:
        """Increments the busy collision counter for the front packet. If the packet's
        collision counter is greater than 10, the packet is dropped. Returns whether
        the packet was dropped.

        Returns:
            bool: Returns True if packet was dropped. False otherwise.
        """
        current_packet = self.packet_queue[0]
        current_packet.busy_collision_counter += 1

        if current_packet.busy_collision_counter > 10:
            self.remove_front_packet()
            return True

        return False

    def reset_busy_collision_counter(self) -> None:
        """Resets the busy collision counter for the packet."""
        self.packet_queue[0].busy_collision_counter = 0


class NodeManager:
    def __init__(
        self,
        node_count: float,
        arrival_rate: float,
        simulation_time: float,
        is_persistent: bool,
    ):
        """Class for the Manager of the Nodes. Controls and manages the insertion of
        packets from the nodes into the bus using CSMA/CD.

        Args:
            node_count (float): The number of nodes to initialize
            arrival_rate (float): The average arrival rate of packets in packet/s
            simulation_time (float): The total simulation time to run for
            is_persistent (bool): Defines whether to use persistent or non-persistent
            CSMA/CD
        """
        self._simulation_time = simulation_time
        self._nodes = [
            Node(arrival_rate, self._simulation_time) for _ in range(node_count)
        ]
        self.successful_transmission_count = 0
        self.transmitted_packet_count = 0
        self.is_persistent = is_persistent
        # Only relevant for the non-persistent case
        self.packet_drop_count = 0

    def _handle_successful_transmission(self, node_id: int):
        """Handles a successful transmission from a node. Removes the packet from the
        node's packet queue, increments the counter, and updates each node's first
        packet's arrival time if it noticed that the medium was busy with the
        successfully transmitted node.

        Args:
            node_id (int): Node ID for the node that was successfully transmitted.
        """
        node = self._nodes[node_id]
        packet_arrival_time = node.packet_queue[0].arrival_time
        self.successful_transmission_count += 1
        node.remove_front_packet()

        for neighbour_node_id, neighbour_node in enumerate(self._nodes):

            if neighbour_node.is_queue_empty():
                continue

            propagation_delay = T_PROPAGATION * abs(neighbour_node_id - node_id)
            # For either case, the arrival time needs to be updated if the packet was
            # sent between T_a + T_prop <= T_b < T_a + T_prop + T_trans
            if self.is_persistent:
                if (
                    packet_arrival_time + propagation_delay
                    <= neighbour_node.packet_queue[0].arrival_time
                    < packet_arrival_time + propagation_delay + T_TRANSMISSION
                ):
                    # For the persistent case, the packet's arrival time is updated to
                    # be right when the medium is sensed as empty, which is once the
                    # packet has passed
                    neighbour_node.packet_queue[0].arrival_time = (
                        packet_arrival_time + propagation_delay + T_TRANSMISSION
                    )
            else:
                while (
                    packet_arrival_time + propagation_delay
                    <= neighbour_node.packet_queue[0].arrival_time
                    < packet_arrival_time + propagation_delay + T_TRANSMISSION
                ):

                    # If the node we are checking is the same node, we don't expect to
                    # attempt transmission when the node is aware that it's currently
                    # transmitting a node. Set this node arrival time to be the time
                    # for the packet to fully transmit.
                    if neighbour_node_id is node_id:
                        neighbour_node.packet_queue[0].arrival_time = (
                            packet_arrival_time + T_TRANSMISSION
                        )
                        break

                    # For the non-persistent case, a busy collision counter is
                    # incremented for the packet until it sense that the medium is
                    # empty. A random wait time is added for when to check again. If
                    # the counter reaches 10, the packet is dropped. If the packet
                    # senses that the medium is empty, then it resets its counter
                    packet_dropped = neighbour_node.increment_busy_collision_counter()

                    # We don't expect the packet to be dropped often. Since the random
                    # wait is in 512 * bittimes and a packet is 1500 bits, it is
                    # expected for the packets to be rarely dropped.
                    if packet_dropped:
                        self.packet_drop_count += 1
                        # Check to see if there are any nodes left. If none, exit
                        if neighbour_node.is_queue_empty():
                            break

                    else:
                        neighbour_node.packet_queue[
                            0
                        ].arrival_time += exponential_backoff(
                            neighbour_node.packet_queue[0].busy_collision_counter
                        )

    def simulate(self) -> Tuple[float, float]:
        """Simulates the CSMA/CD system. Finds earliest arrival node and attempts to
        add it to the bus. If the node experiences a collision, both packets increment
        a counter and reattempt after an exponential backoff. Each attempt of
        transmission by any packet increments a counter. If there is going to be a
        collision, it is recognized by colliding nodes instantaneously. For a
        successful transmission (no collision), a counter is incremented, and then any
        nodes that notice that the bus is busy will wait until the packet passes
        (persistent case), or wait a random time to check again (non-persistent).

        Returns:
            Tuple[float, float]: Returns the efficiency and throughput from the
            simulation.
        """
        while True:
            # Finds the next node to attempt transmission
            node_id, node_arrival_time = self._next_transmission_node()

            # We've run out of valid nodes
            if node_id == -1:
                break

            # Store the node ids that attempt to transmit during this time.
            transmitting_nodes = [node_id]

            # Determine if there will be a collision with any node.
            for neighbour_node_id, neighbour_node in enumerate(self._nodes):

                if neighbour_node_id == node_id or neighbour_node.is_queue_empty():
                    continue

                propagation_delay = T_PROPAGATION * abs(neighbour_node_id - node_id)

                neighbour_arrival_time = neighbour_node.packet_queue[0].arrival_time

                # Collision Case
                if neighbour_arrival_time <= node_arrival_time + propagation_delay:

                    transmitting_nodes.append(neighbour_node_id)

                # Otherwise, the packet does not collide with this node

            self.transmitted_packet_count += len(transmitting_nodes)

            # We've finished iterating over every node. We'll handle the successful
            # transmission case first
            if len(transmitting_nodes) == 1:
                self._handle_successful_transmission(node_id)
                continue

            # Now we'll deal with any collisions that occurred while transmitting this
            # packet. It is simply the sum of the maximum time that a collision was
            # detected at a colliding sending node (time for original packet to reach a
            # different sending node, denoted by max_time_of_collision) and the
            # exponential backoff. If the collision counter is zero after incrementing,
            # it means that the counter was >10 for the node, meaning that the node is
            # dropped, and the wait time is not updated.

            for colliding_node_id in transmitting_nodes:
                colliding_node = self._nodes[colliding_node_id]

                packet_dropped = colliding_node.increment_collision_counter()

                if not packet_dropped:

                    # Add the exponential backoff
                    colliding_node.packet_queue[0].arrival_time += exponential_backoff(
                        colliding_node.packet_queue[0].collision_counter
                    )

                    # Reset the busy collision counter since each of these nodes sensed
                    # that the medium was empty
                    colliding_node.reset_busy_collision_counter()

        efficiency = self.successful_transmission_count / (
            self.transmitted_packet_count + self.packet_drop_count
        )
        throughput = (
            self.successful_transmission_count
            * PACKET_LENGTH
            / self._simulation_time
            / 1e6
        )
        return efficiency, throughput

    def _next_transmission_node(self) -> Tuple[int, float]:
        """Finds the next node to attempt transmission.

        Returns:
            Tuple[int, float]: NodeID to attempt transmission, and the corresponding
            arrival time.
        """
        min_time = float("inf")
        node_id = -1

        for node_i, node in enumerate(self._nodes):
            if not node.is_queue_empty():
                arrival_time = node.packet_queue[0].arrival_time
                if arrival_time < min_time and arrival_time < self._simulation_time:
                    min_time = arrival_time
                    node_id = node_i

        return node_id, min_time


def simulate(
    is_persistent: bool,
    arrival_rates: List[int],
    simulation_time: float,
    print_csv: bool,
):
    """Creates a CSMA/CD instance of NodeManager with the passed parameters. Prints the
    statistics from the attempt.

    Args:
        is_persistent (bool): True is the CSMA/CD is persistent
        arrival_rates (List[int]): List of the arrival rates to attempt in packets/s
        simulation_time (float): Amount of time to run the simulation for
        print_csv (bool): Flag signaling to print in csv format
    """

    if print_csv:
        print("arrival_rate,num_nodes,efficiency,throughput")

    for arrival_rate in arrival_rates:
        for num_nodes in [20, 40, 60, 80, 100]:

            node = NodeManager(
                node_count=num_nodes,
                arrival_rate=arrival_rate,
                simulation_time=simulation_time,
                is_persistent=is_persistent,
            )

            efficiency, throughput = node.simulate()
            if print_csv:
                print(f"{arrival_rate},{num_nodes},{efficiency},{throughput}")
            else:
                print(
                    f"Arrival Rate: {arrival_rate} packets/s\tNumber of Nodes: {num_nodes}"
                    f"\tEfficiency: {efficiency:.5f}\tThroughput: {throughput:.5f} Mbps"
                )


def graph_verification(simulation_time: float, print_csv: bool):
    """Graph verification question for arrival rates of 5 and 12 packets/s

    Args:
        simulation_time (float): Amount of time to run the simulation for
        print_csv (bool): Flag signaling to print in csv format
    """
    print("Graph Verification: Persistent CSMA/CD")
    simulate(
        is_persistent=True,
        arrival_rates=[5, 12],
        simulation_time=simulation_time,
        print_csv=print_csv,
    )


def question_one(simulation_time: float, print_csv: bool):
    """Question 1 for the persistent case, with arrival rates of 7, 10, and 20
    packets/s

    Args:
        simulation_time (float): Amount of time to run the simulation for
        print_csv (bool): Flag signaling to print in csv format
    """
    print("Question 1: Persistent CSMA/CD")
    simulate(
        is_persistent=True,
        arrival_rates=[7, 10, 20],
        simulation_time=simulation_time,
        print_csv=print_csv,
    )


def question_two(simulation_time: float, print_csv: bool):
    """Question 1 for the non-persistent case, with arrival rates of 7, 10, and 20
    packets/s

    Args:
        simulation_time (float): Amount of time to run the simulation for
        print_csv (bool): Flag signaling to print in csv format
    """
    print("Question 2: Non-Persistent CSMA/CD")
    simulate(
        is_persistent=False,
        arrival_rates=[7, 10, 20],
        simulation_time=simulation_time,
        print_csv=print_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--questions",
        nargs="+",
        help="<Optional> Specify which question you'd like to run the simulation for. "
        "Acceptable values: [0, 1, 2]. Defaults to [1, 2].",
        required=False,
        type=int,
        default=[1, 2],
    )
    parser.add_argument(
        "-t",
        "--time-simulation",
        help="<Optional> Specify the simulation time. Defaults to 1000",
        required=False,
        type=int,
        default=200,
    )
    parser.add_argument(
        "--print-csv",
        help="Flag to print the values in csv format",
        action="store_true",
    )

    parser_args = parser.parse_args()
    if 0 in parser_args.questions:
        graph_verification(parser_args.time_simulation, parser_args.print_csv)
    if 1 in parser_args.questions:
        question_one(parser_args.time_simulation, parser_args.print_csv)
    if 2 in parser_args.questions:
        question_two(parser_args.time_simulation, parser_args.print_csv)
