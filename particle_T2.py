import numpy as np
import random
from network import WSN
from chromosome import has_cycle, repair_chromosome


class Particle:
    """
    کلاس معرف یک ذره در الگوریتم PSO.
    هر ذره در فضای پیوسته حرکت می‌کند اما در نهایت به یک درخت مسیریابی گسسته نگاشت می‌شود.
    """

    def __init__(self, num_nodes):
        self.position = np.random.rand(num_nodes)

        self.velocity = np.random.uniform(-0.1, 0.1, num_nodes)

        self.pBest_position = self.position.copy()
        self.pBest_fitness = -1.0

        self.current_chromosome = []
        self.fitness = -1.0
        self.delay = 0.0
        self.energy = 0.0
        self.min_res = 0.0


def decode_position(wsn, position_array):
    """
    نگاشت هوشمند (Continuous-to-Discrete Mapping):
    تبدیل موقعیت پیوسته (اعداد اعشاری) به یک درخت مسیریابی گسسته (کروموزوم)
    """
    chromosome = [-1] * wsn.num_nodes

    for node in wsn.nodes:
        valid_next_hops = []

        for neighbor in node.neighbors:
            if neighbor == wsn.sink:
                valid_next_hops.append("sink")
            elif wsn.distance(neighbor, wsn.sink) < wsn.distance(node, wsn.sink):
                valid_next_hops.append(neighbor.id)

        if not valid_next_hops:
            valid_next_hops.append("sink")

        num_valid_hops = len(valid_next_hops)
        mapped_idx = int(position_array[node.id] * num_valid_hops) % num_valid_hops

        chromosome[node.id] = valid_next_hops[mapped_idx]

    return chromosome


def test_particle_mapping():
    """تابعی برای تست صحت عملکرد ماژول نگاشت و تشکیل ذره"""
    print("Testing Particle Creation and Continuous-to-Discrete Mapping...")

    wsn = WSN(num_nodes=40, comm_radius=35)

    particle = Particle(wsn.num_nodes)
    print("\n[1] Particle Continuous Position (First 5 elements):")
    print(np.round(particle.position[:5], 4))

    raw_chromosome = decode_position(wsn, particle.position)
    print("\n[2] Decoded Raw Chromosome:")
    print(raw_chromosome)

    cycle_exists = has_cycle(raw_chromosome)
    print(f"\n[3] Has Cycle before repair? {cycle_exists}")

    if cycle_exists:
        repaired_chromosome = repair_chromosome(raw_chromosome, wsn)
        print("[4] Repaired Chromosome:")
        print(repaired_chromosome)
        print(f"Has Cycle after repair? {has_cycle(repaired_chromosome)}")
    else:
        print("[4] Chromosome is already valid (No repair needed).")


if __name__ == "__main__":
    test_particle_mapping()