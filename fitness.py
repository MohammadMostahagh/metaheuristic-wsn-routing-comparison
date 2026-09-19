import math
from network import WSN
from chromosome import generate_random_chromosome, has_cycle, repair_chromosome

PACKET_SIZE = 4000


def evaluate_tree(wsn, chromosome):
    """
    تحلیل دقیق درخت مسیریابی (کروموزوم)
    محاسبه ترافیک هر گره، مصرف انرژی دقیق و تأخیر
    """
    num_nodes = wsn.num_nodes
    packets_to_transmit = [1] * num_nodes
    packets_received = [0] * num_nodes
    hops = [0] * num_nodes

    for i in range(num_nodes):
        current = i
        count = 0
        while current != "sink":
            parent = chromosome[current]
            count += 1
            if parent != "sink":
                packets_received[parent] += 1
                packets_to_transmit[parent] += 1
            current = parent
        hops[i] = count

    avg_delay = sum(hops) / num_nodes

    total_energy_consumed = 0
    min_residual_energy = float('inf')

    for i in range(num_nodes):
        node = wsn.nodes[i]
        parent = chromosome[i]

        if parent == "sink":
            dist = wsn.distance(node, wsn.sink)
        else:
            dist = wsn.distance(node, wsn.nodes[parent])

        tx_energy = wsn.calculate_tx_energy(packets_to_transmit[i] * PACKET_SIZE, dist)
        rx_energy = wsn.calculate_rx_energy(packets_received[i] * PACKET_SIZE)

        node_energy_spent = tx_energy + rx_energy
        total_energy_consumed += node_energy_spent

        residual = node.energy - node_energy_spent
        if residual < min_residual_energy:
            min_residual_energy = residual

    return avg_delay, total_energy_consumed, min_residual_energy


def fitness(wsn, chromosome, w_delay=0.3, w_energy=0.4, w_residual=0.3):
    """
     (Golden Fitness Function)
    ترکیبی از 3 هدف اصلی با رویکرد Weighted-Sum
    """
    if has_cycle(chromosome):
        chromosome = repair_chromosome(chromosome, wsn)

    avg_delay, total_energy, min_residual = evaluate_tree(wsn, chromosome)

    norm_delay = avg_delay / 15.0
    norm_energy = total_energy / 0.1

    if min_residual <= 0:
        penalty = 10000
    else:
        penalty = 1.0 / min_residual

    norm_penalty = penalty / (1.0 / (wsn.init_energy * 0.1))

    cost = (w_delay * norm_delay) + (w_energy * norm_energy) + (w_residual * norm_penalty)

    fit_value = 1.0 / (1.0 + cost)

    return fit_value, avg_delay, total_energy, min_residual


def test_fitness():
    print("Testing Golden Fitness Function...")
    wsn = WSN(num_nodes=40, comm_radius=35)
    chrom = generate_random_chromosome(wsn)

    if has_cycle(chrom):
        chrom = repair_chromosome(chrom, wsn)

    fit, delay, energy, min_res = fitness(wsn, chrom)

    print(f"Fitness Score: {fit:.6f}")
    print(f"Average Delay (Hops): {delay:.2f}")
    print(f"Total Energy Consumed (1 Round): {energy:.6f} Joules")
    print(f"Minimum Residual Energy (Bottleneck): {min_res:.6f} Joules")


if __name__ == "__main__":
    test_fitness()