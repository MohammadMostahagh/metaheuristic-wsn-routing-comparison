import numpy as np
import matplotlib.pyplot as plt
import random
import math


class SensorNode:
    def __init__(self, node_id, x, y, energy, is_advanced=False):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = energy
        self.is_advanced = is_advanced
        self.neighbors = []


class WSN:
    def __init__(self, num_nodes=40, field_size=100, init_energy=2.0, comm_radius=35):
        self.num_nodes = num_nodes
        self.field_size = field_size
        self.init_energy = init_energy
        self.comm_radius = comm_radius

        self.E_elec = 50e-9
        self.E_fs = 10e-12
        self.E_mp = 0.0013e-12
        self.d0 = math.sqrt(self.E_fs / self.E_mp)

        self.nodes = []
        self.sink = None

        self._deploy_nodes()
        self._find_neighbors()

    def _deploy_nodes(self):
        sink_x = self.field_size / 2
        sink_y = self.field_size / 2
        self.sink = SensorNode("sink", sink_x, sink_y, float('inf'))

        advanced_ratio = 0.1
        energy_multiplier = 1.5

        for i in range(self.num_nodes):
            x = random.uniform(0, self.field_size)
            y = random.uniform(0, self.field_size)

            is_advanced = random.random() < advanced_ratio
            node_energy = self.init_energy * energy_multiplier if is_advanced else self.init_energy

            node = SensorNode(i, x, y, node_energy, is_advanced)
            self.nodes.append(node)

    def distance(self, node_a, node_b):
        return math.sqrt((node_a.x - node_b.x) ** 2 + (node_a.y - node_b.y) ** 2)

    def _find_neighbors(self):
        """یافتن همسایگان هر گره بر اساس شعاع ارتباطی برای تشکیل گراف"""
        for i, node in enumerate(self.nodes):
            for j, other_node in enumerate(self.nodes):
                if i != j and self.distance(node, other_node) <= self.comm_radius:
                    node.neighbors.append(other_node)

            if self.distance(node, self.sink) <= self.comm_radius:
                node.neighbors.append(self.sink)

    def calculate_tx_energy(self, bits, distance):
        """محاسبه انرژی مصرفی برای ارسال داده (انتقال)"""
        if distance < self.d0:
            return (self.E_elec * bits) + (self.E_fs * bits * (distance ** 2))
        else:
            return (self.E_elec * bits) + (self.E_mp * bits * (distance ** 4))

    def calculate_rx_energy(self, bits):
        """محاسبه انرژی مصرفی برای دریافت داده"""
        return self.E_elec * bits

    def plot_network(self):
        """رسم شبکه به همراه خطوط ارتباطی و تمایز گره‌های پیشرفته"""
        plt.figure(figsize=(8, 8))

        for node in self.nodes:
            for neighbor in node.neighbors:
                plt.plot([node.x, neighbor.x], [node.y, neighbor.y],
                         color='gray', linestyle='--', linewidth=0.3, alpha=0.5)

        for node in self.nodes:
            color = 'green' if node.is_advanced else 'blue'
            marker_size = 50 if node.is_advanced else 20
            plt.scatter(node.x, node.y, c=color, s=marker_size, zorder=5)
            plt.text(node.x + 1, node.y + 1, str(node.id), fontsize=8, zorder=6)

        plt.scatter(self.sink.x, self.sink.y, c='red', marker='s', s=100, zorder=10)
        plt.text(self.sink.x + 1, self.sink.y + 1, "Sink", fontsize=10, fontweight='bold', zorder=10)

        plt.xlim(0, self.field_size)
        plt.ylim(0, self.field_size)
        plt.title("Heterogeneous WSN Deployment with Communication Graph")
        plt.grid(True)
        plt.savefig("wsn_deployment_advanced.png", dpi=300)
        plt.close()
        print("Network graph saved as wsn_deployment_advanced.png")


def test_wsn():
    print("Testing Advanced WSN deployment...")
    wsn = WSN(num_nodes=40, comm_radius=35)
    print(f"Number of sensor nodes: {len(wsn.nodes)}")

    advanced_count = sum(1 for n in wsn.nodes if n.is_advanced)
    print(f"Number of advanced nodes: {advanced_count}")

    tx_energy = wsn.calculate_tx_energy(bits=2000, distance=20)
    print(f"Energy to transmit 2000 bits over 20m: {tx_energy:.6e} Joules")

    wsn.plot_network()


if __name__ == "__main__":
    test_wsn()