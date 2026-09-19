import random
from network import WSN


def generate_random_chromosome(wsn):
    """
    تولید یک کروموزوم معتبر (درخت مسیریابی).
     هر گره فقط از بین 'همسایگان مجاز' در شعاع ارتباطی خود که به سینک نزدیک‌تر هستند، پدر انتخاب می‌کند.
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

        chromosome[node.id] = random.choice(valid_next_hops)

    return chromosome


def has_cycle(chromosome):
    """تشخیص حلقه در درخت مسیریابی"""
    for start in range(len(chromosome)):
        visited = set()
        current = start

        while current != "sink":
            if current in visited:
                return True
            visited.add(current)
            current = chromosome[current]

    return False


def repair_chromosome(chromosome, wsn):
    """
     تابعی برای اصلاح کروموزوم‌های خراب (دارای حلقه).
    این تابع در عملگرهای Crossover و Mutation استفاده خواهد شد تا جواب‌های نامعتبر را به معتبر تبدیل کند.
    """
    for start in range(len(chromosome)):
        visited = set()
        current = start

        while current != "sink":
            if current in visited:

                chromosome[current] = "sink"
                break
            visited.add(current)
            current = chromosome[current]

    return chromosome


def test_chromosome():
    print("Testing Chromosome Generation and Repair...")
    wsn = WSN(num_nodes=40, comm_radius=35)
    chrom = generate_random_chromosome(wsn)

    print("\nGenerated chromosome (Routing Tree):")
    print(chrom)

    cycle_exists = has_cycle(chrom)
    print(f"\nCycle exists initially? {cycle_exists}")

    if not cycle_exists:
        print("Injecting an artificial cycle for testing (Node 0 -> 1 -> 2 -> 0)...")
        chrom[0] = 1
        chrom[1] = 2
        chrom[2] = 0
        print(f"Cycle exists after injection? {has_cycle(chrom)}")

        chrom = repair_chromosome(chrom, wsn)
        print(f"Cycle exists after repair? {has_cycle(chrom)}")
        print("Repaired chromosome:")
        print(chrom)


if __name__ == "__main__":
    test_chromosome()