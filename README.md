# Metaheuristic Algorithms for Energy-Efficient WSN Routing

This repository presents a comparative study of three metaheuristic optimization algorithms for energy-efficient routing in heterogeneous Wireless Sensor Networks (WSNs):

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Simulated Annealing (SA)

The project extends an earlier GA-based implementation by introducing PSO and SA and comparing their behavior under a common network simulation framework.

## Objectives

The main goal is to investigate and compare metaheuristic optimization methods for routing with respect to:

- Energy consumption
- Routing delay
- Residual node energy
- Convergence behavior
- Robustness across multiple runs
- Parameter sensitivity

## Algorithms

### Genetic Algorithm

The GA implementation uses evolutionary operators such as selection, crossover, mutation, elitism, and chromosome repair to search for energy-efficient routing solutions.

### Particle Swarm Optimization

The PSO implementation models candidate routing solutions as particles and iteratively improves them using swarm-based optimization.

### Simulated Annealing

The SA implementation explores the routing search space using temperature-based probabilistic transitions to avoid poor local optima.

## Experimental Analysis

The project includes:

- Multi-run experiments
- Convergence analysis
- Robustness comparison
- Parameter tuning
- Sensitivity analysis
- Statistical summaries
- Comparative visualizations

## Key Visualizations

- Convergence comparison of GA, PSO, and SA
- Robustness boxplots
- PSO parameter sensitivity
- PSO tuning heatmap
- SA tuning analysis

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib

## Academic Context

This project was developed as part of university coursework and is published here as part of my academic and programming portfolio.

It extends my previous GA-based WSN routing project by comparing multiple metaheuristic optimization approaches.
