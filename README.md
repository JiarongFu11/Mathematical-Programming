# Mathematical Programming

Coursework and my own implementations for operations research / optimization. Exact methods, heuristics and nonlinear methods, mostly written from scratch in NumPy.

## Layout

```
mathematical_programming_solver/
├── math_solver/
│   ├── exact/
│   │   ├── boundbranch.py      # branch and bound for integer programs, LP relaxations solved with PuLP,
│   │   │                       #   draws the search tree with graphviz
│   │   └── dynamic_prog.py     # resource allocation DP (sum or product objective, max or min)
│   ├── heuristic/
│   │   ├── ga/                 # genetic algorithm
│   │   │   ├── ga_base.py      #   main loop, subclass it for a specific problem
│   │   │   ├── generate_population.py  # permutation or real-valued chromosomes
│   │   │   ├── crossover.py    #   single point, PMX, OX, position-based, order-based
│   │   │   ├── mutate.py       #   inversion, insertion, swap, 2-opt, 3-opt, interval
│   │   │   └── ga_selection.py #   roulette wheel, tournament, random, with elitism
│   │   └── sa/
│   │       └── sa_base.py      # simulated annealing, reuses the GA mutation operators
│   ├── nonlinear/
│   │   ├── line_search.py      # dichotomous, golden section, Fibonacci
│   │   ├── gradient_descend.py # gradient descent with golden section step size
│   │   └── newton.py           # Newton's method, plain and with line search
│   └── utils/
└── tests/                      # pytest
```

## Setup

```bash
pip install -r requirements.txt
```

The packages that actually matter are `numpy`, `PuLP`, `graphviz` and `pytest`. `newton.py` also imports `matplotlib`, which is not in `requirements.txt`.

`graphviz` also needs the system binary for drawing the tree (`brew install graphviz` on Mac).

## Usage

**Branch and bound**: subclass `BranchBound` and fill in the integer variables, continuous variables, objective and constraints. It runs as soon as it's constructed, prints each node, and saves the tree to `my_branch_bound_tree.png`. See `TestIP_1` to `TestIP_3` at the bottom of `boundbranch.py`.

```bash
python math_solver/exact/boundbranch.py
```

**DP**:

```python
from math_solver.exact.dynamic_prog import ResourceAllocationDP

solver = ResourceAllocationDP(stages_num=3, total_resources=2, mode="Minimize", operator="multiply")
best, allocation = solver.solve(reward_matrix)   # reward_matrix shape: (total_resources + 1, stages_num)
```

**GA / SA**: subclass `GeneticAlgo` or `SimulatedAnnealing` and override the problem-specific parts (population, objective, constraints, termination). `Test1` in `ga_base.py` is an example.

**Line search, gradient descent, Newton**: these import each other as plain modules (`from line_search import ...`), so run them from inside `math_solver/nonlinear/`.

## Tests

```bash
pytest tests/
```

There are tests for DP, crossover, mutation, selection and line search.
