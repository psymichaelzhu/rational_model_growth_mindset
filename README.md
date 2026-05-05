# rational_model_growth_mindset

The project formalizes growth mindset theory as a sequential decision-making problem, where agents decide whether to cultivate or harvest skills over time.

## Structure

```text
code/
├── simulation/
│   ├── core/
│   │   ├── agents.py          # Agent definitions and decision policies
│   │   ├── environment.py     # Sequential decision-making environment
│   │   ├── runner.py          # Simulation runner and result logging
│   │   └── utils.py           # Helper functions
│   └── experiments/
│       ├── belief_grid.py     # Grid search over malleability priors
│       └── environment_grid.py
└── visualization/
    ├── p1.R
    ├── p2.R
    └── p3.R
```

# Reproduce

## Simulation (python)

dependencies:
```bash
pip install numpy pandas tqdm
```

From the repository root, run an experiment script, for example:

```bash
python code/simulation/experiments/belief_grid.py
```

The script runs simulations across combinations of prior and environment parameters. Results are saved as CSV and JSON files under `results/simulation/<experiment_id>/`.

## Visualization (R)

First, open the R project file at the root of the repository.
Then, run the R scripts in the `code/visualization/` directory.


