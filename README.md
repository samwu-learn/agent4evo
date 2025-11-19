# REMIND

A large language model (LLM)-driven evolutionary algorithm framework for automatically generating and optimizing heuristic algorithms for combinatorial optimization problems.

## Introduction

AgentEvo combines evolutionary computation, reinforcement learning concepts, and the generative capabilities of large language models to achieve automatic design and iterative optimization of heuristic algorithms. Through an intelligent planner that dynamically decides exploration/exploitation strategies, and utilizing an experience pool to store and retrieve successful algorithm patterns, AgentEvo can automatically discover high-quality combinatorial optimization solving algorithms.

## Key Features

- **Algorithm Design**: Automatically generate heuristic algorithms for solving combinatorial optimization problems using LLM
- **Evolutionary Optimization**: Iteratively improve algorithm performance through genetic algorithm-style evolutionary processes
- **Strategy Planning and Operator Improvement**: LLM-based planner dynamically decides exploration/exploitation strategies, while automatically triggering improvement processes when operator performance stagnation is detected
- **Experience Pool Mechanism**: Store successful cases and retrieve relevant experiences when generating new algorithms
- **Multi-problem Support**: Support for multiple classic combinatorial optimization problems including TSP, CVRP, knapsack, bin packing, etc.
- **Modular Design**: Clear hierarchical structure, easy to extend with new problems and evolutionary operators
- **Complete Traceability**: Record strategy decision history, operator improvement history, and population evolution process

## Quick Start

### Prerequisites

- Python 3.8+
- Valid OpenAI API Key (or compatible API service)

### Basic Usage

Run TSP constructive heuristic optimization:

```bash
python main.py --config cfg/config.yaml --pool_config cfg/pool.yaml
```

## Configuration

### Main Configuration File (cfg/config.yaml)

```yaml
problem: tsp_constructive  # Problem to solve
algorithm: agentevo           # Algorithm to use

n_pop: 30                 # Number of iterations
pop_size: 10              # Population size
init_pop_size: 30         # Number of candidates during initialization
timeout: 60               # Timeout for single evaluation (seconds)
diversify_init_pop: true  # Whether to diversify initial population

exp:
  output_root: Results    # Output directory
  run_name: auto          # Run name
  summary_name: summary.json
```

### LLM Client Configuration (cfg/llm_client/)

Configure LLM models for different roles:

- `generator_llm.yaml` - Generator model configuration
- `reflector_llm.yaml` - Reflector model configuration
- `planner_llm.yaml` - Planner model configuration

## Supported Problems

AgentEvo currently supports the following combinatorial optimization problems:

### Traveling Salesman Problem (TSP)
- `tsp_constructive` - Constructive heuristics
- `tsp_aco` - Ant Colony Optimization
- `tsp_gls` - Guided Local Search
- `tsp_pomo` - POMO reinforcement learning method
- `tsp_lehd` - LEHD method

### Capacitated Vehicle Routing Problem (CVRP)
- `cvrp_aco` - Ant Colony Optimization
- `cvrp_pomo` - POMO reinforcement learning method
- `cvrp_lehd` - LEHD method

### Other Problems
- `bpp_offline_aco` / `bpp_online` - Bin Packing Problem
- `mkp_aco` - Multidimensional Knapsack Problem
- `op_aco` - Orienteering Problem
- `dpp_ga` - Decoupling Capacitor Placement Problem


## Evolution Process

AgentEvo's evolution process includes the following steps:

1. **Initialize Population**: Generate initial set of candidate algorithms
2. **Evaluate Fitness**: Evaluate performance of each algorithm on test instances
3. **Strategy Planning**: Planner decides exploration/exploitation strategy
4. **Select Parents**: Select excellent individuals based on selection strategy
5. **Apply Operators**: Execute innovation or improvement operators to generate new individuals
6. **Population Management**: Maintain population size and diversity
7. **Iterative Optimization**: Repeat steps 2-6 until reaching the number of iterations
8. **Validate Optimal Solution**: Perform final validation on the best algorithm

## Evolution Operators

AgentEvo provides multiple evolution operators:

- **op1**: Experience-based innovation operator (retrieve from experience pool)
- **op2**: Crossover-based innovation operator
- **op3**: Mutation-based improvement operator
- **op4**: Reflection-based improvement operator (single individual)
- **op5**: Reflection-based improvement operator (multi-individual comparison)

## Selection Strategies

- `prob_rank` - Probabilistic rank selection
- `equal` - Equal probability selection
- `roulette_wheel` - Roulette wheel selection
- `tournament` - Tournament selection

## Advanced Features

### Operator Self-Improvement

When operator performance stagnation is detected, the system automatically triggers an improvement process:

1. Analyze the operator's historical performance
2. Identify failure patterns and improvement opportunities
3. Use LLM to generate improved operator prompts
4. Update operator configuration and continue evolution

### Experience Pool Retrieval

The experience pool uses vector similarity to retrieve relevant successful cases:

1. Convert problem description to vector embedding
2. Retrieve top-k most similar cases from experience pool
3. Inject relevant experience into LLM prompts
4. Generate new algorithms inspired by experience


## Output Results

Run results are saved in the `Results/{problem_name}/{timestamp}/` directory:

- `best/` - Best algorithm code and validation results
- `summary.json` - Evolution process summary
- `planner_stats.json` - Planner decision statistics
- `operator_history.json` - Operator improvement history
- Population information and logs for each generation


## Dependencies

Core dependencies:

```
openai          # LLM API calls
chromadb        # Vector database (optional)
numpy           # Numerical computation
scikit-learn    # Machine learning tools
joblib          # Parallel processing
pyyaml          # YAML configuration parsing
```
