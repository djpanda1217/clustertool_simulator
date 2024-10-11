![image](https://github.com/user-attachments/assets/a41daa95-e106-4f3d-ac2d-91adf00876ad)


# Semiconductor Cluster Tool Scheduling with Reinforcement Learning

This repository implements a reinforcement learning (RL) approach to optimize semiconductor cluster tool scheduling using a Deep Q-Network (DQN) model. The system is designed to improve productivity in semiconductor manufacturing by efficiently scheduling equipment processing.

## Project Overview

This project aims to demonstrate the application of reinforcement learning and imitation learning to cluster tool scheduling tasks in semiconductor manufacturing. The solution leverages a modular simulator, built on OpenAI Gym, to train and evaluate scheduling policies using various approaches, including rule-based methods and DQN-based reinforcement learning.

## Features

- **Modular Environment:** The simulator is built with the OpenAI Gym framework for flexible and easy-to-use interactions.
- **Reinforcement Learning:** Implements a DQN-based approach for scheduling optimization.
- **Imitation Learning Pre-training:** Supports pre-training with rule-based schedulers using imitation learning techniques.
- **Customizable Scheduling Rules:** Users can define and test different rule-based scheduling strategies.

## Project Structure

- `DQN_2.py`: Contains the implementation of the Deep Q-Network used for RL-based scheduling.
- `EQP_Scheduler.py`: Defines the main scheduler class, including methods for rule-based scheduling.
- `EQP_Scheduler_env.py`: Implements the environment for the scheduling simulation, built with OpenAI Gym.

## Getting Started

### Prerequisites

- Python 3.11
- `gym` (OpenAI Gym)
- `torch` (PyTorch)

You can install the required dependencies using the following command:

```bash
pip install gym torch
```

### Running the Simulation

1. Clone the repository:
   ```bash
   git clone https://github.com/hamcheesechoi/clustertool_simulator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd clustertool_simulator
   ```
3. Run the simulation:
   ```bash
   python DQN_2.py
   ```

## Customization

You can modify the scheduling rules in `EQP_Scheduler.py` or configure the environment settings in `EQP_Scheduler_env.py` to experiment with different scenarios and policies.

## Roadmap

- Add more RL algorithms (e.g., PPO, A3C).
- Integrate additional scheduling heuristics for pre-training.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any changes or suggestions.

## License

This project is licensed under the MIT License.
