# Reinforcement Learning course project

# Setup

```console
$ python -m venv venv
$ venv/bin/activate (Linux) ./venv/Scripts/activate (Win)
$ pip install -e .
```

Additionally, you have to install Torch for cuda by yourself:

```console
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Run Training

To run the training, execute the desired algorithm using the command below from the root project directory:

```console
python <algorithm_name>
```

Replace <algorithm_name> with the specific algorithm you want to run, such as ppo, muzero, or dqn.

## 🌳 Project structure

```bash
│   .gitignore # Files to ignore
│   README.md # This file
│   pyproject.toml # Project configuration file
├───checkpoints # Checkpoints for each run
├───scripts # Notebooks
├───dqn # DQN implementation
├───ppo # PPO implementation
├───muzero # MuZero implementation
└───src # Dummy folder for pyproject.toml
```
