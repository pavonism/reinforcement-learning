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

## 🌳 Project structure

```bash
│   .gitignore # Files to ignore
│   README.md # This file
│   pyproject.toml # Project configuration file
├───scripts # Notebooks
├───dqn # DQN implementation
├───ppo # PPO implementation
├───muzero # MuZero implementation
└───src # Dommy folder for pyproject.toml
```
