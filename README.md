# SB3-Burn

[![Continuous Integration](https://github.com/will-maclean/sb3-burn/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/will-maclean/sb3-burn/actions/workflows/rust.yml) [![codecov](https://codecov.io/github/will-maclean/sb3-burn/branch/main/graph/badge.svg?token=1QYQ8E2LXZ)](https://codecov.io/github/will-maclean/sb3-burn)

`sb3-burn` is a reinforcement learning (RL) library written in rust using the [burn](https://github.com/tracel-ai/burn) deep learning library. It is based on the Python/PyTorch library [Stable-baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master) (hence the name) and aims
to bring a fast, flexible RL library to the rust machine learning ecosystem. Features:

**Implemented RL Algorithms**

`sb3-burn` aims to provide understandable, extendable implementations of the common RL algorithms. Although currently a work in progress, the aim is to implement all algorithms available in stable-baselines3.

**Gym-like environments with Rust implementations**

The [gym](https://gymnasium.farama.org/) package has been hugely influential in the Python RL space, providing a common interface for 
RL environments. `sb3-burn` provides a gym-like environment interface, and a set of commonly-used environments have been implemented for extra speed.

**Flexibility**

Different RL environments commonly require tweaking of RL
algorithms, either because of unusual state or action types, or 
customisation of hyper parameters. `sb3-burn` has a strong focus
on utilising rust generics to allow for users to train agents on 
custom environemts with unusual state/action types, without needing to reimplement entire algorithms.


## Project Plan
The project currently contains a working DQN algorithm, as well as a set of implemented environments. The planned works for the
immediate future are:

1. Soft Actor Critic
2. Testing / code coverage
3. Examples / rustdoc / sb3_book
4. Checkpointing / saving / loading / resuming training
5. crates.io
6. Benchmarking performance, including visualisation creation
7. Implementing more common gym environments

## Implemented Works

Algorithms:

| Algorithm | Implementation |
| --------- | -------------- |
| DQN       | Implemented    |
| SAC       | In Progress (on main)       |
| PPO       | Planned        |

Environments:
| Env | Implementation |
| --------- | -------------- |
| Gridworld       | Rust, done    |
| Cartpole       | Rust, done    |
| Pendulum       | Rust, done    |
| MountainCar       | Rust, done    |
| Python gym handler | In progress |
| Multiple probe environments | Rust, done |

## Usage
The `examples` directory shows how algorithms and environemnts can be used.

### GPU Training & Backends
Traditionally, in PyTorch with Python, only Nvidia GPUs are supported with the cuda backend. Burn, the deep learning
library which powers sb3-rust, is more flexible with backends. This is great but does mean that we need to handle 
devices a bit differently. 

If doing CPU only training or inference, the `Ndarray` backend should be fine. However, for GPU training and 
inference, a backend that support GPU is required. The best supported option is `LibTorch`. This requires
libtorch to be installed correctly, which can be a bit of a hassle. Follow [this](https://github.com/tracel-ai/burn/blob/main/crates/burn-tch/README.md) burn guide for installation
instructions, or invsetigate the other burn backends for more specif scenarios. 

## Troubleshooting
1. Run `export RUST_BACKTRACE=1` in your terminal to tell rust to output a backtrace on error - very useful for tracing issues.
