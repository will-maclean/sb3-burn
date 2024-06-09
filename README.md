# SB3-Burn

[![Continuous Integration](https://github.com/will-maclean/sb3-burn/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/will-maclean/sb3-burn/actions/workflows/rust.yml)

Stable-baselines3 port written in rust using the burn library.

WIP, contributors welcome :)

## Project Plan
The goals of the project are:
1. MVP: Establish a minimum viable product (MVP), showing a basic DQN algorithm training on a gridworld or cartpole environemnt.
2. Useability: Make the project a useable library with documentation, crate publishing, and user and contributor guides.
3. Features: Add features to increase the functionality of the project

## Implemented Works

Algorithms:

| Algorithm | Implementation |
| --------- | -------------- |
| DQN       | In Progress    |
| SAC       | Planned        |
| PPO       | Planned        |

Environments:
| Env | Implementation |
| --------- | -------------- |
| Gridworld       | Rust, done    |
| Cartpole       | Rust, done    |
| Gym classic control       | Python, In Progress        |
| Snake       | Python, In Progress        |

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