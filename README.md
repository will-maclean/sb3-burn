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
| Gym classic control       | Python, In Progress        |
| Snake       | Python, In Progress        |

## Usage
The `examples` directory shows how algorithms and environemnts can be used.

## Troubleshooting
1. Run `export RUST_BACKTRACE=1` in your terminal to tell rust to output a backtrace on error - very useful for tracing issues.