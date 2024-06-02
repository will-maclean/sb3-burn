# SB3-Burn
Stable-baselines3 port written in rust using the burn library.

Main design choices remaining are:
- `burn`, `ndarray`, or something else as the non-deep learning math library?
- fork `gym-rs` or reimplement from scratch?
    - Probably going to reimplement from scratch - not super fussed about the graphics, and will be easier to just build something exactly matching what we want.

Major works required:
- gym wrapper implementation
    - includes some envs
- design of algorithm/policy/model structure (no OO in rust!)
- implementation of algorithms

<!-- ## Implemented Works

| Algorithm | Implementation |
|---|---|
| DQN | Planned |
| SAC | Planned |
| PPO | Planned | -->