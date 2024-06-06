from gym_sock_mgr.wrappers.common import *
from gym_sock_mgr.env.base import BaseEnv


def make_full_env(env: BaseEnv, scale=True, frame_stack=True, clip_rewards=False):
    if scale:
        env = ScaledFloatFrame(env)
        env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)

    return env


def make_simple_env(env: BaseEnv):
    # Wrap with any fundamental / required wrappers
    return env