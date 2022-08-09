
__version__ = "0.1.1"

from gym.envs.registration import register

register(
    id='cap-v0',
    entry_point='gym_cap.envs:CapEnv',
)
