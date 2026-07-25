try:
    from gym.envs.registration import register
except ModuleNotFoundError:
    register = None

if register is not None:
    register(
        id='Franka-FMB-v0',
        entry_point='envs.franka_fmb_env:FrankaFMB',
    )
