from gym.envs.registration import register

register(
        id='simenv-v0',
        entry_point='myenv.env:SimEnv'
        )
