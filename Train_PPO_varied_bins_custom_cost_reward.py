from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from typing import Any, List, Mapping, Union
import numpy as np

from citylearn.reward_function import RewardFunction
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper, DiscreteActionWrapper
from citylearn.data import DataSet


def make_discrete_env(schema, action_bins: int = 10, bldg: list = ['Building_1'], CityLearn_kwargs=None):
    """Because ART's attacks are designed for supervised learning they one work with ANNs with a single label or head, using multiple buildings adds an action/head for each"""
    kwargs = CityLearn_kwargs
    env = CityLearnEnv(schema, 
                       central_agent=True,
                       buildings=bldg, 
                       **kwargs)
    #Because ART attacks are made for classification tasks we need a discrete action space 
    env = DiscreteActionWrapper(env, bin_sizes=[{'electrical_storage':action_bins}])
    #Calendar observations are periodically normalized, everything else is min/max normalized 
    env = NormalizedObservationWrapper(env)
    #provides an interface for SB3
    env = StableBaselines3Wrapper(env)
    return env

class CustomReward(RewardFunction):
    """Calculates custom user-defined multi-agent reward.

    Reward is the :py:attr:`net_electricity_consumption_emission`
    for entire district if central agent setup otherwise it is the
    :py:attr:`net_electricity_consumption_emission` each building.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Calculates reward based on the electricity cost

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for o in observations:
            power_cost = -1*o['net_electricity_consumption']*o['electricity_pricing'] #price for power consummed
            reward_list.append(power_cost)

        reward = [sum(reward_list)]

        return reward

dataset_name = 'citylearn_challenge_2022_phase_2'
schema = DataSet.get_schema(dataset_name)
building = list(schema['buildings'].keys())[0] #the first building from the schema's building keys
bins = 20

env_kwargs = {
    'reward_function':CustomReward
}

env = make_discrete_env(schema=schema, 
                        bldg=[building], 
                        action_bins=bins,
                        CityLearn_kwargs=env_kwargs)

eval_kwargs = {
    'reward_function':CustomReward,
    'random_seed':42
}

eval_env = env = make_discrete_env(schema=schema, 
                        bldg=[building], 
                        action_bins=bins,
                        CityLearn_kwargs=env_kwargs)

policy_kwargs = dict(net_arch=[256, 256])
agent = PPO('MlpPolicy', 
            env,
            device='cuda',
            policy_kwargs=policy_kwargs,
            tensorboard_log='logs/Phase1/PPO/',
            )

episodes = 300
T = env.time_steps - 1
agent_name = f'default_PPO_{dataset_name}_{building}_{bins}_bins_power_cost_rwd'

#stop training after consecutive evals with no improvement, after the min eval
early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
#eval agent every 10 episodes
eval_callback = EvalCallback(Monitor(eval_env), eval_freq=10*T, callback_after_eval=early_stopping, verbose=1)

agent.learn(total_timesteps=int(T*episodes), 
            tb_log_name=agent_name,
            callback=eval_callback,
            progress_bar=True)

#update name for number of training episodes, in case training stops early
agent_name = f'default_PPO_{dataset_name}_{building}_{bins}_bins_power_cost_rwd_{int(eval_callback.num_timesteps/T)}'

agent.save(f"Models/Victim/{agent_name}")