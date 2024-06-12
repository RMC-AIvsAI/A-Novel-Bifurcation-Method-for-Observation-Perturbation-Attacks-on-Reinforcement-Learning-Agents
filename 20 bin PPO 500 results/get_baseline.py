AGENT_NAME = 'default_PPO_citylearn_challenge_2022_phase_2_Building_6_20_bins_500'
DATASET_NAME = 'citylearn_challenge_2022_phase_2' #only action is electrical storage

from stable_baselines3 import PPO

from citylearn.data import DataSet

import numpy as np

import KBMproject.utilities as utils

schema = DataSet.get_schema(DATASET_NAME)

agent = PPO.load(path=f"{AGENT_NAME}")
print('Model loaded from storage')
bins = agent.action_space[0].n
env = utils.make_discrete_env(schema=schema,  
                        action_bins=bins,
                        seed=42)

baseline_kpis, baseline_obs, baseline_a = utils.eval_agent(env,agent)

np.savetxt('rebaseline obs.csv', baseline_obs, delimiter=',')
np.savetxt('rebaseline a.csv', baseline_a, delimiter=',')
