from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper, NormalizedObservationWrapper
from citylearn.data import DataSet
from citylearn.reward_function import SolarPenaltyReward

DATASET_NAME = 'citylearn_challenge_2022_phase_2'
EPISODES = 500

def wrap_env(env):
    env = NormalizedObservationWrapper(env) #normalize space -> actions [0,1], not [-1,1]
    env = StableBaselines3Wrapper(env)
    env = Monitor(env)
    return env

schema = DataSet.get_schema(DATASET_NAME)
bldg = list(schema['buildings'].keys())[0], 
kwargs = dict(schema=schema, 
              central_agent=True, 
              buildings=bldg
)


env = CityLearnEnv(
        random_seed=0,
        **kwargs
        )
env = wrap_env(env)

eval_env = CityLearnEnv(
        random_seed=42,
        **kwargs
        )
eval_env = wrap_env(eval_env)

policy_kwargs = dict(net_arch=[256, 256])
agent = SAC('MlpPolicy', 
            env,
            device='cuda',
            policy_kwargs=policy_kwargs,
            tensorboard_log='logs/Phase1/SAC/',
            #buffer_size=int(1e5),
            #batch_size=256,
            #learning_rate=3e-4,
            #gamma=0.99,
            verbose=3,
            #action_noise=, #this could help robustness, if I have time https://stable-baselines3.readthedocs.io/en/master/common/noise.html#stable_baselines3.common.noise.ActionNoise
            )

T = env.time_steps - 1
agent_name = f'SAC_{DATASET_NAME}_{bldg}_default_norm_obs'

print(f'Training: {agent_name}\n Episodes: {EPISODES}')
early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
eval_callback = EvalCallback(Monitor(eval_env), eval_freq=10*T, callback_after_eval=early_stopping, verbose=1)
agent.learn(total_timesteps=int(T*EPISODES), 
            tb_log_name=agent_name,
            callback=eval_callback,
            progress_bar=True)

agent_name += f'_{int(eval_callback.num_timesteps/T)}'

try:
    agent.save(f"Models/Victim/{agent_name}")
except: #in case the dir is wrong somehow
    agent.save(f'{agent_name}')

#print(utils.format_kpis(eval_env)) #raises errer