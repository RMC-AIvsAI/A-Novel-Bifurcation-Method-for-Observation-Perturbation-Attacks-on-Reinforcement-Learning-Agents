from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor


import argparse

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper, DiscreteActionWrapper
from citylearn.data import DataSet


def make_discrete_env(schema, action_bins: int = 10, bldg: list = ['Building_1'], single_agent: bool = True, seed:int = None):
    """Because ART's attacks are designed for supervised learning they one work with ANNs with a single label or head, using multiple buildings adds an action/head for each"""
    env = CityLearnEnv(schema, 
        central_agent=single_agent, 
        buildings=bldg, 
        random_seed=seed)
    #Because ART attacks are made for classification tasks we need a discrete action space 
    env = DiscreteActionWrapper(env, bin_sizes=[{'electrical_storage':action_bins}])
    #Calendar observations are periodically normalized, everything else is min/max normalized 
    env = NormalizedObservationWrapper(env)
    #provides an interface for SB3
    env = StableBaselines3Wrapper(env)
    return env

parser = argparse.ArgumentParser()
parser.add_argument('--bins', default=20, type=int)
parser.add_argument('-eps', '--episodes', default=300, type=int)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--nodes', default=64, type=int)
parser.add_argument('-stop', '--max_no_improvement_evals', default=5, type=int, help='max number of episodes without improvement')

args = parser.parse_args()

dataset_name = 'citylearn_challenge_2022_phase_2'
schema = DataSet.get_schema(dataset_name)
building = list(schema['buildings'].keys())[0] #the first building from the schema's building keys
bins = args.bins

env = make_discrete_env(schema=schema, 
                        bldg=[building], 
                        action_bins=bins)

eval_env = env = make_discrete_env(schema=schema, 
                        bldg=[building], 
                        action_bins=bins,
                        seed=42)

policy_kwargs = dict(net_arch=args.layers*[args.nodes])
agent = PPO('MlpPolicy', 
            env,
            device='cuda',
            policy_kwargs=policy_kwargs,
            tensorboard_log='logs/Phase1/PPO/',
            )

episodes = args.episodes
T = env.time_steps - 1
agent_name = f'default_PPO_{dataset_name}_{building}_{args.layers}_{args.nodes}_arch'

#stop training after 3 consecutive evals with no improvement, starting with the 5th eval
early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=args.max_no_improvement_evals, min_evals=5, verbose=1)
#eval agent every 10 episodes
eval_callback = EvalCallback(Monitor(eval_env), eval_freq=10*T, callback_after_eval=early_stopping, verbose=1)

agent.learn(total_timesteps=int(T*episodes), 
            tb_log_name=agent_name,
            callback=eval_callback,
            progress_bar=True)

#update name for number of training episodes, in case training stops early
agent_name = f'default_PPO_{dataset_name}_{building}_{args.layers}_{args.nodes}_arch_{int(eval_callback.num_timesteps/T)}'

agent.save(f"Models/Victim/{agent_name}")