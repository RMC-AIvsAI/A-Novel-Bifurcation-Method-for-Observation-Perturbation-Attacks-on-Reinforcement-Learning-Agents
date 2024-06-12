from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper, DiscreteActionWrapper
from citylearn.data import DataSet
from citylearn.reward_function import SolarPenaltyReward #IndependentSACReward

#import KBMproject.utilities as utils
from KBMproject import ATLA
from KBMproject import utilities as utils
import torch
from datetime import datetime


DATASET_NAME = 'citylearn_challenge_2022_phase_2'
EPISODES = 500
EVAL_FREQ = 5
SEED = 188

SAVED_MODEL = None #r"Models\optuna\study 107 trial 22.zip"
SAVE_ID = 'study 107 trial 22'

#seeds absent from early testing
torch.manual_seed(SEED) #CPU 
torch.cuda.manual_seed_all(SEED)
#ref: https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False #uses same algo every time, rather than trying multiple and choosing fastest

now = datetime.now()
dtg = f'{now.month}-{now.day}-{now.hour}'

def wrap_env(env, action_bins=20):
    #Because ART attacks are made for classification tasks we need a discrete action space 
    env = DiscreteActionWrapper(env, bin_sizes=[{'electrical_storage':action_bins}])
    #Calendar observations are periodically normalized, everything else is min/max normalized 
    env = NormalizedObservationWrapper(env)
    #provides an interface for SB3
    env = StableBaselines3Wrapper(env)
    env = Monitor(env)
    return env

schema = DataSet.get_schema(DATASET_NAME)
bldg = list(schema['buildings'].keys())[0], 
kwargs = dict(schema=schema, 
              central_agent=True, 
              buildings=bldg,
              reward_function=SolarPenaltyReward, 
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

policy_kwargs = dict(
    net_arch=[256,256,256], #[416, 320], #[256,256] for early trials
    )

if SAVED_MODEL is None:
    agent = PPO('MlpPolicy', 
                env,
                device='cuda',
                policy_kwargs=policy_kwargs,
                tensorboard_log='logs/Phase1/PPO/',
                verbose=0,
                )
else:
    agent = PPO.load(SAVED_MODEL,
                    env=env,
                    print_system_info=True,
                    tensorboard_log='logs/Phase1/SAC/',
                    force_reset=False, #default is true for continued training ref: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO.load
                    )
    
num_ts = agent.num_timesteps #0 for new agent

T = env.time_steps - 1

agent_name = f'{dtg}_PPOd_{DATASET_NAME}_{bldg}_{kwargs["reward_function"].__name__}'

if SAVED_MODEL is not None:
    agent_name += f'_{SAVE_ID}_resumed'

print(f'Training: {agent_name}\n Episodes: {EPISODES}')

early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=3)
eval_callback = EvalCallback(Monitor(eval_env), 
                             eval_freq=EVAL_FREQ*T,
                             best_model_save_path='Models/Victim/',
                             #callback_after_eval=early_stopping, 
                             verbose=3)
modified_h_params = ATLA.HParamCallback(
    ['use_sde','sde_sample_freq','n_steps','ent_coef','learning_rate','gamma','max_grad_norm','gae_lambda','batch_size','vf_coef',]
    ) #the default clip range is a function, which will cause an error if recorded, it's recoded under train in tb


agent.learn(total_timesteps=int(T*EPISODES) + num_ts, 
            tb_log_name=agent_name,
            callback=[eval_callback,modified_h_params],
            progress_bar=True, #does this affect training time?
            )

agent_name += f'_{int(eval_callback.num_timesteps/T)}'

try:
    agent.save(f"Models/Victim/{agent_name}")
except: #in case the dir is wrong somehow
    agent.save(f'{agent_name}')

#print(utils.format_kpis(eval_env)) #raises errer