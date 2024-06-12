from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.wrappers import NormalizedSpaceWrapper
#from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.data import DataSet
from citylearn.reward_function import SolarPenaltyReward #IndependentSACReward

#import KBMproject.utilities as utils
from KBMproject import ATLA
import torch
from datetime import datetime


DATASET_NAME = 'citylearn_challenge_2022_phase_2'
EPISODES = 500
N_STEPS = 2048 #for early trials
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

def wrap_env(env):
    env = NormalizedSpaceWrapper(env) #normalize space -> actions [0,1], not [-1,1]
    #try norm obs next time?
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
    full_std=True, #Whether to use (n_features x n_actions) parameters for the std instead of only (n_features,) when using gSDE, default True
        #performance was worse for 2-15-15 without full std, compared to 2-15-14 with it.
    #squash_output=True, #Whether to squash the output using a tanh function, this allows to ensure boundaries when using gSDE. Try True?
        #with squashed outputs and expn, my std tensor was all NaNs during training with this error raised:
        #ValueError: Expected parameter scale (Tensor of shape (256, 1)) of distribution Normal(loc: torch.Size([256, 1]), scale:
        #torch.Size([256, 1])) to satisfy the constraint GreaterThan(lower_bound=0.0), but found invalid values:
        #inside self.weights_dist = Normal(th.zeros_like(std), std)
        #squashing might be the issue, since I'm already using tanh?
    use_expln=True #Use expln() function instead of exp() to ensure a positive standard deviation (cf paper). It allows to keep variance above zero and prevent it from growing too fast. In practice, exp() is usually enough.
    )

#PPO HParam ranges: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

early_hparams = dict( #attempted with default reward
    ent_coef=0.01, #increased from 0 to improve exploration
                    #though ent_loss is almost always 0 during training, so this may not help at all
                use_sde=True, #cite if this helps: https://proceedings.mlr.press/v164/raffin22a.html, set here not in policy kwargs
                    #as of 2-15-14, this seemse to speed convergence for norm-space (action and obs) performance with expln -> test norm-obs
                sde_sample_freq=N_STEPS, #sample multiple times per episode as per: https://paperswithcode.com/method/gsde, increasing frequency increases exploration
                    #set to n_steps/4 in 2-15-23? might need to reduce n_steps too...
                n_steps=N_STEPS,
                learning_rate=3e-4, #decresed in 2-15-17 from 3e-4 to improve exploration, decrease by OOM, then try schedule
                    #using 5e-6 in 2-15-21 did not improve exploration, but drastically slowed learning, going back 2 3e-5 for 2-15-23
                clip_range=0.2,#increased in 2-15-17 to improve exploration, but this caused forgetting and reduced performance, stick with 0.2
                    #this may be responsible for a large dips in 2-15-17, particularly the drops during evals which maybe forgetting
                vf_coef=0.5, #medium article suggests 0.5 or 1, tried 1 with 2-16-8 to minimal effect
)
#Continuous PPO solar reward (id=107)
trial_22_hparams = dict(
    n_epochs = 8,
    gamma = 0.9062274459056439,
    max_grad_norm = 0.45487141703809475,
    gae_lambda = 0.9639143359823201,
    n_steps = 512, 
    sde_sample_freq = 2048,
    learning_rate = 8.377741144413337e-05,
    ent_coef = 1.5635546620789673e-05,
    vf_coef = 0.5,
    clip_range = 0.2,
    seed=SEED,
)

if SAVED_MODEL is None:
    agent = PPO('MlpPolicy', 
                env,
                device='cuda',
                policy_kwargs=policy_kwargs,
                **trial_22_hparams,
                tensorboard_log='logs/Phase1/SAC/',
                verbose=0,
                )
else:
    agent = PPO.load(SAVED_MODEL,
                    env=env,
                    print_system_info=True,
                    tensorboard_log='logs/Phase1/SAC/',
                    force_reset=False, #default is true for continued training ref: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO.load
                        #this resets gSDE, is this a good thing?
                    )
    
num_ts = agent.num_timesteps #0 for new agent

T = env.time_steps - 1
#agent_name = f'{dtg}_PPOc_{DATASET_NAME}_{bldg}_gSDE_norm_space_{kwargs["reward_function"].__name__}_{policy_kwargs["net_arch"][0]}_{policy_kwargs["net_arch"][1]}'
#agent_name = f'{dtg}_PPOc'
agent_name = f'{dtg}_PPOc_{DATASET_NAME}_{bldg}_gSDE_norm_space_{kwargs["reward_function"].__name__}_deep_net_256'

if SAVED_MODEL is not None:
    agent_name += f'_{SAVE_ID}_resumed'

print(f'Training: {agent_name}\n Episodes: {EPISODES}')

early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=3)
eval_callback = EvalCallback(Monitor(eval_env), 
                             eval_freq=EVAL_FREQ*T,
                             best_model_save_path='Models/Victim/',
                             callback_after_eval=early_stopping, 
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