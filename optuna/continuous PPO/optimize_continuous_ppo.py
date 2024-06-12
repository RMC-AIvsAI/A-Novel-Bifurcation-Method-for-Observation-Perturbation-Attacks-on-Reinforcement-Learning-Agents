""" 
ref: https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py

"""
from typing import Any
from typing import Dict

import gym

import optuna
from optuna.samplers import TPESampler
#from optuna.pruners import HyperbandPruner
from optuna import logging

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import KBMproject.utilities as utils

import torch
import torch.nn as nn

from citylearn.data import DataSet

# from citylearn.citylearn import CityLearnEnv
# from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper, DiscreteActionWrapper


N_TRIALS = 1000
N_STARTUP_TRIALS = 10 #trials with random params rather than heuristically selected
N_EVALUATIONS = 10
EPISODE = 8760
TRAIN_TIMESTEPS = 50*EPISODE
#EVAL_FREQ = int(TRAIN_TIMESTEPS / N_EVALUATIONS)
EVAL_FREQ = 5*EPISODE
N_EVAL_EPISODES = 1
SEED = 188
JOBS = 1
DIRECTION = 'maximize'

STORAGE = None

ENV_ID = 'citylearn_challenge_2022_phase_2'

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "device": 'cuda',
    "use_sde": True, #ref https://proceedings.mlr.press/v164/raffin22a.html
    "seed": SEED,
    "policy_kwargs": {
            #"net_arch":[256,256], 
            "activation_fn": nn.Tanh, #default
            "ortho_init": True, #default
            "use_expln": True, #additional stability: Use expln() function instead of exp() to ensure a positive standard deviation (cf paper)
        },
}


class NoImprovementPruner(optuna.pruners.BasePruner): 
    #ref: https://optuna.readthedocs.io/en/stable/_modules/optuna/pruners/_base.html#BasePruner
    def __init__(self, direction:str, warmup_steps:int=2,patience:int=2, min_improvement:float=1.0):
        """Prunes trials when the objective value stops improving
        only tested for single objectives
        direction: string of either maximize or minimize
        warmup speficies the number of evals before the trail can be pruned
        patience specifies the number of evals with no improvement before pruning
        tolerance increases the improvement required for a new best value"""
        self.warmup_steps = warmup_steps
        dir_map = {'minimize':1.0,'maximize':-1.0} #prune assumes a minimization
        assert direction in dir_map.keys(), f'Direction must be in {dir_map.keys()}'
        self.direction = dir_map[direction] #score coeff avoids rewriting for different directions
            #could direction be inferred from the study object in prune?
        self.patience = patience
        self.best_intermediate_result = self.direction*float('inf')
        self.best_step = 0
        self.tol = min_improvement*self.direction
        #self.trial_num = None


    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool: #I don't think this resets between trials
        #if this is a new trial, reset trial variables
        # if self.trial_num != trial.number:
        #     self.best_intermediate_result = self.direction*float('inf')
        #     self.best_step = 0
        #     self.trial_num = trial.number

        # Get the latest intermediate result (the value at the current step)
        step = trial.last_step
        if step is None:
            return False
        current_value = trial.intermediate_values[step]
        

        # During warmup, update the best intermediate result and do not prune
        if step < self.warmup_steps:
            self.best_intermediate_result = self.direction*min(self.direction*self.best_intermediate_result, 
                                                               self.direction*current_value)
            return False

        # After warmup, if the current value is better, update the best intermediate result and step
        if self.direction*current_value < self.direction*(self.best_intermediate_result - self.tol):
            self.best_intermediate_result = current_value
            self.best_step = step

        #would it be betteter to check len(trial.intermediate_values) > 1, then trial.intermediate_values[step] > trial.intermediate_values[step -1] or something?
            #awkward for looking multiple steps back...

        # Prune the trial if the number of evaluations without improvement exceeds the patience, reset trial variables
        if step - self.best_step > self.patience:
            self.best_intermediate_result = self.direction*float('inf')
            self.best_step = 0
            return True

        return False

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters"""
    #PPO Hyperparameters
    #PPO HParam ranges: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    n_epochs = trial.suggest_int("n_epochs", 3, 10) #should've been up to 30, change for future studies
    gamma = 1.0 - trial.suggest_float("gamma", 0.0003, 0.2, log=True) #discount [0.9997,0.8]
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 0.5, log=True) 
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.1, log=True) #[0.9,1]
    exponent_n_steps = trial.suggest_int("exponent_n_steps", 2, 12)
    n_steps = 2**exponent_n_steps
    batch_size = 2**trial.suggest_int("exponent_batch_size", 2, exponent_n_steps)
        #keeps the batch size evenly divisable and less than n_steps, to address truncated steps warning 
    target_kl = trial.suggest_categorical('target_kl', [None] + [3e-4*x for x in range(1, 10)]) #default None
        #suggests either none or one of ten values from [0.0003,0.003]
    sde_sample_freq = 2**trial.suggest_int("sde_sample_freq", 8, 12) #at least twice per episode [256,4096]
    learning_rate = trial.suggest_float("lr", 5e-6, 3e-3, log=True) 
    ent_coef = trial.suggest_float("ent_coef", 1e-9, 0.01, log=True) 
    vf_coef = trial.suggest_categorical("vf_coef", [0.5,1.0]) 
    clip_range = trial.suggest_categorical("clip_range", [0.1,0.2,0.3])
    
    #MLPpolicy hyperparameters
    #ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = [trial.suggest_int("layer_0_units", 256, 512, step=16),
                trial.suggest_int("layer_1_units", 256, 512, step=16)]

    #activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    full_std = trial.suggest_categorical("full_std", [False, True])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("sde_sample_freq", sde_sample_freq)
    trial.set_user_attr("clip_range", clip_range)
    trial.set_user_attr("vf_coef", vf_coef)

    return {
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "sde_sample_freq": sde_sample_freq,
        "batch_size": batch_size, 
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "target_kl": target_kl,
        "max_grad_norm": max_grad_norm,
        "clip_range": clip_range,
        "policy_kwargs": {
            "net_arch": net_arch,
            "full_std": full_std,
        },
    }

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 1,
        eval_freq: int = EVAL_FREQ,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

#TODO Try SAC Rwd?

class objective():
    def __init__(self,env_name, default_hparams, eval_freq:int, train_ts:int, n_eval_episodes:int=1) -> None:
        self.env_name = env_name
        self.default_hparams = default_hparams
        self.eval_freq = eval_freq
        self.train_ts = train_ts
        self.n_eval_episodes = n_eval_episodes

    def __call__(self, trial) -> Any:
        #set optuna's verbosity during trials
        logging.set_verbosity(logging.INFO) 
        kwargs = self.default_hparams.copy()
        # Sample hyperparameters.
        kwargs.update(sample_ppo_params(trial))
        # Create the RL model.
        model = PPO(env=utils.make_continuous_env(self.env_name, seed=0),
                    **kwargs)
        # Create env used for evaluation.
        eval_env = Monitor(utils.make_continuous_env(self.env_name, seed=42,))
        # Create the callback that will periodically evaluate and report the performance.
        eval_callback = TrialEvalCallback(
            eval_env, trial, n_eval_episodes=self.n_eval_episodes, eval_freq=self.eval_freq, deterministic=True
        )

        nan_encountered = False
        try:
            model.learn(self.train_ts, callback=eval_callback, progress_bar=False)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)
            nan_encountered = True
        finally:
            # Free memory.
            #eval_env.evaulate could be called here, and cost functions could be the returned metrics
            model.env.close()
            eval_env.close()

        # Tell the optimizer that the trial failed.
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        score = eval_callback.last_mean_reward

        #save best model
            #error here failed tiral 13
        if len([t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
            #best value only exists if a trial has completed
            if score > trial.study.best_value: # assumes maximization
                model.save('best_agent')
        else: #otherwise, first complete model is the best
            model.save('best_agent')

        return score


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training, suggested by reference example
    #torch.set_num_threads(1) #only for CPU?
    torch.manual_seed(SEED) #CPU 
    torch.cuda.manual_seed_all(SEED)

    #For reproducibility, but may hurt performance. But we need this to be replroducible for continued training
    #ref: https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #uses same algo every time, rather than trying multiple and choosing fastest

    rdb = optuna.storages.RDBStorage(
        url=STORAGE,
        heartbeat_interval=180,
        grace_period=None,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )

    study = optuna.create_study(sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS), #TPE suggested for uncorrelated hyperparamenters and less than 1000 trials
                                #reccomended pruner for TPE sampler
                                pruner=NoImprovementPruner(DIRECTION,
                                                           warmup_steps=0,
                                                           patience=2,
                                                           min_improvement=5), 
                                direction=DIRECTION, 
                                storage=rdb, 
                                study_name='Continuous PPO 1',
                                load_if_exists=True #allows runing the file multiple time for multiprocessing
                                )

    study.set_user_attr('Sampler',study.sampler.__class__.__name__)
    study.set_user_attr('Pruner',study.pruner.__class__.__name__)
    study.set_user_attr('Random seed',SEED)

    try:
        study.optimize(objective(DataSet.get_schema(ENV_ID), DEFAULT_HYPERPARAMS,EVAL_FREQ,TRAIN_TIMESTEPS),
                    show_progress_bar=True,
                    n_trials=N_TRIALS, #trials to run for each thread/process
                    gc_after_trial=True, #runs garbage collector after each trial and may help with memory consumption and stability
                    n_jobs=JOBS,
                    ) 
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))