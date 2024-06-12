""" 
ref: https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using PPO implementation from Stable-Baselines3
on CityLearn.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

This is the second optimization, informed by the parater impoartances of the first.

Repeated params:
    n_steps: explore smaller values as the min was best in the first attempt
    ent_coeff, clip_range, and activation had improtance greater than 0.10 but no clear trend

    n_epochs and action_bins (for the environment) are new

"""
from typing import Any
from typing import Dict

import gym

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna import logging

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper, DiscreteActionWrapper


N_TRIALS = 10 #per thread, so multiply by 24
N_STARTUP_TRIALS = 10 #trials with random params rather than heuristically selected
N_EVALUATIONS = 10
EPISODE = 8760
TRAIN_TIMESTEPS = 100*EPISODE
EVAL_FREQ = int(TRAIN_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 1

BINS=20

STORAGE = None

ENV_ID = 'citylearn_challenge_2022_phase_2'

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "device": 'cuda',
    "policy_kwargs": {
            "net_arch":[256,256], #best result from 10 bin study and previous works
            "activation_fn": nn.Tanh, #tanh performed best in study 1
            "ortho_init": True, #True performed best in study 1
        },
    #add policy kwargs for net arch
}


def make_discrete_env(env_id, action_bins: int = 10, bldg: list = ['Building_6'], single_agent: bool = True, seed:int = 0):
    """Because ART's attacks are designed for supervised learning they one work with ANNs with a single label or head, using multiple buildings adds an action/head for each"""
    env = CityLearnEnv(env_id, 
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


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""


    #should I add # action bins?
     #PPO Hyperparameters
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.01, log=True) #narrowed range based on study 1 results (10 bins)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0, log=True) #best value from 10bin study was 0.76, so max reduced from 5 to 1
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.02, log=True) #best result of first study were between these values
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 2, 10) #from prev study, best value was the lowest of 8, I'm not sure if this holds with more bins
    #learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True) #low importance in intiial 10 bin study
    #ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True) #low importance in intiial 10 bin study
    vf_coef = trial.suggest_float("vf_coef", 0.5, 0.8) #later trials on 10 bin study preffered higher values, so we'll search between the default and 0.8 instead of 0.7
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3) #fisrt study had good results at both ends of this range
    #MLPpolicy hyperparameters
    #ortho_init = trial.suggest_categorical("ortho_init", [False, True]) #low importance in intiial 10 bin study
    #net_arch = trial.suggest_categorical("net_arch", list(archs.keys())) #low importance in intiial 10 bin study
    #activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"]) #low importance in intiial 10 bin study


    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    

    #net_arch = [
    #    {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    #]

    #activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": n_steps, #remove truncated batches warning, we are only using one env so the rollout is n_steps
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        #"learning_rate": learning_rate,
        #"ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "clip_range": clip_range,
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
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


#I don't like the use of globals here, but study needs an objective with a positional rag trail, menaing we can initialize kwargs afterwards.
def objective(trial: optuna.Trial) -> float:
    #set optuna's verbosity during trials
    logging.set_verbosity(logging.INFO) 
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(env=make_discrete_env(ENV_ID, action_bins=BINS),
                **kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(make_discrete_env(ENV_ID, seed=42, action_bins=BINS))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(TRAIN_TIMESTEPS, callback=eval_callback, progress_bar=True)
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

    #save the best model trained
    try: 
        if score > trial.study.best_value:
            model.save(f'models/best_model_trial_{trial.number}')
            print('Best model saved')
    except Exception as e: #meant to catch ValueError: Record does not exist. When no best value is defined after the first trial, I'm unsure which other errors are possible
        print(f'The following exception occurred: {e}/n while trying to save the best model')
        model.save(f'models/model_trial_{trial.number}')
        print('model saved')
    

    return score


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training, suggested by reference example
    #torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS) #TPE suggested for uncorrelated hyperparamenters and less than 1000 trials
    
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, 
                          n_warmup_steps=N_EVALUATIONS // 3 # Do not prune before 1/3 of the max budget is used.
                          )

    study = optuna.create_study(sampler=sampler, 
                                pruner=pruner, 
                                direction="maximize", 
                                storage=STORAGE, 
                                study_name='PPO Victim optimization 20 Bin',
                                load_if_exists=True #allows runing the file multiple time for multiprocessing
                                )

    try:
        study.optimize(objective,
                    show_progress_bar=True,
                    n_trials=N_TRIALS, #trials to run for each thread/process
                    gc_after_trial=True #runs garbage collector after each trial and may help with memory consumption and stability
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