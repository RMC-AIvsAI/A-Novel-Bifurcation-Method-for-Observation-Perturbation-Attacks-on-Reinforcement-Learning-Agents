""" 
ref: https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a Gymnasium environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
from typing import Any
from typing import Dict

import json

import gym
#from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor #hack for multiprocessing
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

from os import cpu_count
#from concurrent.futures import ProcessPoolExecutor

N_TRIALS = 4 #per thread, so multiply by 24
N_STARTUP_TRIALS = 1
N_EVALUATIONS = 1
EPISODE = 8760
TRAIN_TIMESTEPS = 100*EPISODE
EVAL_FREQ = int(TRAIN_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 1

STORAGE = None

ENV_ID = 'citylearn_challenge_2022_phase_2'

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "device": 'cuda',
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

    archs = {
        'default': [64,64],
        'deep': [64,64,64],
        'wider':[128,128],
        'widest':[256,256]
    }
    #should I add # action bins?
    #PPO Hyperparameters
    #gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10) 
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 0.7)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    #MLPpolicy hyperparameters
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", list(archs.keys()))
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])


    # Display true values.
    #trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    

    #net_arch = [
    #    {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    #]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": n_steps, #remove truncated batches warning, we are only using one env so the rollout is n_steps
        #"gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "clip_range": clip_range,
        "policy_kwargs": {
            #"net_arch": net_arch,
            "net_arch":archs[net_arch],
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
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
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    #set optuna's verbosity during trials
    logging.set_verbosity(logging.INFO) 
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(env=make_discrete_env(ENV_ID, seed=1), #shouldn't use globals here...
                **kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(make_discrete_env(ENV_ID, seed=42))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(TRAIN_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    #torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS) #suggested for uncorrelated hyperparamenters for less than 1000 trials
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize", storage=STORAGE, study_name=f'_Victim_PPO_{ENV_ID}_{N_TRIALS}_trials_of_{TRAIN_TIMESTEPS}_steps',load_if_exists=True)

    #with ThreadPoolExecutor() as pool: #actually process pool execcutor
    #    for cpu in range(cpu_count()):
    #        pool.submit(study.optimize(objective, #maybe objective should have some arguments/initialization
    #                n_trials=N_TRIALS, #trials to run for each thread
    #                ))

    try:
        study.optimize(objective, #maybe objective should have some arguments/initialization
                    n_trials=N_TRIALS, #trials to run for each thread
                    #timeout=600, #timelimit in s
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

    #save best hyperparams
    with open(f'optuna/Victim/PPO_{ENV_ID}_{N_TRIALS}_trials_of_{TRAIN_TIMESTEPS}_steps.json', 'w') as outfile:
        json.dump({**trial.params, **trial.user_attrs}, outfile)