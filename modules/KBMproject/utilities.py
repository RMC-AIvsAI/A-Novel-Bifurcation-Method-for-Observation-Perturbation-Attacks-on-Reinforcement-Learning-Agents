from citylearn.citylearn import CityLearnEnv, EvaluationCondition
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper, DiscreteActionWrapper, NormalizedSpaceWrapper

from art.estimators.classification import PyTorchClassifier as classifier
from art.utils import to_categorical
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn

from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import isnan

np.random.seed(42)

def make_discrete_env(schema, action_bins: int = 10, bldg: list = None, single_agent: bool = True, seed:int = 0, T=None):
    """Because ART's attacks are designed for supervised learning they one work with ANNs with a single label or head, using multiple buildings adds an action/head for each"""
    
#TODO support custom rewards
    if bldg is None:
        bldg = list(schema['buildings'].keys())[0] #the first building from the schema's building keys
    
    env = CityLearnEnv(schema, 
        central_agent=single_agent, 
        buildings=bldg, 
        random_seed=seed,
        episode_time_steps=T,)
    #Because ART attacks are made for classification tasks we need a discrete action space 
    env = DiscreteActionWrapper(env, bin_sizes=[{'electrical_storage':action_bins}])
    #Calendar observations are periodically normalized, everything else is min/max normalized 
    env = NormalizedObservationWrapper(env)
    #provides an interface for SB3
    env = StableBaselines3Wrapper(env)
    return env

def make_continuous_env(schema, bldg: list = None, single_agent: bool = True, seed:int = 0, T=None, env_kwargs:dict=dict()):
    """Because ART's attacks are designed for supervised learning they one work with ANNs with a single label or head, using multiple buildings adds an action/head for each"""
    
    #TODO support custom rewards
    if bldg is None:
        bldg = list(schema['buildings'].keys())[0] #the first building from the schema's building keys

    kwargs = env_kwargs
    
    env = CityLearnEnv(schema, 
        central_agent=single_agent, 
        buildings=bldg, 
        random_seed=seed,
        episode_time_steps=T,
        **kwargs)
    #Calendar observations are periodically normalized, everything else is min/max normalized 
    env = NormalizedSpaceWrapper(env)
    #provides an interface for SB3
    env = StableBaselines3Wrapper(env)
    return env


def format_kpis(env, eval_condition=None):
    """displays the KPIs from the evnironment's most recent timestep.
    This function can be called after an agent runs in a test env to evaluate performance"""

    if eval_condition is None:
        eval_condition = EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PARTIAL_LOAD_AND_PV

    kpis = env.evaluate(baseline_condition=eval_condition).pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    kpis = kpis['District']
    kpis = kpis[kpis != 0]
    return kpis



def eval_agent(env, agent):
    """evaluates the input agent for one episode
    returns a df containing the KPIs, and arrays containing the observations and actions"""
    obs_list = []
    a_list = []

    observations = env.reset()

    while not env.done:
        obs_list.append(observations)
        actions, _ = agent.predict(observations, deterministic=True)
        a_list.append(actions)
        observations, _, _, _ = env.step(actions)
    
    return format_kpis(env), np.array(obs_list), np.array(a_list)



def extract_actor(agent):
    """Extracts the policy network from and SB3 actor critic algorithm 
    returns a pytorch seuqential network"""
    from copy import deepcopy
    policy_net = deepcopy(agent.policy.mlp_extractor.policy_net) #copies shared net rather than referencing
    policy_net.add_module('actions', agent.policy.action_net)
    return policy_net

def extract_SACtor(sac_agent):
    """Extracts the policy network from and SB3 Soft Actor Critic (SAC) algorithm 
    returns a pytorch seuqential network"""
    from copy import deepcopy
    policy_net = deepcopy(sac_agent.actor.latent_pi) #copies shared net rather than referencing/changing the agent
    policy_net.add_module('4', sac_agent.actor.mu)
    return policy_net


def get_feature_permutations(agent, observations:np.array, actions: np.array, FeaturePermuation_kwargs: dict=None) -> np.array:
    """takes an agent and it's observations and actions during evaluation, and returns the importance of each feature in a np.array"""
    #import extract actor?
    from captum.attr import FeaturePermutation
    import torch

    tensor_obs = torch.from_numpy(observations[:actions.shape[0]]).type(torch.FloatTensor).to('cuda')
    actor = extract_actor(agent)
    fp = FeaturePermutation(actor)
    attr = fp.attribute(tensor_obs, 
                    target=actions.flatten().tolist())
    attr = attr.detach().cpu().numpy()

    return np.mean(attr, axis=0)

def get_integrated_gradradients(agent, observations:np.array, actions: np.array, IntegratedGradients_kwargs: dict=None) -> np.array:
    from captum.attr import IntegratedGradients
    import torch

    tensor_obs = torch.from_numpy(observations[:actions.shape[0]]).type(torch.FloatTensor).to('cuda')
    actor = extract_actor(agent)
    ig = IntegratedGradients(actor)
    attr = ig.attribute(tensor_obs, 
                    target=actions.flatten().tolist())
    attr = attr.detach().cpu().numpy()

    return np.mean(attr, axis=0)


def eval_rand_attack(agent, env, eps=0.05):
    """evaluates an agent for one episode with random noise in the observations
    returns a df of KPIS, and arrays of the observations and actions"""
    obs_list = []
    a_list = []
    asr = 0

    observations = env.reset()
    
    while not env.done:
        obs_list.append(observations)
        noisey_obs = observations + np.random.rand(*observations.shape)*eps
        a_adv, _ = agent.predict(noisey_obs, deterministic=True)
        actions, _ = agent.predict(observations, deterministic=True)
        a_list.append(actions)
        if a_adv!=actions: #check if the perturbation changed the agent's action
            asr+=1
        observations, _, _, _ = env.step(a_adv)

    asr/=env.time_steps
    #print(f'The Adversarial success rate is: {asr}')
    return format_kpis(env), np.array(obs_list), np.array(a_list), asr



def define_attack(agent, ART_atk, loss_fn=CrossEntropyLoss(), ART_kwargs=None):
    """returns an ART attack function based on the input gym enviornment, SB3 Agent and ART attack class"""
    
    agent_policy = extract_actor(agent)

    victim_policy = classifier(
        model=agent_policy,
        loss=loss_fn, 
        nb_classes=agent.action_space[0].n, #could I just use agent here?
        input_shape=agent.observation_space.shape,
        device_type='gpu',
        clip_values = (agent.observation_space.low.min(),agent.observation_space.high.max()) #min and max values of each feature, brendle bethge attack only supports floats values and not array
        )
    kwargs = ART_kwargs
    return ART_atk(estimator=victim_policy, **kwargs)



def eval_untargeted_attack(agent, env, atk, time_steps:int=None, mask:list=None):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack"""
    obs_list = []
    adv_obs_list = []
    asr = 0
    n_features = agent.observation_space.shape[0]

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(n_features) #1 for all features

    pbar = tqdm(total=time_steps)
    for step in tqdm(range(time_steps)):

        obs_list.append(observations)
        actions = agent.predict(observations, deterministic=True)

        adv_obs = np.expand_dims(observations, axis=0) #ART atks expect a 2d array
        #would using the true label/action imporve the asr? it would hurt adversarial training: https://arxiv.org/abs/1611.01236
        #the 'true label' is the prediction generated by the model, which is used when no target is provided
        adv_obs = atk.generate(adv_obs, mask=mask)
        adv_obs = np.squeeze(adv_obs) #CityLearn envs expect a 1d array
        
        a_adv, _ = agent.predict(adv_obs, deterministic=True)
        if a_adv[0]!=actions[0]: #check if an adversarial example was crafted
            asr+=1
            adv_obs_list.append(adv_obs)
        else:
            adv_obs_list.append(np.array([np.nan]*n_features)) #same shape as observations

        observations, _, _, _ = env.step(a_adv)

        #update progress bar including asr
        pbar.update(1)
        pbar.set_postfix({'ASR': asr/(step + 1)}, refresh=True)
        if env.done:
            break
    
    pbar.close()
    #obs_list.append(observations) #ignore final obs as no action is taken
    #adv_obs_list.append(adv_obs)
    asr/=time_steps
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), asr

def eval_untargeted_attack_on_step(agent, env, atk, atk_steps:list,time_steps:int=None, mask:list=None):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack
    only perturbs observation on timestpes in the list atk_stps"""
    obs_list = []
    adv_obs_list = []
    asr = 0
    n_features = agent.observation_space.shape[0]

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(n_features) #1 for all features

    #pbar = tqdm(total=time_steps)
    for step in tqdm(range(time_steps)):

        obs_list.append(observations)
        actions = agent.predict(observations, deterministic=True)

        if step in atk_steps:
            adv_obs = np.expand_dims(observations, axis=0) #ART atks expect a 2d array

            adv_obs = atk.generate(adv_obs, mask=mask)
            adv_obs = np.squeeze(adv_obs) #CityLearn envs expect a 1d array
            
            a_adv, _ = agent.predict(adv_obs, deterministic=True)
        else:
            a_adv = actions

        if a_adv[0]!=actions[0]: #check if an adversarial example was crafted
            asr+=1
            adv_obs_list.append(adv_obs)
        else:
            adv_obs_list.append(np.array([np.nan]*n_features)) #same shape as observations

        observations, _, _, _ = env.step(a_adv)

        #update progress bar including asr
        # pbar.update(1)
        # pbar.set_postfix({'ASR': asr/(step + 1)}, refresh=True)
        if env.done:
            break
    
    #pbar.close()
    asr/=len(atk_steps)
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), asr


def eval_untargeted_attack_with_action_distance(agent, env, atk, time_steps:int=None, mask:list=None):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack"""
    obs_list = []
    asr = 0
    avg_action_dist = 0

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(agent.observation_space.shape[0]) #1 for all features

    for i in tqdm(range(time_steps)):

        obs_list.append(observations)
        actions = agent.predict(observations, deterministic=True)

        adv_obs = np.expand_dims(observations, axis=0) #ART atks expect a 2d array
        #would using the true label/action imporve the asr? it would hurt adversarial training: https://arxiv.org/abs/1611.01236
        #the 'true label' is the prediction generated by the model, which is used when no target is provided
        adv_obs = atk.generate(adv_obs, mask=mask)
        adv_obs = np.squeeze(adv_obs) #CityLearn envs expect a 1d array
        
        a_adv, _ = agent.predict(adv_obs, deterministic=True)
        if a_adv[0]!=actions[0]: #check if an adversarial example was successful
            asr+=1

        observations, _, _, _ = env.step(a_adv)

        avg_action_dist = (avg_action_dist*(i) + abs(a_adv[0]-actions[0]))/(i+1)

        if env.done:
            break
    
    obs_list.append(observations)
    asr/=time_steps
    #print(f'The Adversarial success rate is: {asr}')
    #print(f'The average distance between optinmal and adversarial actions is: {avg_action_dist}')
    return format_kpis(env), np.array(obs_list), asr, avg_action_dist


def extract_critic(agent):
    """Extracts the policy network from and SB3 actor critic algorithm 
    returns a pytorch seuqential network"""
    from copy import deepcopy
    value_net = deepcopy(agent.policy.mlp_extractor.value_net) #copies shared net rather than referencing
    value_net.add_module('value', agent.policy.value_net)
    return value_net


def eval_untargeted_value_attack(agent, env, atk, time_steps:int=None, mask:list=None, value_threshold:float=-70):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack,
    when the agent's value for the state is above a threshold"""
    obs_list = []
    asr = 0
    perturbation_rate = 0

    value_net = extract_critic(agent)
    value = classifier(model=value_net,
                   nb_classes=env.action_space[0].n,
                   loss=CrossEntropyLoss(),
                   input_shape=agent.observation_space.shape,
                   device_type='gpu')

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(agent.observation_space.shape[0]) #1 for all features

    for _ in tqdm(range(time_steps)):

        obs_list.append(observations)

        if value.predict(observations) >= value_threshold:

            perturbation_rate+=1

            actions = agent.predict(observations, deterministic=True)

            adv_obs = np.expand_dims(observations, axis=0) #ART atks expect a 2d array
            #would using the true label/action imporve the asr? it would hurt adversarial training: https://arxiv.org/abs/1611.01236
            #the 'true label' is the prediction generated by the model, which is used when no target is provided
            adv_obs = atk.generate(adv_obs, mask=mask)
            adv_obs = np.squeeze(adv_obs) #CityLearn envs expect a 1d array
            
            a_adv, _ = agent.predict(adv_obs, deterministic=True)
            if a_adv[0]!=actions[0]: #check if an adversarial example was successful
                asr+=1
            actions = a_adv
        else:
            actions, _ = agent.predict(observations, deterministic=True)

        observations, _, _, _ = env.step(actions)

        if env.done:
            break
    
    obs_list.append(observations)
    asr/=perturbation_rate
    perturbation_rate/=time_steps
    print(f'The Adversarial success rate is: {asr}')
    print(f'The perturbation rate was: {perturbation_rate}')
    return format_kpis(env), np.array(obs_list)



def eval_untargeted_attack_rand_step(agent, env, atk, time_steps:int=None, mask:list=None, rand_threshold:float=0.5):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack,
    perturbing random observations of a set proportion"""
    obs_list = []
    asr = 0
    perturbation_rate = 0


    observations = env.reset()

    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(agent.observation_space.shape[0]) #1 for all features

    for _ in tqdm(range(time_steps)):

        obs_list.append(observations)

        if np.random.rand() >= rand_threshold: #randomly shoose when to attack

            perturbation_rate+=1

            actions = agent.predict(observations, deterministic=True)

            adv_obs = np.expand_dims(observations, axis=0) #ART atks expect a 2d array
            adv_obs = atk.generate(adv_obs, mask=mask)
            adv_obs = np.squeeze(adv_obs) #CityLearn envs expect a 1d array
            
            a_adv, _ = agent.predict(adv_obs, deterministic=True)
            if a_adv[0]!=actions[0]: #check if an adversarial example was crafted
                asr+=1
            actions = a_adv
        else:
            actions, _ = agent.predict(observations, deterministic=True)

        observations, _, _, _ = env.step(actions)

        if env.done:
            break
    
    obs_list.append(observations)
    asr/=perturbation_rate
    perturbation_rate/=time_steps
    print(f'The Adversarial success rate is: {asr}')
    print(f'The perturbation rate was: {perturbation_rate}')
    return format_kpis(env), np.array(obs_list)


def describe_list(lst):
    import statistics

    # Check if list is empty
    if not lst:
        return "List is empty."

    # Calculate descriptive statistics
    count = len(lst)
    mean = sum(lst) / count
    median = statistics.median(lst)
    min_val = min(lst)
    max_val = max(lst)
    std_dev = statistics.stdev(lst)

    # Return results
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "std_dev": std_dev,
    }

def dynamic_distortion(victim_policy, atk_candidates:list, generate_kwargs:dict=None):
    """finds the minimum successful distortion epsilon from a given SORTED list, 
    by iteratively removing list items until one remains with a binary search
    expects a lits of tupples, containing an attack obsect with gereate() attribute, and an epsilon value.
    returns adversarial example, adversarial action, eps used to produce them, and the difference between the original and adversarial actions"""

    idx = (len(atk_candidates))//2 #middle
    best_candidate_idx = list(range((len(atk_candidates)))) #assigned an index for each candidate
    kwargs = generate_kwargs
    atk_idx = 0
    eps_idx = 1
    min_eps = None
    
    kwargs = generate_kwargs
    if kwargs['x'].ndim == 1: #ART requires 2d input
        kwargs['x'] = np.expand_dims(kwargs['x'], axis=0)
    min_eps_sample = kwargs['x']
    a = a_min_eps = np.argmax(victim_policy.predict(kwargs['x'], training_mode=False)) #nn outputs -> largest index corresponds to the chosen action, agent.predict()

    while len(atk_candidates) >= 1: #list is not empty, we remove all tested values
        adv_sample = atk_candidates[idx][atk_idx].generate(**kwargs)
        a_adv = np.argmax(victim_policy.predict(adv_sample, training_mode=False))
        
        if a!=a_adv: #adversarial example successful
            min_eps = atk_candidates[idx][eps_idx]
            atk_candidates = atk_candidates[:idx] #this worked, so only keep lower values, slice is exclusive because we save the values below
            min_eps_sample = adv_sample
            a_min_eps = a_adv
        else: #no adversarial sample found
            atk_candidates = atk_candidates[idx+1:] #This didn't work, so only keep higher values, slice is exclusive
            best_candidate_idx = best_candidate_idx[idx+1:]

        idx = (len(atk_candidates))//2
    a_dist = abs(a - a_min_eps)
    return np.squeeze(min_eps_sample), np.array([a_min_eps]), min_eps, a_dist #agent.predict returns an array

def eval_untargeted_dynamic_distortion_value_attack(agent, env, ART_atk, ART_atk_kwargs:dict=None, 
                                    eps_candidates:list=[0.05], init_step_coeff:float=None, time_steps:int=None, mask:list=None,
                                    value_threshold:float=-70):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack,
    using the smallest distortion from a list"""

    from math import isnan

    obs_list = []
    adv_obs_list = []
    asr = 0
    eps_list = []
    avg_action_dist = 0
    atk_list = [] #list of attacks for each eps candidate
    eps_candidates.sort() #sort list in ascending order

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is not None:
        generate_kwargs = dict(mask=mask)
    else:
        generate_kwargs = dict()

    victim_policy = classifier(
                            model=extract_actor(agent),
                            loss=CrossEntropyLoss(), 
                            nb_classes=env.action_space[0].n,
                            input_shape=agent.observation_space.shape,
                            device_type='gpu'
                            )
    value_net = extract_critic(agent)
    value = classifier(model=value_net,
                   nb_classes=env.action_space[0].n,
                   loss=CrossEntropyLoss(),
                   input_shape=agent.observation_space.shape,
                   device_type='gpu')
    
    kwargs = ART_atk_kwargs
    kwargs['verbose'] = False #can crash jupyter

    #define list of attacks for each candidate epsilon
    if init_step_coeff is None:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))
    else:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            kwargs['eps_step'] = init_step_coeff*eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))

    for i in tqdm(range(time_steps)):
        obs_list.append(observations) #next state

        if value.predict(observations) >= value_threshold:
            generate_kwargs['x'] = observations #sample

            adv_obs, a_adv, min_eps, a_dist = dynamic_distortion(victim_policy=victim_policy,
                                                                generate_kwargs=generate_kwargs,
                                                                atk_candidates=atk_list)
            adv_obs_list.append(adv_obs)
            if a_dist!=0:
                asr+=1
                eps_list.append(min_eps)
            else:
                eps_list.append(np.nan)

            observations, _, _, _ = env.step(a_adv)

            avg_action_dist = (avg_action_dist*(i) + a_dist)/(i+1)
        else:
            actions, _ = agent.predict(observations, deterministic=True)
            observations, _, _, _ = env.step(actions)
        if env.done:
            break
    
    obs_list.append(observations)
    asr/=time_steps
    print(f'The Adversarial success rate is: {asr}')
    print(f'The average distance between optinmal and adversarial actions is: {avg_action_dist}')
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(eps_list)


def eval_untargeted_dynamic_distortion_attack(agent, env, ART_atk, ART_atk_kwargs:dict=None, 
                                    eps_candidates:list=[0.05], init_step_coeff:float=None, 
                                    time_steps:int=None, mask:list=None):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack,
    using the smallest distortion from a list"""

    from math import isnan

    obs_list = []
    adv_obs_list = []
    asr = 0
    eps_list = []
    avg_action_dist = 0
    atk_list = [] #list of attacks for each eps candidate
    eps_candidates.sort() #sort list in ascending order

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is not None:
        generate_kwargs = dict(mask=mask)
    else:
        generate_kwargs = dict()

    victim_policy = classifier(
                            model=extract_actor(agent),
                            loss=CrossEntropyLoss(), 
                            nb_classes=env.action_space[0].n,
                            input_shape=agent.observation_space.shape,
                            device_type='gpu'
                            )
    kwargs = ART_atk_kwargs
    kwargs['verbose'] = False #can crash jupyter
    #define list of attacks for each candidate epsilon
    if init_step_coeff is None:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))
    else:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            kwargs['eps_step'] = init_step_coeff*eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))

    for i in tqdm(range(time_steps)):
        obs_list.append(observations) #next state
        generate_kwargs['x'] = observations  #sample

        adv_obs, a_adv, min_eps, a_dist = dynamic_distortion(victim_policy=victim_policy,
                                                            generate_kwargs=generate_kwargs,
                                                            atk_candidates=atk_list)
        adv_obs_list.append(adv_obs)
        if a_dist!=0:
            asr+=1
            eps_list.append(min_eps)
        else:
            eps_list.append(np.nan)

        observations, _, _, _ = env.step(a_adv)

        avg_action_dist = (avg_action_dist*(i) + a_dist)/(i+1)

        if env.done:
            break
    
    obs_list.append(observations)
    asr/=time_steps
    print(f'The Adversarial success rate is: {asr}')
    print(f'The average distance between optinmal and adversarial actions is: {avg_action_dist}')
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(eps_list)

def eval_untargeted_dynamic_distortion_attack_rand_step(agent, env, ART_atk, ART_atk_kwargs:dict=None, 
                                    eps_candidates:list=[0.05], init_step_coeff:float=None, time_steps:int=None, 
                                    mask:list=None, rand_threshold:float=0.5,):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack,
    using the smallest distortion from a list"""

    from math import isnan

    obs_list = []
    adv_obs_list = []
    asr = perturbation_rate = 0
    eps_list = []
    avg_action_dist = 0
    atk_list = [] #list of attacks for each eps candidate
    eps_candidates.sort() #sort list in ascending order

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is not None:
        generate_kwargs = dict(mask=mask)
    else:
        generate_kwargs = dict()

    victim_policy = classifier(
                            model=extract_actor(agent),
                            loss=CrossEntropyLoss(), 
                            nb_classes=env.action_space[0].n,
                            input_shape=agent.observation_space.shape,
                            device_type='gpu'
                            )
    kwargs = ART_atk_kwargs
    kwargs['verbose'] = False #can crash jupyter
    #define list of attacks for each candidate epsilon
    if init_step_coeff is None:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))
    else:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            kwargs['eps_step'] = init_step_coeff*eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))

    for i in tqdm(range(time_steps)):
        obs_list.append(observations) #next state
        generate_kwargs['x'] = observations  #sample
        if  np.random.rand() >= rand_threshold:
            perturbation_rate+=1
            adv_obs, a_adv, min_eps, a_dist = dynamic_distortion(victim_policy=victim_policy,
                                                                generate_kwargs=generate_kwargs,
                                                                atk_candidates=atk_list)
            adv_obs_list.append(adv_obs)
            if a_dist != 0:
                asr+=1
                eps_list.append(min_eps)
            else:
                eps_list.append(np.nan)

            observations, _, _, _ = env.step(a_adv)
        else:
            a, _ = agent.predict(observations, deterministic=True)
            observations, _, _, _ = env.step(a)
            a_dist = 0

        avg_action_dist = (avg_action_dist*(i) + a_dist)/(i+1)

        if env.done:
            break
    
    obs_list.append(observations)
    asr/=perturbation_rate
    print(f'The Adversarial success rate is: {asr}')
    print(f'The average distance between optinmal and adversarial actions is: {avg_action_dist}')
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(eps_list)

def eval_untargeted_dynamic_distortion_attack_value_threshold(agent, env, ART_atk, ART_atk_kwargs:dict=None, 
                                    eps_candidates:list=[0.05], init_step_coeff:float=None, time_steps:int=None, 
                                    mask:list=None, value_threshold:float=-70,):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack,
    using the smallest distortion from a list"""

    from math import isnan

    obs_list = []
    adv_obs_list = []
    asr = perturbation_rate = 0
    eps_list = []
    avg_action_dist = 0
    atk_list = [] #list of attacks for each eps candidate
    eps_candidates.sort() #sort list in ascending order

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is not None:
        generate_kwargs = dict(mask=mask)
    else:
        generate_kwargs = dict()

    victim_policy = classifier(
                            model=extract_actor(agent),
                            loss=CrossEntropyLoss(), 
                            nb_classes=env.action_space[0].n,
                            input_shape=agent.observation_space.shape,
                            device_type='gpu'
                            )
    
    value = classifier(model=extract_critic(agent),
                   nb_classes=env.action_space[0].n,
                   loss=CrossEntropyLoss(),
                   input_shape=agent.observation_space.shape,
                   device_type='gpu')
    
    kwargs = ART_atk_kwargs
    kwargs['verbose'] = False #can crash jupyter
    #define list of attacks for each candidate epsilon
    if init_step_coeff is None:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))
    else:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            kwargs['eps_step'] = init_step_coeff*eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))

    for i in tqdm(range(time_steps)):
        obs_list.append(observations) #next state

        if  value.predict(observations) >= value_threshold:
            generate_kwargs['x'] = observations  #sample
            perturbation_rate+=1
            adv_obs, a_adv, min_eps, a_dist = dynamic_distortion(victim_policy=victim_policy,
                                                                generate_kwargs=generate_kwargs,
                                                                atk_candidates=atk_list)
            adv_obs_list.append(adv_obs)
            if a_dist != 0:
                asr+=1
                eps_list.append(min_eps)
            else:
                eps_list.append(np.nan)

            observations, _, _, _ = env.step(a_adv)
        else:
            a, _ = agent.predict(observations, deterministic=True)
            observations, _, _, _ = env.step(a)
            a_dist = 0

        avg_action_dist = (avg_action_dist*(i) + a_dist)/(i+1)

        if env.done:
            break
    
    obs_list.append(observations)
    asr/=perturbation_rate
    print(f'The Adversarial success rate is: {asr}')
    print(f'The average distance between optinmal and adversarial actions is: {avg_action_dist}')
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(eps_list)

def eval_targeted_dynamic_distortion_attack(agent, adversary, env, ART_atk, ART_atk_kwargs:dict=None, 
                                    eps_candidates:list=[0.05], init_step_coeff:float=None, time_steps:int=None, mask:list=None):
    """Evaluates an SB3 agent subject to targeted observation perturbations chosen by the adversary
      generated by an ART evasion attack, using the smallest distortion from a list"""

    obs_list = []
    adv_obs_list = []
    asr = 0
    eps_list = []
    avg_action_dist = 0
    atk_list = [] #list of attacks for each eps candidate
    eps_candidates.sort() #sort list in ascending order
    generate_kwargs = dict(targeted=True)

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is not None:
        generate_kwargs['mask'] = mask

    victim_policy = classifier(
                            model=extract_actor(agent),
                            loss=CrossEntropyLoss(), 
                            nb_classes=agent.action_space[0].n,
                            input_shape=agent.observation_space.shape,
                            device_type='gpu',
                            clip_values = (agent.observation_space.low.min(),agent.observation_space.high.max())
                            )
    kwargs = ART_atk_kwargs
    kwargs['verbose'] = False #can crash jupyter
    #define list of attacks for each candidate epsilon
    if init_step_coeff is None:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))
    else:
        for eps in eps_candidates:
            kwargs['eps'] = eps
            kwargs['eps_step'] = init_step_coeff*eps
            atk = ART_atk(victim_policy, **kwargs)
            atk_list.append((atk, eps))

    for i in tqdm(range(time_steps)):
        obs_list.append(observations) #next state
        generate_kwargs['x'] = observations  #sample
        generate_kwargs['y'], _ = adversary.predict(observations, deterministic=True) #target

        adv_obs, a_adv, min_eps, a_dist = dynamic_distortion(victim_policy=victim_policy,
                                                            generate_kwargs=generate_kwargs,
                                                            atk_candidates=atk_list)
        adv_obs_list.append(adv_obs) #clean obs if unsuccessful
        if a_adv==generate_kwargs['y']:
            asr+=1
            eps_list.append(min_eps)
        else:
            eps_list.append(np.nan)

        observations, _, _, _ = env.step(a_adv)

        avg_action_dist = (avg_action_dist*(i) + a_dist)/(i+1)

        if env.done:
            break
    
    obs_list.append(observations)
    asr/=time_steps
    print(f'The Adversarial success rate is: {asr}')
    print(f'The average distance between optinmal and adversarial actions is: {avg_action_dist}')
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(eps_list)

def eval_targeted_attack(agent, adversary, env, ART_atk, time_steps:int=None, mask:list=None, x_adv_init:dict=None):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack,
    starting p[oints are added for the bb attack, which requires initialization from a sample of the target class"""

    obs_list = []
    adv_obs_list = []
    target_list = []
    action_list = []
    asr = 0
    kwargs = dict()

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is not None:
        kwargs['mask'] = mask
    if isinstance(x_adv_init, dict):
        initialized = True
    else:
        initialized = False

    pbar = tqdm(total=time_steps)
    for step in range(time_steps):

        obs_list.append(observations)

        target, _ = adversary.predict(observations, deterministic=True)
        target_list.append(target)
        kwargs['y'] = to_categorical(target, nb_classes=adversary.action_space[0].n) #one-hot encode int

        if initialized: #assume dict where keys are actions and values are adv inits
            if target[0] in x_adv_init.keys():
                kwargs['x_adv_init'] = x_adv_init[target[0]].astype(observations.dtype)
                if(kwargs['x_adv_init'].shape[0] > 1): #multiple samples fo this target, pick the closest
                    l2dist = np.linalg.norm(kwargs['x_adv_init'] - observations, ord=2, axis=1) 
                    kwargs['x_adv_init'] = kwargs['x_adv_init'][np.argmin(l2dist)]

        adv_obs = np.expand_dims(observations, axis=0) #ART atks expect a 2d array
        adv_obs = ART_atk.generate(adv_obs, **kwargs)
        adv_obs = np.squeeze(adv_obs) #CityLearn envs expect a 1d array
        
        adv_a, _ = agent.predict(adv_obs, deterministic=True)
        action_list.append(adv_a)
        if target[0]==adv_a[0]: #check if the action matches the intended target
            asr+=1
            
        adv_obs_list.append(adv_obs)

        observations, _, _, _ = env.step(adv_a)

        #update progress bar including asr
        pbar.update(1)
        pbar.set_postfix({'ASR': asr/(step + 1)}, refresh=True)
        if env.done:
            break
    
    pbar.close()
    #obs_list.append(observations)
    asr/=time_steps
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(action_list), np.array(target_list), asr

class RegressorWrapper(nn.Module):
    """wraps a regressor with a symetrical positive and negative output, e.g. [-1,1]
    and replaces the single output with 2 logits, one is positive when the output is
    the other when it's negative"""
    def __init__(self, base_model):
        super(RegressorWrapper, self).__init__()
        self.base_model = base_model


    def forward(self, x):
        output = self.base_model(x)
        
        logits = torch.cat((output,-1.0*output), dim=1).float()
        return logits
    
class RegressorLinearWrapper(nn.Module):
    """wraps a regressor
    and replaces the single output with 2 logits, one is maximized at 0 
    the other at 1 (by default)
    y= m*x + b"""
    def __init__(self, base_model, m1=1.0, b1=0.0, m2=-1.0, b2=0.0):
        super(RegressorLinearWrapper, self).__init__()
        self.base_model = base_model
        self.m1 = m1
        self.m2 = m2
        self.b1 = b1
        self.b2 = b2


    def forward(self, input):
        x = self.base_model(input)
        
        logits = torch.cat((self.m1*x + self.b1, self.m2*x + self.b2), dim=1).float()
        return logits
    

class CWLoss(nn.Module):
    """Carlini and Wagner or Difference Logits loss FOR UNTARGETED ATTACKS
    where the loss is difference between the target/clean
    logit and any other"""
    def __init__(self, reduction=None):
        super(CWLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target_one_hot):
        target_logits = torch.sum(target_one_hot * logits, dim=1)
        max_non_target_logits = torch.max((1 - target_one_hot) * logits, dim=1)[0]
        loss = max_non_target_logits - target_logits

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  #reduction is None
            return loss
        
class MaximumBifuricationWrapper(nn.Module):
    def __init__(self, base_model):
        super(MaximumBifuricationWrapper, self).__init__()
        self.base_model = base_model


    def forward(self, x):
        logits = self.base_model(x)
        lower_half, higher_half = torch.split(logits, logits.size(1) // 2, dim=1)
        
        # get the max of the lower and higher halves
        lower_max = torch.max(lower_half, dim=1)[0]
        higher_max = torch.max(higher_half, dim=1)[0]
        
        # concatenate the max of the lower and higher halves into a single tensor
        output = torch.cat((lower_max.unsqueeze(1), higher_max.unsqueeze(1)), dim=1)
        return output
    
def eval_continuous_attack(agent, env, atk, time_steps:int=None, mask:list=None): 
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack
    agent has a continuous action space wrapped to provide logits"""
    obs_list = []
    adv_obs_list = []
    a_list = []
    adv_a_list = []
    mae = 0
    n_features = agent.observation_space.shape[0]

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(n_features) #1 for all features

    pbar = tqdm(total=time_steps)
    for step in tqdm(range(time_steps)):

        obs_list.append(observations)
        actions = agent.predict(observations, deterministic=True)
        a_list.append(actions[0])

        adv_obs = np.squeeze(
            atk.generate(np.expand_dims(observations, axis=0), 
                            mask=mask)
                            )
        adv_obs_list.append(adv_obs)
        
        a_adv, _ = agent.predict(adv_obs, deterministic=True)
        a_dist = abs(a_adv[0] - actions[0])[0]
        mae += a_dist

        adv_a_list.append(a_adv[0])
        observations, _, _, _ = env.step(a_adv)

        #update progress bar including asr
        pbar.update(1)
        pbar.set_postfix({'MAE': mae/(step + 1)}, refresh=True)
        if env.done:
            break
    
    pbar.close()
    mae/=time_steps
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(a_list), np.array(adv_a_list), mae 

def eval_toggle_bifurcation_continuous_attack(agent, env, atk, time_steps:int=None, mask:list=None): 
    """Evaluates an SB3 agent subject to targeted observation perturbations generated by an ART evasion attack
    agent has a continuous action space wrapped to provide logits. the target is either 0 or 1 conrresponding to one of 
    the logits"""
    obs_list = []
    adv_obs_list = []
    a_list = []
    adv_a_list = []
    mae = 0
    n_features = agent.observation_space.shape[0]

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(n_features) #1 for all features

    pbar = tqdm(total=time_steps)
    for step in tqdm(range(time_steps)):

        obs_list.append(observations)
        actions = agent.predict(observations, deterministic=True)
        a_list.append(actions[0])

        adv_obs = np.squeeze(
            atk.generate(np.expand_dims(observations, axis=0),
                         y=np.array([step%2]), #might need one-hot encoding, but I think ART handles that
                         mask=mask)
            )
        adv_obs_list.append(adv_obs)
        
        a_adv, _ = agent.predict(adv_obs, deterministic=True)
        a_dist = abs(a_adv[0] - actions[0])[0]
        mae += a_dist

        adv_a_list.append(a_adv[0])
        observations, _, _, _ = env.step(a_adv)

        #update progress bar including asr
        pbar.update(1)
        pbar.set_postfix({'MAE': mae/(step + 1)}, refresh=True)
        if env.done:
            break
    
    pbar.close()
    mae/=time_steps
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(a_list), np.array(adv_a_list), mae 

def eval_toggle_bifurcation_attack(agent, env, atk, time_steps:int=None, mask:list=None):
    """Evaluates an SB3 agent subject to utargeted observation perturbations generated by an ART evasion attack,
    model must have bifurcation wrapper (or two logits)"""
    obs_list = []
    adv_obs_list = []
    a_list = []
    adv_a_list = []
    asr = 0
    n_features = agent.observation_space.shape[0]

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(n_features) #1 for all features

    pbar = tqdm(total=time_steps)
    for step in tqdm(range(time_steps)):

        obs_list.append(observations)
        actions = agent.predict(observations, deterministic=True)
        a_list.append(actions[0])

        adv_obs = np.squeeze(
            atk.generate(np.expand_dims(observations, axis=0),
                         y=np.array([step%2]), #might need one-hot encoding, but I think ART handles that
                         mask=mask)
            )
        adv_a_list.append(a_adv[0])
        a_adv, _ = agent.predict(adv_obs, deterministic=True)
        if a_adv[0]!=actions[0]: #check if an adversarial example was crafted
            asr+=1
            adv_obs_list.append(adv_obs)
        else:
            adv_obs_list.append(np.array([np.nan]*n_features)) #same shape as observations

        observations, _, _, _ = env.step(a_adv)

        #update progress bar including asr
        pbar.update(1)
        pbar.set_postfix({'ASR': asr/(step + 1)}, refresh=True)
        if env.done:
            break
    
    pbar.close()
    #obs_list.append(observations) #ignore final obs as no action is taken
    #adv_obs_list.append(adv_obs)
    asr/=time_steps
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), np.array(a_list), np.array(adv_a_list), asr


def obs_heatmap(df1, df2, row_ids, title='Comparison of Clean and Adversarial Observations', fig_size:tuple=(50,10)):
    
    fig, axs = plt.subplots(1, 3, 
                            figsize=fig_size,
                            sharex=True, 
                            sharey=True,
                            )
    df_diff = (df1 - df2).abs()

    # Plot row from df1
    sns.heatmap(df1.iloc[row_ids], 
                ax=axs[0], 
                cmap='viridis',
                cbar=False)
    axs[0].set_title('Clean Observations')

    # Plot row from df2
    sns.heatmap(df2.iloc[row_ids], 
                ax=axs[1], 
                cmap='viridis',
                cbar=False
                )
    axs[1].set_title('Adversarial Observations')

    # Plot the difference
    sns.heatmap(df_diff.iloc[row_ids], 
                ax=axs[2], 
                cmap='viridis',
                vmin=0, 
                vmax=1,
                )
    axs[2].set_title('Absolute Difference')

    fig.text(0.5, -0.65, 'Features (min-max normalized)', ha='center', fontdict={'size':22})
    fig.text(0.1, 0.5, 'Time Step (hours)', va='center', rotation='vertical', fontdict={'size':22})
    fig.suptitle(title,
                 fontsize='x-large')
    plt.show()

def obs_heatmap_columns(df1, df2, row_ids, title='Comparison of Clean and Adversarial Observations', fig_size:tuple=(10,40)):

    fig, axs = plt.subplots(3, 1, 
                            figsize=fig_size,
                            sharex=True, 
                            sharey=True,
                            )
    df_diff = (df1 - df2).abs()

    # Plot row from df1
    sns.heatmap(df1.iloc[row_ids].T, 
                ax=axs[0], 
                cmap='viridis',
                cbar=False)
    axs[0].set_title('Clean Observations')

    # Plot row from df2
    sns.heatmap(df2.iloc[row_ids].T, 
                ax=axs[1], 
                cmap='viridis',
                cbar=False
                )
    axs[1].set_title('Adversarial Observations')

    # Plot the difference
    cax = sns.heatmap(df_diff.iloc[row_ids].T, 
                ax=axs[2], 
                cmap='viridis',
                vmin=0, 
                vmax=1,
                cbar_ax=fig.add_axes([0.92, 0.1, 0.02, 0.8])
                )
    axs[2].set_title('Absolute Difference')

    fig.text(-0.65, 0.5, 'Features (min-max normalized)', va='center', rotation='vertical', fontdict={'size':28})
    fig.text(0.5, 0.05, 'Time Step (hours)', ha='center', fontdict={'size':28})
    fig.suptitle(title,
                 fontsize='x-large')
    plt.show()


def eval_toggle_targeted_attack(agent, env, atk, time_steps:int=None, mask:list=None):
    """Evaluates an SB3 agent subject to untargeted observation perturbations generated by an ART evasion attack
    this attack expects a model with 2 outputs i.e. from a bifurcated attack, and alternates between 2 targets"""
    obs_list = []
    adv_obs_list = []
    asr = 0
    n_features = agent.observation_space.shape[0]

    observations = env.reset()
    if time_steps is None:
        time_steps = env.time_steps - 1
    if mask is None:
        mask=np.ones(n_features) #1 for all features

    pbar = tqdm(total=time_steps)
    for step in tqdm(range(time_steps)):

        obs_list.append(observations)
        actions = agent.predict(observations, deterministic=True)

        if step%2 == 0: #toggle target for even/odd ts
            target = np.array([0])
        else:
            target =  np.array([1])
        #adv_obs = np.expand_dims(observations, axis=0) #ART atks expect a 2d array
        adv_obs = np.squeeze(atk.generate(np.expand_dims(observations, axis=0),
                                            y=target, 
                                            mask=mask)
                                            )
        #adv_obs = np.squeeze(adv_obs) #CityLearn envs expect a 1d array
        
        a_adv, _ = agent.predict(adv_obs, deterministic=True)
        if a_adv[0]!=actions[0]: #check if an adversarial example was crafted
            asr+=1
            adv_obs_list.append(adv_obs)
        else:
            adv_obs_list.append(np.array([np.nan]*n_features)) #same shape as observations

        observations, _, _, _ = env.step(a_adv)

        #update progress bar including asr
        pbar.update(1)
        pbar.set_postfix({'ASR': asr/(step + 1)}, refresh=True)
        if env.done:
            break
    
    pbar.close()
    #obs_list.append(observations) #ignore final obs as no action is taken
    #adv_obs_list.append(adv_obs)
    asr/=time_steps
    return format_kpis(env), np.array(obs_list), np.array(adv_obs_list), asr

