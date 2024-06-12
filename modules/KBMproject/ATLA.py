import gym
import numpy as np
from typing import Any, Callable
from numpy.distutils.misc_util import is_sequence

from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from copy import deepcopy

    
class BReplace():
    def __init__(self,atla_instance) -> None:
        self.min = atla_instance.env.observation_space.low
        self.max = atla_instance.env.observation_space.high

    def __call__(self, atla_instance, perturbation:np.ndarray):
        #NB perturbation must be the same shape as the observation
        return np.clip(perturbation,self.min,self.max)


class BSum():
    """Adds a pretubation to the current observation
    mask is an array of 0 or 1 values, )s mean a feature is not perturbed"""
    def __init__(self,atla_instance) -> None:
        self.min = atla_instance.env.observation_space.low
        self.max = atla_instance.env.observation_space.high
        self.mask = atla_instance.mask

    def __call__(self, atla_instance, perturbation:np.ndarray): #the instance will be self inside the adversary ATLA wrapper
        obs = atla_instance.obs_list[-1]
        obs[self.mask] += perturbation
        return np.clip(obs,self.min,self.max)
    

class BScaledSum():
    """Adds a pretubation to the current observation
    mask is an array of 0 or 1 values, )s mean a feature is not perturbed"""
    def __init__(self,atla_instance,max_perturbation:np.ndarray) -> None:
        self.obs_min = atla_instance.env.observation_space.low
        self.obs_max = atla_instance.env.observation_space.high
        self.mask = atla_instance.mask
        self.max_perturbation = max_perturbation

    def __call__(self, atla_instance, action:np.ndarray): #the instance will be self inside the adversary ATLA wrapper
        obs = atla_instance.obs_list[-1]
        obs[self.mask] += action*self.max_perturbation
        return np.clip(obs,self.obs_min,self.obs_max)
    

class BSumPrevProj():
    """clips the obs + perturbation within a certain distance of the previous observation"""
    def __init__(self,atla_instance,clip_bound:np.ndarray,) -> None:
        self.mask = atla_instance.mask
        self.clip_bound = clip_bound
        self.obs_min = atla_instance.env.observation_space.low
        self.obs_max = atla_instance.env.observation_space.high

    def __call__(self, atla_instance, perturbation:np.ndarray): #the instance will be self inside the adversary ATLA wrapper
        """atla_instance must have an obs_list attribute"""
        obs = atla_instance.obs_list[-1]
        if len(atla_instance.obs_list) > 1: #use try except instead?
            prev_obs = atla_instance.obs_list[-2] #not defined for first step!
        else:
            prev_obs = obs
        # what if the difference between prev obs and obs is outide the boundary?
        prev_clip_min = prev_obs - self.clip_bound
        prev_clip_max = prev_obs + self.clip_bound
        min_diff = np.maximum(np.minimum(prev_clip_min, obs), #use obs value if it's outside the clip range of the last obs
                              self.obs_min) #stay in observation space
        
        max_diff = np.minimum(np.maximum(prev_clip_max, obs),
                              self.obs_max) #stay in observation space
        #apply anc clip perturbation
        obs[self.mask] += perturbation
        return np.clip(obs,min_diff,max_diff)
    

class BScaledSumPrevProj():
    """clips the obs + perturbation within a certain distance of the previous observation
    for an agent with an action space in [-1,1] for each feature (symmectrical action space)
    The user provides a max perturbation which is scaled by the action"""
    def __init__(self,atla_instance,clip_bound:np.ndarray,max_perturbation:np.ndarray) -> None:
        self.clip_bound = clip_bound
        self.max_perturbation = max_perturbation
        self.mask = atla_instance.mask
        self.obs_min = atla_instance.env.observation_space.low
        self.obs_max = atla_instance.env.observation_space.high
        #TODO confirm env action space is [-1,1]

    def __call__(self, atla_instance, action:np.ndarray): #the instance will be self inside the adversary ATLA wrapper
        """atla_instance must have an obs_list attribute"""
        obs = atla_instance.obs_list[-1]
        if len(atla_instance.obs_list) > 1: #replace with try except?
            prev_obs = atla_instance.obs_list[-2] #not defined for first step!
        else:
            prev_obs = obs
        #boundries for adv obs based on previous observation, so the change is in the normal distribution
        prev_clip_min = prev_obs - self.clip_bound
        prev_clip_max = prev_obs + self.clip_bound
        min_diff = np.maximum(np.minimum(prev_clip_min, obs), #use obs value if it's outside the clip range of the last obs
                              self.obs_min) #stay in observation space
        
        max_diff = np.minimum(np.maximum(prev_clip_max, obs),
                              self.obs_max) #stay in observation space
        #apply perturbation
        perturbation = action*self.max_perturbation
        obs[self.mask] += perturbation
        #clip perturbation
        return np.clip(obs,min_diff,max_diff)

#****************************************Perturbation Functions*****************************************************

class sb3_perturbation():
    def __init__(self,adversary, B:Callable=None) -> None:
        self.adversary = adversary
        if B is None: #extract B from adversary's environment
            B = adversary.get_env().get_attr('B')[0]
        self.B = B
        self.obs_list = []
    
    def __call__(self, obs) -> np.ndarray:
        self.obs_list.append(obs)
        adv_obs, _ = self.adversary.predict(obs, deterministic=True)
        return self.B(self, adv_obs)
    
    #does this need a reset method for the obs list?

class summed_perturbation():
    def __init__(self, perturbation_func, obs_space, perturbation_kwargs:dict={}) -> None:
        self.perturbation = perturbation_func
        self.kwargs = perturbation_kwargs
        self.low = obs_space.low
        self.high = obs_space.high

    def __call__(self, obs) -> Any:
        kwargs = self.kwargs
        return np.clip(obs + self.perturbation(**kwargs).astype(obs.dtype),
                       self.low,
                       self.high)
    
class ARTperturbation():
    def __init__(self, art_attack, generate_kwargs) -> None:
        self.attack = art_attack
        self.kwargs = generate_kwargs

    def __call__(self, obs) -> Any:
        kwargs = self.kwargs
        adv_obs = np.expand_dims(obs, axis=0)
        adv_obs = self.attack.generate(adv_obs, **kwargs)
        adv_obs = np.squeeze(adv_obs)
        return adv_obs

#*******************************************Define adversarial rewards ************************************************

def neg_reward(r, adv_obs, obs):
            return -r  

class NormScaleReward():
    """scales the default reward by the distance between the original and adversarial observations
    r=-r*(1-dist/max_dist)**exp
    inputs: gym environment, order for calculating the distance, exponent for the scaling:float"""
    def __init__(self, env, ord, exp:float=1) -> None:
        self.ord = ord
        self.exp = exp #could this be scheduled, start small and increase as the training budget is expended?
        self.max_norm = float(np.linalg.norm(env.observation_space.high - env.observation_space.low,
                                      ord=self.ord,
                                      )) #cast to float, otherwise it's a numpy object which breaks the logger in summary.py

    def __call__(self, r, adv_obs, obs):
         #Lp distance between clean and adv observations
         norm_factor = np.linalg.norm(obs - adv_obs,
                                      ord=self.ord,
                                      #axis=1,
                                      )
         norm_factor = min(1, norm_factor/self.max_norm) #scale to [0,1]
         # subtracting the factor from 1 means the reward is larger when the norm/distance is smaller
         norm_factor = (1 - norm_factor)**self.exp #exponent changes the influence of the norm on the reward
         return -r*norm_factor

#********************************************Define custom wrappers****************************************************************
#gym source for wrappers: https://github.com/openai/gym/blob/master/gym/core.py#L213

class AdversaryATLAWrapper(gym.Wrapper): #should this combine separate reward and action wrappers?, No, need the obs to calculate the reward
    """Wrapps an environment for the ATLA advserary
    modifies the step function
    inputs: SB3 agent, function for generating rewards for the adversary
    from the environment's reward
    B is a function which maps adversarial actions to the perturbation space
    the feature mask is an array of indices which corespond to features the adversary can modify"""
    def __init__(self, env, victim, adv_reward:Callable=neg_reward, ord=np.inf,B:Callable=BReplace, 
                 action_space=None,feature_mask:np.ndarray=None, B_kwargs:dict={}):
        super().__init__(env)
        
        self.victim = victim
        self.adv_reward = adv_reward
        self.ord = ord
        self.obs_list = []
        #TODO check mask
        #should the feature masked be a list of indexes which can be perturbed?
        #length equal to the diff between the adv action space and observation spaces
        #values must be smaller than the length of the action space
        if feature_mask is None:
            self.mask = np.arange(0,self.env.observation_space.shape[0])#np.ones(self.env.observation_space.shape)
        else:
            self.mask = feature_mask
        if action_space is None:
             #assumes box action space
             #action_space = self.env.observation_space
            action_space = gym.spaces.Box(
                low=self.env.observation_space.low[feature_mask],
                high=self.env.observation_space.high[feature_mask],
                dtype=self.env.observation_space.high.dtype
            )
        self.action_space = action_space
        kwargs = B_kwargs
        self.B = B(self, **kwargs)
        
        
        #assert victim.action_space == env.action_space, "Victim and Environment have mismatched action spaces!" #no need to check when masked
        assert victim.observation_space == env.observation_space, "Victim and Environment have mismatched observation spaces!"

    def step(self, adv_obs:np.array):
        """modifies the step method to take an adversarial observation
        and provide the corresponding action from the victim to the env
        then modifies the resulting reward with the adversarial reward
        function"""


        adv_obs = self.B(self,adv_obs)
        action = self.victim.predict(adv_obs, deterministic=True)
        obs, reward, done, info = self.env.step(action)
        self.obs_list.append(obs)
        if is_sequence(reward): #sometimes step returns an array, but not consistently, changes during episode
            reward = reward[0]
        reward = self.adv_reward(reward, adv_obs, self.obs_list[-2]) 
        self.perturbation_size = self.calculate_adv_dist_metric(adv_obs)
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """saves the initial observation for reward calculation in step
        as a workaround because env.observations does not return 
        normalized values"""
        self.obs_list = []
        obs = super().reset(**kwargs)
        self.obs_list.append(obs)
        return self.obs_list[-1]
    
    def calculate_adv_dist_metric(self, adv_obs):
         """size of the perturbations in Lp space"""
         return np.linalg.norm(self.obs_list[-2] - adv_obs, self.ord)

class VictimATLAWrapper(gym.Wrapper): 
    """wrapps an environment for the ATLA victim
    inputs: function for generating advsersarial observations and associated kwargs
    modifies the env step function to return adversarial observations
    The feature mask must match that of the adversary, as only some features are modified"""
    def __init__(self, env, obs_perturb_func:Callable, obs_perturb_kwargs:dict={}, feature_mask:np.ndarray=None, ):
        super().__init__(env)

        self.perturbation = obs_perturb_func
        self.perturb_kwargs = obs_perturb_kwargs

    def step(self, action): 
        """Modifies the step method to return observations
        with adversarial perturbations"""
        obs, reward, done, info = self.env.step(action)
        
        kwargs = self.perturb_kwargs
        adv_obs = self.perturbation(obs, **kwargs)

        return adv_obs, reward, done, info

#****************************************define custom callbacks *************************************************

class AdvDistanceTensorboardCallback(BaseCallback): # why does this log every value at the end of the episode?
     """Logs the adversarial perturbation size during training using TesnorBoard"""
     def __init__(self, verbose=2):
        super().__init__(verbose)
        self.perturbation_sizes_list = []

     def _on_step(self):
        """Stores the L2 perturbation nom on each step"""
        self.perturbation_sizes_list.append(self.training_env.get_attr('perturbation_size')[0])
        return True
     
     def _on_rollout_end(self):
        """Logs the mean perturbation size and resets the list of sizes"""
        self.logger.record('mean_perturbation_size', np.mean(self.perturbation_sizes_list))
        self.perturbation_sizes_list = []



class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    Ref: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    """
    def __init__(self, hparam_keys:list=['gamma','learning_rate',], policy_keys:list=None, verbose=2):
        super().__init__(verbose)
        self.hparam_keys = hparam_keys
        self.policy_keys = policy_keys

    def _on_training_start(self) -> None:
        hparam_dict = vars(self.model) #All attributes
        hparam_dict = {k: hparam_dict[k] for k in self.hparam_keys if k in hparam_dict} #selection of attributes
        hparam_dict['algorithm'] = self.model.__class__.__name__
        if 'use_sde' in vars(self.model.policy):
            if self.model.policy.use_sde:
                hparam_dict.update(self.model.policy.dist_kwargs)
        metric_dict = {
            "rollout/ep_rew_mean": 0.0,
            "eval/mean_reward": 0.0,
        }            
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict,),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class NormRwdHParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    Ref: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    includes unique parameters for the norm scaled reward
    """
    def __init__(self, hparam_keys:list=['gamma','learning_rate'], verbose=2):
        super().__init__(verbose)
        self.hparam_keys = hparam_keys

    def _on_training_start(self) -> None:
        hparam_dict = vars(self.model) #All attributes
        hparam_dict = {k: hparam_dict[k] for k in self.hparam_keys if k in hparam_dict} #selection of attributes
        hparam_dict['algorithm'] = self.model.__class__.__name__

        if self.model.get_env().env_is_wrapped(AdversaryATLAWrapper): #returns true for victim wrapper...
            adv_reward_dict = deepcopy(vars(self.model.env.get_attr('adv_reward')[0])) #attirbutes for reward function
            adv_reward_dict['ADV Dist Norm'] = deepcopy(self.model.env.get_attr('ord')[0])
            adv_reward_dict['name'] = self.model.env.get_attr('adv_reward')[0].__class__.__name__
            for key, value in adv_reward_dict.items():
                if value==np.inf:
                    adv_reward_dict[key] = str(value)
            adv_reward_dict = {'adversarial reward ' + key: value for key, value in adv_reward_dict.items()} #add prefix
            hparam_dict.update(adv_reward_dict) #combine
        #define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        #Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_rew_mean": 0.0,
        }
        if self.model.get_env().env_is_wrapped(AdversaryATLAWrapper):
            metric_dict['mean_perturbation_size'] = 0.0
            
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict,),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class PauseOnStepCallback(BaseCallback):
    def __init__(self, max_ts: int):
        super().__init__()
        self.max_ts = max_ts

    def _on_step(self) -> bool:
        # pause learninging after reaching max steps
        return self.num_timesteps < self.max_ts
    
class PeriodicPauseCallback(BaseCallback):
    """ Repeatedly pauses training after a specified number of timesteps"""
    def __init__(self, train_ts: int):
        super().__init__()
        self.pause_ts = self.train_ts = train_ts

    def _on_step(self) -> bool:
        # pause learning after the training period elapses
        keep_training = self.num_timesteps < self.pause_ts
        if not keep_training:
            self.pause_ts += self.train_ts
        return keep_training
