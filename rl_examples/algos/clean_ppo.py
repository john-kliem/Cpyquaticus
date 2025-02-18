import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

@dataclass
class PPOArgs:
    total_timesteps: int = None
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = None
    """the number of parallel game environments"""
    num_steps: int = None
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

# Agent Class

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPO:
    def __init__(self, agent,config, observation_space, action_space, device):
        self.device = device
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.network = agent(observation_space, action_space)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate, eps=1e-5)

        #Create Buffers Used For Training
        #print("shapes: ", self.config.num_steps)
        self.observations = torch.zeros((self.config.num_steps, self.config.num_envs)+self.observation_space.shape).to(device)
        self.actions = torch.zeros((self.config.num_steps, self.config.num_envs) + self.action_space.shape).to(device)
        self.logprobs = torch.zeros((self.config.num_steps, self.config.num_envs)).to(device)
        self.rewards = torch.zeros((self.config.num_steps, self.config.num_envs)).to(device)
        self.dones = torch.zeros((self.config.num_steps, self.config.num_envs)).to(device)
        self.values = torch.zeros((self.config.num_steps, self.config.num_envs)).to(device) 
        self.buffer_step = 0
        
        #Start Game Buffers
        self.returns = None
        self.advantages = None

        #Vars Used in Tracking
        self.explained_var = np.nan
        self.v_loss = np.nan
        self.pg_loss = np.nan
        self.entropy_loss = np.nan
        self.old_approx_kl = np.nan
        self.approx_kl = np.nan
        self.clipfracs = np.nan

    def get_value(self, x):
        return self.network.get_value(x)
    def get_action_and_value(self, obs, action=None):
        return self.network.get_action_and_value(obs, action)
    def get_action_and_value_train(self, obs, done, action=None):
        #TODO: Might need to flatten x before passing into get_action_and_value
        obs = torch.tensor(obs).to(self.device)
        self.observations[self.buffer_step] = obs
        self.dones[self.buffer_step] = torch.tensor(done).to(self.device)
        action, logprob, entropy, value = self.get_action_and_value(obs, action)
        self.actions[self.buffer_step] = action
        self.logprobs[self.buffer_step] = logprob
        self.values[self.buffer_step] = value.flatten()
        return action, logprob, entropy, value
    def set_reward(self, reward):
        self.rewards[self.buffer_step] = torch.tensor(reward).to(self.device).view(-1)
        self.buffer_step += 1
        return
    def anneal_lr(self, iteration, num_iterations):
        if self.config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * self.config.learning_rate
            self.optimizer.param_groups[0]['lr'] = lrnow
        return
    def bootstrap(self, num_steps, next_obs, next_done):
        with torch.no_grad():
            next_obs = torch.tensor(next_obs).to(self.device)
            #next_done = torch.tensor(next_done).to(self.device)
            next_value = self.get_value(next_obs).reshape(1, -1)
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            self.returns = self.advantages + self.values
        return
    def train(self, batch):
        #TODO: Understand Bootstrap
        if self.returns == None:
            assert "Bootstrap Value First"
        #assert isinstance(self.returns, None), "Bootstrap Value First"
        # Flatten the batch
        b_obs = self.observations.reshape((-1,) + self.observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_space.shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        _, newlogprob, entropy, newvalue = self.network.get_action_and_value(b_obs[batch], b_actions.long()[batch])
        logratio = newlogprob - b_logprobs[batch]
        ratio = logratio.exp()
        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio-1.0).abs() > self.config.clip_coef).float().mean().item()]
        mb_advantages = b_advantages[batch]
        if self.config.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        # Policy Loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-self.config.clip_coef, 1+self.config.clip_coef)
        
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value Loss
        newvalue = newvalue.view(-1)
        if self.config.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[batch])**2
            v_clipped = b_values[batch] + torch.clamp(
                    newvalue - b_values[batch],
                    -self.config.clip_coef,
                    self.config.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns[batch]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[batch]) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        #Save variables for plotting
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        self.explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self.v_loss = v_loss
        self.pg_loss = pg_loss
        self.entropy_loss = entropy_loss
        self.old_approx_kl = old_approx_kl
        self.approx_kl = approx_kl
        self.clipfracs = clipfracs
        return 
    def get_plotting_vars(self, ):
        return self.explained_var, self.v_loss, self.pg_loss, self.entropy_loss, self.old_approx_kl, self.approx_kl, self.clipfracs
    def save(self, path):
        torch.save(self.network.state_dict(), path)
    def load(self, path):
        self.network.load_state_dict(torch.load(path,map_location=self.device))
