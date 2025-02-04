from faster_envs import Cpyquaticus
import time
import argparse
import gymnasium as gym
import numpy as np
import os
import logging

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
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 2400
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class DoNothing:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
    def get_action_and_value(self, x):
        length = x.shape[0]
        actions = torch.ones((length,))*17
        return actions, None, None, None
class RandomAction:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
    def get_action_and_value(self, x):
        length = x.shape[0]
        actions = torch.ones((length,))*action_space.sample()
        return action, None, None, None


def make_env():
    def thunk():
        env = Cpyquaticus(c_load='linux')
        return env

    return thunk

#def batch

from sequential import SequentialMultiEnv
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    vec_envs = SequentialMultiEnv([make_env() for i in range(args.num_envs)])
    num_agents = len(vec_envs.possible_agents)
    agents = []
    optimizers = []
    obs_shape = []
    act_shape = []
    for a in vec_envs.possible_agents:
        obs_space = vec_envs.observation_spaces[a]
        obs_shape.append(obs_space.sample())
        act_space = vec_envs.action_spaces[a]
        act_shape.append(np.array([act_space.sample(), act_space.sample()]))
        if a == 'agent_0':
            agents.append(Agent(obs_space, act_space).to(device))
            optimizers.append(optim.Adam(agents[-1].parameters(), lr=args.learning_rate, eps=1e-5))
        else:
            agents.append(DoNothing(obs_space, act_space).to(device))
        #optimizers.append(optim.Adam(agents[-1].parameters(), lr=args.learning_rate, eps=1e-5))
    
    obs_shape = np.array(obs_shape)
    act_shape = np.array(act_shape)
    #print("Action Space: ", act_space)
    # ALGO Logic: Storage Setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    #print("obs shape: ",obs.shape)
    #print("Act shape: ",actions.shape)
    #Convert for multi-agents
    logprobs = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    #print("Values: ", values.shape)
    #print("Dones: ", dones.shape)
    #print("Rewards: ", rewards.shape)
    #print("logprobs: ", logprobs.shape)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = vec_envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_agents = len(vec_envs.possible_agents)

    for iteration in range(1, args.num_iterations + 1):
        print("Iteration: ", iteration)
        # Annealing the rate if instructed to do so.
        for i, a in enumerate(vec_envs.possible_agents):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizers[i].param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            #print("Global Step: ", global_step, " Out of ", args.total_timesteps)
            global_step += args.num_envs
            #print("Next Obs Shape: ", next_obs.shape)
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            acts = []
            lgps = []
            vals = []
            for i, a in enumerate(vec_envs.possible_agents):
                with torch.no_grad():
                    #print("next Obs shape: ", next_obs.shape)
                    action, logprob, _, value = agents[i].get_action_and_value(next_obs[:,i,:])
                    #print("Action: ", action)
                    
                    acts.append(action)
                    if isinstance(a, Agent):
                        lgps.append(logprob)
                        vals.append(torch.tensor(value.flatten()))

            #print("Vals: ", vals, " my shape: ", vals.shape)
            #print(" values shape: ", torch.stack(vals))
            #print()
            values[step] = torch.stack(vals).T
            #print("Actions shape: ", actions[step].shape)
            actions[step] = torch.stack(acts).T
            logprobs[step] = torch.stack(lgps).T
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = vec_envs.step(actions[step].cpu().numpy())
            
            next_done = np.logical_or(terminations[:,0], truncations[:,0]).astype(int)
            #print("Next Done: ", next_done)
            rewards[step] = torch.tensor(reward).to(device)#.view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
#                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
 #                       writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        for ind,a in enumerate(agents):
            if isinstance(a, RandomAction) or isinstance(a, DoNothing):
                continue
            with torch.no_grad():
                next_value = a.get_value(next_obs[:,ind]).reshape(1, -1)
                advantages = torch.zeros_like(rewards[:,:,ind]).to(device)
                #print("\n\n\n\n\n")
                #print("rewards Shape: ", rewards.shape)
                #print("next_value", next_value.shape)
                #print("Dones: ", dones[t+1])
                #print("Rewards at t: ", rewards[t])
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1,:,ind]
                    #print("Values Shape: ", values[t].shape, " Vals: ", values[t,:,ind])
                    #print("Rewards shape: ", rewards[t].shape, " vals rew ", rewards[t,:,ind])
                    #print("nextvalues: ", nextvalues.shape, " vals: ", nextvalues)
                    #print("nextnonterminal shape: ", nextnonterminal.shape, " vals ", nextnonterminal)
                    delta = rewards[t, :,ind] + args.gamma * nextvalues * nextnonterminal - values[t,:,ind]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                
                returns = advantages + values[t,:,ind]

            # flatten the batch
            b_obs = obs[:,:,ind].reshape((-1,) + vec_envs.envs[0].observation_space('agent_'+str(i)).shape)
            b_logprobs = logprobs[:,:,ind].reshape(-1)
            b_actions = actions[:,:,ind].reshape((-1, 1)) #+ vec_envs.envs[0].action_space('agent_'+str(i)).shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            #print("Values Shape: ", values.shape)
            b_values = values[:,:,ind].reshape(-1)
            #print("B_values: ",b_values.shape)
           # print("b_obs: ", b_obs.shape)
           # print("b_obs: ", b_obs[0])
           # print("b_actions: ", b_actions.shape)
           # print("b_actions: ", b_actions)
           # print("Actions.: ", actions.shape)
           # print("actions: ", actions)
            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = a.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(a.parameters(), args.max_grad_norm)
                    optimizers[i].step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            #if True:
            #    print("Saving Checkpoint")

            #os.mkdir('./clean_checkpoint_'+str(global_step))
             #   for i,a in enumerate(agents):

              #      torch.save(a.state_dict(), './agent_'+str(i)+'_'+str(global_step/100000)+'.pt')

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        if iteration % 5 == 0 or iteration == args.num_iterations-1:
            for i, a in enumerate(agents):
                torch.save(a.state_dict(), './agent_'+str(i)+'_'+str(iteration)+'.pt')

        #print("Iteration: ", iteration, "/", args.num_iterations)
        #print("Global Steps: ", global_step)
        #if True:
        #    print("Saving Checkpoint")
            
            #os.mkdir('./clean_checkpoint_'+str(global_step))
 #           for i,a in enumerate(agents):
#
  #              torch.save(a.state_dict(), './agent_'+str(i)+'_'+str(global_step/100000)+'.pt')

