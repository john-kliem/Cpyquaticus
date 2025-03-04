
# grimyRL is a modifed CleanRL Trainer to work with decentralized multi-agent training with different algorithms
# Algorithm code is taken from cleanRL and modifed in classes please see examples 
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
from multi_env.dict_based_sequential import SequentialMultiEnv
from algos.clean_ppo import PPO, PPOAgent, PPOArgs
from cpyquaticus.envs.c_pyquaticus import Cpyquaticus
from cpyquaticus.base_policies.BasePolicies import DoNothing, Random
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
    wandb_project_name: str = "cpyquaticus"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = 'cpyquaticus'#"CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    num_envs: int = 20
    """the number of parallel game environments"""
    num_steps: int = 6000#2400
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 16
    """the K epochs to update the policy"""
    
    #Agent Config
    possible_agents = {'agent_0':(PPO, PPOAgent, PPOArgs),'agent_1':(PPO, PPOAgent, PPOArgs)}#(DoNothing,None, None)}#'agent_1':(PPO, PPOAgent, PPOArgs)}
    """the agent ID and the policy you want to use"""
    """For gym envs use default agent ID"""
    to_train = ['agent_0', 'agent_1']
    """the agent IDs for policies that need to be trained"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def make_cpyquaticus_env():
    def thunk():
        env = Cpyquaticus(num_steps=600,c_load='linux')
        return env
    return thunk()


if __name__ == "__main__":

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agents = {}
    summary_writers = {}
    # env setup
    action_space = None
    if args.env_id == 'cpyquaticus':
        envs = SequentialMultiEnv([make_cpyquaticus_env for i in range(args.num_envs)])
        #Assign Algorithm Total Timesteps, num_envs, and steps to conform with trainer settings
        for a in args.possible_agents:
            run_name = f"{args.env_id}_{a}_{time.time()}"
            if a in args.to_train:
                args.possible_agents[a][2].total_timesteps = args.total_timesteps
                args.possible_agents[a][2].num_envs = args.num_envs
                args.possible_agents[a][2].num_steps = args.num_steps
                agents[a] = args.possible_agents[a][0](args.possible_agents[a][1], args.possible_agents[a][2], envs.envs[0].observation_space(a), envs.envs[0].action_space(a), device)
                summary_writers[a] = SummaryWriter(f"runs/{run_name}")
            else:
                agents[a] = args.possible_agents[a][0](envs.envs[0].observation_space(a), envs.envs[0].action_space(a), device)
            action_space = envs.envs[0].action_space(a)
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
        )
        for a in args.possible_agents:
            run_name = f"{args.env_id}_{a}_{time.time()}"
            args.possible_agents[a][2].total_timesteps = args.total_timesteps
            args.possible_agents[a][2].num_envs = args.num_envs
            args.possible_agents[a][2].num_steps = args.num_steps
            summary_writers[a] = SummaryWriter(f"runs/{run_name}")
            agents[a] = args.possible_agents[a][0](args.possible_agents[a][1], args.possible_agents[a][2], envs.single_observation_space, envs.single_action_space, device)
        action_space = envs.single_action_space
    assert isinstance(action_space, gym.spaces.Discrete), "only discrete action space is supported"
    for a in agents:
        if a in args.to_train:
            summary_writers[a].add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(agents[a].config).items()])),
            global_step=0, 
            )
    #TODO: Add device passthrough

    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    #next_obs = #torch.Tensor(next_obs)
    next_done = torch.zeros(args.num_envs)


    #TODO add next_done and next_obs to algo
    
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        for a in agents:
            agents[a].anneal_lr(iteration, args.num_iterations)
            
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # obs[step] = next_obs
            # dones[step] = next_done

            # ALGO LOGIC: action logic
            actions = []
            with torch.no_grad():
                agent_actions = []
                for ind,a in enumerate(agents):
                    if step == 0:
                        agents[a].buffer_step = 0
                    if isinstance(envs, gym.vector.SyncVectorEnv):
                        action, logprob, _, value = agents[a].get_action_and_value_train(next_obs, next_done)
                    else:
                        agent_next_obs  = torch.tensor([d[a] for d in next_obs if a in d])
                        action, logprob, _, value = agents[a].get_action_and_value_train(agent_next_obs, next_done)
                        agent_actions.append(action)            
            if isinstance(envs, gym.vector.SyncVectorEnv):
                actions = action
            else:
                actions = torch.stack(agent_actions).T


            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(actions.cpu().numpy())
            #TODO Update next_done call for correct shapes
            next_done = terminations
            for ind,a in enumerate(agents):
                if isinstance(envs, gym.vector.SyncVectorEnv):
                    agents[a].set_reward(reward)
                    next_done = np.logical_or(terminations, truncations)
                else:
                    agent_reward  = torch.tensor([d[a] for d in reward if a in d])
                    agents[a].set_reward(agent_reward)
                    agent_terms = torch.tensor([d[a] for d in terminations if a in d])
                    agent_truncs = torch.tensor([d[a] for d in truncations if a in d])
                    next_done = np.logical_or(agent_terms, agent_truncs)
            # next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(next_done)
            if True in next_done:
                for a in args.to_train:
                    agent_return = agents[a].rewards.sum(dim=0).reshape(1, args.num_envs)
                    #Average num envs
                    avg_agent_return = agent_return.mean(dim=1)
                    if isinstance(envs, gym.vector.SyncVectorEnv):
                        for info in infos['final_info']:
                            if info and 'episode' in info:
                                summary_writers[a].add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                                summary_writers[a].add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    else:
                        summary_writers[a].add_scalar("charts/episodic_return", avg_agent_return, global_step)
                    # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        # bootstrap value if not done
        for a in args.to_train:
            #TODO: Grab only specific agent next obs and next done
            if isinstance(envs, gym.vector.SyncVectorEnv):
                agent_next_obs = next_obs
                next_done = torch.tensor(next_done).float()
                agents[a].bootstrap(args.num_steps, next_done, next_done)
            else:
                agent_next_obs  = torch.tensor([d[a] for d in next_obs if a in d])
                agents[a].bootstrap(args.num_steps, agent_next_obs, next_done)
        
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # print("Epoch Training")
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                for a in args.to_train:
                    agents[a].train(mb_inds)

        for a in agents:
            if a in args.to_train:
                explained_var, v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = agents[a].get_plotting_vars()
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                summary_writers[a].add_scalar("charts/learning_rate", agents[a].optimizer.param_groups[0]["lr"], global_step)
                summary_writers[a].add_scalar("losses/value_loss", v_loss.item(), global_step)
                summary_writers[a].add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                summary_writers[a].add_scalar("losses/entropy", entropy_loss.item(), global_step)
                summary_writers[a].add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                summary_writers[a].add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                summary_writers[a].add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                summary_writers[a].add_scalar("losses/explained_variance", explained_var, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                summary_writers[a].add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if iteration % 100 == 0:
           for a in agents:
               if a in args.to_train:
                   agents[a].save(f'./CleanMulti/run_2_{a}_{global_step}')
    envs.close()
    for a in args.to_train:
        summary_writers[a].close()
    for a in agents:
        if a in args.to_train:
            agents[a].save(f'./CleanMulti/run_2_{a}_{args.total_timesteps}')
    
