import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from src.nets.actor import GNNActor
from src.nets.critic import GNNValue
from src.algos.reb_flow_solver import solveRebFlow
import os 
from tqdm import trange
import sys

from src.helperfunctions.skip_actor import skip_actor


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.0
args.log_interval = 10
#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, input_size, cfg, parser, train_dir, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device
        self.act_dim = env.nNodes
        
        self.actor = self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic = GNNValue(self.input_size, self.hidden_size)
        self.parser = parser
        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.train_dir = train_dir
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.train_mask = []
        self.to(self.device)
    
    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parser.parse_obs(obs).to(self.device)
        
        # actor: computes concentration parameters of a Dirichlet distribution
        action, log_prob = self.actor(x.x, x.edge_index)

        # critic: estimates V(s_t)
        value = self.critic(x)
        return action, log_prob, value
    
    def select_action(self, obs):
        action, log_prob, value = self.forward(obs)
        
        self.saved_actions.append(SavedAction(log_prob, value))
        return action.detach().cpu().numpy().squeeze()
    

    def assign_discrete_actions(self, total_agents, action_rl):
        desired_agent_dist = np.floor(action_rl * total_agents).astype(int)

        remaining_agents = total_agents - np.sum(desired_agent_dist)

        fractional_parts = (action_rl * total_agents) - desired_agent_dist
        top_indices = np.argpartition(-fractional_parts, int(remaining_agents))[:int(remaining_agents)]

        desired_agent_dist[top_indices] += 1
        return desired_agent_dist
    


    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R, mask in zip(saved_actions, returns, self.train_mask):
            if mask:
                advantage = R - value.item()

                # calculate actor (policy) loss 
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.train_mask[:]

        return [loss.item() for loss in value_losses], [loss.item() for loss in policy_losses]
    
    def learn(self, cfg):

        if cfg.model.load_from_ckeckpoint:
            self.load_checkpoint(path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}.pth"))
            print("last checkpoint loaded")

        train_episodes = cfg.model.max_episodes #set max number of training episodes
        T = cfg.model.max_steps #set episode length
        epochs = trange(train_episodes)     # epoch iterator
        best_reward = -np.inf   # set best reward
        self.train()   # set model in train mode
        self.num_tasks_finished_store = []
        
        for i_episode in epochs:
    
            self.episode_num_tasks_finished = 0
            episode_reward = 0
            
            obs, rew, _ = self.env.reset()
           
            #self.rewards.append(rew)
            for step in range(T):
                
                action_rl = self.select_action(obs)
                # action_rl = skip_actor(self.env, obs)

                total_agents = sum(obs["free_agents_per_node"])
                desired_agent_dist = self.assign_discrete_actions(total_agents, action_rl)

                reb_action = solveRebFlow(
                    self.env,
                    obs,
                    desired_agent_dist,
                    self.cplexpath,
                ) 
                action_dict = {"reb_action": reb_action}

                new_obs, reward_dict, done = self.env.step(action_dict)

                rew = reward_dict["A*-distance"] + reward_dict["idle-agents"]
                self.rewards.append(rew)
                episode_reward += rew

                self.episode_num_tasks_finished += reward_dict["task-finished"]

                # train mask
                if total_agents > 0:
                    self.train_mask.append(True)
                else:
                    self.train_mask.append(False)

                print(f"Episode {i_episode+1} | Step {step+1} | Free agents (before step): {sum(obs['free_agents_per_node'])} | A*-reward: {reward_dict['A*-distance']} | Idle agents reward: {reward_dict['idle-agents']} | Total reward: {rew} | Tasks finished (after step): {reward_dict["task-finished"]}")
                # print(f"\nxxxxxxxxxxxxxxxxxxx STEP {step} started xxxxxxxxxxxxxxxxxxx")

                # print(f"obs:     free_agents_per_node: {obs['free_agents_per_node']}")
                # print(f"new_obs: free_agents_per_node: {new_obs['free_agents_per_node']}")
                # print(f"free_tasks_per_node : {obs['free_tasks_per_node']}")
                # print(f"agents_per_node     : {obs['agents_per_node']}")  
                # print("action_rl (result from skip_actor): ", action_rl.tolist())
                # print(f"desired_agent_dist  : {desired_agent_dist.tolist()}")
                # for i in range(self.env.nNodes):
                #     for j in range(self.env.nNodes):
                #         if reb_action[(i,j)] > 0:
                #             print(f"reb_action node {i} to {j}: {reb_action[(i,j)]}")


                if done:
                    break
                obs = new_obs
                
            
            # perform on-policy backprop
            v_loss, p_loss = self.training_step()

            # Send current statistics to screen
            epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Tasks Finished: {self.episode_num_tasks_finished}")
            
            if cfg.model.wandb:
                self.wandb.log({"Reward": episode_reward, "Policy Loss": p_loss, "Value Loss": v_loss})

            self.num_tasks_finished_store.append(self.episode_num_tasks_finished)

            if cfg.model.tensorboard:
                self.tensorboard.add_scalar("Reward", episode_reward, i_episode)
                self.tensorboard.add_scalar("Tasks Finished", self.episode_num_tasks_finished, i_episode)
                self.tensorboard.add_scalar("Policy Loss", np.array(p_loss).mean(), i_episode)
                self.tensorboard.add_scalar("Value Loss", np.array(v_loss).mean(), i_episode)
            
            self.save_checkpoint(
                path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}.pth")
            )
            if episode_reward > best_reward: 
                best_reward = episode_reward
                self.save_checkpoint(
                    path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}_best.pth")
                )
        print("Average reward over all episodes: ", np.mean(self.num_tasks_finished_store))
    
    def test_agent(self, test_episodes, env, cplexpath, matching_steps, agent_name):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        episode_rebalanced_vehicles = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            eps_rebalancing_veh = 0
            actions = []
            done = False
            # Reset the environment
            obs = env.reset()   # initialize environment
            if self.sim == 'sumo':
                if self.env.scenario.is_meso:
                    try:
                        traci.simulationStep()
                    except Exception as e:
                        print(f"FatalTraCIError during initial step: {e}")
                        traci.close()
                        break
                while not done:
                    sumo_step = 0
                    try:
                        while sumo_step < matching_steps:
                            traci.simulationStep()
                            sumo_step += 1
                    except Exception as e:
                        print(f"FatalTraCIError during matching steps: {e}")
                        traci.close()
                        break
                obs, paxreward, done, info = env.pax_step(CPLEXPATH=cplexpath, PATH=f'scenario_lux/{agent_name}')
                eps_reward += paxreward

                o = self.parser.parse_obs(obs)

                action_rl = self.select_action(o, deterministic=True)
                actions.append(action_rl)

                desired_acc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + env.tstep)) for i in range(len(env.region))}

                reb_action = solveRebFlow(env, f'scenario_lux/{agent_name}', desired_acc, cplexpath)

                # Take action in environment
                try:
                    _, rebreward, done, info = env.reb_step(reb_action)
                except Exception as e:
                    print(f"FatalTraCIError during rebalancing step: {e}")
                    if self.sim == 'sumo':
                        traci.close()
                    break

                eps_reward += rebreward
                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
                eps_rebalancing_veh += info["rebalanced_vehicles"]
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            episode_rebalanced_vehicles.append(eps_rebalancing_veh)

            # stop episode if terminating conditions are met
            if done and self.sim == 'sumo':
                traci.close()

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )
    

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=1e-4)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=1e-4)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)
