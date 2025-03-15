import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from src.algos.reb_flow_solver import solveRebFlow
from src.nets.actor import GNNActorTD3
from src.nets.critic import GNNCriticTD3
import random
from tqdm import trange
import os
import sys
import time
from src.helperfunctions.skip_actor import skip_actor


#import concurrent.futures
#import threading

from copy import deepcopy
from joblib import Parallel, delayed

class timer:
    def __init__(self):
        self.now = time.time()
        self.outerLoop = 0
        self.selectAction = 0
        self.solveReb = 0
        self.step = 0
        self.rest = 0
        self.learning = 0

    def addTime(self):
        ret = time.time() - self.now
        self.now = time.time()
        return ret
    
    def printAvgTimes(self, iEpisode):
        print(f"\n Times: Episode {iEpisode} | Avg outer-loop: {self.outerLoop/iEpisode:.2f} | Avg select-action: {self.selectAction/iEpisode:.2f} | Avg solve-reb: {self.solveReb/iEpisode:.2f} | Avg step: {self.step/iEpisode:.2f} | Avg rest: {self.rest/iEpisode:.2f} | Avg learning: {self.learning/iEpisode:.2f}")
   

class PairData(Data):
    """
    Store 2 graphs in one Data object (s_t and s_t+1)
    """

    def __init__(self, edge_index_s=None, x_s=None, reward=None, action=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """
    Replay buffer for SAC agents
    """

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(self, data1, action, reward, data2):
        self.data_list.append(PairData(data1.edge_index, data1.x, torch.as_tensor(
            reward), torch.as_tensor(action), data2.edge_index, data2.x))
        self.rewards.append(reward)

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32, norm=False):
        data = random.sample(self.data_list, batch_size)
        if norm:
            mean = np.mean(self.rewards)
            std = np.std(self.rewards)
            batch = Batch.from_data_list(data, follow_batch=['x_s', 'x_t'])
            batch.reward = (batch.reward-mean)/(std + 1e-16)
            return batch.to(self.device)
        else:
            return Batch.from_data_list(data, follow_batch=['x_s', 'x_t']).to(self.device)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant

#########################################
############## TD3 AGENT ################
#########################################
class TD3(nn.Module):
    def __init__(
        self,
        env,
        input_size,
        cfg, 
        parser,
        train_dir,
        device=torch.device("cpu"),
    ):

        super(TD3, self).__init__()
        self.env = env
        self.eps = np.finfo(np.float32).eps.item(),
        self.input_size = input_size
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nNodes

        self.parser = parser

        # TD3 parameters
        self.max_action = 1.0
        self.min_action = 0.0 + 1e-4
        self.discount = 0.99
        self.tau = 0.1 # 0.1
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 1
        self.lr = 1.00e-3
        self.l2 = 1e-2
        # self.grad_clip = 100.0

        # Replay buffer
        self.replay_buffer = ReplayData(device=device)

        # Networks
        self.actor = GNNActorTD3(self.input_size, self.hidden_size, act_dim=self.act_dim, layer_norm=cfg.actor_layer_norm)
        self.critic_1 = GNNCriticTD3(self.input_size, self.hidden_size, act_dim=self.act_dim, layer_norm=cfg.q_layer_norm)
        self.critic_2 = GNNCriticTD3(self.input_size, self.hidden_size, act_dim=self.act_dim, layer_norm=cfg.q_layer_norm)
        
       

        self.actor_target = deepcopy(self.actor)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)

        for p in self.critic_1_target.parameters():
            p.requires_grad = False
        for p in self.critic_2_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr, weight_decay=self.l2)
        self.critic_1_optimizer = torch.optim.AdamW(self.critic_1.parameters(), lr=self.lr, weight_decay=self.l2)
        self.critic_2_optimizer = torch.optim.AdamW(self.critic_2.parameters(), lr=self.lr, weight_decay=self.l2)

        # Other
        self.directory = cfg.directory
        self.agent_name = cfg.agent_name
        self.cplexpath = cfg.cplexpath
        self.train_dir = train_dir

        self.entropy_factor = cfg.entropy_factor

        self.total_it = 0

    def select_action(self, data, deterministic=True):
        with torch.no_grad():
            a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return a
    
    def assign_discrete_actions(self, total_agents, action_rl):
        desired_agent_dist = np.floor(action_rl * total_agents).astype(int)

        remaining_agents = total_agents - np.sum(desired_agent_dist)

        fractional_parts = (action_rl * total_agents) - desired_agent_dist
        top_indices = np.argpartition(-fractional_parts, int(remaining_agents))[:int(remaining_agents)]

        desired_agent_dist[top_indices] += 1
        return desired_agent_dist

    def update(self, data):
        self.total_it += 1

        (
            state_batch,
            edge_index,
            next_state_batch,
            edge_index2,
            reward_batch,
            action_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.x_t,
            data.edge_index_t,
            data.reward,
            data.action.reshape(-1, self.act_dim),
        )

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action_batch) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state_batch, edge_index2, True)[0]
            next_action = (next_action + noise).clamp(self.min_action, self.max_action)
            next_action = next_action / next_action.sum(dim=-1, keepdim=True)

            # Compute the target Q value
            target_Q1 = self.critic_1_target(next_state_batch, edge_index2, next_action) 
            target_Q2 = self.critic_2_target(next_state_batch, edge_index2, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state_batch, edge_index, action_batch)
        current_Q2 = self.critic_2(state_batch, edge_index, action_batch)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        critic_loss.backward()

        # with torch.autograd.detect_anomaly():
        #         critic_loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        # torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)

        # for name, param in self.critic_1.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"Critic 1.")
        #             print(f"Gradient of {name} is nan.")
        #             print(param.grad)
        
        # for name, param in self.critic_2.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"Critic 2.")
        #             print(f"Gradient of {name} is nan.")
        #             print(param.grad)

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        self.LogCriticLoss.append(critic_loss.item())

        # Delayed policy updates
        if (self.total_it % self.policy_freq == 0):

            # Compute actor loss
            actor_action = self.actor(state_batch, edge_index, True)[0]
            q_loss = -self.critic_1(state_batch, edge_index, actor_action).mean() 
            if self.entropy_factor == 0:
                actor_loss = q_loss
            else:
                raise NotImplementedError("Entropy factor not implemented yet.")
                actor_entropy = (actor_action * actor_action.log()).sum(dim=-1)
                actor_loss = q_loss + self.entropy_factor*actor_entropy.mean()

            # actor_loss = -self.critic_1(state_batch, edge_index, self.actor(state_batch, edge_index, True)[0]).mean() 
            # - self.actor(state_batch, edge_index, True)[0].log().mean()
            # actor_loss = -self.actor(state_batch, edge_index, True)[0].log().mean()

            # actor_action = self.actor(state_batch, edge_index, True)[0]
            # actor_entropy = - actor_action * actor_action.log()
            # actor_loss = - self.critic_1(state_batch, edge_index, actor_action).mean() - actor_entropy.mean()

            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # with torch.autograd.detect_anomaly():
            #     actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

            # for name, param in self.actor.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any():
            #             print(f"Actor.")
            #             print(f"Gradient of {name} is nan.")
            #             print(param.grad)

            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.LogPolicyLoss.append(actor_loss.item())

    def learn(self, cfg):
        
        train_episodes = cfg.model.max_episodes  # set max number of training episodes
        epochs = trange(train_episodes)  # epoch iterator
        best_reward = -np.inf  # set best reward
        self.train()  # set model in train mode
        myTimer = timer()
        self.LogPolicyLoss = []
        self.LogCriticLoss = []
    

        for i_episode in epochs:

         
            obs, rew, _ = self.env.reset()  # initialize environment
            
            obs_parsed = self.parser.parse_obs(obs).to(self.device)
            episode_reward = 0
            episode_reward += rew
            episode_num_tasks_finished = 0
            task_search_durations = []
            task_distances = []

            done = False

            myTimer.outerLoop += myTimer.addTime()

            while not done:
                # actor step
                print("free agents per node", obs["free_agents_per_node"])
                action_rl = self.select_action(obs_parsed)
                # action_rl = skip_actor(self.env, obs)
                
                # create discrete action distribution
                total_agents = sum(obs["free_agents_per_node"])
                desired_agent_dist = self.assign_discrete_actions(total_agents, action_rl)
                myTimer.selectAction += myTimer.addTime()
                
                # solve rebalancing
                reb_action = solveRebFlow(
                    self.env,
                    obs,
                    desired_agent_dist,
                    self.cplexpath,
                )
                action_dict = {"reb_action": reb_action}
                myTimer.solveReb += myTimer.addTime()

                # step
                new_obs, reward_dict, done, info = self.env.step(action_dict)
                myTimer.step += myTimer.addTime()

                # reward
                rew = reward_dict["A*-distance"] + reward_dict["idle-agents"]
                
                # store in replay buffer
                new_obs_parsed = self.parser.parse_obs(new_obs).to(self.device)
                if not cfg.model.mask_impactless_actions or total_agents > 0:
                    self.replay_buffer.store(obs_parsed, action_rl, rew, new_obs_parsed)
                myTimer.rest += myTimer.addTime()
       
                # learn
                if i_episode > 10:
                    batch = self.replay_buffer.sample_batch(cfg.model.batch_size)
                    self.update(data=batch)
                    myTimer.learning += myTimer.addTime()

                # obs = new_obs
                obs = new_obs
                obs_parsed = new_obs_parsed
                
                # save infos
                episode_reward += rew
                episode_num_tasks_finished += reward_dict["task-finished"]
                task_search_durations.extend(info["task-search-durations"])
                task_distances.extend(info["task-distances"])
                # if info["task-search-durations"]:
                #     print("Task assigned after steps", info["task-search-durations"])
                myTimer.rest += myTimer.addTime()

                avg_value, max_value = parameter_value(self.actor)

            epochs.set_description(
                f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | NumTasksFinishe: {episode_num_tasks_finished:.1f} | Checkpoint: {cfg.model.checkpoint_path}"
            )

            if self.wandb is not None:
                self.wandb.log({"Reward": episode_reward, "Num Tasks finished": episode_num_tasks_finished, "Task search duration": np.mean(task_search_durations), "Task distance": np.mean(task_distances), "Step": i_episode, "Critic Loss": np.mean(self.LogCriticLoss), "Policy Loss": np.mean(self.LogPolicyLoss)})
            if self.tensorboard is not None:
                self.tensorboard.add_scalar("Reward", episode_reward, i_episode)
                self.tensorboard.add_scalar("Num Tasks finished", episode_num_tasks_finished, i_episode)
                self.tensorboard.add_scalar("Task search duration", np.mean(task_search_durations), i_episode)
                self.tensorboard.add_scalar("Task distance", np.mean(task_distances), i_episode)
                self.tensorboard.add_scalar("Critic Loss", np.mean(self.LogCriticLoss), i_episode)
                self.tensorboard.add_scalar("Policy Loss", np.mean(self.LogPolicyLoss), i_episode)
                

            self.save_checkpoint(
                path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}.pth")
            )
            if episode_reward > best_reward: 
                best_reward = episode_reward
                self.save_checkpoint(
                    path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}_best.pth")
                )


    def test(self, test_episodes, env):
        sim = env.cfg.name
        if sim == "sumo":
            # traci.close(wait=False)
            os.makedirs(f'saved_files/sumo_output/{env.cfg.city}/', exist_ok=True)
            matching_steps = int(env.cfg.matching_tstep * 60 / env.cfg.sumo_tstep)  # sumo steps between each matching
            if env.scenario.is_meso:
                matching_steps -= 1

            sumo_cmd = [
                "sumo", "--no-internal-links", "-c", env.cfg.sumocfg_file,
                "--step-length", str(env.cfg.sumo_tstep),
                "--device.taxi.dispatch-algorithm", "traci",
                "--summary-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.agent_name + "_dua_meso.static.summary.xml",
                "--tripinfo-output", "saved_files/sumo_output/" + env.cfg.city + "/" + self.agent_name + "_dua_meso.static.tripinfo.xml",
                "--tripinfo-output.write-unfinished", "true",
                "-b", str(env.cfg.time_start * 60 * 60), "--seed", "10",
                "-W", 'true', "-v", 'false',
            ]
            assert os.path.exists(env.cfg.sumocfg_file), "SUMO configuration file not found!"
        epochs = trange(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        episode_rebalanced_vehicles = []
        episode_actions = []
        episode_inflows = []
        for i_episode in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            eps_rebalancing_veh = 0
            done = False
            if sim =='sumo':
                traci.start(sumo_cmd)
            obs, rew = env.reset()  # initialize environment
            obs = self.parser.parse_obs(obs)
            eps_reward += rew
            eps_served_demand += rew
            actions = []
            inflow = np.zeros(len(env.region))
            while not done:
                
                action_rl = self.select_action(obs, deterministic=True)
                actions.append(action_rl)
                desiredAcc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(self.env.region))
                }
                reb_action = solveRebFlow(
                    self.env,
                    self.env.cfg.directory,
                    desiredAcc,
                    self.cplexpath,
                )
                new_obs, rew, done, info = env.step(reb_action=reb_action)
                #calculate inflow to each node in the graph
               
                for k in range(len(env.edges)):
                    i,j = env.edges[k]
                    inflow[j] += reb_action[k]

                if not done:
                    obs = self.parser.parse_obs(new_obs)
                
                eps_reward += rew
                eps_served_demand += info["profit"]
                eps_rebalancing_cost += info["rebalancing_cost"]
                #eps_rebalancing_veh += info["rebalanced_vehicles"]
            epochs.set_description(
                f"Test Episode {i_episode+1} | Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}"
            )
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            episode_actions.append(np.mean(actions, axis=0))
            episode_inflows.append(inflow)
            #episode_rebalanced_vehicles.append(eps_rebalancing_veh)
        

        return (
            episode_reward,
            episode_served_demand,
            episode_rebalancing_cost,
            episode_inflows,
        )

    def save_checkpoint(self, path="ckpt.pth"):
        
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        checkpoint["actor_optimizer"] = self.actor_optimizer.state_dict()
        checkpoint["critic_1_optimizer"] = self.critic_1_optimizer.state_dict()
        checkpoint["critic_2_optimizer"] = self.critic_2_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        
        checkpoint = torch.load(path)
        try:
            # Attempt to load the model state dict as is
            self.load_state_dict(checkpoint["model"])
            #print(checkpoint["model"].keys())
        except RuntimeError as e:
        
            model_state_dict = checkpoint["model"]
            new_state_dict = {}
            # Remapping the keys
            for key in model_state_dict.keys():
                if "conv1.weight" in key:
                    new_key = key.replace("conv1.weight", "conv1.lin.weight")
                #elif "lin.bias" in key:
                #    new_key = key.replace("lin.bias", "bias")
                else:
                    new_key = key
                new_state_dict[new_key] = model_state_dict[key]

            self.load_state_dict(new_state_dict)
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)


def temp(env, parser, select_action, cplexpath, rew_scale, n, acc_new):

    # Convert prob to vehicles count
    total_reb_vehicles = dictsum(env.acc, env.time + 1)

    # acc_new = (acc_new * total_reb_vehicles).round()
    # acc_diff = acc_new.sum() - total_reb_vehicles
    # i_max = np.argmax(acc_new)
    # acc_new[i_max] = acc_new[i_max] - acc_diff
    acc_new = convert_prob_to_count(acc_new, int(total_reb_vehicles))

    # Update state
    for i in env.region:
        env.acc[i][env.time + 1] = int(acc_new[i])

    # RL Section
    obs_unparsed = (env.acc, env.time, env.dacc, env.demand)
    obs = parser.parse_obs(obs_unparsed)

    action_rl = select_action(obs)

    # check if nan 
    if np.isnan(action_rl).any():
        print("Nan in action_rl")
        print(action_rl)
        print("Obs: ", obs)
        nan_break = True
        return None, None, None, None, nan_break

    desiredAcc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
        for i in range(len(env.region))
    }
    
    try:
        reb_action = solveRebFlow(
            env,
            env.cfg.directory,
            desiredAcc,
            cplexpath,
            n
        )

    except:
        print("Error in solveRebFlow; Solution does not exist.")
        print("Total Reb Vehicles: ", int(total_reb_vehicles), ", Desired Acc: ", desiredAcc)

        reb_action = [0.0]*(env.nregion*env.nregion)

    new_obs, rew, done, _ = env.step(reb_action=reb_action)
    new_obs = parser.parse_obs(new_obs)

    if done:
        raise ValueError("Environment is done, check before calling this function failed.")

    return obs, action_rl, rew_scale * rew, new_obs, False

def temp_2(env, desiredAcc, cplexpath, n):

    try:
        reb_action = solveRebFlow(
            env,
            env.cfg.directory,
            desiredAcc,
            cplexpath,
            n
        )

    except:
        print("Error in solveRebFlow; Solution does not exist.")
        reb_action = [0.0]*(env.nregion*env.nregion)

    new_obs_unparsed, rew_unscaled, done, _ = env.step(reb_action=reb_action)

    if done:
        raise ValueError("Environment is done, check before calling this function failed.")

    return  (rew_unscaled, new_obs_unparsed)

def process_task(env, parser, select_action, cplexpath, rew_scale, replay_buffer, lock, n):
    # Copy the environment
    env_copy = deepcopy(env)

    # Perform the task
    obs, action_rl, rew, new_obs = temp(env_copy, parser, select_action, cplexpath, rew_scale, n)

    # Add transition to the replay buffer in a thread-safe manner
    with lock:
        replay_buffer.store(obs, action_rl, rew, new_obs)


def generate_probability_dirichlet(d, K):
    # Initialize the grid with the uniform probability vector
    grid = [np.ones(d) / d]
    
    # As K increases, sample from the Dirichlet distribution
    if K > 1:
        alphas = np.linspace(10, 0.1, K-1)  # Concentration parameters decreasing
        for alpha in alphas:
            grid.append(np.random.dirichlet([alpha] * d))
    
    return np.array(grid)

def generate_probability_uniform(d, k):
    # Initialize the grid
    grid=[]

    for _ in range(k):
        prob = np.random.rand(d) # Random uniform probability vector
        prob = prob / np.sum(prob)
        grid.append(prob)

    return np.array(grid)

def generate_probability_dirichlet_v2(d, K):

    grid = []
    
    alphas = np.linspace(2, 0.25, K)
    for alpha in alphas:
        grid.append(np.random.dirichlet([alpha] * d))
    
    return np.array(grid)

def generate_probability_uniform_v2(d, k):
    
    grid=[]

    for _ in range(k):
        prob = np.random.dirichlet([1.0] * d)
        grid.append(prob)

    return np.array(grid)

def convert_prob_to_count(prob_vector, total_count):
    # Step 1: Scale the probabilities to the total count
    scaled_counts = prob_vector * total_count
    
    # Step 2: Floor the values to get initial integers
    int_vector = np.floor(scaled_counts).astype(int)
    
    # Step 3: Calculate the difference to be adjusted
    difference = total_count - int_vector.sum()
    
    # Step 4: Distribute the difference based on largest fractional parts
    fractional_parts = scaled_counts - int_vector
    indices = np.argsort(-fractional_parts)  # Sort in descending order of fractional part
    int_vector[indices[:difference]] += 1
    
    return int_vector

def parameter_value(model):
    total_sum = 0.0
    total_params = 0
    max_value = 0.0
    
    for param in model.parameters():
        total_sum += param.data.abs().sum().item()
        total_params += param.numel()
        max_value = max(max_value, param.data.abs().max().item())
    
    avg_value = total_sum / total_params if total_params > 0 else 0.0
    return avg_value, max_value

def nan_check(model):
    count = 0
    for param in model.parameters():
        if torch.isnan(param).any():
            count += 1
    print(f"Number of NaN parameters: {count}")
    return