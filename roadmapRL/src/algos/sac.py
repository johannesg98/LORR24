import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import random
from tqdm import trange
import os
import sys
import pickle
import time
import wandb
import matplotlib.pyplot as plt
import seaborn as sns



class timer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.now = time.time()
        self.outerLoop = 0
        self.selectAction = 0
        self.step = 0
        self.rest = 0
        self.learning = 0
        self.batchSampling = 0
        self.test_cycles = 0

    def addTime(self):
        ret = time.time() - self.now
        self.now = time.time()
        return ret
    
    def printAvgTimes(self, iEpisode):
        print(f"\n Times: Episode {iEpisode} | Avg outer-loop: {self.outerLoop/iEpisode:.2f} | Avg select-action: {self.selectAction/iEpisode:.2f} | Avg step: {self.step/iEpisode:.2f} | Avg rest: {self.rest/iEpisode:.2f} | Avg learning: {self.learning/max(1,iEpisode-(self.cfg.model.start_training_at_episode+1)):.2f} | Avg batch sampling: {self.batchSampling/max(1,iEpisode-(self.cfg.model.start_training_at_episode+1)):.2f} | Avg test-cycles: {self.test_cycles/iEpisode:.2f} |")
        
    

class PairData(Data):
    """
    Store 2 graphs in one Data object (s_t and s_t+1)
    """

    def __init__(self, edge_index_s=None, edge_attr_s=None, x_s=None, reward=None, action=None, edge_index_t=None, edge_attr_t=None, x_t=None, next_action=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        #self.next_action = next_action

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

    def __init__(self, device, capacity=10000):
        self.device = device
        self.capacity = capacity
        self.data_list = []
        self.rewards = []
        self.position = 0

    def store(self, data1, action, reward, data2):
        pair_data = PairData(data1.edge_index, 
                             data1.edge_attr, 
                             data1.x, 
                             torch.as_tensor(reward), 
                             torch.as_tensor(action), 
                             data2.edge_index, 
                             data2.edge_attr, 
                             data2.x)

        if len(self.data_list) < self.capacity:
            self.data_list.append(pair_data)
            self.rewards.append(reward)
        else:
            self.data_list[self.position] = pair_data
            self.rewards[self.position] = reward
            
        self.position = (self.position + 1) % self.capacity
    
    def create_dataset(self, edge_index, memory_path, rew_scale, size=60000):
        w = open(f"data/{memory_path}.pkl", "rb")
        data = pickle.load(w)
   
        (state_batch, action_batch, reward_batch, next_state_batch) = (
            data["state"],
            data["action"],
            rew_scale * data["reward"],
            data["next_state"],
            #data["next_action"]
        )

        for i in range(len(state_batch)):
            self.data_list.append(
                PairData(
                    edge_index,
                    state_batch[i],
                    reward_batch[i],
                    action_batch[i],
                    edge_index,
                    next_state_batch[i],
                    #next_action_batch[i]
                )
            )

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32):
        data = random.sample(self.data_list, min(batch_size, len(self.data_list)))
        # Only move data to GPU when needed for the batch
        batch = Batch.from_data_list(data, follow_batch=['x_s', 'x_t']).to(self.device)
        return batch

class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant

#########################################
############## SAC AGENT ################
#########################################
class SAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the LRR control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        cfg, 
        parser,
        train_dir,
        device=torch.device("cpu"),
    ):
        super(SAC, self).__init__()
        self.env = env
        self.eps = np.finfo(np.float32).eps.item(),
        self.input_size = input_size
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nRoadmapNodes
        self.cfg = cfg

        # SAC parameters
        self.alpha = cfg.alpha
        self.polyak = 0.995
        self.env = env
        self.BATCH_SIZE = cfg.batch_size
        self.p_lr = cfg.p_lr
        self.q_lr = cfg.q_lr
        self.gamma = 0.99 #0.99
        self.use_automatic_entropy_tuning = cfg.auto_entropy
        self.clip = cfg.clip
        self.parser = parser

        self.cplexpath = cfg.cplexpath
        self.directory = cfg.directory
        self.train_dir = train_dir
        self.agent_name = cfg.agent_name
        self.step = 0
        self.nodes = env.nRoadmapNodes

        self.tensorboard = None
        self.wandb = None
        self.last_best_checkpoint = None

        self.replay_buffer = ReplayData(device=device, capacity=100000)
        

        if cfg.net == "GCNConv":
            from src.nets.GCNConv import GNNActor, GNNCritic
        elif cfg.net == "GCNConvPenta":
            from src.nets.GCNConvPenta import GNNActor, GNNCritic
        elif cfg.net == "NNConv":
            from src.nets.NNConv import GNNActor, GNNCritic
        elif cfg.net == "GATConv":
            from src.nets.GATConv import GNNActor, GNNCritic
        elif cfg.net == "TransformerConv":
            from src.nets.TransformerConv import GNNActor, GNNCritic
        elif cfg.net == "TransformerConvDouble":
            from src.nets.TransformerConvDouble import GNNActor, GNNCritic
  
        self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim, edge_feature_dim=cfg.edge_feature_dim)
        self.critic1 = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim, edge_feature_dim=cfg.edge_feature_dim)
        self.critic2 = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim, edge_feature_dim=cfg.edge_feature_dim)
        self.critic1_target = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim, edge_feature_dim=cfg.edge_feature_dim)
        self.critic2_target = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim, edge_feature_dim=cfg.edge_feature_dim)

        assert self.critic1.parameters() != self.critic2.parameters()

        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

        #if min_q_weigth in cfg set it 
        if hasattr(cfg, 'min_q_weight'):
            self.min_q_weight = cfg.min_q_weight
        if hasattr(cfg, 'temp'):
            self.temp = cfg.temp
        if hasattr(cfg, 'num_random'):
            self.num_random = cfg.num_random
    
    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), deterministic)
        a = a.squeeze(0)
        a = a.detach().cpu().numpy().astype(int)
        return a
    
    
    def compute_loss_q(self, data, conservative=False):
        (
            state_batch,
            edge_index,
            edge_attr,
            next_state_batch,
            edge_index2,
            edge_attr2,
            reward_batch,
            action_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.edge_attr_s,
            data.x_t,
            data.edge_index_t,
            data.edge_attr_t,
            data.reward,
            data.action.reshape(-1, self.nodes),
        )

        q1 = self.critic1(state_batch, edge_index, edge_attr, action_batch)
        q2 = self.critic2(state_batch, edge_index, edge_attr, action_batch)

        self.LogQ1.append(q1.mean().item())
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(next_state_batch, edge_index2, edge_attr2)
            q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, edge_attr2, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, edge_attr2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)


            
            # times = next_state_batch.reshape(-1, self.act_dim, self.input_size)[:, 0, -3] - state_batch.reshape(-1, self.act_dim, self.input_size)[:, 0, -3]
            # if self.cfg.normalise_obs:
            #     times *= self.cfg.max_steps
            # gamma = self.gamma ** (times)

            backup = reward_batch + self.gamma * (q_pi_targ - self.alpha * logp_a2)


        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        if conservative:
            batch_size = action_batch.shape[0]
            action_dim = action_batch.shape[-1]

            (
                random_log_prob,
                current_log,
                next_log,
                q1_rand,
                q2_rand,
                q1_current,
                q2_current,
                q1_next,
                q2_next,
            ) = self._get_action_and_values(data, 10, batch_size, action_dim)

            """Importance sampled version"""
            cat_q1 = torch.cat(
                [
                    q1_rand - random_log_prob.detach(),
                    q1_next - next_log.detach(),
                    q1_current - current_log.detach(),
                ],
                1,
            )
            cat_q2 = torch.cat(
                [
                    q2_rand - random_log_prob.detach(),
                    q2_next - next_log.detach(),
                    q2_current - current_log.detach(),
                ],
                1,
            )

            min_qf1_loss = (
                torch.logsumexp(
                    cat_q1 / self.temp,
                    dim=1,
                ).mean()
                * self.min_q_weight
                * self.temp
            )
            min_qf2_loss = (
                torch.logsumexp(
                    cat_q2 / self.temp,
                    dim=1,
                ).mean()
                * self.min_q_weight
                * self.temp
            )

            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - q1.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - q2.mean() * self.min_q_weight

            loss_q1 = loss_q1 + min_qf1_loss
            loss_q2 = loss_q2 + min_qf2_loss

        return loss_q1, loss_q2

    def compute_loss_pi(self, data):
        state_batch, edge_index, edge_attr = (
            data.x_s,
            data.edge_index_s,
            data.edge_attr_s
        )

        actions, logp_a = self.actor(state_batch, edge_index, edge_attr)
        q1_1 = self.critic1(state_batch, edge_index, edge_attr, actions)
        q2_a = self.critic2(state_batch, edge_index, edge_attr, actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()
        return loss_pi

    def update(self, data, conservative=False, only_q=False):
        loss_q1, loss_q2 = self.compute_loss_q(data, conservative)

        self.optimizers["c1_optimizer"].zero_grad()

        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        loss_q1.backward(retain_graph=True)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        self.LogQ1Loss.append(loss_q1.item())
        if not only_q:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False

            # one gradient descent step for policy network
            self.optimizers["a_optimizer"].zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.optimizers["a_optimizer"].step()

            # Unfreeze Q-networks
            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

            self.LogPolicyLoss.append(loss_pi.item())

    def _get_action_and_values(self, data, num_actions, batch_size, action_dim):
      
        alpha = torch.ones(action_dim)
        d = torch.distributions.Dirichlet(alpha)
        random_actions = d.sample((batch_size * self.num_random,))
        random_log_prob = (
            d.log_prob(random_actions).view(batch_size, num_actions, 1).to(self.device)
        )
        random_actions = random_actions.to(self.device)
        data_list = data.to_data_list()
        data_list = data_list * num_actions
        batch_temp = Batch.from_data_list(data_list).to(self.device)
        current_actions, current_log = self.actor(
            batch_temp.x_s, batch_temp.edge_index_s
        )
        current_log = current_log.view(batch_size, num_actions, 1)

        next_actions, next_log = self.actor(batch_temp.x_t, batch_temp.edge_index_t)
        next_log = next_log.view(batch_size, num_actions, 1)

        q1_rand = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, random_actions
        ).view(batch_size, num_actions, 1)

        q2_rand = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, random_actions
        ).view(batch_size, num_actions, 1)

        q1_current = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, current_actions
        ).view(batch_size, num_actions, 1)

        q2_current = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, current_actions
        ).view(batch_size, num_actions, 1)

        q1_next = self.critic1(
            batch_temp.x_s, batch_temp.edge_index_s, next_actions
        ).view(batch_size, num_actions, 1)

        q2_next = self.critic2(
            batch_temp.x_s, batch_temp.edge_index_s, next_actions
        ).view(batch_size, num_actions, 1)
        
        return (
            random_log_prob,
            current_log,
            next_log,
            q1_rand,
            q2_rand,
            q1_current,
            q2_current,
            q1_next,
            q2_next,
        )
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)

        return optimizers
    
    
    def learn(self, cfg, Dataset=None):
        if cfg.model.load_from_ckeckpoint:
            self.load_checkpoint(path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}.pth"))
            print("last checkpoint loaded")

        if cfg.model.load_external_actor:
            self.load_external_actor(cfg)
        
        train_episodes = cfg.model.max_episodes  # set max number of training episodes
        epochs = trange(train_episodes)  # epoch iterator
        self.train()  # set model in train mode
        myTimer = timer(cfg)
        if cfg.model.visu_episode_list:
            curr_visu_idx = 0
            outputFile = os.path.join(self.train_dir, "../outputs/cont_outputs/", cfg.model.checkpoint_path, str(cfg.model.visu_episode_list[curr_visu_idx])+".json")
            os.makedirs(os.path.join(self.train_dir, "../outputs/cont_outputs/", cfg.model.checkpoint_path), exist_ok=True)

        episode_tasks_finished_sum = 0

        for i_episode in epochs:
            self.i_episode = i_episode
            if cfg.model.visu_episode_list:
                obs, rew, _ = self.env.reset(outputFile_=outputFile)
                if cfg.model.visu_episode_list[curr_visu_idx] == i_episode and curr_visu_idx < len(cfg.model.visu_episode_list)-1:
                    curr_visu_idx += 1
                    outputFile = os.path.join(self.train_dir, "../outputs/cont_outputs/", cfg.model.checkpoint_path, str(cfg.model.visu_episode_list[curr_visu_idx])+".json")
            else:
                obs, rew, _ = self.env.reset()
            step = 0
            obs_parsed = self.parser.parse_obs(obs)
            episode_reward = 0
            episode_reward += rew
            episode_num_tasks_finished = 0
            task_search_durations = []
            task_distances = []
            self.LogQ1 = []
            self.LogQ1Loss = []
            self.LogPolicyLoss = []
            road_active_sum = 0
            road_inactive_sum = 0
            done = False
            myTimer.outerLoop += myTimer.addTime()

            while not done:
                # actor step
                action_rl = self.select_action(obs_parsed, cfg.model.deterministic_actor)
                action_rl = np.zeros(self.nodes).astype(int)
                myTimer.selectAction += myTimer.addTime()

                # step
                action_dict = {"roadmap_activation": action_rl.tolist()}
                new_obs, reward_dict, done, info = self.env.step(action_dict)
                myTimer.step += myTimer.addTime()
                
                # reward
                rew = cfg.model.rew_w_roadmap_progress * reward_dict["roadmap-progress-reward"] + cfg.model.rew_w_task_finish * reward_dict["task-finished"]       # dist-reward, A*-distance, task-finished
                if done:
                    rew += cfg.model.rew_w_final_tasks_finished * (episode_num_tasks_finished + reward_dict["task-finished"])
                episode_reward += rew

                # store in replay buffer
                new_obs_parsed = self.parser.parse_obs(new_obs)
                self.replay_buffer.store(obs_parsed, action_rl, rew, new_obs_parsed)
                myTimer.rest += myTimer.addTime()


                # learn
                if cfg.model.train and i_episode > cfg.model.start_training_at_episode:
                    batch = self.replay_buffer.sample_batch(cfg.model.batch_size)
                    myTimer.batchSampling += myTimer.addTime()
                    if i_episode < cfg.model.only_q_steps:
                        self.update(data=batch, only_q=True)
                    else:
                        self.update(data=batch)
                    myTimer.learning += myTimer.addTime()

                # obs = new_obs
                obs = new_obs
                obs_parsed = new_obs_parsed
                
                # save infos
                episode_num_tasks_finished += reward_dict["task-finished"]
                road_active_sum += action_rl.sum()
                road_inactive_sum += (1 - action_rl).sum()
                step+= 1
                myTimer.rest += myTimer.addTime()      

        
            
            # print episode infos
            epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | NumTasksFinishe: {episode_num_tasks_finished:.1f} | Checkpoint: {cfg.model.checkpoint_path}")

            # log wandb + tensorboard
            myTimer.printAvgTimes(i_episode+1)
            if self.wandb is not None:
                self.wandb.log({"Reward": episode_reward, "Num Tasks finished": episode_num_tasks_finished, "Roads active (%)": road_active_sum/(road_active_sum+road_inactive_sum)*100, "Step": i_episode, "Q1 Loss": np.mean(self.LogQ1Loss), "Policy Loss": np.mean(self.LogPolicyLoss), "Q1": np.mean(self.LogQ1)}, step=i_episode)
                # self.wandb_policy_logger(i_episode)
            if self.tensorboard is not None:
                self.tensorboard.add_scalar("Reward", episode_reward, i_episode)
                self.tensorboard.add_scalar("Num Tasks finished", episode_num_tasks_finished, i_episode)
                self.tensorboard.add_scalar("Q1 Loss", np.mean(self.LogQ1Loss), i_episode)
                self.tensorboard.add_scalar("Policy Loss", np.mean(self.LogPolicyLoss), i_episode)
                self.tensorboard.add_scalar("Q1", np.mean(self.LogQ1), i_episode)

            episode_tasks_finished_sum += episode_num_tasks_finished
            print("Avg num_task_fisnished reward: ", episode_tasks_finished_sum / (i_episode + 1))
            
            
                
            # save checkpoints   
            if cfg.model.train:
                self.checkpoint_handler(i_episode, episode_num_tasks_finished, cfg)

            myTimer.outerLoop += myTimer.addTime()

            # test agent
            if i_episode % 20 == 5:
                self.test_during_training(i_episode)
            if i_episode % 100 == 0:
                self.test_best_checkpoint(i_episode, cfg)

            myTimer.test_cycles += myTimer.addTime()

    def test_best_checkpoint(self, i_episode, cfg):
        num_tests = 5

        #load best checkpoint
        if self.last_best_checkpoint is not None:
            self.load_checkpoint(path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}_best_{self.last_best_checkpoint}.pth"))
            
            num_tasks_finished_sum = 0
            for test_i in range(num_tests):
                obs, _, _ = self.env.reset()
                obs_parsed = self.parser.parse_obs(obs).to(self.device)
                done = False
                while not done:
                    action_rl = self.select_action(obs_parsed, deterministic=True)
                    
                    
                    action_dict = {"roadmap_activation": action_rl.tolist()}
                    
                    # step
                    new_obs, reward_dict, done, info = self.env.step(action_dict)

                    obs = new_obs
                    obs_parsed = self.parser.parse_obs(obs).to(self.device)
                    
                    # save infos
                    num_tasks_finished_sum += reward_dict["task-finished"]
            
            #load most recent checkpoint
            self.load_checkpoint(path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}.pth"))

            if self.wandb is not None:
                self.wandb.log({"Test best checkpoint (num_tasks_finished)": num_tasks_finished_sum / num_tests}, step=i_episode)

    def test_during_training(self, i_episode):
        obs, rew, _ = self.env.reset()
        obs_parsed = self.parser.parse_obs(obs).to(self.device)
        episode_num_tasks_finished = 0
        done = False
        while not done:
            action_rl = self.select_action(obs_parsed, deterministic=True)
            
            
            action_dict = {"roadmap_activation": action_rl.tolist()}
            
            # step
            new_obs, reward_dict, done, info = self.env.step(action_dict)

            obs = new_obs
            obs_parsed = self.parser.parse_obs(obs).to(self.device)
            
            # save infos
            episode_num_tasks_finished += reward_dict["task-finished"]
        if self.wandb is not None:
            self.wandb.log({"Test during training (num_tasks_finished)": episode_num_tasks_finished}, step=i_episode)

   

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path)
        try:
            # Attempt to load the model state dict as is
            self.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            print("dunno should work without this stuff")
            # model_state_dict = checkpoint["model"]
            # new_state_dict = {}
            # Remapping the keys
            # for key in model_state_dict.keys():
            #     if "conv1.weight" in key:
            #         new_key = key.replace("conv1.weight", "conv1.lin.weight")
            #     elif "lin.bias" in key:
            #        new_key = key.replace("lin.bias", "bias")
            #     else:
            #         new_key = key
            #     new_state_dict[new_key] = model_state_dict[key]

            # self.load_state_dict(new_state_dict)
        
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def load_external_actor(self, cfg):
        self.actor.load_state_dict(torch.load(os.path.join(self.train_dir, cfg.model.external_actor_path), map_location=self.device))
        print("External actor loaded")

    def checkpoint_handler(self, i_episode, episode_num_tasks_finished, cfg):
        if i_episode == 0:
            self.last_20_episode_tasks_finished = [0]*20
            self.best_num_tasks_finished = -np.inf
            self.last_best_checkpoint = None
        self.save_checkpoint(path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}.pth"))
        self.last_20_episode_tasks_finished.pop(0)
        self.last_20_episode_tasks_finished.append(episode_num_tasks_finished)
        if i_episode > 20 and np.mean(self.last_20_episode_tasks_finished) > self.best_num_tasks_finished:
            self.best_num_tasks_finished = np.mean(self.last_20_episode_tasks_finished)
            self.save_checkpoint(
                path=os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}_best_{i_episode}.pth")
            )
            if self.last_best_checkpoint is not None:
                os.remove(os.path.join(self.train_dir, f"ckpt/{cfg.model.checkpoint_path}_best_{self.last_best_checkpoint}.pth"))
            self.last_best_checkpoint = i_episode

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
