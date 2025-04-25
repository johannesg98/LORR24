import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.distributions import Dirichlet
import numpy as np
import math
import time
import wandb
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Device configuration (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Your GNNActor model definition
class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False, return_dist=False):
        out = F.relu(self.conv1(state, edge_index))
        if torch.isnan(out).any():
            print("NaN values detected in out!")
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
        if return_dist:
            return Dirichlet(concentration + 1e-20)
        if deterministic:
            action = concentration / (concentration.sum(dim=-1, keepdim=True) + 1e-20)  # Normalize
            log_prob = None
        else:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)
        regularize = concentration.abs().mean()
        return action, log_prob, regularize

def assign_discrete_actions(total_agents, action_rl):
        desired_agent_dist = np.floor(action_rl * total_agents).astype(int)

        remaining_agents = total_agents - np.sum(desired_agent_dist)

        fractional_parts = (action_rl * total_agents) - desired_agent_dist
        top_indices = np.argpartition(-fractional_parts, int(remaining_agents))[:int(remaining_agents)]

        desired_agent_dist[top_indices] += 1
        return desired_agent_dist


def do_one_training(dataset, batch_size = 32, lr = 0.001, num_epochs = 200, loss_fn = nn.MSELoss(), perc_data_used = 1):
    # Unpack the dataset
    nAgents = dataset['nAgents']
    normalized = dataset['normalise_obs']
    obs_vec = dataset['obs']  # Shape: n_data x 79 x 3
    action_vec = dataset['actions']  # Shape: n_data x 79
    edge_index = dataset['edge_index'].to(device)  # Shape: appropriate for edge_index, e.g., (2, num_edges)

    # Convert the data to a TensorDataset for DataLoader
    sample_size = int(obs_vec.shape[0] * perc_data_used)
    dataset = TensorDataset(obs_vec[:sample_size], action_vec[:sample_size])

    # 2. Split dataset into training (90%) and testing (10%)
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # 3. Prepare DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize the model
    in_channels = obs_vec.shape[2]  # Assuming 3 channels in the input
    nNodes = action_vec.shape[1]
    model = GNNActor(in_channels=in_channels, hidden_size=256, act_dim=nNodes)  # Action dim is 79

    # Move the model to the selected device
    model.to(device)

    # 5. Define Loss Function and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare testing and save results
    agent_scale_fac = 1/math.ceil(nAgents/nNodes*4)
    test_results = np.zeros(num_epochs)

    # 6. Loop
    for epoch in range(num_epochs):

        # 6.1. Testing
        model.eval()  # Set model to evaluation mode
        total_test_loss = 0.0
        total_assignments = 0
        wrong_assignments = 0
        start_time = time.time()
        with torch.no_grad():  # No need to compute gradients during evaluation
            for batch_idx, (obs, actions) in enumerate(test_loader):
                # Move data to device (GPU if available)
                obs, actions = obs.to(device), actions.to(device)

                

                # Forward pass
                action_pred, log_prob, regularize = model(obs, edge_index, deterministic=True, return_dist=False)
                
                # Compute test loss
                if isinstance(loss_fn, nn.CosineEmbeddingLoss):
                    target = torch.ones(actions.size(0), device=device)  # Assuming all pairs are similar
                    loss = loss_fn(action_pred, actions, target)
                else:
                    loss = loss_fn(action_pred, actions)
                
                total_test_loss += loss.item()
                for i in range(len(actions)):  # len(actions) should be equal to the batch size
                    one_action = actions[i]
                    one_pred_action = action_pred[i]
                    one_obs = obs[i]
                    total_agents = one_obs[:,1].sum().cpu().numpy()
                    if normalized:
                        total_agents = round(total_agents/agent_scale_fac)
                    diff = assign_discrete_actions(total_agents, one_pred_action.cpu().numpy()) - assign_discrete_actions(total_agents, one_action.cpu().numpy())
                    total_assignments += total_agents
                    wrong_assignments += np.sum(np.abs(diff))

        # Print test loss after each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {total_test_loss / len(test_loader)}, Wrong assignments: {wrong_assignments} of {total_assignments} = {wrong_assignments/total_assignments*100:.2f}%, , Time: {time.time()-start_time}")
        test_results[epoch] = wrong_assignments/total_assignments*100

        # 7. Training
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        total_reg = 0.0
        start_time = time.time()
        for batch_idx, (obs, actions) in enumerate(train_loader):

            # Move data to device (GPU if available)
            obs, actions = obs.to(device), actions.to(device)

            # Forward pass
            action_pred, log_prob, regularize = model(obs, edge_index, deterministic=True, return_dist=False)
        
            # Compute training loss (mean squared error for now, adjust as necessary)
            if isinstance(loss_fn, nn.CosineEmbeddingLoss):
                target = torch.ones(actions.size(0), device=device)  # Assuming all pairs are similar
                loss = loss_fn(action_pred, actions, target)
            else:
                loss = loss_fn(action_pred, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_reg += regularize.item()
        
        # Print training loss after each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_train_loss / len(train_loader)}, Reg: {total_reg / len(train_loader)}, Time: {time.time()-start_time}")

    return test_results

    # torch.save(model.state_dict(), "gnn_actor_model.pth")





#####    Define Parameters    #####

dataset = torch.load(os.path.join(script_dir, "data/skip_dataset_normalized1000.pt"))
batch_size = 32
lr = 1e-3
loss_fn = nn.MSELoss()          # nn.L1Loss()
num_epochs = 100
n_repeats = 3
perc_data_used = 0.3


##### Lists of parameters to test #####
lr_list = [1e-2, 1e-3, 1e-4, 1e-5]
batch_size_list = [16, 32, 64, 128]
loss_fn_list = [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss(), nn.HuberLoss(), nn.CosineEmbeddingLoss(), nn.KLDivLoss(), nn.CrossEntropyLoss(), nn.BCELoss(), nn.BCEWithLogitsLoss()] 



#####   Iterate over parameters   #####
for loss_fn in loss_fn_list:
    test_results = np.zeros(num_epochs)

    for i in range(n_repeats):
        results = do_one_training(dataset, batch_size, lr, num_epochs, loss_fn, perc_data_used)
        test_results += results

    test_results /= n_repeats

    ##### Initialize WandB #####
    wandb1 = wandb.init(
                project="nn-optimization",
                entity="johannesg98",
                name=f"lossfn_{loss_fn}_{n_repeats}Repeats"
            )
    
    for i in range(num_epochs):
        wandb1.log({"test wrong assignments (%)": test_results[i]}, step=i)
    wandb1.finish()






