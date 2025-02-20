import hydra
from omegaconf import DictConfig
import os 
import torch
import json
from hydra import initialize, compose

import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(script_dir, "build")
sys.path.append(build_path)

import envWrapper
from src.helperfunctions.LRRParser import LRRParser




def setup_model(cfg, env, parser, device):
    model_name = cfg.model.name
    cfg = cfg.model
    if model_name == "a2c":
        from src.algos.a2c import A2C
        return A2C(env=env, input_size=cfg.input_size,cfg=cfg, parser=parser, device=device).to(device)
    # elif model_name == "sac" or model_name =="cql":
    #     from src.algos.sac import SAC
    #     return SAC(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser, device=device).to(device)
    else:
        raise ValueError(f"Unknown model or baseline: {model_name}")




def load_actor_weights(model, path):
    full_model_state = torch.load(f"ckpt/{path}.pth")

    actor_encoder_state = {
        k.replace("actor.", ""): v
        for k, v in full_model_state["model"].items()
        if "actor" in k
    }
    model.actor.load_state_dict(actor_encoder_state)
    return model

@hydra.main(version_base=None, config_path="src/config/", config_name="config")
def main(cfg: DictConfig):
    
    env = envWrapper.LRRenv(
    inputFile="./example_problems/custom_warehouse.domain/warehouse_4x3_100.json",
    outputFile="./outputs/trainRL.json",
    simulationTime=cfg.model.max_steps,
    planTimeLimit=300,
    preprocessTimeLimit=30000,
    observationTypes={"node-basics"}
    )
    env.make_env_params_available()
    
    parser = LRRParser(env, cfg.model)
    
    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = setup_model(cfg, env, parser, device)
    
    model.wandb = None
    if cfg.model.wandb: 
        import wandb
        config = {}
        for key in cfg.model.keys():
            config[key] = cfg.model[key]
        wandb = wandb.init(
            project="",
            entity="",
            config=config,
        )
        model.wandb = wandb

    model.learn(cfg) #online RL

if __name__ == "__main__":
    main()
