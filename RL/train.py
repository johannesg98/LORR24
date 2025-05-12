import hydra
from omegaconf import DictConfig
import os 
import torch
import sys
from profilehooks import profile

script_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(script_dir, "../envWrapper/build")
sys.path.append(build_path)

import envWrapper
from src.helperfunctions.LRRParser import LRRParser




def setup_model(cfg, env, parser, device):
    model_name = cfg.model.name
    cfg = cfg.model
    if model_name == "a2c":
        from src.algos.a2c import A2C
        return A2C(env=env, input_size=cfg.input_size,cfg=cfg, parser=parser, train_dir=script_dir, device=device).to(device)
    elif model_name == "sac":
        from src.algos.sac import SAC
        return SAC(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser, train_dir=script_dir, device=device).to(device)
    elif model_name == "td3":
        from src.algos.td3 import TD3
        return TD3(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser, train_dir=script_dir, device=device).to(device)
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


@hydra.main(version_base=None, config_path=os.path.join(script_dir, "src/config/"), config_name="config")
def main(cfg: DictConfig):

    # json_path = "../example_problems/custom_warehouse.domain/warehouse_8x6.json"
    
    env = envWrapper.LRRenv(
        inputFile=os.path.join(script_dir, cfg.model.map_path),
        outputFile=os.path.join(script_dir, "../outputs/trainRL.json"),
        simulationTime=cfg.model.max_steps,
        planTimeLimit=120,
        preprocessTimeLimit=30000,
        observationTypes={"node-basics"},
        random_agents_and_tasks="true"
    )
    env.make_env_params_available()
    
    parser = LRRParser(env, cfg.model)
    
    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = setup_model(cfg, env, parser, device)

    if cfg.model.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        model.tensorboard = SummaryWriter(os.path.join(script_dir, "logs/", cfg.model.checkpoint_path, datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    
    if cfg.model.wandb: 
        import wandb
        config = {}
        for key in cfg.model.keys():
            config[key] = cfg.model[key]
        wandb5 = wandb.init(
            project=cfg.model.map_path.split("/")[-1].replace(".json", "") + "_ag" + str(env.nAgents),
            entity="johannesg98",
            name=cfg.model.checkpoint_path,
            config=config
        )
        model.wandb = wandb5

    model.learn(cfg) #online RL

if __name__ == "__main__":
    main()
