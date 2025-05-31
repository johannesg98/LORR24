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
    if model_name == "sac":
        from src.algos.sac import SAC
        return SAC(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser, train_dir=script_dir, device=device).to(device)
    else:
        raise ValueError(f"Unknown model or baseline: {model_name}")




@hydra.main(version_base=None, config_path=os.path.join(script_dir, "src/config/"), config_name="config")
def main(cfg: DictConfig):

    
    env = envWrapper.LRRenv(
        inputFile=os.path.join(script_dir, cfg.model.map_path),
        outputFile=os.path.join(script_dir, "../outputs/trainRoadmapRL.json"),
        simulationTime=cfg.model.max_steps,
        planTimeLimit=70,
        preprocessTimeLimit=30000,
        observationTypes={"roadmap-activation"},
        random_agents_and_tasks="true",
        use_dummy_goals_for_idle_agents=cfg.model.use_dummy_goals_for_idle_agents,
        scheduler_type="ILP",
        planner_type="default",
        guarantee_planner_time = True
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
            project= "Roadmap " + cfg.model.map_path.split("/")[-1].replace(".json", "") + "_ag" + str(env.nAgents),
            entity="johannesg98",
            name=cfg.model.checkpoint_path,
            config=config
        )
        model.wandb = wandb5
    
    model.learn(cfg) #online RL

if __name__ == "__main__":
    main()
