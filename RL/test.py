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
    elif model_name == "activation_sac":
        from src.algos.activation_sac import ActivationSAC
        return ActivationSAC(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser, train_dir=script_dir, device=device).to(device)
    elif model_name == "scaling_sac":
        from src.algos.scaling_sac import ScalingSAC
        return ScalingSAC(env=env, input_size=cfg.input_size, cfg=cfg, parser=parser, train_dir=script_dir, device=device).to(device)
    else:
        raise ValueError(f"Unknown model or baseline: {model_name}")




@hydra.main(version_base=None, config_path=os.path.join(script_dir, "src/config/"), config_name="config")
def main(cfg: DictConfig):

    # json_path = "../example_problems/custom_warehouse.domain/warehouse_8x6.json"
    
    env = envWrapper.LRRenv(
        inputFile=os.path.join(script_dir, cfg.model.map_path),
        outputFile=os.path.join(script_dir, "../outputs/trainRL.json"),
        simulationTime=cfg.model.test_max_steps,
        planTimeLimit=cfg.model.test_time_per_step,
        preprocessTimeLimit=30000,
        observationTypes={"node-basics", "node-advanced"},
        random_agents_and_tasks="true",
        message_passing_edge_limit=cfg.model.message_passing_edge_limit,
        distance_until_agent_avail_MAX=cfg.model.distance_until_agent_avail_MAX,
        use_dummy_goals_for_idle_agents=cfg.model.use_dummy_goals_for_idle_agents,
        backtrack_reward_type = cfg.model.backtrack_reward_type,
        scheduler_type=cfg.model.scheduler_type,
        planner_type="default",
        guarantee_planner_time = True,
        allow_task_change=cfg.model.test_allow_task_change
    )
    
    env.make_env_params_available()
    
    parser = LRRParser(env, cfg.model)
    
    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = setup_model(cfg, env, parser, device)
    
    model.test_seperate(cfg)

if __name__ == "__main__":
    main()
