import os
import sys
import hydra
from omegaconf import OmegaConf, DictConfig
from shutil import copyfile

from train import main as train_main

script_dir = os.path.dirname(os.path.abspath(__file__))
slurm_task_id = int(sys.argv[1])
sys.argv = sys.argv[:1]
iterator = 1

@hydra.main(version_base=None, config_path=os.path.join(script_dir, "src/config/"), config_name="config")
def main(cfg: DictConfig):

    ##########################################################
    ############## Iterate over experiments here #############
    ########### only change sum of exp in run.slurm ##########

    map_path_list = [
        "../example_problems/custom_warehouse.domain/warehouse_8x6.json",
        "../example_problems/custom_warehouse.domain/warehouse_4x3.json"
    ]
    use_markovian_new_obs_list = [False, True]
    rew_w_list = [5,10,20,30,50,100,200,300,500,750,1000,1500,3000,6000,12000,20000,50000]
    trys = range(2)

    

    # for map_path in map_path_list:
    #     cfg.model.map_path = map_path

    #     for rew_w_immitation in rew_w_immitation_list:
    #         cfg.model.rew_w_immitation = rew_w_immitation

    for i,rew_w_roadmap_progress in enumerate(rew_w_list):
        cfg.model.rew_w_roadmap_progress = rew_w_roadmap_progress
        
        cfg.model.checkpoint_path = f"CPU_roadmap-progress{rew_w_roadmap_progress}_id{slurm_task_id}"
        if run_training(cfg):
            return



        # for use_markovian_new_obs in use_markovian_new_obs_list:
        #     cfg.model.use_markovian_new_obs = use_markovian_new_obs

        #     for rew_w_idle in rew_w_idle_list:
        #         cfg.model.rew_w_idle = rew_w_idle

        #         for rew_w_backtrack in rew_w_backtrack_list:
        #             cfg.model.rew_w_backtrack = rew_w_backtrack

        #             cfg.model.checkpoint_path = f"mrkv{use_markovian_new_obs}_rIdle{rew_w_idle}_rBktr1Dtime{rew_w_backtrack}"
        #             if run_training(cfg):
        #                 return

            

        
            



    ############## Iterate over experiments here #############
    ##########################################################
    

def run_training(cfg):
    global iterator
    run_done = False
    if iterator == slurm_task_id:
        train_main(cfg)
        run_done = True
    else:
        iterator += 1
    return run_done


if __name__ == "__main__":
    main()
