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
    max_episodes_list = [10, 20, 50]

    for map_path in map_path_list:
        cfg.model.map_path = map_path

        for i in range(2):
            cfg.model.checkpoint_path = f"markovian_new_obs_{i}"
            if run_training(cfg):
                return

        
            



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
