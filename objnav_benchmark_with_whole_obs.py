import argparse
import csv
import os

import habitat
from config_utils import hm3d_config, mp3d_config
from mapper_with_whole_obs import Instruct_Mapper
from mapping_utils.transform import habitat_camera_intrinsic
from objnav_agent_with_whole_obs import HM3D_Objnav_Agent
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--mapper_resolution", type=float, default=0.05)
    parser.add_argument("--path_resolution", type=float, default=0.2)
    parser.add_argument("--path_scale", type=int, default=5)
    parser.add_argument("--image_perceiver",default="glee",choices=["glee","ramsam","dinosam"])
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*::.*")

    args = get_args()
    habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
    habitat_env = habitat.Env(habitat_config)
    habitat_mapper = Instruct_Mapper(habitat_camera_intrinsic(habitat_config),
                                     pcd_resolution=args.mapper_resolution,
                                     grid_resolution=args.path_resolution,
                                     grid_size=args.path_scale,
                                     perceiver=args.image_perceiver)

    habitat_agent = HM3D_Objnav_Agent(habitat_env, habitat_mapper)
    evaluation_metrics = []
    for i in tqdm(range(args.eval_episodes)):
        print(f"Processing episode: i = {i}")

        habitat_agent.reset()
        if i == 0:
            continue

        habitat_agent.make_plan_mod(idx=i)
        flag = True
        while flag:
            flag = habitat_agent.step_mod(idx=i)
        # while not habitat_env.episode_over and habitat_agent.episode_steps < 495:
        #     habitat_agent.step_mod()
        habitat_agent.save_trajectory("./debug/episode-%d/" % i)
        # evaluation_metrics.append({
        #     'success': habitat_agent.metrics['success'],
        #     'spl': habitat_agent.metrics['spl'],
        #     'distance_to_goal': habitat_agent.metrics['distance_to_goal'],
        #     'object_goal': habitat_agent.instruct_goal
        # })
        # write_metrics(evaluation_metrics)
        exit(0)
