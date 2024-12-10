import argparse
import copy
import datetime
import json
import os
import shutil
import time
from glob import glob

import datasets
import robosuite as suite

import mimicgen.utils.file_utils as MG_FileUtils

from robosuite import load_controller_config

from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper

from utils.collect_human_demo import collect_human_trajectory, gather_demonstrations_as_hdf5

import env


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",type=str,
                        default=os.path.join(datasets.demo_root, "demo_4"))
    parser.add_argument("--demoNum", type=int, default=1, help="演示的次数")
    parser.add_argument("--environment", type=str, default="LiftAny_D0", help="选择那个环境")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="选择哪个机器人")

    parser.add_argument("--arm", type=str, default="right", help="双臂机器人，选择控制哪个臂，左或右")
    parser.add_argument("--camera", type=str, default="agentview", help="使用哪种相机来收集演示")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="用户输入的位置要缩放多少")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="用户输入的旋转要缩放多少")

    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="选择控制器.'IK_POSE' or 'OSC_POSE'")
    parser.add_argument("--device", type=str, default="keyboard",
                        help="选择设备，keyboard 或 spacemouse")

    args = parser.parse_args()

    # 获取控制器配置
    controller_config = load_controller_config(default_controller=args.controller)

    # 环境配置
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # 检查我们是否使用多臂环境，如果是，使用env_configuration参数
    if "TwoArm" in args.environment:
        config["env_configuration"] = "bimanual"
    else:
        config["env_configuration"] = "single-arm-opposed"

    # camera = "frontview"



    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        obj_name="wood_ball",
    )

    # 可视化的环境包装器
    env = VisualizationWrapper(env)

    # 获取对控制器配置的引用并将其转换为json编码的字符串
    # 增加 环境类型 robosuite  默认为1
    env_args = dict(
        env_name=args.environment,
        type=1,
    )
    env_kwargs = dict(has_renderer=True,
                      has_offscreen_renderer=False,
                      render_camera=args.camera,
                      ignore_done=True,
                      use_camera_obs=False,
                      reward_shaping=True,
                      control_freq=20,
                      obj_name="wood_ball",
                      controller_configs=controller_config,
                      robots=args.robots,
                      )

    env_args.update(env_kwargs=env_kwargs)

    env_args = json.dumps(env_args)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize devicerrr
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    now_day, now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split("_")
    new_dir = os.path.join(args.directory, "{}/{}".format(now_day,now_time))
    os.makedirs(new_dir)

    # collect demonstrations
    for _ in range(args.demoNum):
        collect_human_trajectory(env, device, args.arm, config["env_configuration"])
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_args)