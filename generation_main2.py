import argparse
import json
import os.path
import traceback


import generation_config as cfg
from utils.generate_config import get_gen_config_from_source_dataset
from utils.generate_dataset import generate_dataset

from env.mimicgen_env.lift import *


def set_arg():
    parser = argparse.ArgumentParser()
    # 必要的参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="./datasets/source/square.hdf5",
        help="hdf5 dataset的路径",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="square",
        help="选择源数据集要使用的环境",
    )
    parser.add_argument(
        "--env_interface_name",
        type=str,
        default="MG_Square",
        help="选择源数据集要使用的环境接口名称",
    )
    parser.add_argument(
        "--env_interface_type",
        type=str,
        default="robosuite",
        help="选择源数据集要使用的环境接口类别",
    )

    # extra 参数
    parser.add_argument(
        "--debug",action='store_true',help="设置此标志, 以运行快速生成过程以进行调试")
    parser.add_argument(
        "--auto-remove-exp",action='store_true',help="如果实验文件夹存在，则强制删除")
    parser.add_argument(
        "--render",action='store_true',help="在屏幕上呈现每个数据生成尝试",)
    parser.add_argument(
        "--pause_subtask",action='store_true',help="在生成每个子任务后暂停以进行调试-仅在render标志下")
    parser.add_argument(
        "--video_path",type=str,help="如果提供，将生成数据保存到视频路径", default=None)
    parser.add_argument(
        "--video_skip",type=int,help="编写视频时跳过每n帧", default=5)
    parser.add_argument(
        "--render_image_names",type=str,nargs='+',default=None,
        help="(optional) 用于在屏幕上或视频上渲染的摄像机名称. 默认是none, 每个env类型对应一个预定义的相机")
    parser.add_argument(
        "--source",type=str,help="源数据集的路径，可以覆盖配置中的路径", default=None)
    parser.add_argument(
        "--task_name",type=str,help="用于数据生成的环境名，可以覆盖配置中的环境名",default=None)
    parser.add_argument(
        "--folder",type=str,help="将使用新数据创建的文件夹，以覆盖配置中的文件夹",default=None)
    parser.add_argument(
        "--num_demos",type=int,help="要生成或尝试生成的演示数量，以覆盖配置中设定的演示数量",default=None)
    parser.add_argument(
        "--seed",type=int,help="种子, 可以覆盖配置中的种子",default=None)

    args = parser.parse_args()
    return args


def main(args):
    config_path = os.path.dirname(os.path.dirname(args.dataset)) + "/config"

    if os.path.exists(config_path) and os.path.isdir(config_path):
        files = os.listdir(config_path)
        if len(files) > 0:
            config_file = None
            for filename in os.listdir(config_path):
                if filename.endswith('.json') and args.env in filename:
                    config_file= os.path.join(config_path, filename)
                    break
            if config_file is None:
                print("==== 无法获取对应的配置文件，开始生成 ====")
                config_file = get_gen_config_from_source_dataset(
                    args.dataset,
                    config_path,
                    args.env_interface_name,
                    args.env_interface_type,
                    args.env,
                    filter_key=None,
                    n=None,
                    cfg=cfg,
                )
        else:
            config_file = get_gen_config_from_source_dataset(
                args.dataset,
                config_path,
                args.env_interface_name,
                args.env_interface_type,
                args.env,
                filter_key=None,
                n=None,
                cfg=cfg,
            )
        # 读取配置文件（json）
        with open(config_file, 'r') as file:
            try:
                ext_cfg = json.load(file)
                print(f"文件 {config_file} 内容：", ext_cfg)  # 或者其他的处理
            except json.JSONDecodeError:
                print(f"文件 {config_file} 无法解析为JSON")

        mg_config = cfg.get_mg_config(args, ext_cfg)

        # 生成数据集并捕捉错误
        important_stats = None
        try:
            important_stats = generate_dataset(
                mg_config=mg_config,
                auto_remove_exp=args.auto_remove_exp,
                render=args.render,
                video_path=args.video_path,
                video_skip=args.video_skip,
                render_image_names=args.render_image_names,
                pause_subtask=args.pause_subtask,
            )
            res_str = "finished run successfully!"
        except Exception as e:
            res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())

        print(res_str)

        if important_stats is not None:
            important_stats = json.dumps(important_stats, indent=4)
            print("\nFinal Data Generation Stats")
            print(important_stats)



if __name__ == "__main__":
    args = set_arg()
    main(args)








