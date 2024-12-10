import json
import os

import h5py
import numpy as np
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

from mimicgen.env_interfaces.base import make_interface
import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.scripts.prepare_src_dataset import extract_datagen_info_from_trajectory


def pre_process_dataset(
        dataset_path,
        env_interface_name,
        env_interface_type,
        filter_key=None,
        start=None,
        n=None
    ):
    """
    This script is used to pre-process the dataset.
    It will extract the information from the source dataset and save it in a new dataset.
    """
    # 创建用于数据收集的环境与环境接口
    # dataset_path = os.path.expanduser(dataset_path)
    # f = h5py.File(dataset_path, "r")
    #
    # env_meta = json.loads(f["data"].attrs["env_args"])

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta, camera_names=[],
        camera_height=84, camera_width=84, reward_shaping=False)
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        env=env.base_env,
    )
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # 从源数据中获取源demo列表 ——> list[demo1, demo2 ...]
    demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=dataset_path,
        filter_key=filter_key,
        start=start,
        n=n)

    f = h5py.File(dataset_path, "a")

    for ind in tqdm(range(len(demos))):
        ep = demos[ind]
        ep_grp = f["data/{}".format(ep)]

        # 从源数据中获取 state
        states = ep_grp["states"][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = ep_grp.attrs["model_file"]

        # 将state带入仿真环境, 获取源 datagen_info
        actions = ep_grp["actions"][()]
        datagen_info = extract_datagen_info_from_trajectory(
            env=env,
            env_interface=env_interface,
            initial_state=initial_state,
            states=states,
            actions=actions,
        )


        # delete old dategen info if it already exists
        if "datagen_info" in ep_grp:
            del ep_grp["datagen_info"]

        for k in datagen_info:
            if k in ["object_poses", "subtask_term_signals"]:
                # handle dict
                for k2 in datagen_info[k]:
                    ep_grp.create_dataset("datagen_info/{}/{}".format(k, k2), data=np.array(datagen_info[k][k2]))
            else:
                ep_grp.create_dataset("datagen_info/{}".format(k), data=np.array(datagen_info[k]))

        # remember the env interface used too
        ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name
        ep_grp["datagen_info"].attrs["env_interface_type"] = env_interface_type

    print("Modified {} trajectories to include datagen info.".format(len(demos)))
    f.close()

