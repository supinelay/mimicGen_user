import os
import json
import shutil

import robomimic
from robomimic.utils.hyperparam_utils import ConfigGenerator

import mimicgen
import mimicgen.utils.config_utils as ConfigUtils
from utils.preprocess_dataset import pre_process_dataset





def make_generator(config_file, settings, out_config_path, **kwargs):
    """
    Implement this function to setup your own hyperparameter scan.
    Each config generator is created using a base config file (@config_file)
    and a @settings dictionary that can be used to modify which parameters
    are set.
    """
    generator = ConfigGenerator(
        base_config_file=config_file,
        generated_config_dir=out_config_path,    # todo
        script_file="", # will be overriden in next step
    )

    # set basic settings
    ConfigUtils.set_basic_settings(
        generator=generator,
        group=0,
        source_dataset_path=settings["dataset_path"],
        source_dataset_name=settings["dataset_name"],
        generation_path=settings["generation_path"],
        guarantee=kwargs["cfg"].GUARANTEE,
        num_traj=kwargs["cfg"].NUM_TRAJ,
        num_src_demos=10,
        max_num_failures=25,
        num_demo_to_render=10,
        num_fail_demo_to_render=25,
        verbose=False,
    )

    # set settings for subtasks
    ConfigUtils.set_subtask_settings(
        generator=generator,
        group=0,
        base_config_file=config_file,
        select_src_per_subtask=settings["select_src_per_subtask"],
        subtask_term_offset_range=settings["subtask_term_offset_range"],
        selection_strategy=settings.get("selection_strategy", None),
        selection_strategy_kwargs=settings.get("selection_strategy_kwargs", None),
        # default settings: action noise 0.05, with 5 interpolation steps
        action_noise=0.05,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        verbose=False,
    )

    # optionally set env interface to use, and type
    # generator.add_param(
    #     key="experiment.task.interface",
    #     name="",
    #     group=0,
    #     values=[settings["task_interface"]],
    # )
    # generator.add_param(
    #     key="experiment.task.interface_type",
    #     name="",
    #     group=0,
    #     values=["robosuite"],
    # )

    # set task to generate data on
    generator.add_param(
        key="experiment.task.name",
        name="task",
        group=1,
        values=settings["tasks"],
        value_names=settings["task_names"],
    )

    # optionally set robot and gripper that will be used for data generation (robosuite-only)
    if settings.get("robots", None) is not None:
        generator.add_param(
            key="experiment.task.robot",
            name="r",
            group=2,
            values=settings["robots"],
        )
    if settings.get("grippers", None) is not None:
        generator.add_param(
            key="experiment.task.gripper",
            name="g",
            group=2,
            values=settings["grippers"],
        )

    # set observation collection settings
    ConfigUtils.set_obs_settings(
        generator=generator,
        group=-1,
        collect_obs=True,
        camera_names=kwargs["cfg"].CAMERA_NAMES,
        camera_height=kwargs["cfg"].CAMERA_SIZE[0],
        camera_width=kwargs["cfg"].CAMERA_SIZE[1],
    )

    if kwargs["cfg"].DEBUG:
        # set debug settings
        ConfigUtils.set_debug_settings(
            generator=generator,
            group=-1,
        )

    # seed
    generator.add_param(
        key="experiment.seed",
        name="",
        group=1000000,
        values=[1],
    )

    return generator


def get_gen_config_from_source_dataset(
    dataset_path,
    config_path,
    env_interface_name,
    env_interface_type,
    env,
    filter_key=None,
    start=None,
    n=None,
    **kwargs,
):
    """
    介绍：1.预处理源数据集，2.生成配置文件
    """

    pre_process_dataset(
        dataset_path=dataset_path,
        env_interface_name=env_interface_name,
        env_interface_type=env_interface_type,
        filter_key=filter_key,
        start=start,
        n=n
    )

    # make config generator
    generator = make_generator(kwargs["cfg"].BASE_CONFIGS[env],
                               kwargs["cfg"].all_settings[env],
                               config_path,
                               cfg=kwargs["cfg"])

    json_file = generator._generate_jsons()

    return json_file[0]
