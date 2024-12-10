import os
import mimicgen
from mimicgen.configs import config_factory, get_all_registered_configs

# set path to folder containing src datasets
SRC_DATA_DIR = "./datasets/source"

# set base folder for where to copy each base config and generate new config files for data generation
CONFIG_DIR = "./tmp/core_configs"

# set base folder for newly generated datasets
OUTPUT_FOLDER = "./tmp/core_datasets"

# number of trajectories to generate (or attempt to generate)
NUM_TRAJ = 3

# whether to guarantee that many successful trajectories (e.g. keep running until that many successes, or stop at that many attempts)
GUARANTEE = False

# whether to run a quick debug run instead of full generation
DEBUG = False

# camera settings for collecting observations
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
CAMERA_SIZE = (84, 84)

# path to base config(s)
BASE_BASE_CONFIG_PATH = "./exps/templates/robosuite"

BASE_CONFIGS = {
    "lift": os.path.join(BASE_BASE_CONFIG_PATH, "lift.json"),
    "stack": os.path.join(BASE_BASE_CONFIG_PATH, "stack.json"),
    "stack_three": os.path.join(BASE_BASE_CONFIG_PATH, "stack_three.json"),
    "square": os.path.join(BASE_BASE_CONFIG_PATH, "square.json"),
    "threading": os.path.join(BASE_BASE_CONFIG_PATH, "threading.json"),
    "three_piece_assembly": os.path.join(BASE_BASE_CONFIG_PATH, "three_piece_assembly.json"),
    "coffee": os.path.join(BASE_BASE_CONFIG_PATH, "coffee.json"),
    "coffee_preparation": os.path.join(BASE_BASE_CONFIG_PATH, "coffee_preparation.json"),
    "nut_assembly": os.path.join(BASE_BASE_CONFIG_PATH, "nut_assembly.json"),
    "pick_place": os.path.join(BASE_BASE_CONFIG_PATH, "pick_place.json"),
    "hammer_cleanup": os.path.join(BASE_BASE_CONFIG_PATH, "hammer_cleanup.json"),
    "mug_cleanup": os.path.join(BASE_BASE_CONFIG_PATH, "mug_cleanup.json"),
    "kitchen": os.path.join(BASE_BASE_CONFIG_PATH, "kitchen.json"),
}


all_settings = {
        "lift":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "lift.hdf5"),
            dataset_name="lift",
            generation_path="{}/lift".format(OUTPUT_FOLDER),
            tasks=["LiftAny_D0"],
            task_names=["D0"],
            select_src_per_subtask=True,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[10,20], None],
        ),
        # stack
        "stack":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "stack.hdf5"),
            dataset_name="stack",
            generation_path="{}/stack".format(OUTPUT_FOLDER),
            # task_interface="MG_Stack",
            tasks=["Stack_D0", "Stack_D1"],
            task_names=["D0", "D1"],
            select_src_per_subtask=True,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            subtask_term_offset_range=[[10, 20], None],
        ),
        # stack_three
        "stack_three":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "stack_three.hdf5"),
            dataset_name="stack_three",
            generation_path="{}/stack_three".format(OUTPUT_FOLDER),
            # task_interface="MG_StackThree",
            tasks=["StackThree_D0", "StackThree_D1"],
            task_names=["D0", "D1"],
            select_src_per_subtask=True,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            subtask_term_offset_range=[[10, 20], [10, 20], [10, 20], None],
        ),
        # square
        "square":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "square.hdf5"),
            dataset_name="square",
            generation_path="{}/square".format(OUTPUT_FOLDER),
            # task_interface="MG_Square",
            tasks=["Square_D0"],
            task_names=["D0"],
            select_src_per_subtask=False,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            subtask_term_offset_range=[[10, 20], None],
        ),
        # threading
        "threading":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "threading.hdf5"),
            dataset_name="threading",
            generation_path="{}/threading".format(OUTPUT_FOLDER),
            # task_interface="MG_Threading",
            tasks=["Threading_D0", "Threading_D1", "Threading_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 10], None],
        ),
        # three_piece_assembly
        "three_piece_assembly":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "three_piece_assembly.hdf5"),
            dataset_name="three_piece_assembly",
            generation_path="{}/three_piece_assembly".format(OUTPUT_FOLDER),
            # task_interface="MG_ThreePieceAssembly",
            tasks=["ThreePieceAssembly_D0", "ThreePieceAssembly_D1", "ThreePieceAssembly_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 10], [5, 10], [5, 10], None],
        ),
        # coffee
        "coffee":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "coffee.hdf5"),
            dataset_name="coffee",
            generation_path="{}/coffee".format(OUTPUT_FOLDER),
            # task_interface="MG_Coffee",
            tasks=["Coffee_D0", "Coffee_D1", "Coffee_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 10], None],
        ),
        # coffee_preparation
        "coffee_preparation":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "coffee_preparation.hdf5"),
            dataset_name="coffee_preparation",
            generation_path="{}/coffee_preparation".format(OUTPUT_FOLDER),
            # task_interface="MG_CoffeePreparation",
            tasks=["CoffeePreparation_D0", "CoffeePreparation_D1"],
            task_names=["D0", "D1"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 10], [5, 10], [5, 10], [5, 10], None],
        ),
        # nut_assembly
        "nut_assembly":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "nut_assembly.hdf5"),
            dataset_name="nut_assembly",
            generation_path="{}/nut_assembly".format(OUTPUT_FOLDER),
            # task_interface="MG_NutAssembly",
            tasks=["NutAssembly_D0"],
            task_names=["D0"],
            select_src_per_subtask=False,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            subtask_term_offset_range=[[10, 20], [10, 20], [10, 20], None],
        ),
        # pick_place
        "pick_place":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "pick_place.hdf5"),
            dataset_name="pick_place",
            generation_path="{}/pick_place".format(OUTPUT_FOLDER),
            # task_interface="MG_PickPlace",
            tasks=["PickPlace_D0"],
            task_names=["D0"],
            select_src_per_subtask=True,
            # NOTE: selection strategy is set by default in the config template, and we will not change it
            # selection_strategy="nearest_neighbor_object",
            # selection_strategy_kwargs=dict(nn_k=3),
            subtask_term_offset_range=[[10, 20], None, [10, 20], None, [10, 20], None, [10, 20], None],
        ),
        # hammer_cleanup
        "hammer_cleanup":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "hammer_cleanup.hdf5"),
            dataset_name="hammer_cleanup",
            generation_path="{}/hammer_cleanup".format(OUTPUT_FOLDER),
            # task_interface="MG_HammerCleanup",
            tasks=["HammerCleanup_D0", "HammerCleanup_D1"],
            task_names=["D0", "D1"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[10, 20], [10, 20], None],
        ),
        # mug_cleanup
        "mug_cleanup":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "mug_cleanup.hdf5"),
            dataset_name="mug_cleanup",
            generation_path="{}/mug_cleanup".format(OUTPUT_FOLDER),
            # task_interface="MG_MugCleanup",
            tasks=["MugCleanup_D0", "MugCleanup_D1", "MugCleanup_O1", "MugCleanup_O2"],
            task_names=["D0", "D1", "O1", "O2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[10, 20], [10, 20], None],
        ),
        # kitchen
        "kitchen":dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "kitchen.hdf5"),
            dataset_name="kitchen",
            generation_path="{}/kitchen".format(OUTPUT_FOLDER),
            # task_interface="MG_Kitchen",
            tasks=["Kitchen_D0", "Kitchen_D1"],
            task_names=["D0", "D1"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[10, 20], [10, 20], [10, 20], [10, 20], [10, 20], [10, 20], None],
        ),
}

# update config with external json ——> mg_config


def get_mg_config(args, ext_cfg):
    if "meta" in ext_cfg:
        del ext_cfg["meta"]

    test1 = ext_cfg["type"]

    all_config = get_all_registered_configs()

    mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])

    with mg_config.values_unlocked():

        mg_config.update(ext_cfg)
        # We assume that the external config specifies all subtasks, so
        # delete any subtasks not in the external config.
        source_subtasks = set(mg_config.task.task_spec.keys())
        new_subtasks = set(ext_cfg["task"]["task_spec"].keys())
        for subtask in (source_subtasks - new_subtasks):
            print("deleting subtask {} in original config".format(subtask))
            del mg_config.task.task_spec[subtask]

        # maybe override some settings
        if args.task_name is not None:
            mg_config.experiment.task.name = args.task_name

        if args.source is not None:
            mg_config.experiment.source.dataset_path = args.source

        if args.folder is not None:
            mg_config.experiment.generation.path = args.folder

        if args.num_demos is not None:
            mg_config.experiment.generation.num_trials = args.num_demos

        if args.seed is not None:
            mg_config.experiment.seed = args.seed

        # maybe modify config for debugging purposes
        if args.debug:
            # shrink length of generation to test whether this run is likely to crash
            mg_config.experiment.source.n = 3
            mg_config.experiment.generation.guarantee = False
            mg_config.experiment.generation.num_trials = 2

            # send output to a temporary directory
            mg_config.experiment.generation.path = "/tmp/tmp_mimicgen"

    return mg_config
