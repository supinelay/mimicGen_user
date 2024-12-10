from robosuite import make

# Manipulation environments
from robosuite import Lift
from robosuite import Stack
from robosuite import NutAssembly
from robosuite import PickPlace
from robosuite import Door
from robosuite import Wipe
from robosuite import ToolHang
from robosuite import TwoArmLift
from robosuite import TwoArmPegInHole
from robosuite import TwoArmHandover
from robosuite import TwoArmTransport

from robosuite import ALL_ENVIRONMENTS
from robosuite import ALL_CONTROLLERS, load_controller_config
from robosuite import ALL_ROBOTS
from robosuite import ALL_GRIPPERS





# 自定义的robosuite环境导入
from env.robosuite_env.lift import LiftAny
from env.robosuite_env.pick_place import PickPlaceAny, PickPlaceAny2

# 自定义的MimicGen环境导入
from env.mimicgen_env.lift import LiftAny_D0


__version__ = "1.4.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
