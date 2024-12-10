#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : hpb
# @time    : 2024/12/9 14:16
# @function: the script is used to do something.
# @version : V1

import numpy as np

from mimicgen.env_interfaces.base import MG_EnvInterface
from mimicgen import RobosuiteInterface


"""
描述： 自定义任务场景的配置
        1. 获取目标物的位姿
        2. 获取子任务的结束信号
        

"""


class MG_Lift(RobosuiteInterface):
    """
    Corresponds to robosuite Lift task and variants.
    """

    def get_object_poses(self):
        """
        Returns: 目标位姿
        """
        object_poses = dict()
        object_poses["cube"] = self.get_object_pose(obj_name=self.env.cube.root_body, obj_type="body")
        return object_poses

    def get_subtask_term_signals(self):
        """
        Returns: 终止信号
        """
        signals = dict()

        # checks which objects are on their correct pegs and records them in @self.objects_on_pegs
        self.env._check_success()

        signals["grasp"] = int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube))

        # signals["grasp_{}".format(obj_name)] = int(self.env._check_grasp(
        #     gripper=self.env.robots[0].gripper,
        #     object_geoms=[g for g in self.env.objects[obj_id].contact_geoms])
        # )

        return signals


