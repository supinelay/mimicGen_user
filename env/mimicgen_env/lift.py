#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : hpb
# @time    : 2024/12/6 11:14
# @function: 自定义的拿取任务.
# @version : V1

import numpy as np
from env.mimicgen_env.single_arm_env_mg import SingleArmEnv_MG_C

from robosuite.models.arenas.table_arena import TableArena

from env.robosuite_env.lift import *
from models.arenas.desk_arena import DeskArena


class LiftAny_D0(LiftAny, SingleArmEnv_MG_C):
    """
    自定义的拿取任务，可以自定义拿取单个对象，初始化给入对象名称

    """

    def __init__(self,**kwargs):
        LiftAny.__init__(self, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG_C.edit_model_xml(self, xml_str)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = DeskArena()
        # mujoco_arena = TableArena(
        #     table_full_size=self.table_full_size,
        #     table_friction=self.table_friction,
        #     table_offset=self.table_offset,
        # )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.cube = AnyObject(name=self.obj_name)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()


            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():


                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
