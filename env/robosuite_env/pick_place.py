# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Slight variant of pick place task.
"""
import random

import numpy as np
from robosuite import PickPlace
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import BinsArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from robosuite.models.objects import (
    BreadObject,
    CanObject,
    CerealObject,
    MilkObject,
)


from models.objects import WoodBallObject, AnyObject

OBJECT_LIBRARY = {
    "milk": 0,
    "bread": 1,
    "cereal": 2,
    "can": 3,
    "wood_ball": 4,
}


class PickPlace_D0(PickPlace):
    """
    Slightly easier task where we limit z-rotation to 0 to 90 degrees for all object initializations (instead of full 360).
    """
    def __init__(self, **kwargs):
        assert "z_rotation" not in kwargs
        super().__init__(
            z_rotation=(0., np.pi / 2.),
            **kwargs,
        )


class PickPlaceAny(PickPlace):
    """
    Slightly easier task where we limit z-rotation to 0 to 90 degrees for all object initializations (instead of full 360).
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.39, 0.49, 0.82),
        table_friction=(1, 0.005, 0.0001),
        bin1_pos=(0.1, -0.25, 0.8),
        bin2_pos=(0.1, 0.28, 0.8),
        z_offset=0.,
        z_rotation=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        single_object_mode=1,
        object_type=None,
        has_visual_object=False,   # 额外增加的， object =（visual模型 + 模型） or (模型)
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):

        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = OBJECT_LIBRARY
        self.object_id_to_sensors = {}  # Maps object id to sensor names for that object
        self.obj_names = list(OBJECT_LIBRARY.keys())

        # todo  后续自动化添加其他物体
        if object_type is not None:
            assert object_type in self.object_to_id.keys(), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[object_type]  # use for convenient indexing

        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # settings for bin position
        self.bin1_pos = np.array(bin1_pos)
        self.bin2_pos = np.array(bin2_pos)
        self.z_offset = z_offset  # z offset for initializing items in bin
        self.z_rotation = z_rotation  # z rotation for initializing items in bin

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # 是否使用visual object
        self.has_visual_object = has_visual_object

        # 显式继承 SingleArmEnv
        SingleArmEnv.__init__(self,
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )


    def _load_model(self):
        """
        重写PickPlace的加载方法， Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = BinsArena(
            bin1_pos=self.bin1_pos, table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size = mujoco_arena.table_full_size

        # make objects
        if self.has_visual_object:
            self._construct_visual_objects()

        self._construct_objects()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects if not self.has_visual_object else self.objects+self.visual_objects,
        )

        # Generate placement initializer
        self._get_placement_initializer()

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # can sample anywhere in bin
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=self.z_offset,
            )
        )

        if self.has_visual_object:
            # each visual object should just be at the center of each target bin
            index = 0
            for vis_obj in self.visual_objects:

                # get center of target bin
                bin_x_low = self.bin2_pos[0]
                bin_y_low = self.bin2_pos[1]
                if index == 0 or index == 2:
                    bin_x_low -= self.bin_size[0] / 2
                if index < 2:
                    bin_y_low -= self.bin_size[1] / 2
                bin_x_high = bin_x_low + self.bin_size[0] / 2
                bin_y_high = bin_y_low + self.bin_size[1] / 2
                bin_center = np.array(
                    [
                        (bin_x_low + bin_x_high) / 2.0,
                        (bin_y_low + bin_y_high) / 2.0,
                    ]
                )

                # placement is relative to object bin, so compute difference and send to placement initializer
                rel_center = bin_center - self.bin1_pos[:2]

                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{vis_obj.name}ObjectSampler",
                        mujoco_objects=vis_obj,
                        x_range=[rel_center[0], rel_center[0]],
                        y_range=[rel_center[1], rel_center[1]],
                        rotation=0.0,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.bin1_pos,
                        z_offset=self.bin2_pos[2] - self.bin1_pos[2],
                    )
                )
                index += 1


    def _construct_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.objects = []
        for obj_cls, obj_name in zip(
                (MilkObject, BreadObject, CerealObject, CanObject, WoodBallObject),
                self.obj_names,
        ):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        SingleArmEnv._setup_references(self)

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # 判断需要处理的所有对象
        if self.has_visual_object:
            target_object = self.visual_objects + self.objects
        else:
            target_object = self.objects

        # object-specific ids
        for obj in target_object:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.objects))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i, obj in enumerate(self.objects):
            bin_id = i
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.0
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.0
            bin_x_low += self.bin_size[0] / 4.0
            bin_y_low += self.bin_size[1] / 4.0
            self.target_bin_placements[i, :] = [bin_x_low, bin_y_low, self.bin2_pos[2]]



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
                # Set the visual object body locations
                # if "visual" in obj.name.lower():
                self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                # else:
                #     # Set the collision object joints
                #     self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Set the bins to the desired position
        self.sim.model.body_pos[self.sim.model.body_name2id("bin1")] = self.bin1_pos
        self.sim.model.body_pos[self.sim.model.body_name2id("bin2")] = self.bin2_pos

        # Move objects out of the scene depending on the mode
        obj_names = {obj.name for obj in self.objects}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(obj_names))
            for obj_type, i in self.object_to_id.items():
                if obj_type.lower() in self.obj_to_use.lower():
                    self.object_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.objects[self.object_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            self.clear_objects(list(obj_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.object_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active object, else False
                    self._observables[name].set_enabled(i == self.object_id)
                    self._observables[name].set_active(i == self.object_id)

class PickPlaceAny2(PickPlace):
    """
    Slightly easier task where we limit z-rotation to 0 to 90 degrees for all object initializations (instead of full 360).
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.39, 0.49, 0.82),
        table_friction=(1, 0.005, 0.0001),
        bin1_pos=(0.1, -0.25, 0.8),
        bin2_pos=(0.1, 0.28, 0.8),
        z_offset=0.,
        z_rotation=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        single_object_mode=0,
        has_visual_object=False,   # 额外增加的， object =（visual模型 + 模型） or (模型)
        obj_name_list=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):

        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = OBJECT_LIBRARY
        self.object_id_to_sensors = {}  # Maps object id to sensor names for that object
        self.obj_names = obj_name_list

        # todo  后续自动化添加其他物体
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # settings for bin position
        self.bin1_pos = np.array(bin1_pos)
        self.bin2_pos = np.array(bin2_pos)
        self.z_offset = z_offset  # z offset for initializing items in bin
        self.z_rotation = z_rotation  # z rotation for initializing items in bin

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # 是否使用visual object
        self.has_visual_object = has_visual_object

        # 显式继承 SingleArmEnv
        SingleArmEnv.__init__(self,
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )


    def _load_model(self):
        """
        重写PickPlace的加载方法， Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = BinsArena(
            bin1_pos=self.bin1_pos, table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # store some arena attributes
        self.bin_size = mujoco_arena.table_full_size

        # make objects
        if self.has_visual_object:
            self._construct_visual_objects()

        self._construct_objects()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects if not self.has_visual_object else self.objects+self.visual_objects,
        )

        # Generate placement initializer
        self._get_placement_initializer()

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # can sample anywhere in bin
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=self.z_offset,
            )
        )

        if self.has_visual_object:
            # each visual object should just be at the center of each target bin
            index = 0
            for vis_obj in self.visual_objects:

                # get center of target bin
                bin_x_low = self.bin2_pos[0]
                bin_y_low = self.bin2_pos[1]
                if index == 0 or index == 2:
                    bin_x_low -= self.bin_size[0] / 2
                if index < 2:
                    bin_y_low -= self.bin_size[1] / 2
                bin_x_high = bin_x_low + self.bin_size[0] / 2
                bin_y_high = bin_y_low + self.bin_size[1] / 2
                bin_center = np.array(
                    [
                        (bin_x_low + bin_x_high) / 2.0,
                        (bin_y_low + bin_y_high) / 2.0,
                    ]
                )

                # placement is relative to object bin, so compute difference and send to placement initializer
                rel_center = bin_center - self.bin1_pos[:2]

                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{vis_obj.name}ObjectSampler",
                        mujoco_objects=vis_obj,
                        x_range=[rel_center[0], rel_center[0]],
                        y_range=[rel_center[1], rel_center[1]],
                        rotation=0.0,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.bin1_pos,
                        z_offset=self.bin2_pos[2] - self.bin1_pos[2],
                    )
                )
                index += 1


    def _construct_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.objects = []
        for obj_name in self.obj_names:
            obj = AnyObject(name=obj_name)
            self.objects.append(obj)


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        SingleArmEnv._setup_references(self)

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # 判断需要处理的所有对象
        if self.has_visual_object:
            target_object = self.visual_objects + self.objects
        else:
            target_object = self.objects

        # object-specific ids
        for obj in target_object:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        # keep track of which objects are in their corresponding bins
        self.objects_in_bins = np.zeros(len(self.objects))

        # target locations in bin for each object type
        self.target_bin_placements = np.zeros((len(self.objects), 3))
        for i, obj in enumerate(self.objects):
            bin_id = i
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if bin_id == 0 or bin_id == 2:
                bin_x_low -= self.bin_size[0] / 2.0
            if bin_id < 2:
                bin_y_low -= self.bin_size[1] / 2.0
            bin_x_low += self.bin_size[0] / 4.0
            bin_y_low += self.bin_size[1] / 4.0
            self.target_bin_placements[i, :] = [bin_x_low, bin_y_low, self.bin2_pos[2]]



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
                # Set the visual object body locations
                # if "visual" in obj.name.lower():
                self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                # else:
                #     # Set the collision object joints
                #     self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Set the bins to the desired position
        self.sim.model.body_pos[self.sim.model.body_name2id("bin1")] = self.bin1_pos
        self.sim.model.body_pos[self.sim.model.body_name2id("bin2")] = self.bin2_pos

        # Move objects out of the scene depending on the mode
        obj_names = {obj.name for obj in self.objects}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(obj_names))
            for obj_type, i in self.object_to_id.items():
                if obj_type.lower() in self.obj_to_use.lower():
                    self.object_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.objects[self.object_id].name
        if self.single_object_mode in {1, 2}:
            obj_names.remove(self.obj_to_use)
            self.clear_objects(list(obj_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.object_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active object, else False
                    self._observables[name].set_enabled(i == self.object_id)
                    self._observables[name].set_active(i == self.object_id)













