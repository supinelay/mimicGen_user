#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : hpb
# @time    : 2024/12/4 15:34
# @function: the script is used to do something.
# @version : V1
import numpy as np
from robomimic.envs.env_robosuite import EnvRobosuite

def step_process(
        env,
        env_interface,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None):

    max_step = 300

    write_video = (video_writer is not None)
    video_count = 0

    states = []
    actions = []
    observations = []
    datagen_infos = []

    success = {k: False for k in env.is_success()}  # success metrics

    # iterate over waypoints in each sequence
    for j in range(max_step):
        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        # current waypoint
        waypoint = seq[j]

        # current state and obs
        state = env.get_state()["states"]
        obs = env.get_observation()

        # convert target pose to arm action
        action_pose = env_interface.target_pose_to_action(target_pose=waypoint.pose)

        # maybe add noise to action
        if waypoint.noise is not None:
            action_pose += waypoint.noise * np.random.randn(*action_pose.shape)
            action_pose = np.clip(action_pose, -1., 1.)

        # add in gripper action
        play_action = np.concatenate([action_pose, waypoint.gripper_action], axis=0)

        # store datagen info too
        datagen_info = env_interface.get_datagen_info(action=play_action)

        # step environment
        env.step(play_action)

        # collect data
        states.append(state)
        play_action_record = play_action
        actions.append(play_action_record)
        observations.append(obs)
        datagen_infos.append(datagen_info)

        cur_success_metrics = env.is_success()
        for k in success:
            success[k] = success[k] or cur_success_metrics[k]

    results = dict(
        states=states,
        observations=observations,
        datagen_infos=datagen_infos,
        actions=np.array(actions),
        success=bool(success["task"]),
    )

    return results





if __name__ == "__main__":

    # 内置的环境
    envs = ["Lift", "Stack", "NutAssembly", "NutAssemblySingle",
            "NutAssemblySquare", "NutAssemblyRound", "PickPlace",
            "PickPlaceSingle", "PickPlaceMilk", "PickPlaceBread",
            "PickPlaceCereal", "PickPlaceCan", "Door", "Wipe", "ToolHang",
            "TwoArmLift", "TwoArmPegInHole", "TwoArmHandover", "TwoArmTransport", "MyCustomTask"]

    # 内置的机器人
    robot = ["Panda", "Sawyer", "Baxter"]

    # 针对"TwoArm"任务的  双手机器人仅"Baxter", 其余都用单手配置
    config = ["bimanual", "single-arm-opposed"]


    env_config = {"env_name": "PickPlace",
                  "robots": robot[0],
                  "env_configuration": config[1]}

    robosuite_env = EnvRobosuite(env_name="Stack",
                                 env_configuration=config[1],
                                 robots=robot[0],
                                 has_renderer=True,  # no on-screen renderer
                                 has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
                                 render_camera="frontview",
                                 ignore_done=True,  # (optional) never terminates episode
                                 use_camera_obs=True,  # use camera observations
                                 camera_heights=84,  # set camera height
                                 camera_widths=84,  # set camera width
                                 camera_names="agentview",  # use "agentview" camera
                                 use_object_obs=False,  # no object feature when training on pixels
                                 reward_shaping=True,  # (optional) using a shaping reward,
                                 render=True,
                                 render_offscreen=False,
                                 use_image_obs=False,
                                 use_depth_obs=False,
                                 postprocess_visual_obs=True)


    action_min, action_max = robosuite_env.env.action_spec

    robosuite_env.reset()

    for i in range(1000):

        action = np.random.uniform(action_min, action_max)

        obs, reward, done, info = robosuite_env.step(action)
        # 渲染
        robosuite_env.render()

        print(f"奖励为{reward}")

    robosuite_env.env.close()


