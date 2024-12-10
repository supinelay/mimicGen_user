from models import robosuite as suite
from env.robosuite_env.pick_place import *
from env.robosuite_env.lift import *



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
    # 其余任务都用
    config2 = None

    env_config = {"env_name": "Stack",
                  "robots": robot[0],
                  "env_configuration": config[1]}

    # Create environment
    env = suite.make(
        **env_config,
        has_renderer=True,  # no on-screen renderer
        has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
        render_camera="frontview",
        ignore_done=True,  # (optional) never terminates episode
        use_camera_obs=True,  # use camera observations
        camera_heights=84,  # set camera height
        camera_widths=84,  # set camera width
        camera_names=["agentview","robot0_robotview"],  # use "agentview" camera
        use_object_obs=False,  # no object feature when training on pixels
        reward_shaping=True,  # (optional) using a shaping reward
        # obj_name="wood_ball",
    )

    # 重置环境，初始化状态
    obs = env.reset()

    # get action range
    action_min, action_max = env.action_spec
    assert action_min.shape == action_max.shape

    # Get robot prefix
    pr = env.robots[0].robot_model.naming_prefix

    # run 10 random actions
    for _ in range(500):
        test1 = env.sim.data
        assert pr + "proprio-state" in obs
        assert obs[pr + "proprio-state"].ndim == 1

        # assert "agentview_image" in obs
        # assert obs["agentview_image"].shape == (84, 84, 3)

        assert "object-state" not in obs

        action = np.random.uniform(action_min, action_max)

        obs, reward, done, info = env.step(action)
        # 渲染
        env.render()

        print(f"奖励为{reward}")

    env.close()



