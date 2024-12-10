import os

from models import robosuite as suite
from env.robosuite_env.pick_place import *
from env.robosuite_env.lift import *



if __name__ == "__main__":

    env_config = {"env_name": "LiftAny",
                  "robots": "Panda",
                  "env_configuration": None}

    # 指定文件夹路径
    folder_path = '../models/assets/objects'

    # 创建一个空列表用于存储文件名
    obj_name = []

    for filename in os.listdir(folder_path):
        # 如果是XML文件（以.xml结尾）
        if filename.endswith('.xml'):
            obj_name.append(str(filename)[:-4])

    # Create environment
    env = suite.make(
        **env_config,
        obj_name=obj_name[0],
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_heights=84,
        camera_widths=84,
        camera_names=["agentview"],
        use_object_obs=False,
        reward_shaping=True,

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
        assert pr + "proprio-state" in obs
        assert obs[pr + "proprio-state"].ndim == 1

        assert "agentview_image" in obs
        assert obs["agentview_image"].shape == (84, 84, 3)

        assert "object-state" not in obs

        action = np.random.uniform(action_min, action_max)

        obs, reward, done, info = env.step(action)
        # 渲染
        env.render()

        print(f"奖励为{reward}")

    env.close()



