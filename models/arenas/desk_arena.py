import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class DeskArena(Arena):
    def __init__(
            self,
            table_full_size=(1.54, 0.615, 0.733),
            table_friction=(1, 0.005, 0.0001),
            table_offset=(0, 0, 0.733),
            xml="/home/exploit-01/project/minicgen_new/models/assets/arenas/table_scene/scene_test_temp.xml"
    ):
        super().__init__(xml)
        self.has_legs = False
        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2.0
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = self.bottom_pos + np.array([0, 0, -self.table_full_size[2]]) + self.table_offset

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.configure_location()

    def configure_location(self):
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set("pos", array_to_string(np.array([0, 0, self.table_half_size[2]])))

    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset
