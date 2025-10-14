from isaaclab.utils import configclass
from dataclasses import MISSING
# import isaaclab_tasks.manager_based.locomotion.velocity.terrain as terrain
from . import terrain
#import isaaclab_tasks.manager_based.locomotion.velocity.hf_terrains as hf_terrains 
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg


@configclass
class HfUniformDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain with different obstacle types."""

    function = terrain.uniform_discrete_obstacles_terrain
    seed: float = 0
    """Env seed."""
    
    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    
    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    
    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    
    obstacles_distance: float = MISSING
    """The minimum distance between obstacles (in m)."""
    
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    
    avoid_positions: list[list[float, float]] = []
    """List of positions to avoid generating obstacles."""


# class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
#     """Configuration for a discrete obstacles height field terrain."""

#     function = hf_terrains.discrete_obstacles_terrain

#     obstacle_height_mode: str = "choice"
#     """The mode to use for the obstacle height. Defaults to "choice".

#     The following modes are supported: "choice", "fixed".
#     """
#     obstacle_width_range: tuple[float, float] = MISSING
#     """The minimum and maximum width of the obstacles (in m)."""
#     # obstacle_height_range: tuple[float, float] = MISSING
#     obstacle_height_range: list = MISSING
#     """The minimum and maximum height of the obstacles (in m)."""
#     obstacle_height_probability: list = MISSING
#     num_obstacles: int = MISSING
#     """The number of obstacles to generate."""
#     platform_width: float = 1.0
#     """The width of the square platform at the center of the terrain. Defaults to 1.0."""
