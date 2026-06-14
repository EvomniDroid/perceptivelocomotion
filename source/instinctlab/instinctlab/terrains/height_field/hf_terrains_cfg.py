from dataclasses import MISSING
from typing import List

from isaaclab.terrains.height_field import (
    HfDiscreteObstaclesTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfTerrainBaseCfg,
    HfWaveTerrainCfg,
)
from isaaclab.utils import configclass

from . import hf_terrains


class WallTerrainCfgMixin:
    wall_prob: List[float] = [0.0, 0.0, 0.0, 0.0]  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls


@configclass
class PerlinPlaneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_plane_terrain

    noise_scale: float | List[float] = 0.05
    noise_frequency: int = 20

    fractal_octaves = 2
    fractal_lacunarity = 2.0
    fractal_gain = 0.25

    centering = False  # If True, the noise will be centered around 0


@configclass
class PerlinPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_pyramid_sloped_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinInvertedPyramidSlopedTerrainCfg(HfInvertedPyramidSlopedTerrainCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_pyramid_sloped_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_pyramid_stairs_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinInvertedPyramidStairsTerrainCfg(HfInvertedPyramidStairsTerrainCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_pyramid_stairs_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_discrete_obstacles_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinWaveTerrainCfg(HfWaveTerrainCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_wave_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinSteppingStonesTerrainCfg(HfSteppingStonesTerrainCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_stepping_stones_terrain
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


# -- Newly added terrain configurations for parkour terrains-- #
@configclass
class PerlinParapetTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a parapet terrain, can be used for jump and hurdle tasks."""

    function = hf_terrains.perlin_parapet_terrain
    parapet_height: tuple[float, float] | float = (0.1, 0.3)
    parapet_length: tuple[float, float] | float = (0.1, 0.3)
    parapet_width: float | None = None
    curved_top_rate: float | None = None
    """The rate to generate curved top. If None, the top will be flat."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinGutterTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a gutter parkour terrain."""

    function = hf_terrains.perlin_gutter_terrain
    gutter_length: tuple[float, float] | float = (0.5, 1.5)  # the distance between gutters
    gutter_depth: tuple[float, float] | float = (0.1, 0.3)  # the depth of the gutter
    gutter_width: float | None = None  # the length of the gutter
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinBowlPitTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a raised mound/hill terrain on flat ground.

    The terrain is a raised circular platform/hill where:
    - Top of the mound is flat at specified height
    - Edges have vertical walls (step up)
    - Robot must climb up and over the mound
    """

    function = hf_terrains.perlin_bowl_pit_terrain
    pit_depth: tuple[float, float] | float = (0.05, 1.0)  # height of the mound (5cm to 100cm)
    pit_radius: tuple[float, float] | float = (0.5, 1.5)  # radius of the mound
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinPitTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a pit/crater terrain below ground level.

    The terrain is a flat-bottomed pit where:
    - Bottom of the pit is below ground (negative height)
    - Edges have vertical walls (90-degree drop)
    - Robot spawns in the pit and must jump out
    """

    function = hf_terrains.perlin_pit_terrain
    pit_depth: tuple[float, float] | float = (0.05, 1.0)  # depth of the pit (5cm to 100cm below ground)
    pit_radius: tuple[float, float] | float = (0.5, 1.5)  # radius of the pit
    raise_surrounding_ground: bool = False
    """Whether to raise the surrounding ground by the pit depth, so the pit bottom stays near ground height."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinStairsUpDownTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs up and down parkour terrain."""

    function = hf_terrains.perlin_stairs_up_down_terrain
    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinStairsDownUpTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a stairs down and up parkour terrain."""

    function = hf_terrains.perlin_stairs_down_up_terrain
    per_step_height: tuple[float, float] | float = MISSING
    """The height of each step. Could be a fixed value or a range (min, max)."""
    per_step_width: float | None = None
    """The width of each step. If None, it will be equal to the width of the terrain."""
    per_step_length: tuple[float, float] | float = MISSING
    """The length of each step along the y-axis."""
    num_steps: tuple[int, int] | int = MISSING
    """The number of steps. Could be a fixed value or a range (min, max)."""

    platform_length: float = 1.0
    """The length of the platform at the bottom of the stairs."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinTiltTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilt terrain."""

    function = hf_terrains.perlin_tilt_terrain
    wall_height: tuple[float, float] | float = MISSING
    wall_width: float | None = None
    wall_length: tuple[float, float] | float = MISSING
    wall_opening_angle: tuple[float, float] | float = MISSING  # in degrees
    wall_opening_width: tuple[float, float] | float = MISSING
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinTiltedRampTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a tilted ramp terrain."""

    function = hf_terrains.perlin_tilted_ramp_terrain
    tilt_angle: tuple[float, float] | float = MISSING  # in degrees
    tilt_height: tuple[float, float] | float = MISSING
    tilt_width: tuple[float, float] | float = MISSING
    tilt_length: tuple[float, float] | float = MISSING
    switch_spacing: tuple[float, float] | float = MISSING
    spacing_curriculum: bool | None = None
    overlap_size: float | None = None
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinSlopeTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a slope up and down terrain with a flat ground in the middle."""

    function = hf_terrains.perlin_slope_terrain
    slope_angle: tuple[float, float] | float = MISSING  # in degrees
    per_slope_length: tuple[float, float] | float = MISSING
    platform_length: float = 1.0
    slope_width: float | None = None
    up_down: bool | None = None  # If True or None, the slope will be up and down, otherwise it will be down and up.
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinCrossStoneTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a cross stone terrain."""

    function = hf_terrains.perlin_cross_stone_terrain
    stone_size: tuple[float, float] = MISSING
    stone_height: tuple[float, float] | float = MISSING
    stone_spacing: tuple[float, float] | float = MISSING
    ground_depth: float = -0.5
    platform_width: float = 1.5
    xy_random_ratio: float = 0.2
    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinSquareGapTerrainCfg(HfTerrainBaseCfg, WallTerrainCfgMixin):
    function = hf_terrains.perlin_square_gap_terrain

    gap_distance_range: tuple[float, float] = (0.1, 0.5)
    gap_depth: tuple[float, float] = (0.2, 0.5)
    platform_width: float = 1.5
    border_width: float = 0.0

    perlin_cfg: PerlinPlaneTerrainCfg | None = None


@configclass
class PerlinCircleTrackTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a circular track terrain.

    The terrain is a flat circular ring path where the robot walks in circles.
    The center of the terrain is flat but the robot follows a circular path.
    This encourages the robot to learn turning skills.
    """

    function = hf_terrains.perlin_circle_track_terrain
    track_radius: tuple[float, float] | float = (3.0, 6.0)
    track_width: float = 1.5
    track_depth: tuple[float, float] | float = (0.1, 0.3)
    center_size: float = 2.0
    perlin_cfg: PerlinPlaneTerrainCfg | None = None
