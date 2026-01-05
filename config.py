from dataclasses import dataclass


@dataclass
class AppConfig:
    sim_width: int = 384
    sim_height: int = 384
    display_width: int = 1024
    display_height: int = 768
    tracking_width: int = 640
    tracking_height: int = 360

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    max_hands: int = 2
    detection_confidence: float = 0.7
    velocity_scale: float = 2.0
    density_amount: int = 150
    radius_min: int = 10
    radius_max: int = 100

    diffusion: float = 0.0002
    viscosity: float = 0.00001
    decay: float = 0.995
    pressure_iterations: int = 8

    visualization_mode: str = "density"
    colormap: str = "water"
    colormaps: tuple = ("water", "fire", "plasma", "cool", "gray")

    velocity_arrow_step: int = 8
    velocity_arrow_scale: float = 2.0
    particles_count: int = 2000
    streamlines_num_lines: int = 30
    streamlines_length: int = 50

    combined_streamlines_num_lines: int = 20
    combined_streamlines_length: int = 45
    combined_particles_count: int = 400

    glow_intensity: float = 0.3
    effect_stride: int = 2
    sim_step_stride: int = 2

    fluid_blend: float = 0.85
    camera_blend: float = 0.15
    hands_blend: float = 0.2
