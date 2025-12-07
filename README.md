# Realistic Fluid Simulation with Hand Control

A visually dramatic, real-time fluid simulation system that responds to hand movements tracked via your laptop's webcam. This system combines advanced computer vision, realistic physics simulation, and sophisticated visualization techniques to create an interactive experience.

## Features

### Core Functionality

- **Real-time Hand Tracking**: Uses MediaPipe for robust hand detection and tracking from webcam feed
- **Realistic Fluid Physics**: Implements Navier-Stokes equations with:
  - Semi-Lagrangian advection for stable fluid transport
  - Pressure projection for incompressibility
  - Viscous diffusion for realistic fluid behavior
  - Velocity decay and dissipation

- **Interactive Control**: Hand movements directly control fluid dynamics:
  - Hand position determines force application point
  - Hand velocity determines force magnitude and direction
  - Multiple hands can interact simultaneously
  - Hand size affects influence radius

### Visualization Modes

1. **Density Visualization**: Shows fluid concentration with color gradients
2. **Velocity Field**: Displays velocity vectors as arrows
3. **Particle Effects**: Renders particles traced through the fluid
4. **Streamlines**: Shows fluid flow paths with gradient coloring
5. **Combined Mode**: Blends multiple visualizations for dramatic effect

### Visual Effects

- **Glow Effects**: Bloom and glow for enhanced visual drama
- **Motion Blur**: Velocity-based motion blur for fluid smoothness
- **Refraction Mapping**: Density-based distortion effects
- **Caustics**: Animated water caustics patterns
- **Light Rays**: God rays effect from light sources
- **Foam Effects**: High-velocity region visualization
- **Vorticity Visualization**: Shows rotation and swirl patterns

### Color Schemes

- **Water**: Blue-cyan gradient for realistic water appearance
- **Fire**: Red-yellow-white for hot fluid effects
- **Plasma**: Vibrant plasma-like colors
- **Cool**: Cyan-magenta gradient
- **Viridis**: Perceptually uniform scientific colormap
- **Turbo**: High-contrast turbo colormap
- **Temperature**: Cold-to-hot temperature visualization

## System Architecture

### Components

1. **hand_tracker.py**: Computer vision module for hand detection and tracking
   - MediaPipe-based hand landmark detection
   - Position and velocity calculation
   - Hand radius estimation

2. **fluid_simulator.py**: Physics simulation engine
   - Navier-Stokes solver implementation
   - Semi-Lagrangian advection
   - Pressure projection
   - Diffusion and viscosity

3. **visualizer.py**: Rendering and visualization
   - Multiple visualization modes
   - Color mapping techniques
   - Effect rendering (glow, blur, etc.)

4. **advanced_effects.py**: Advanced visual effects
   - Refraction, caustics, light rays
   - Foam and vorticity visualization
   - Performance optimization utilities

5. **fluid_interaction.py**: Main application
   - System integration
   - Real-time rendering loop
   - User interface and controls

## Installation

### Requirements

- Python 3.11 (not 3.14)
- Webcam
- Linux/macOS/Windows

### Setup

```bash
# Create virtual environment (macOS/Linux, Python 3.11)
python3.11 -m venv fluid_sim_env
source fluid_sim_env/bin/activate  # On Windows: fluid_sim_env\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe scipy numpy

# Run the application
python fluid_interaction.py
```

Quick fix (macOS, using python3.11 if installed):

```bash
deactivate
rm -rf fluid_sim_env
python3.11 -m venv fluid_sim_env
source fluid_sim_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python fluid_interaction.py
```

## Usage

### Keyboard Controls

| Key | Action |
|-----|--------|
| **1** | Density visualization mode |
| **2** | Velocity field mode |
| **3** | Particle effects mode |
| **4** | Streamlines mode |
| **5** | Combined visualization mode |
| **C** | Cycle through color schemes |
| **H** | Toggle hand visualization overlay |
| **I** | Toggle information display |
| **SPACE** | Pause/Resume simulation |
| **R** | Reset simulation |
| **Q** | Quit application |

### Hand Interaction

1. **Position Control**: Move your hand to apply force at that location
2. **Velocity Control**: Faster hand movements create stronger fluid disturbances
3. **Multi-Hand**: Use both hands simultaneously for complex fluid patterns
4. **Hand Size**: Larger hand gestures affect larger fluid regions

## Configuration

Edit parameters in `fluid_interaction.py` to customize behavior:

```python
# Simulation resolution (higher = more detailed, slower)
sim_width = 512
sim_height = 512

# Display resolution
display_width = 1024
display_height = 768

# Fluid properties
config = FluidConfig(
    diffusion=0.0002,      # Higher = more diffuse
    viscosity=0.00001,     # Higher = more viscous
    decay=0.995,           # Velocity decay per frame
    pressure_iterations=20 # More = more accurate
)
```

## Physics Details

### Navier-Stokes Implementation

The simulator solves the incompressible Navier-Stokes equations:

```
∂u/∂t + (u·∇)u = -∇p + ν∇²u + f
∇·u = 0
```

Where:
- **u**: Velocity field
- **p**: Pressure
- **ν**: Kinematic viscosity
- **f**: External forces (hand interactions)

### Numerical Methods

1. **Advection**: Semi-Lagrangian with bilinear interpolation
2. **Diffusion**: Jacobi iteration for stable diffusion
3. **Pressure**: Jacobi iteration for pressure projection
4. **Boundary Conditions**: Dirichlet (constant value) at edges

## Performance Optimization

The system includes adaptive optimization:

- **Adaptive Resolution**: Automatically adjusts simulation resolution based on FPS
- **Particle Culling**: Reduces particle count if performance drops
- **Substep Reduction**: Decreases simulation steps for lower-end systems

Monitor performance via the on-screen FPS counter.

## Tips for Best Results

1. **Lighting**: Ensure good lighting for reliable hand tracking
2. **Distance**: Position yourself 1-2 meters from the camera
3. **Hand Visibility**: Keep hands in frame and visible
4. **Smooth Movements**: Smooth hand movements create more realistic fluid effects
5. **Combined Mode**: Use visualization mode 5 for most dramatic effects
6. **Glow Effects**: These enhance visual drama significantly

## Advanced Usage

### Customizing Physics

Modify `FluidConfig` in `fluid_simulator.py`:

```python
config = FluidConfig(
    width=512,
    height=512,
    diffusion=0.0002,      # Increase for more diffuse fluid
    viscosity=0.00001,     # Increase for thicker fluid
    decay=0.995,           # Decrease for longer-lasting effects
    pressure_iterations=20 # Increase for more accurate incompressibility
)
```

### Adding Custom Effects

Extend `visualizer.py` with new visualization methods:

```python
def custom_effect(self, density, vel_x, vel_y):
    # Your effect code here
    return effect_image
```

## Troubleshooting

### Hand Not Detected

- Ensure adequate lighting
- Keep hands fully visible in frame
- Check webcam is working: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

### Low FPS

- Reduce simulation resolution
- Switch to simpler visualization mode (1 or 2)
- Reduce particle count in `advanced_effects.py`
- Close other applications

### Simulation Unstable

- Decrease hand velocity influence
- Increase viscosity in `FluidConfig`
- Increase pressure iterations

## Technical Specifications

| Aspect | Specification |
|--------|---------------|
| **Physics Engine** | Navier-Stokes solver |
| **Hand Tracking** | MediaPipe (21 landmarks) |
| **Simulation Grid** | 512×512 (configurable) |
| **Display Resolution** | 1024×768 (configurable) |
| **Target FPS** | 30+ |
| **Max Hands** | 2 (configurable) |
| **Color Spaces** | RGB, HSV, Temperature |

## Performance Metrics

- **Typical FPS**: 25-35 (depending on hardware)
- **Latency**: ~50-100ms (hand to effect)
- **Memory Usage**: ~200-300MB
- **CPU Usage**: 30-50% (single core)

## Future Enhancements

- 3D fluid simulation
- GPU acceleration (CUDA/OpenGL)
- Sound synthesis from fluid dynamics
- Network multiplayer interaction
- VR integration
- Machine learning gesture recognition

## License

This project is provided as-is for educational and entertainment purposes.

## References

- Stam, J. (1999). "Stable Fluids"
- Bridson, R. (2015). "Fluid Simulation for Computer Graphics"
- MediaPipe Hand Tracking: https://google.github.io/mediapipe/

## Support

For issues or questions, check that:
1. All dependencies are installed correctly
2. Webcam is accessible and working
3. System meets minimum requirements
4. Hand is clearly visible to camera

Enjoy creating fluid art!
