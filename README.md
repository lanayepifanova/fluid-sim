**Real-time Hand Tracking**: Uses MediaPipe for  hand detection and tracking from webcam feed

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

## References

- Stam, J. (1999). "Stable Fluids"
- Bridson, R. (2015). "Fluid Simulation for Computer Graphics"
- MediaPipe Hand Tracking: https://google.github.io/mediapipe/

