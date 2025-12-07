# Quick Start Guide

## Installation (5 minutes)

Use Python 3.11 (not 3.14).

### 1. Create Virtual Environment
```bash
cd /home/ubuntu
python3.11 -m venv fluid_sim_env
source fluid_sim_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install opencv-python mediapipe scipy numpy
```

### 3. Run the Application
```bash
python fluid_interaction.py
```

### Quick Fix (macOS, using python3.11 if installed)
```bash
deactivate
rm -rf fluid_sim_env
python3.11 -m venv fluid_sim_env
source fluid_sim_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python fluid_interaction.py
```

## First Run

1. **Allow Camera Access**: Grant permission when prompted
2. **Position Yourself**: Sit 1-2 meters from camera, hands visible
3. **Good Lighting**: Ensure adequate lighting for hand detection
4. **Start Interacting**: Move your hands to create fluid effects

## Basic Controls

| Key | Action |
|-----|--------|
| **1-5** | Change visualization mode |
| **C** | Cycle colors |
| **H** | Show/hide hands |
| **SPACE** | Pause simulation |
| **R** | Reset |
| **Q** | Quit |

## Visualization Modes

1. **Density** - Colored fluid concentration (best for beginners)
2. **Velocity** - Arrow field showing flow direction
3. **Particles** - Particle effects (most dramatic)
4. **Streamlines** - Flow paths visualization
5. **Combined** - All effects blended (most beautiful)

## Tips for Best Results

### Hand Tracking
- Keep hands fully in frame
- Ensure good lighting (natural light preferred)
- Move hands smoothly and deliberately
- Use both hands for complex patterns

### Visual Quality
- Use **Mode 5 (Combined)** for best visuals
- Cycle through **colormaps (C key)** to find your favorite
- **Water** colormap is most realistic
- **Fire** colormap is most dramatic

### Performance
- If FPS drops below 20, switch to Mode 1 (Density)
- Close other applications
- Reduce simulation resolution in code if needed

## Troubleshooting

### "Could not open webcam"
```bash
# Check if webcam is accessible
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAILED')"
```

### Hand Not Detected
- Increase lighting
- Move hands closer to camera
- Ensure hands are fully visible
- Try different hand positions

### Low FPS
- Switch to simpler visualization (Mode 1 or 2)
- Reduce particle count
- Close other applications
- Update GPU drivers

### Simulation Unstable
- Decrease hand movement speed
- Press R to reset
- Increase viscosity in code

## Next Steps

1. Explore different visualization modes
2. Try different colormaps
3. Experiment with two-hand interactions
4. Check README.md for advanced configuration
5. Modify physics parameters in `fluid_simulator.py`

## File Structure

```
/home/ubuntu/
├── fluid_interaction.py      # Main application
├── hand_tracker.py           # Hand detection
├── fluid_simulator.py        # Physics engine
├── visualizer.py             # Rendering
├── advanced_effects.py       # Visual effects
├── README.md                 # Full documentation
└── QUICKSTART.md            # This file
```

## Performance Expectations

- **Target FPS**: 30+
- **Latency**: ~100ms (hand to effect)
- **Memory**: ~200-300MB
- **CPU**: 30-50% single core

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Hand not detected | Improve lighting, move closer |
| Low FPS | Use simpler visualization mode |
| Jerky movements | Ensure smooth hand motion |
| Unstable fluid | Reduce hand velocity influence |
| Black screen | Check webcam permissions |

## Advanced Tips

### Physics Tuning
Edit `fluid_interaction.py` line 28-35:
- Lower `viscosity` = more turbulent
- Higher `viscosity` = more smooth
- Lower `decay` = longer-lasting effects

### Resolution Adjustment
Edit `fluid_interaction.py` line 20-22:
- Higher `sim_width/height` = more detail (slower)
- Lower = faster but less detailed

### Effect Intensity
Edit `visualizer.py` methods:
- `add_glow_effect()` - Adjust intensity parameter
- `add_motion_blur()` - Modify blend factors

## Getting Help

1. Check README.md for detailed documentation
2. Review inline code comments
3. Experiment with different settings
4. Check system requirements

## Enjoy!

You now have a real-time fluid simulation system controlled by your hand gestures. Have fun creating beautiful fluid patterns!

For more details, see README.md
