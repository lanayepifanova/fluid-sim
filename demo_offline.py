#!/usr/bin/env python3
"""
Offline Demo - Test fluid simulation without webcam
Useful for testing and development without camera access
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from fluid_simulator import FluidSimulator, FluidConfig
from visualizer import FluidVisualizer
from advanced_effects import AdvancedEffects


class OfflineDemo:
    """Offline demonstration of fluid simulation"""
    
    def __init__(self, width: int = 512, height: int = 512):
        """Initialize demo"""
        self.width = width
        self.height = height
        
        config = FluidConfig(
            width=width,
            height=height,
            diffusion=0.0002,
            viscosity=0.00001,
            decay=0.995,
            pressure_iterations=20
        )
        
        self.fluid_sim = FluidSimulator(config)
        self.visualizer = FluidVisualizer(width, height)
        
        self.time_step = 0
        self.mode = 0
        self.colormap = 'water'
        
    def add_random_forces(self):
        """Add random forces to simulate hand interaction"""
        # Add forces at random locations
        for _ in range(3):
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            
            vx = np.random.randn() * 10
            vy = np.random.randn() * 10
            
            radius = np.random.randint(20, 50)
            
            self.fluid_sim.add_velocity(x, y, vx, vy, radius)
            self.fluid_sim.add_density(x, y, amount=100, radius=radius)
    
    def add_circular_motion(self):
        """Add circular motion pattern"""
        center_x = self.width // 2
        center_y = self.height // 2
        radius = 100
        
        angle = self.time_step * 0.05
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        vx = -radius * np.sin(angle) * 0.1
        vy = radius * np.cos(angle) * 0.1
        
        self.fluid_sim.add_velocity(x, y, vx, vy, 30)
        self.fluid_sim.add_density(x, y, amount=150, radius=30)
    
    def add_wave_pattern(self):
        """Add wave pattern"""
        for i in range(5):
            x = int(self.width * (0.2 + i * 0.15))
            y = int(self.height * (0.5 + 0.2 * np.sin(self.time_step * 0.05 + i)))
            
            self.fluid_sim.add_velocity(x, y, 5, 0, 20)
            self.fluid_sim.add_density(x, y, amount=100, radius=20)
    
    def add_explosion(self):
        """Add explosion effect"""
        if self.time_step % 60 == 0:
            x = self.width // 2
            y = self.height // 2
            
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                vx = 20 * np.cos(angle)
                vy = 20 * np.sin(angle)
                
                self.fluid_sim.add_velocity(x, y, vx, vy, 40)
            
            self.fluid_sim.add_density(x, y, amount=200, radius=50)
    
    def render_frame(self):
        """Render current frame"""
        density = self.fluid_sim.get_density_field()
        vel_x = self.fluid_sim.velocity_x
        vel_y = self.fluid_sim.velocity_y
        
        if self.mode == 0:
            # Density
            frame = self.visualizer.density_to_color(density, self.colormap)
        elif self.mode == 1:
            # Velocity
            frame = self.visualizer.velocity_to_arrows(vel_x, vel_y, step=8, scale=2.0)
        elif self.mode == 2:
            # Particles
            frame = self.visualizer.create_particle_effect(density, vel_x, vel_y, num_particles=2000)
        elif self.mode == 3:
            # Streamlines
            frame = self.visualizer.create_streamlines(vel_x, vel_y, num_lines=30, length=50)
        else:
            # Combined
            density_vis = self.visualizer.density_to_color(density, self.colormap)
            particle_vis = self.visualizer.create_particle_effect(density, vel_x, vel_y, num_particles=1000)
            streamline_vis = self.visualizer.create_streamlines(vel_x, vel_y, num_lines=15, length=40)
            
            frame = self.visualizer.blend_images(density_vis, particle_vis, 0.6)
            frame = self.visualizer.blend_images(frame, streamline_vis, 0.4)
        
        # Add effects
        frame = self.visualizer.add_glow_effect(frame, intensity=0.3)
        frame = self.visualizer.add_motion_blur(frame, vel_x, vel_y)
        
        return frame
    
    def draw_ui(self, frame):
        """Draw UI on frame"""
        modes = ['Density', 'Velocity', 'Particles', 'Streamlines', 'Combined']
        
        cv2.putText(frame, f"Mode: {modes[self.mode]} | Colormap: {self.colormap}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {self.time_step}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        instructions = [
            "Controls:",
            "1-5: Change mode",
            "C: Cycle colormap",
            "SPACE: Pause",
            "R: Reset",
            "Q: Quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, self.height - 150 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Run demo"""
        print("Starting Offline Demo...")
        print("Press 'Q' to quit")
        
        paused = False
        pattern = 0  # 0: random, 1: circular, 2: wave, 3: explosion
        
        try:
            while True:
                if not paused:
                    # Add forces based on pattern
                    if pattern == 0:
                        self.add_random_forces()
                    elif pattern == 1:
                        self.add_circular_motion()
                    elif pattern == 2:
                        self.add_wave_pattern()
                    else:
                        self.add_explosion()
                    
                    # Simulate
                    self.fluid_sim.step()
                    self.time_step += 1
                
                # Render
                frame = self.render_frame()
                frame = self.draw_ui(frame)
                
                # Display
                cv2.imshow('Offline Fluid Demo', frame)
                
                # Handle input
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.mode = 0
                elif key == ord('2'):
                    self.mode = 1
                elif key == ord('3'):
                    self.mode = 2
                elif key == ord('4'):
                    self.mode = 3
                elif key == ord('5'):
                    self.mode = 4
                elif key == ord('c'):
                    colormaps = ['water', 'fire', 'plasma', 'cool']
                    idx = colormaps.index(self.colormap)
                    self.colormap = colormaps[(idx + 1) % len(colormaps)]
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('r'):
                    self.fluid_sim.reset()
                    self.time_step = 0
                elif key == ord('p'):
                    pattern = (pattern + 1) % 4
                    self.fluid_sim.reset()
                
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cv2.destroyAllWindows()
            print("Demo complete")


if __name__ == '__main__':
    demo = OfflineDemo(width=512, height=512)
    demo.run()
