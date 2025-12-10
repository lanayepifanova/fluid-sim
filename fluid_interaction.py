#!/usr/bin/env python3
"""
Realistic Fluid Simulation with Hand Control
Hand movements control water physics in real-time
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hand_tracker import HandTracker, HandData
from fluid_simulator import FluidSimulator, FluidConfig
from visualizer import FluidVisualizer


class FluidInteractionSystem:
    """Main system integrating hand tracking and fluid simulation"""
    
    def __init__(self, sim_width: int = 512, sim_height: int = 512, 
                 display_width: int = 1024, display_height: int = 768):
        """
        Initialize the system
        
        Args:
            sim_width, sim_height: Simulation resolution
            display_width, display_height: Display resolution
        """
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.display_width = display_width
        self.display_height = display_height
        
        # Initialize components
        self.hand_tracker = HandTracker(max_hands=2, detection_confidence=0.7)
        
        config = FluidConfig(
            width=sim_width,
            height=sim_height,
            diffusion=0.0002,
            viscosity=0.00001,
            decay=0.995,
            pressure_iterations=20
        )
        self.fluid_sim = FluidSimulator(config)
        self.visualizer = FluidVisualizer(sim_width, sim_height)
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # UI state
        self.visualization_mode = 'density'  # density, velocity, particles, streamlines, combined
        self.colormap = 'water'  # water, fire, plasma, cool
        self.paused = False
        self.show_hands = True
        self.show_info = True
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        
    def process_hand_interaction(self, hand_data_list):
        """
        Process hand tracking data and apply to fluid
        
        Args:
            hand_data_list: List of detected hands
        """
        for hand_data in hand_data_list:
            if hand_data.is_detected:
                # Scale hand position to simulation resolution
                x = int(hand_data.position[0] * self.sim_width / self.display_width)
                y = int(hand_data.position[1] * self.sim_height / self.display_height)
                
                # Clamp to bounds
                x = np.clip(x, 0, self.sim_width - 1)
                y = np.clip(y, 0, self.sim_height - 1)
                
                # Get hand radius for influence area
                radius = self.hand_tracker.get_hand_radius(hand_data) * (self.sim_width / self.display_width)
                radius = np.clip(radius, 10, 100)
                
                # Add velocity to fluid
                vx = hand_data.velocity[0] * 2.0
                vy = hand_data.velocity[1] * 2.0
                
                self.fluid_sim.add_velocity(x, y, vx, vy, radius)
                
                # Add density for visualization
                self.fluid_sim.add_density(x, y, amount=150, radius=radius)
                
    def render_frame(self, camera_frame):
        """
        Render a complete frame with all visualizations
        
        Args:
            camera_frame: Input camera frame
            
        Returns:
            Rendered output frame
        """
        # Get simulation fields
        density = self.fluid_sim.get_density_field()
        vel_x = self.fluid_sim.velocity_x
        vel_y = self.fluid_sim.velocity_y
        
        # Create visualization based on mode
        if self.visualization_mode == 'density':
            fluid_vis = self.visualizer.density_to_color(density, self.colormap)
            
        elif self.visualization_mode == 'velocity':
            fluid_vis = self.visualizer.velocity_to_arrows(vel_x, vel_y, step=8, scale=2.0)
            
        elif self.visualization_mode == 'particles':
            fluid_vis = self.visualizer.create_particle_effect(density, vel_x, vel_y, num_particles=2000)
            
        elif self.visualization_mode == 'streamlines':
            fluid_vis = self.visualizer.create_streamlines(vel_x, vel_y, num_lines=30, length=50)
            
        elif self.visualization_mode == 'combined':
            # Combine multiple visualizations
            density_vis = self.visualizer.density_to_color(density, self.colormap)
            particle_vis = self.visualizer.create_particle_effect(density, vel_x, vel_y, num_particles=1000)
            streamline_vis = self.visualizer.create_streamlines(vel_x, vel_y, num_lines=15, length=40)
            
            fluid_vis = self.visualizer.blend_images(density_vis, particle_vis, 0.6)
            fluid_vis = self.visualizer.blend_images(fluid_vis, streamline_vis, 0.4)
        
        # Add effects
        fluid_vis = self.visualizer.add_glow_effect(fluid_vis, intensity=0.3)
        fluid_vis = self.visualizer.add_motion_blur(fluid_vis, vel_x, vel_y)
        
        # Resize to display resolution
        display_fluid = cv2.resize(fluid_vis, (self.display_width, self.display_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Overlay camera feed (optional)
        camera_resized = cv2.resize(camera_frame, (self.display_width, self.display_height))
        
        # Blend camera with fluid visualization
        output = cv2.addWeighted(display_fluid, 0.85, camera_resized, 0.15, 0)
        
        return output, display_fluid
        
    def draw_ui(self, frame, hand_data_list):
        """
        Draw UI elements on frame
        
        Args:
            frame: Output frame
            hand_data_list: List of detected hands
            
        Returns:
            Frame with UI
        """
        if not self.show_info:
            return frame
        
        # Draw mode info
        mode_text = f"Mode: {self.visualization_mode} | Colormap: {self.colormap}"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Draw hand count
        hand_text = f"Hands: {len([h for h in hand_data_list if h.is_detected])}"
        cv2.putText(frame, hand_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "1-5: Change visualization mode",
            "C: Cycle colormap",
            "H: Toggle hand visualization",
            "I: Toggle info",
            "SPACE: Pause/Resume",
            "R: Reset",
            "Q: Quit"
        ]
        
        y_offset = self.display_height - len(instructions) * 25
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
        
    def run(self):
        """Main application loop"""
        import time
        
        print("Starting Fluid Interaction System...")
        print("Press 'Q' to quit, 'H' for help")
        
        last_time = time.time()
        
        try:
            while True:
                ret, camera_frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Flip for mirror effect
                camera_frame = cv2.flip(camera_frame, 1)
                
                # Track hands
                hand_data_list = self.hand_tracker.process_frame(camera_frame)
                
                # Update simulation
                if not self.paused:
                    self.process_hand_interaction(hand_data_list)
                    self.fluid_sim.step(dt=1.0)
                
                # Render
                output_frame, fluid_vis = self.render_frame(camera_frame)
                
                # Draw hand tracking
                if self.show_hands:
                    camera_resized = cv2.resize(camera_frame, (self.display_width, self.display_height))
                    camera_with_hands = self.hand_tracker.draw_hands(camera_resized, hand_data_list)
                    output_frame = cv2.addWeighted(output_frame, 0.8, camera_with_hands, 0.2, 0)
                
                # Draw UI
                output_frame = self.draw_ui(output_frame, hand_data_list)
                
                # Display
                cv2.imshow('Fluid Interaction System', output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.visualization_mode = 'density'
                elif key == ord('2'):
                    self.visualization_mode = 'velocity'
                elif key == ord('3'):
                    self.visualization_mode = 'particles'
                elif key == ord('4'):
                    self.visualization_mode = 'streamlines'
                elif key == ord('5'):
                    self.visualization_mode = 'combined'
                elif key == ord('c'):
                    colormaps = ['water', 'fire', 'plasma', 'cool', 'gray']
                    idx = colormaps.index(self.colormap)
                    self.colormap = colormaps[(idx + 1) % len(colormaps)]
                elif key == ord('h'):
                    self.show_hands = not self.show_hands
                elif key == ord('i'):
                    self.show_info = not self.show_info
                elif key == ord(' '):
                    self.paused = not self.paused
                elif key == ord('r'):
                    self.fluid_sim.reset()
                
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - last_time)
                    self.frame_count = 0
                    last_time = current_time
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("System shutdown complete")


if __name__ == '__main__':
    try:
        system = FluidInteractionSystem(
            sim_width=512,
            sim_height=512,
            display_width=1024,
            display_height=768
        )
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
