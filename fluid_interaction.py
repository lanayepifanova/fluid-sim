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
        self.tracking_width = 640
        self.tracking_height = 360
        
        # Initialize components
        self.hand_tracker = HandTracker(max_hands=2, detection_confidence=0.7)
        
        config = FluidConfig(
            width=sim_width,
            height=sim_height,
            diffusion=0.0002,
            viscosity=0.00001,
            decay=0.995,
            pressure_iterations=8
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
        self.sim_frame_index = 0
        self.prev_density = None
        self.prev_vel_x = None
        self.prev_vel_y = None
        self.effect_frame_index = 0
        self.effect_stride = 2
        self.last_effect_vis = None
        
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
                
    def render_frame(self, camera_frame, density=None, vel_x=None, vel_y=None):
        """
        Render a complete frame with all visualizations
        
        Args:
            camera_frame: Input camera frame
            density: Optional density field override
            vel_x, vel_y: Optional velocity field overrides
            
        Returns:
            Rendered output frame
        """
        # Get simulation fields
        if density is None:
            density = self.fluid_sim.get_density_field()
        if vel_x is None or vel_y is None:
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
            streamline_vis = self.visualizer.create_streamlines(vel_x, vel_y, num_lines=20, length=45)
            particle_vis = self.visualizer.create_particle_effect(density, vel_x, vel_y, num_particles=400)
            
            fluid_vis = self.visualizer.blend_images(density_vis, streamline_vis, 0.7)
            fluid_vis = self.visualizer.blend_images(fluid_vis, particle_vis, 0.2)
        
        # Add effects (throttled to every N frames)
        if self.effect_frame_index % self.effect_stride == 0 or self.last_effect_vis is None:
            effect_vis = self.visualizer.add_glow_effect(fluid_vis, intensity=0.3)
            effect_vis = self.visualizer.add_motion_blur(effect_vis, vel_x, vel_y)
            self.last_effect_vis = effect_vis
        else:
            effect_vis = self.last_effect_vis
        self.effect_frame_index += 1
        fluid_vis = effect_vis
        
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
                tracking_frame = cv2.resize(camera_frame, (self.tracking_width, self.tracking_height))
                hand_data_list = self.hand_tracker.process_frame(tracking_frame)
                
                if hand_data_list:
                    scale_x = self.display_width / self.tracking_width
                    scale_y = self.display_height / self.tracking_height
                    for hand_data in hand_data_list:
                        hand_data.position[0] *= scale_x
                        hand_data.position[1] *= scale_y
                        hand_data.velocity[0] *= scale_x
                        hand_data.velocity[1] *= scale_y
                        if hand_data.landmarks is not None:
                            hand_data.landmarks[:, 0] *= scale_x
                            hand_data.landmarks[:, 1] *= scale_y
                
                # Update simulation
                did_sim_step = False
                if not self.paused:
                    if self.sim_frame_index % 2 == 0:
                        self.prev_density = self.fluid_sim.density.copy()
                        self.prev_vel_x = self.fluid_sim.velocity_x.copy()
                        self.prev_vel_y = self.fluid_sim.velocity_y.copy()
                        self.process_hand_interaction(hand_data_list)
                        self.fluid_sim.step(dt=1.0)
                        did_sim_step = True
                    self.sim_frame_index += 1
                
                # Render with interpolation on non-sim frames
                if self.prev_density is not None and not self.paused:
                    alpha = 1.0 if did_sim_step else 0.5

                    interp_density = (
                        self.prev_density * (1.0 - alpha) + self.fluid_sim.density * alpha
                    )
                    interp_vel_x = (
                        self.prev_vel_x * (1.0 - alpha) + self.fluid_sim.velocity_x * alpha
                    )
                    interp_vel_y = (
                        self.prev_vel_y * (1.0 - alpha) + self.fluid_sim.velocity_y * alpha
                    )
                    density_field = np.clip(interp_density, 0, 255).astype(np.uint8)
                    output_frame, fluid_vis = self.render_frame(
                        camera_frame,
                        density=density_field,
                        vel_x=interp_vel_x,
                        vel_y=interp_vel_y
                    )
                else:
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
            sim_width=384,
            sim_height=384,
            display_width=1024,
            display_height=768
        )
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
