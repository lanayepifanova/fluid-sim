import time

import cv2
import numpy as np

from capture import CameraCapture
from config import AppConfig
from controls import ControlState, get_instructions, handle_key
from fluid_simulator import FluidSimulator, FluidConfig
from hand_tracker import HandTracker
from visualizer import FluidVisualizer


class FluidInteractionSystem:
    """Main system integrating hand tracking and fluid simulation."""

    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config

        self.hand_tracker = HandTracker(
            max_hands=config.max_hands,
            detection_confidence=config.detection_confidence,
        )

        sim_config = FluidConfig(
            width=config.sim_width,
            height=config.sim_height,
            diffusion=config.diffusion,
            viscosity=config.viscosity,
            decay=config.decay,
            pressure_iterations=config.pressure_iterations,
        )
        self.fluid_sim = FluidSimulator(sim_config)
        self.visualizer = FluidVisualizer(config.sim_width, config.sim_height)
        self.capture = CameraCapture(config)

        self.controls = ControlState(
            visualization_mode=config.visualization_mode,
            colormap=config.colormap,
        )

        self.frame_count = 0
        self.fps = 0.0
        self.sim_frame_index = 0
        self.prev_density = None
        self.prev_vel_x = None
        self.prev_vel_y = None
        self.effect_frame_index = 0
        self.last_effect_vis = None

    def process_hand_interaction(self, hand_data_list):
        for hand_data in hand_data_list:
            if not hand_data.is_detected:
                continue

            x = int(hand_data.position[0] * self.config.sim_width / self.config.display_width)
            y = int(hand_data.position[1] * self.config.sim_height / self.config.display_height)
            x = np.clip(x, 0, self.config.sim_width - 1)
            y = np.clip(y, 0, self.config.sim_height - 1)

            radius = (
                self.hand_tracker.get_hand_radius(hand_data)
                * (self.config.sim_width / self.config.display_width)
            )
            radius = np.clip(radius, self.config.radius_min, self.config.radius_max)

            vx = hand_data.velocity[0] * self.config.velocity_scale
            vy = hand_data.velocity[1] * self.config.velocity_scale
            self.fluid_sim.add_velocity(x, y, vx, vy, radius)
            self.fluid_sim.add_density(x, y, amount=self.config.density_amount, radius=radius)

    def render_frame(self, camera_frame, density=None, vel_x=None, vel_y=None):
        if density is None:
            density = self.fluid_sim.get_density_field()
        if vel_x is None or vel_y is None:
            vel_x = self.fluid_sim.velocity_x
            vel_y = self.fluid_sim.velocity_y

        mode = self.controls.visualization_mode
        colormap = self.controls.colormap

        if mode == "density":
            fluid_vis = self.visualizer.density_to_color(density, colormap)
        elif mode == "velocity":
            fluid_vis = self.visualizer.velocity_to_arrows(
                vel_x,
                vel_y,
                step=self.config.velocity_arrow_step,
                scale=self.config.velocity_arrow_scale,
            )
        elif mode == "particles":
            fluid_vis = self.visualizer.create_particle_effect(
                density,
                vel_x,
                vel_y,
                num_particles=self.config.particles_count,
            )
        elif mode == "streamlines":
            fluid_vis = self.visualizer.create_streamlines(
                vel_x,
                vel_y,
                num_lines=self.config.streamlines_num_lines,
                length=self.config.streamlines_length,
            )
        elif mode == "combined":
            density_vis = self.visualizer.density_to_color(density, colormap)
            streamline_vis = self.visualizer.create_streamlines(
                vel_x,
                vel_y,
                num_lines=self.config.combined_streamlines_num_lines,
                length=self.config.combined_streamlines_length,
            )
            particle_vis = self.visualizer.create_particle_effect(
                density,
                vel_x,
                vel_y,
                num_particles=self.config.combined_particles_count,
            )
            fluid_vis = self.visualizer.blend_images(density_vis, streamline_vis, 0.7)
            fluid_vis = self.visualizer.blend_images(fluid_vis, particle_vis, 0.2)
        else:
            fluid_vis = self.visualizer.density_to_color(density, colormap)

        if self.effect_frame_index % self.config.effect_stride == 0 or self.last_effect_vis is None:
            effect_vis = self.visualizer.add_glow_effect(fluid_vis, intensity=self.config.glow_intensity)
            effect_vis = self.visualizer.add_motion_blur(effect_vis, vel_x, vel_y)
            self.last_effect_vis = effect_vis
        else:
            effect_vis = self.last_effect_vis
        self.effect_frame_index += 1

        display_fluid = cv2.resize(
            effect_vis,
            (self.config.display_width, self.config.display_height),
            interpolation=cv2.INTER_LINEAR,
        )

        camera_resized = cv2.resize(camera_frame, (self.config.display_width, self.config.display_height))
        output = cv2.addWeighted(
            display_fluid,
            self.config.fluid_blend,
            camera_resized,
            self.config.camera_blend,
            0,
        )
        return output, display_fluid

    def draw_ui(self, frame, hand_data_list):
        if not self.controls.show_info:
            return frame

        mode_text = f"Mode: {self.controls.visualization_mode} | Colormap: {self.controls.colormap}"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        hand_text = f"Hands: {len([h for h in hand_data_list if h.is_detected])}"
        cv2.putText(frame, hand_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        instructions = get_instructions()
        y_offset = self.config.display_height - len(instructions) * 25
        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
        return frame

    def run(self):
        print("Starting Fluid Interaction System...")
        print("Press 'Q' to quit, 'H' for help")

        last_time = time.time()

        try:
            while True:
                camera_frame = self.capture.read()
                if camera_frame is None:
                    print("Failed to read frame")
                    break

                camera_frame = cv2.flip(camera_frame, 1)

                tracking_frame = cv2.resize(
                    camera_frame, (self.config.tracking_width, self.config.tracking_height)
                )
                hand_data_list = self.hand_tracker.process_frame(tracking_frame)

                if hand_data_list:
                    scale_x = self.config.display_width / self.config.tracking_width
                    scale_y = self.config.display_height / self.config.tracking_height
                    for hand_data in hand_data_list:
                        hand_data.position[0] *= scale_x
                        hand_data.position[1] *= scale_y
                        hand_data.velocity[0] *= scale_x
                        hand_data.velocity[1] *= scale_y
                        if hand_data.landmarks is not None:
                            hand_data.landmarks[:, 0] *= scale_x
                            hand_data.landmarks[:, 1] *= scale_y

                did_sim_step = False
                if not self.controls.paused:
                    if self.sim_frame_index % self.config.sim_step_stride == 0:
                        self.prev_density = self.fluid_sim.density.copy()
                        self.prev_vel_x = self.fluid_sim.velocity_x.copy()
                        self.prev_vel_y = self.fluid_sim.velocity_y.copy()
                        self.process_hand_interaction(hand_data_list)
                        self.fluid_sim.step(dt=1.0)
                        did_sim_step = True
                    self.sim_frame_index += 1

                if self.prev_density is not None and not self.controls.paused:
                    alpha = 1.0 if did_sim_step else 0.5
                    interp_density = self.prev_density * (1.0 - alpha) + self.fluid_sim.density * alpha
                    interp_vel_x = self.prev_vel_x * (1.0 - alpha) + self.fluid_sim.velocity_x * alpha
                    interp_vel_y = self.prev_vel_y * (1.0 - alpha) + self.fluid_sim.velocity_y * alpha
                    density_field = np.clip(interp_density, 0, 255).astype(np.uint8)
                    output_frame, _ = self.render_frame(
                        camera_frame,
                        density=density_field,
                        vel_x=interp_vel_x,
                        vel_y=interp_vel_y,
                    )
                else:
                    output_frame, _ = self.render_frame(camera_frame)

                if self.controls.show_hands:
                    camera_resized = cv2.resize(
                        camera_frame, (self.config.display_width, self.config.display_height)
                    )
                    camera_with_hands = self.hand_tracker.draw_hands(camera_resized, hand_data_list)
                    output_frame = cv2.addWeighted(
                        output_frame,
                        1.0 - self.config.hands_blend,
                        camera_with_hands,
                        self.config.hands_blend,
                        0,
                    )

                output_frame = self.draw_ui(output_frame, hand_data_list)
                cv2.imshow("Fluid Interaction System", output_frame)

                key = cv2.waitKey(1) & 0xFF
                action = handle_key(key, self.controls, self.config.colormaps, self.fluid_sim)
                if action == "quit":
                    break

                self.frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - last_time)
                    self.frame_count = 0
                    last_time = current_time
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.capture.release()
            cv2.destroyAllWindows()
            print("System shutdown complete")


def main():
    system = FluidInteractionSystem(AppConfig())
    system.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
