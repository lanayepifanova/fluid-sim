import cv2
import numpy as np
from typing import Tuple, Optional
import colorsys

class FluidVisualizer:
    """Advanced visualization for fluid simulation"""
    
    def __init__(self, width: int = 256, height: int = 256):
        """
        Initialize visualizer
        
        Args:
            width, height: Dimensions of visualization
        """
        self.width = width
        self.height = height
        
    def density_to_color(self, density: np.ndarray, colormap: str = 'water') -> np.ndarray:
        """
        Convert density field to RGB image
        
        Args:
            density: Density field (0-255)
            colormap: Color scheme ('water', 'fire', 'plasma', 'cool')
            
        Returns:
            RGB image
        """
        # Normalize to 0-1
        normalized = density.astype(np.float32) / 255.0
        
        if colormap == 'water':
            # Blue-cyan gradient for water effect
            r = np.zeros_like(normalized)
            g = normalized * 0.7
            b = normalized
            
        elif colormap == 'fire':
            # Red-yellow-white gradient
            r = np.ones_like(normalized)
            g = np.clip(normalized * 2 - 0.5, 0, 1)
            b = np.clip(normalized * 4 - 3, 0, 1)
            
        elif colormap == 'plasma':
            # Plasma colormap
            r = np.clip(4 * normalized - 2, 0, 1)
            g = np.clip(4 * normalized - 1, 0, 1)
            b = np.clip(4 * normalized, 0, 1)
            
        elif colormap == 'cool':
            # Cyan-magenta gradient
            r = normalized
            g = np.ones_like(normalized)
            b = 1 - normalized
            
        else:
            # Grayscale
            r = g = b = normalized
        
        # Convert to 8-bit BGR (OpenCV format)
        image = np.stack([
            (b * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (r * 255).astype(np.uint8)
        ], axis=2)
        
        return image
        
    def velocity_to_arrows(self, vel_x: np.ndarray, vel_y: np.ndarray, 
                          step: int = 16, scale: float = 1.0) -> np.ndarray:
        """
        Visualize velocity field as arrows
        
        Args:
            vel_x, vel_y: Velocity components
            step: Spacing between arrows
            scale: Arrow scale factor
            
        Returns:
            RGB image with velocity arrows
        """
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                vx = vel_x[y, x] * scale
                vy = vel_y[y, x] * scale
                
                magnitude = np.sqrt(vx**2 + vy**2)
                
                if magnitude > 0.5:
                    # Color based on magnitude
                    color_val = min(int(magnitude * 10), 255)
                    color = (color_val, 255 - color_val, 128)
                    
                    # Draw arrow
                    end_x = int(x + vx)
                    end_y = int(y + vy)
                    
                    cv2.arrowedLine(image, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        
        return image
        
    def create_particle_effect(self, density: np.ndarray, vel_x: np.ndarray, 
                              vel_y: np.ndarray, num_particles: int = 1000) -> np.ndarray:
        """
        Create particle-based visualization
        
        Args:
            density: Density field
            vel_x, vel_y: Velocity components
            num_particles: Number of particles to render
            
        Returns:
            RGB image with particle effects
        """
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sample random particles from high-density regions
        density_normalized = density.astype(np.float32) / 255.0
        density_flat = density_normalized.flatten()
        
        # Use density as probability distribution
        if density_flat.sum() > 0:
            probabilities = density_flat / density_flat.sum()
            indices = np.random.choice(len(density_flat), size=min(num_particles, len(density_flat)), 
                                      p=probabilities, replace=True)
            
            for idx in indices:
                y = idx // self.width
                x = idx % self.width
                
                # Get velocity at this point
                vx = vel_x[y, x]
                vy = vel_y[y, x]
                
                # Color based on velocity magnitude
                vel_mag = np.sqrt(vx**2 + vy**2)
                hue = (vel_mag / 10.0) % 1.0
                saturation = 0.8
                value = density_normalized[y, x]
                
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                
                # Draw particle
                radius = int(1 + value * 3)
                cv2.circle(image, (x, y), radius, 
                          (int(b*255), int(g*255), int(r*255)), -1)
        
        return image
        
    def blend_images(self, img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blend two images
        
        Args:
            img1, img2: Input images
            alpha: Blend factor (0-1)
            
        Returns:
            Blended image
        """
        return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
        
    def add_glow_effect(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Add glow effect to image
        
        Args:
            image: Input image
            intensity: Glow intensity
            
        Returns:
            Image with glow effect
        """
        # Create blurred version for glow
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        
        # Blend with original
        result = cv2.addWeighted(image, 1.0, blurred, intensity, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def add_motion_blur(self, image: np.ndarray, vel_x: np.ndarray, vel_y: np.ndarray) -> np.ndarray:
        """
        Add motion blur based on velocity field
        
        Args:
            image: Input image
            vel_x, vel_y: Velocity components
            
        Returns:
            Image with motion blur
        """
        # Create motion blur kernel based on average velocity
        avg_vel_x = np.mean(np.abs(vel_x))
        avg_vel_y = np.mean(np.abs(vel_y))
        
        if avg_vel_x > 0.1 or avg_vel_y > 0.1:
            kernel_size = int(3 + (avg_vel_x + avg_vel_y) * 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return cv2.addWeighted(image, 0.7, result, 0.3, 0)
        
        return image
        
    def create_streamlines(self, vel_x: np.ndarray, vel_y: np.ndarray, 
                          num_lines: int = 20, length: int = 50) -> np.ndarray:
        """
        Create streamline visualization
        
        Args:
            vel_x, vel_y: Velocity components
            num_lines: Number of streamlines
            length: Length of each streamline
            
        Returns:
            RGB image with streamlines
        """
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Random starting points
        for _ in range(num_lines):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            
            points = [(x, y)]
            
            # Trace streamline
            for _ in range(length):
                if 0 <= int(y) < self.height and 0 <= int(x) < self.width:
                    vx = vel_x[int(y), int(x)]
                    vy = vel_y[int(y), int(x)]
                    
                    mag = np.sqrt(vx**2 + vy**2)
                    if mag > 0.1:
                        x += vx * 0.5
                        y += vy * 0.5
                        points.append((int(x), int(y)))
                    else:
                        break
                else:
                    break
            
            # Draw streamline with gradient color
            if len(points) > 1:
                for i in range(len(points) - 1):
                    alpha = i / len(points)
                    color_val = int(255 * alpha)
                    cv2.line(image, points[i], points[i+1], 
                            (255, color_val, 100), 1)
        
        return image
