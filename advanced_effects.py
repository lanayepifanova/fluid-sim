import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class AdvancedEffects:
    """Advanced visual effects for fluid simulation"""
    
    @staticmethod
    def create_refraction_map(density: np.ndarray, intensity: float = 0.05) -> np.ndarray:
        """
        Create refraction distortion based on density gradients
        
        Args:
            density: Density field
            intensity: Refraction intensity
            
        Returns:
            Displacement map
        """
        # Compute gradients
        grad_x = np.gradient(density.astype(np.float32), axis=1)
        grad_y = np.gradient(density.astype(np.float32), axis=0)
        
        # Normalize
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        grad_x = grad_x / magnitude * intensity
        grad_y = grad_y / magnitude * intensity
        
        return grad_x, grad_y
    
    @staticmethod
    def apply_refraction(image: np.ndarray, disp_x: np.ndarray, disp_y: np.ndarray) -> np.ndarray:
        """
        Apply refraction distortion to image
        
        Args:
            image: Input image
            disp_x, disp_y: Displacement maps
            
        Returns:
            Refracted image
        """
        h, w = image.shape[:2]
        yy, xx = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (xx + disp_x).astype(np.float32)
        map_y = (yy + disp_y).astype(np.float32)
        
        # Remap
        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def create_caustics(density: np.ndarray, time_step: int = 0, scale: float = 1.0) -> np.ndarray:
        """
        Generate caustics pattern based on fluid dynamics
        
        Args:
            density: Density field
            time_step: Animation frame number
            scale: Pattern scale
            
        Returns:
            Caustics pattern
        """
        h, w = density.shape
        
        # Create animated noise pattern
        yy, xx = np.meshgrid(np.arange(w), np.arange(h))
        
        # Multiple sine waves for caustics
        caustics = np.zeros_like(density, dtype=np.float32)
        
        for i in range(3):
            freq = 0.01 * (i + 1) * scale
            amp = 0.3 / (i + 1)
            
            wave1 = np.sin(xx * freq + time_step * 0.01) * np.cos(yy * freq)
            wave2 = np.sin(yy * freq - time_step * 0.015) * np.cos(xx * freq)
            
            caustics += amp * (wave1 + wave2)
        
        # Modulate by density
        caustics = (caustics + 1) / 2  # Normalize to 0-1
        caustics = caustics * (density.astype(np.float32) / 255.0)
        
        return (caustics * 255).astype(np.uint8)
    
    @staticmethod
    def create_light_rays(density: np.ndarray, light_pos: tuple = None) -> np.ndarray:
        """
        Create god rays / light rays effect
        
        Args:
            density: Density field
            light_pos: Light source position (x, y)
            
        Returns:
            Light rays image
        """
        h, w = density.shape
        
        if light_pos is None:
            light_pos = (w // 2, h // 2)
        
        # Create radial gradient from light source
        yy, xx = np.meshgrid(np.arange(w), np.arange(h))
        
        dist = np.sqrt((xx - light_pos[0])**2 + (yy - light_pos[1])**2)
        max_dist = np.sqrt(h**2 + w**2)
        
        # Inverse distance falloff
        rays = np.exp(-dist / (max_dist / 3))
        
        # Modulate by density
        rays = rays * (density.astype(np.float32) / 255.0)
        
        return (rays * 255).astype(np.uint8)
    
    @staticmethod
    def create_foam_effect(vel_x: np.ndarray, vel_y: np.ndarray, 
                          threshold: float = 2.0) -> np.ndarray:
        """
        Create foam/bubble effect in high-velocity regions
        
        Args:
            vel_x, vel_y: Velocity components
            threshold: Velocity threshold for foam
            
        Returns:
            Foam pattern
        """
        vel_mag = np.sqrt(vel_x**2 + vel_y**2)
        
        # Foam appears where velocity is high
        foam = np.zeros_like(vel_mag, dtype=np.float32)
        foam[vel_mag > threshold] = 1.0
        
        # Smooth and add some randomness
        foam = gaussian_filter(foam, sigma=2.0)
        
        # Add turbulence
        noise = np.random.rand(*foam.shape) * 0.3
        foam = np.clip(foam + noise, 0, 1)
        
        return (foam * 255).astype(np.uint8)
    
    @staticmethod
    def create_edge_detection(density: np.ndarray) -> np.ndarray:
        """
        Detect edges in density field for contour visualization
        
        Args:
            density: Density field
            
        Returns:
            Edge map
        """
        # Sobel edge detection
        edges_x = cv2.Sobel(density, cv2.CV_32F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(density, cv2.CV_32F, 0, 1, ksize=3)
        
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        
        return edges
    
    @staticmethod
    def create_vorticity_visualization(vel_x: np.ndarray, vel_y: np.ndarray) -> np.ndarray:
        """
        Visualize vorticity (rotation) in the fluid
        
        Args:
            vel_x, vel_y: Velocity components
            
        Returns:
            Vorticity field visualization
        """
        # Compute vorticity: ∂v/∂x - ∂u/∂y
        dv_dx = np.gradient(vel_y, axis=1)
        du_dy = np.gradient(vel_x, axis=0)
        
        vorticity = dv_dx - du_dy
        
        # Normalize
        vort_mag = np.abs(vorticity)
        vort_mag = np.clip(vort_mag / (np.max(vort_mag) + 1e-6), 0, 1)
        
        # Color positive and negative vorticity differently
        image = np.zeros((*vorticity.shape, 3), dtype=np.uint8)
        
        # Positive vorticity (counter-clockwise) -> Red
        pos_mask = vorticity > 0
        image[pos_mask, 2] = (vort_mag[pos_mask] * 255).astype(np.uint8)
        
        # Negative vorticity (clockwise) -> Blue
        neg_mask = vorticity < 0
        image[neg_mask, 0] = (vort_mag[neg_mask] * 255).astype(np.uint8)
        
        return image


class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def adaptive_resolution(target_fps: float = 30.0, current_fps: float = 30.0,
                           base_width: int = 512, base_height: int = 512) -> tuple:
        """
        Adaptively adjust simulation resolution based on FPS
        
        Args:
            target_fps: Target FPS
            current_fps: Current FPS
            base_width, base_height: Base resolution
            
        Returns:
            Adjusted (width, height)
        """
        if current_fps < target_fps * 0.8:
            # Reduce resolution
            scale = 0.75
        elif current_fps < target_fps * 0.9:
            scale = 0.9
        elif current_fps > target_fps * 1.2:
            # Increase resolution
            scale = 1.1
        else:
            scale = 1.0
        
        new_width = int(base_width * scale)
        new_height = int(base_height * scale)
        
        # Keep dimensions as multiples of 16 for efficiency
        new_width = (new_width // 16) * 16
        new_height = (new_height // 16) * 16
        
        return new_width, new_height
    
    @staticmethod
    def reduce_simulation_steps(fps: float, base_steps: int = 1) -> int:
        """
        Reduce simulation substeps if FPS is low
        
        Args:
            fps: Current FPS
            base_steps: Base number of steps
            
        Returns:
            Adjusted number of steps
        """
        if fps < 20:
            return max(1, base_steps // 2)
        elif fps < 25:
            return max(1, base_steps - 1)
        else:
            return base_steps
    
    @staticmethod
    def optimize_particle_count(fps: float, base_count: int = 2000) -> int:
        """
        Adjust particle count based on performance
        
        Args:
            fps: Current FPS
            base_count: Base particle count
            
        Returns:
            Adjusted particle count
        """
        if fps < 20:
            return base_count // 4
        elif fps < 25:
            return base_count // 2
        elif fps > 50:
            return int(base_count * 1.5)
        else:
            return base_count


class EnhancedColorMapping:
    """Advanced color mapping techniques"""
    
    @staticmethod
    def perceptual_colormap(density: np.ndarray, colormap_type: str = 'viridis') -> np.ndarray:
        """
        Apply perceptually uniform colormaps
        
        Args:
            density: Density field (0-255)
            colormap_type: Type of colormap
            
        Returns:
            RGB image
        """
        # Normalize to 0-1
        normalized = density.astype(np.float32) / 255.0
        
        if colormap_type == 'viridis':
            # Viridis-like colormap
            r = np.clip(4 * normalized - 2, 0, 1)
            g = np.clip(4 * normalized - 1, 0, 1)
            b = np.clip(4 * normalized, 0, 1)
            
        elif colormap_type == 'turbo':
            # Turbo colormap approximation
            r = np.clip(0.13572 * np.sin(normalized * 25.9) + 0.5, 0, 1)
            g = np.clip(0.13572 * np.sin(normalized * 25.9 + 2.0) + 0.5, 0, 1)
            b = np.clip(0.13572 * np.sin(normalized * 25.9 + 4.0) + 0.5, 0, 1)
            
        elif colormap_type == 'twilight':
            # Twilight cyclic colormap
            angle = normalized * 2 * np.pi
            r = (np.sin(angle) + 1) / 2
            g = (np.sin(angle + 2.0) + 1) / 2
            b = (np.sin(angle + 4.0) + 1) / 2
            
        else:
            r = g = b = normalized
        
        # Convert to 8-bit BGR
        image = np.stack([
            (b * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (r * 255).astype(np.uint8)
        ], axis=2)
        
        return image
    
    @staticmethod
    def temperature_colormap(density: np.ndarray) -> np.ndarray:
        """
        Temperature-based colormap (cold to hot)
        
        Args:
            density: Density field
            
        Returns:
            RGB image
        """
        normalized = density.astype(np.float32) / 255.0
        
        # Cold (blue) -> Warm (red)
        r = np.clip(2 * normalized - 0.5, 0, 1)
        g = np.clip(2 * np.sin(normalized * np.pi / 2), 0, 1)
        b = np.clip(2 * (1 - normalized), 0, 1)
        
        image = np.stack([
            (b * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (r * 255).astype(np.uint8)
        ], axis=2)
        
        return image
