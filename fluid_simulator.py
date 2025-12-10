import numpy as np
from scipy.ndimage import convolve
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FluidConfig:
    """Configuration for fluid simulation"""
    width: int = 256
    height: int = 256
    diffusion: float = 0.0001  # Diffusion coefficient
    viscosity: float = 0.0001  # Viscosity coefficient
    decay: float = 0.995  # Velocity decay per frame
    pressure_iterations: int = 20  # Iterations for pressure solving
    

class FluidSimulator:
    """
    Realistic 2D fluid simulator based on Navier-Stokes equations
    Uses semi-Lagrangian advection and pressure projection
    """
    
    def __init__(self, config: FluidConfig = None):
        """
        Initialize fluid simulator
        
        Args:
            config: FluidConfig object with simulation parameters
        """
        self.config = config or FluidConfig()
        
        # Velocity fields (u and v components)
        self.velocity_x = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        self.velocity_y = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        
        # Pressure and divergence
        self.pressure = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        self.divergence = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        
        # Density field (for visualization)
        self.density = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        
        # Temporary fields
        self.velocity_x_temp = np.zeros_like(self.velocity_x)
        self.velocity_y_temp = np.zeros_like(self.velocity_y)
        self.density_temp = np.zeros_like(self.density)
        
    def add_velocity(self, x: int, y: int, vx: float, vy: float, radius: float = 20.0):
        """
        Add velocity to a region (from hand interaction)
        
        Args:
            x, y: Center position
            vx, vy: Velocity components
            radius: Radius of influence
        """
        # Create Gaussian kernel for smooth velocity addition
        yy, xx = np.ogrid[:self.config.height, :self.config.width]
        distance = np.sqrt((xx - x)**2 + (yy - y)**2)
        kernel = np.exp(-(distance**2) / (2 * radius**2))
        kernel = kernel / (kernel.max() + 1e-6)
        
        self.velocity_x += vx * kernel
        self.velocity_y += vy * kernel
        
    def add_density(self, x: int, y: int, amount: float = 100.0, radius: float = 20.0):
        """
        Add density to a region (for visualization)
        
        Args:
            x, y: Center position
            amount: Amount of density to add
            radius: Radius of influence
        """
        yy, xx = np.ogrid[:self.config.height, :self.config.width]
        distance = np.sqrt((xx - x)**2 + (yy - y)**2)
        kernel = np.exp(-(distance**2) / (2 * radius**2))
        kernel = kernel / (kernel.max() + 1e-6)
        
        self.density += amount * kernel
        self.density = np.clip(self.density, 0, 255)
        
    def advect(self, field: np.ndarray, vel_x: np.ndarray, vel_y: np.ndarray) -> np.ndarray:
        """
        Semi-Lagrangian advection
        
        Args:
            field: Field to advect
            vel_x, vel_y: Velocity components
            
        Returns:
            Advected field
        """
        h = 1.0
        yy, xx = np.meshgrid(np.arange(self.config.width), np.arange(self.config.height))
        
        # Backtrack position
        x = np.clip(xx - vel_x * h, 0, self.config.width - 1).astype(np.float32)
        y = np.clip(yy - vel_y * h, 0, self.config.height - 1).astype(np.float32)
        
        # Bilinear interpolation
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
        
        x0 = np.clip(x0, 0, self.config.width - 1)
        x1 = np.clip(x1, 0, self.config.width - 1)
        y0 = np.clip(y0, 0, self.config.height - 1)
        y1 = np.clip(y1, 0, self.config.height - 1)
        
        sx = x - np.floor(x)
        sy = y - np.floor(y)
        
        result = (
            field[y0, x0] * (1 - sx) * (1 - sy) +
            field[y0, x1] * sx * (1 - sy) +
            field[y1, x0] * (1 - sx) * sy +
            field[y1, x1] * sx * sy
        )
        
        return result
        
    def diffuse(self, field: np.ndarray, coefficient: float) -> np.ndarray:
        """
        Diffusion using Jacobi iteration
        
        Args:
            field: Field to diffuse
            coefficient: Diffusion coefficient
            
        Returns:
            Diffused field
        """
        a = coefficient
        result = field.copy()
        
        # Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        
        for _ in range(5):  # Jacobi iterations
            laplacian = convolve(result, kernel, mode='constant')
            result = (field + a * laplacian) / (1 + 4 * a)
        
        return result
        
    def compute_divergence(self):
        """Compute velocity divergence"""
        # Central differences
        div_x = np.zeros_like(self.velocity_x)
        div_y = np.zeros_like(self.velocity_y)
        
        div_x[:, 1:-1] = (self.velocity_x[:, 2:] - self.velocity_x[:, :-2]) / 2.0
        div_y[1:-1, :] = (self.velocity_y[2:, :] - self.velocity_y[:-2, :]) / 2.0
        
        self.divergence = div_x + div_y
        
    def solve_pressure(self):
        """Solve pressure using Jacobi iteration"""
        self.compute_divergence()
        
        # Laplacian kernel for pressure
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        
        for _ in range(self.config.pressure_iterations):
            laplacian = convolve(self.pressure, kernel, mode='constant')
            self.pressure = (laplacian - self.divergence) / 4.0
            
    def subtract_pressure_gradient(self):
        """Subtract pressure gradient from velocity"""
        # Compute pressure gradient
        grad_p_x = np.zeros_like(self.pressure)
        grad_p_y = np.zeros_like(self.pressure)
        
        grad_p_x[:, 1:-1] = (self.pressure[:, 2:] - self.pressure[:, :-2]) / 2.0
        grad_p_y[1:-1, :] = (self.pressure[2:, :] - self.pressure[:-2, :]) / 2.0
        
        self.velocity_x -= grad_p_x
        self.velocity_y -= grad_p_y
        
    def step(self, dt: float = 1.0):
        """
        Perform one simulation step
        
        Args:
            dt: Time step
        """
        # Apply viscosity (diffusion)
        self.velocity_x = self.diffuse(self.velocity_x, self.config.viscosity)
        self.velocity_y = self.diffuse(self.velocity_y, self.config.viscosity)
        
        # Pressure projection (incompressibility)
        self.solve_pressure()
        self.subtract_pressure_gradient()
        
        # Advection
        self.velocity_x = self.advect(self.velocity_x, self.velocity_x, self.velocity_y)
        self.velocity_y = self.advect(self.velocity_y, self.velocity_x, self.velocity_y)
        
        # Advect density
        self.density = self.advect(self.density, self.velocity_x, self.velocity_y)
        
        # Apply diffusion to density
        self.density = self.diffuse(self.density, self.config.diffusion)
        
        # Decay
        self.velocity_x *= self.config.decay
        self.velocity_y *= self.config.decay
        self.density *= 0.998
        
    def get_density_field(self) -> np.ndarray:
        """
        Get density field for visualization
        
        Returns:
            Normalized density field (0-255)
        """
        return np.clip(self.density, 0, 255).astype(np.uint8)
        
    def get_velocity_magnitude(self) -> np.ndarray:
        """
        Get velocity magnitude field
        
        Returns:
            Velocity magnitude field
        """
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        
    def reset(self):
        """Reset all fields"""
        self.velocity_x.fill(0)
        self.velocity_y.fill(0)
        self.pressure.fill(0)
        self.divergence.fill(0)
        self.density.fill(0)
