import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from typing import Tuple, List, Optional
import time


class BlackHole:
    G = 1.0  
    c = 1.0  
    
    def __init__(self, mass: float, position: Tuple[float, float] = (0.0, 0.0)):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self._update_properties()
    
    def _update_properties(self):
        # Schwarzschild radius: r_s = 2GM/c²
        self.schwarzschild_radius = 2 * self.G * self.mass / (self.c ** 2)
        self.event_horizon = self.schwarzschild_radius
        # ISCO (Innermost Stable Circular Orbit) at 3 r_s
        self.isco = 3 * self.schwarzschild_radius
    
    def accrete_mass(self, mass_to_add: float):
        if mass_to_add > 0:
            self.mass += mass_to_add
            self._update_properties()
        
    def gravitational_potential(self, position: np.ndarray) -> float:
        r = np.linalg.norm(position - self.position)
        if r < self.event_horizon:
            return -np.inf 
        return -self.G * self.mass / r
    
    def gravitational_force(self, position: np.ndarray, mass: float = 1.0) -> np.ndarray:
        displacement = position - self.position
        r = np.linalg.norm(displacement)
        
        if r < self.event_horizon:
            return np.zeros(2)  
        force_magnitude = self.G * self.mass * mass / (r ** 2)
        gr_correction = 1.0 + 3.0 * self.schwarzschild_radius / r
        force_magnitude *= gr_correction
        
        direction = -displacement / r
        return force_magnitude * direction
    
    def time_dilation_factor(self, position: np.ndarray) -> float:
        r = np.linalg.norm(position - self.position)
        if r <= self.schwarzschild_radius:
            return 0.0  # Time stops at event horizon
        return np.sqrt(1.0 - self.schwarzschild_radius / r)
    
    def deflection_angle(self, impact_parameter: float) -> float:
        if impact_parameter < self.schwarzschild_radius * 1.5:
            return np.pi  
        return 4 * self.G * self.mass / (self.c ** 2 * impact_parameter)
    
    def calculate_lensing_map(self, grid_size: Tuple[int, int] = (50, 50),
                              view_radius: float = None) -> Tuple[np.ndarray, np.ndarray]:
        if view_radius is None:
            view_radius = 15 * self.schwarzschild_radius
        
        nx, ny = grid_size
        x = np.linspace(-view_radius, view_radius, nx)
        y = np.linspace(-view_radius, view_radius, ny)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt((X - self.position[0])**2 + (Y - self.position[1])**2)
        R = np.maximum(R, self.schwarzschild_radius * 0.1)
        deflection_magnitude = 4 * self.G * self.mass / (self.c ** 2 * R)
        strong_field_mask = R < 1.5 * self.schwarzschild_radius
        STRONG_FIELD_ENHANCEMENT = 2.0
        deflection_magnitude[strong_field_mask] *= STRONG_FIELD_ENHANCEMENT
        dx = X - self.position[0]
        dy = Y - self.position[1]
        deflection_x = -deflection_magnitude * dx / R
        deflection_y = -deflection_magnitude * dy / R
        distorted_x = X + deflection_x * view_radius * 0.5
        distorted_y = Y + deflection_y * view_radius * 0.5
        
        return distorted_x, distorted_y
    
    def calculate_precession_rate(self, radius: float) -> float:
        if radius < self.event_horizon:
            return 0.0
        precession = 6.0 * np.pi * self.G * self.mass / (self.c ** 2 * radius)
        
        return precession
    
    def calculate_periapsis(self, position: np.ndarray, velocity: np.ndarray) -> float:
        r_vec = position - self.position
        r = np.linalg.norm(r_vec)
        
        if r == 0:
            return 0.0
        L = r_vec[0] * velocity[1] - r_vec[1] * velocity[0]
        v_squared = np.dot(velocity, velocity)
        E = 0.5 * v_squared + self.gravitational_potential(position)
        if abs(L) < 1e-10:  
            return 0.0  
        r_min = abs(L) / np.sqrt(2.0 * self.G * self.mass)
        radial_velocity = np.dot(r_vec, velocity) / r
        if radial_velocity > 0:
            return min(r, r_min)
        
        return r_min


def calculate_blackbody_color(temperature: float) -> Tuple[float, float, float]:
    temp = np.clip(temperature, 1000, 40000)
    temp = temp / 100.0
    if temp <= 66:
        red = 1.0
    else:
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = np.clip(red / 255.0, 0.0, 1.0)
    if temp <= 66:
        green = temp
        green = 99.4708025861 * np.log(green) - 161.1195681661
        green = np.clip(green / 255.0, 0.0, 1.0)
    else:
        green = temp - 60
        green = 288.1221695283 * (green ** -0.0755148492)
        green = np.clip(green / 255.0, 0.0, 1.0)
    if temp >= 66:
        blue = 1.0
    elif temp <= 19:
        blue = 0.0
    else:
        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        blue = np.clip(blue / 255.0, 0.0, 1.0)
    
    return (red, green, blue)


def calculate_disk_temperature(blackhole_mass: float, radius: float, 
                               accretion_rate: float = 1.0) -> float:
    r_s = 2.0 * blackhole_mass  
    if radius < r_s * 1.5:
        radius = r_s * 1.5
    base_temp = 10000.0  
    temp_factor = (blackhole_mass ** -0.25) * ((radius / r_s) ** -0.75)
    accretion_factor = accretion_rate ** 0.25
    temperature = base_temp * temp_factor * accretion_factor
    temperature = np.clip(temperature, 1000, 40000)
    
    return temperature


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, 
                 mass: float = 0.0, is_photon: bool = False):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.mass = mass
        self.is_photon = is_photon
        self.trajectory = [self.position.copy()]
        self.alive = True
        
    def update(self, acceleration: np.ndarray, dt: float, time_dilation: float = 1.0, 
               precession_rate: float = 0.0):
        if not self.alive:
            return
        dt_proper = dt * time_dilation
        self.velocity += acceleration * dt_proper
        if self.is_photon:
            speed = np.linalg.norm(self.velocity)
            if speed > 0:
                self.velocity = self.velocity / speed * BlackHole.c
        if precession_rate != 0.0 and not self.is_photon:
            precession_angle = precession_rate * dt_proper
            cos_theta = np.cos(precession_angle)
            sin_theta = np.sin(precession_angle)
            vx_new = self.velocity[0] * cos_theta - self.velocity[1] * sin_theta
            vy_new = self.velocity[0] * sin_theta + self.velocity[1] * cos_theta
            self.velocity = np.array([vx_new, vy_new])        
        self.position += self.velocity * dt_proper
        self.trajectory.append(self.position.copy())


class RayTracer:
    def __init__(self, blackhole: BlackHole):
        self.blackhole = blackhole
        
    def trace_ray(self, start_pos: np.ndarray, direction: np.ndarray, 
                  max_steps: int = 1000, dt: float = 0.01) -> List[np.ndarray]:
        direction = direction / np.linalg.norm(direction) * self.blackhole.c
        
        photon = Particle(start_pos, direction, mass=0.0, is_photon=True)
        trajectory = [photon.position.copy()]
        periapsis = self.blackhole.calculate_periapsis(photon.position, photon.velocity)
        if periapsis < 1.5 * self.blackhole.schwarzschild_radius:
            photon.alive = False
            return []         
        for _ in range(max_steps):
            r = np.linalg.norm(photon.position - self.blackhole.position)
            
            if r < self.blackhole.event_horizon:
                photon.alive = False
                break
            if r > 50 * self.blackhole.schwarzschild_radius:
                break
            displacement = photon.position - self.blackhole.position
            r_vec = displacement / r
            v_perp = photon.velocity - np.dot(photon.velocity, r_vec) * r_vec
            deflection = -3.0 * self.blackhole.G * self.blackhole.mass / (r ** 3) * v_perp
            radial_acc = -self.blackhole.G * self.blackhole.mass / (r ** 2) * r_vec
            radial_acc *= (1.0 + 3.0 * self.blackhole.schwarzschild_radius / r)
            
            total_acc = radial_acc + deflection
            time_dilation = self.blackhole.time_dilation_factor(photon.position)
            photon.update(total_acc, dt, time_dilation)
            trajectory.append(photon.position.copy())
            
        return trajectory


class AccretionDisk:
    def __init__(self, blackhole: BlackHole, n_particles: int = 500,
                 inner_radius: float = None, outer_radius: float = None,
                 replenish: bool = True):
        self.blackhole = blackhole
        self.particles: List[Particle] = []
        self.n_particles = n_particles
        self.replenish = replenish
        
        if inner_radius is None:
            inner_radius = blackhole.isco
        if outer_radius is None:
            outer_radius = blackhole.isco * 10
        
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self._spawn_particles(n_particles)
    
    def _spawn_particles(self, count: int):
        self.inner_radius = self.blackhole.isco
        self.outer_radius = self.blackhole.isco * 10
        for _ in range(count):
            u = np.random.random()
            if u < 0.3:
                r_min = self.blackhole.event_horizon * 1.2
                r_max = self.blackhole.event_horizon * 2.0
                r = r_min + (r_max - r_min) * np.random.random()
            else:
                r = self.inner_radius + (self.outer_radius - self.inner_radius) * (np.random.random() ** 0.5)
            
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta) + self.blackhole.position[0]
            y = r * np.sin(theta) + self.blackhole.position[1]
            position = np.array([x, y])
            v_magnitude = np.sqrt(self.blackhole.G * self.blackhole.mass / r)
            vx = -v_magnitude * np.sin(theta)
            vy = v_magnitude * np.cos(theta)
            velocity = np.array([vx, vy])
            perturbation_strength = 0.1 if r < self.blackhole.isco else 0.05
            velocity += np.random.normal(0, v_magnitude * perturbation_strength, 2)
            particle_mass = self.blackhole.mass * 0.0001  
            particle = Particle(position, velocity, mass=particle_mass)
            self.particles.append(particle)
    
    def update(self, dt: float):
        """
        Update all particles in the accretion disk.
        Includes general relativistic precession and event horizon shadow capture.
        
        Args:
            dt: Time step
            
        Returns:
            Total mass consumed in this time step
        """
        mass_consumed = 0.0
        particles_consumed = 0
        
        for particle in self.particles:
            if not particle.alive:
                continue
            
            # Check periapsis for event horizon shadow capture
            # Particles with periapsis < 1.5 × Rs are captured
            r_vec = particle.position - self.blackhole.position
            r = np.linalg.norm(r_vec)
            
            periapsis = self.blackhole.calculate_periapsis(particle.position, particle.velocity)
            if periapsis < 1.5 * self.blackhole.schwarzschild_radius:
                # Particle is captured by event horizon shadow
                particle.alive = False
                mass_consumed += particle.mass
                particles_consumed += 1
                continue
            
            # Calculate gravitational force
            force = self.blackhole.gravitational_force(particle.position, particle.mass)
            acceleration = force / particle.mass
            
            # Get time dilation factor
            time_dilation = self.blackhole.time_dilation_factor(particle.position)
            
            # Calculate precession rate for this orbital radius
            precession_rate = self.blackhole.calculate_precession_rate(r)
            
            # Update particle with precession
            particle.update(acceleration, dt, time_dilation, precession_rate)
            
            # Apply viscous drag after update (directly to velocity)
            # This simulates accretion disk friction and angular momentum loss
            # Drag primarily removes tangential velocity (angular momentum)
            # while allowing radial infall
            r_vec = particle.position - self.blackhole.position
            r = np.linalg.norm(r_vec)
            
            if r > 0:
                r_unit = r_vec / r
                
                # Decompose velocity into radial and tangential components
                v_radial = np.dot(particle.velocity, r_unit)
                v_radial_vec = v_radial * r_unit
                v_tangential_vec = particle.velocity - v_radial_vec
                
                # Distance-dependent drag on tangential velocity
                # (this removes angular momentum, causing inward spiral)
                if r < self.blackhole.isco:
                    # Strong drag near ISCO
                    tangential_drag = 0.80  # Retain 80% of tangential velocity
                elif r < self.blackhole.isco * 2:
                    # Medium drag in transition zone
                    tangential_drag = 0.92
                else:
                    # Light drag in outer disk
                    tangential_drag = 0.96
                
                # Apply drag to tangential component only
                v_tangential_vec *= tangential_drag
                
                # Reconstruct velocity (radial component unchanged)
                particle.velocity = v_radial_vec + v_tangential_vec
            
            # Check if particle crossed event horizon
            r_new = np.linalg.norm(particle.position - self.blackhole.position)
            if r_new < self.blackhole.event_horizon:
                particle.alive = False
                # Add particle's mass to the consumed total
                mass_consumed += particle.mass
                particles_consumed += 1
        
        # Add consumed mass to the black hole
        if mass_consumed > 0:
            self.blackhole.accrete_mass(mass_consumed)
        
        # Replenish particles to maintain disk density
        if self.replenish and particles_consumed > 0:
            self._spawn_particles(particles_consumed)
        
        return mass_consumed
    
    def get_positions(self) -> np.ndarray:
        """Get positions of all alive particles."""
        return np.array([p.position for p in self.particles if p.alive])
    
    def get_velocities(self) -> np.ndarray:
        """Get velocities of all alive particles."""
        return np.array([np.linalg.norm(p.velocity) for p in self.particles if p.alive])
    
    def get_temperatures(self, accretion_rate: float = 0.5) -> np.ndarray:
        """
        Get temperature-based values for all alive particles.
        
        Args:
            accretion_rate: Current accretion rate (0-1, relative to Eddington)
            
        Returns:
            Array of temperatures in Kelvin
        """
        temperatures = []
        for particle in self.particles:
            if particle.alive:
                r = np.linalg.norm(particle.position - self.blackhole.position)
                temp = calculate_disk_temperature(self.blackhole.mass, r, accretion_rate)
                temperatures.append(temp)
        return np.array(temperatures)
    
    def get_colors(self, accretion_rate: float = 0.5) -> np.ndarray:
        """
        Get RGB colors for particles based on blackbody temperature.
        
        Args:
            accretion_rate: Current accretion rate (0-1, relative to Eddington)
            
        Returns:
            Array of RGB colors (Nx3)
        """
        colors = []
        for particle in self.particles:
            if particle.alive:
                r = np.linalg.norm(particle.position - self.blackhole.position)
                temp = calculate_disk_temperature(self.blackhole.mass, r, accretion_rate)
                rgb = calculate_blackbody_color(temp)
                colors.append(rgb)
        return np.array(colors)


class BlackHoleSimulation:
    """
    Main simulation class that orchestrates all components.
    """
    
    def __init__(self, blackhole_mass: float = 10.0, 
                 enable_accretion: bool = True,
                 enable_ray_tracing: bool = True):
        """
        Initialize the black hole simulation.
        
        Args:
            blackhole_mass: Mass of the black hole
            enable_accretion: Whether to simulate accretion disk
            enable_ray_tracing: Whether to compute ray tracing
        """
        self.blackhole = BlackHole(blackhole_mass, position=(0.0, 0.0))
        self.ray_tracer = RayTracer(self.blackhole)
        self.accretion_disk = None
        self.light_rays: List[List[np.ndarray]] = []
        
        # Track black hole growth statistics
        self.initial_mass = blackhole_mass
        self.total_mass_accreted = 0.0
        self.accretion_events = 0
        
        if enable_accretion:
            self.accretion_disk = AccretionDisk(self.blackhole, n_particles=1000)
        
        if enable_ray_tracing:
            self._compute_light_rays()
    
    def _compute_light_rays(self):
        """
        Compute light ray trajectories for visualization.
        Creates a grid of light rays from different angles.
        """
        n_rays = 12
        distance = 30 * self.blackhole.schwarzschild_radius
        
        for i in range(n_rays):
            angle = 2 * np.pi * i / n_rays
            start_x = distance * np.cos(angle)
            start_y = distance * np.sin(angle)
            start_pos = np.array([start_x, start_y])
            
            # Direction toward black hole (with slight offset)
            direction = -start_pos / np.linalg.norm(start_pos)
            
            # Add perpendicular component for interesting trajectories
            perp = np.array([-direction[1], direction[0]])
            direction = direction + 0.3 * perp
            
            trajectory = self.ray_tracer.trace_ray(start_pos, direction, max_steps=2000, dt=0.01)
            self.light_rays.append(trajectory)
    
    def update(self, dt: float):
        """
        Update simulation by one time step.
        
        Args:
            dt: Time step
        """
        if self.accretion_disk:
            mass_consumed = self.accretion_disk.update(dt)
            if mass_consumed > 0:
                self.total_mass_accreted += mass_consumed
                self.accretion_events += 1
    
    def visualize(self, show_rays: bool = True, show_disk: bool = True, 
                  show_time_dilation: bool = True):
        """
        Create a visualization of the black hole simulation.
        
        Args:
            show_rays: Whether to show light ray trajectories
            show_disk: Whether to show accretion disk
            show_time_dilation: Whether to show time dilation field
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        
        # Set limits
        view_radius = 15 * self.blackhole.schwarzschild_radius
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-view_radius, view_radius)
        
        # Background
        ax.set_facecolor('#000510')
        fig.patch.set_facecolor('#000510')
        
        # Time dilation field (background grid)
        if show_time_dilation:
            n_grid = 50
            x = np.linspace(-view_radius, view_radius, n_grid)
            y = np.linspace(-view_radius, view_radius, n_grid)
            X, Y = np.meshgrid(x, y)
            
            # Calculate time dilation at each point
            Z = np.zeros_like(X)
            for i in range(n_grid):
                for j in range(n_grid):
                    pos = np.array([X[i, j], Y[i, j]])
                    Z[i, j] = self.blackhole.time_dilation_factor(pos)
            
            # Plot time dilation as contours
            contours = ax.contour(X, Y, Z, levels=10, colors='cyan', alpha=0.2, linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, colors='cyan')
        
        # Draw photon sphere (1.5 r_s)
        photon_sphere = Circle(self.blackhole.position, 
                              1.5 * self.blackhole.schwarzschild_radius,
                              fill=False, edgecolor='orange', 
                              linewidth=1, linestyle='--', alpha=0.5,
                              label='Photon Sphere')
        ax.add_patch(photon_sphere)
        
        # Draw ISCO (3 r_s)
        isco_circle = Circle(self.blackhole.position, 
                           self.blackhole.isco,
                           fill=False, edgecolor='yellow', 
                           linewidth=1, linestyle='--', alpha=0.5,
                           label='ISCO')
        ax.add_patch(isco_circle)
        
        # Draw event horizon
        event_horizon = Circle(self.blackhole.position, 
                              self.blackhole.event_horizon,
                              color='black', zorder=100,
                              label='Event Horizon')
        ax.add_patch(event_horizon)
        
        # Draw singularity
        ax.plot(self.blackhole.position[0], self.blackhole.position[1], 
               'w*', markersize=15, zorder=101)
        
        # Draw light rays (gravitational lensing)
        if show_rays and self.light_rays:
            for ray in self.light_rays:
                if len(ray) > 1:
                    ray_array = np.array(ray)
                    ax.plot(ray_array[:, 0], ray_array[:, 1], 
                           color='#FFA500', linewidth=0.8, alpha=0.7)
        
        # Draw accretion disk with temperature-based color grading
        if show_disk and self.accretion_disk:
            positions = self.accretion_disk.get_positions()
            
            if len(positions) > 0:
                # Calculate accretion rate (based on growth)
                accretion_rate = min(1.0, self.total_mass_accreted / (self.initial_mass * 0.1 + 0.01))
                
                # Get temperature-based colors
                colors = self.accretion_disk.get_colors(accretion_rate)
                
                # Scatter plot with blackbody colors
                scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                                   c=colors, 
                                   s=2, alpha=0.8)
                
                # Add colorbar to show temperature gradient
                # Create a custom colorbar by showing temperature range
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                temperatures = self.accretion_disk.get_temperatures(accretion_rate)
                norm = Normalize(vmin=temperatures.min(), vmax=temperatures.max())
                sm = ScalarMappable(cmap='hot', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='Temperature (K)', pad=0.02)
                cbar.ax.yaxis.set_tick_params(color='white')
                cbar.ax.yaxis.label.set_color('white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Title and labels
        ax.set_title('2D Black Hole Simulation with Ray Tracing\n' + 
                    f'Mass: {self.blackhole.mass:.1f} M☉, ' +
                    f'Schwarzschild Radius: {self.blackhole.schwarzschild_radius:.2f}',
                    color='white', fontsize=14, pad=20)
        ax.set_xlabel('Distance', color='white')
        ax.set_ylabel('Distance', color='white')
        
        # Legend
        ax.legend(loc='upper right', facecolor='#000510', 
                 edgecolor='white', labelcolor='white')
        
        # Styling
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        plt.tight_layout()
        return fig, ax
    
    def animate(self, frames: int = 200, interval: int = 50, 
                filename: Optional[str] = None):
        """
        Create an animation of the simulation.
        
        Args:
            frames: Number of frames to animate
            interval: Interval between frames in ms
            filename: Optional filename to save animation
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        
        view_radius = 15 * self.blackhole.schwarzschild_radius
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-view_radius, view_radius)
        ax.set_facecolor('#000510')
        fig.patch.set_facecolor('#000510')
        
        # Static elements
        event_horizon = Circle(self.blackhole.position, 
                              self.blackhole.event_horizon,
                              color='black', zorder=100)
        ax.add_patch(event_horizon)
        
        isco_circle = Circle(self.blackhole.position, 
                           self.blackhole.isco,
                           fill=False, edgecolor='yellow', 
                           linewidth=1, linestyle='--', alpha=0.5)
        ax.add_patch(isco_circle)
        
        ax.plot(self.blackhole.position[0], self.blackhole.position[1], 
               'w*', markersize=15, zorder=101)
        
        # Initialize scatter plot for particles
        scatter = ax.scatter([], [], s=2, c=[], cmap='hot', 
                           alpha=0.8, vmin=0, vmax=1.5)
        
        # Light rays
        ray_lines = []
        if self.light_rays:
            for ray in self.light_rays:
                if len(ray) > 1:
                    ray_array = np.array(ray)
                    line, = ax.plot(ray_array[:, 0], ray_array[:, 1], 
                                  color='#FFA500', linewidth=0.8, alpha=0.7)
                    ray_lines.append(line)
        
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          color='white', verticalalignment='top',
                          fontsize=12, family='monospace')
        
        ax.set_title('2D Black Hole Simulation - Real-time Evolution',
                    color='white', fontsize=14)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
            return scatter, time_text
        
        def update_frame(frame):
            # Update simulation
            self.update(dt=0.05)
            
            # Update particle positions
            if self.accretion_disk:
                positions = self.accretion_disk.get_positions()
                velocities = self.accretion_disk.get_velocities()
                
                if len(positions) > 0:
                    scatter.set_offsets(positions)
                    scatter.set_array(velocities)
            
            # Update time display
            time_text.set_text(f'Frame: {frame}\nParticles: {len(self.accretion_disk.get_positions()) if self.accretion_disk else 0}')
            
            return scatter, time_text
        
        anim = FuncAnimation(fig, update_frame, init_func=init,
                           frames=frames, interval=interval, 
                           blit=True, repeat=True)
        
        if filename:
            anim.save(filename, writer='pillow', fps=20)
            print(f"Animation saved to {filename}")
        
        return anim
    
    def run_active_simulator(self, dt: float = 0.05, interval: int = 50,
                            show_rays: bool = True, show_disk: bool = True,
                            show_fps: bool = True):
        """
        Run an active, continuously updating simulator in a window.
        The simulation runs indefinitely until the window is closed.
        
        Args:
            dt: Time step for simulation updates
            interval: Update interval in milliseconds
            show_rays: Whether to show light ray trajectories
            show_disk: Whether to show accretion disk
            show_fps: Whether to show FPS counter
        
        Returns:
            The animation object (keeps window alive)
        """
        print("\n" + "="*60)
        print("ACTIVE SIMULATOR MODE")
        print("="*60)
        print("\nStarting continuous simulation...")
        print("  - Real-time physics updates")
        print("  - Live particle dynamics")
        print("  - Close the window to stop simulation")
        print("="*60 + "\n")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        
        view_radius = 15 * self.blackhole.schwarzschild_radius
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-view_radius, view_radius)
        ax.set_facecolor('#000510')
        fig.patch.set_facecolor('#000510')
        
        # Static elements
        event_horizon = Circle(self.blackhole.position, 
                              self.blackhole.event_horizon,
                              color='black', zorder=100,
                              label='Event Horizon')
        ax.add_patch(event_horizon)
        
        isco_circle = Circle(self.blackhole.position, 
                           self.blackhole.isco,
                           fill=False, edgecolor='yellow', 
                           linewidth=1, linestyle='--', alpha=0.5,
                           label='ISCO')
        ax.add_patch(isco_circle)
        
        photon_sphere = Circle(self.blackhole.position, 
                              1.5 * self.blackhole.schwarzschild_radius,
                              fill=False, edgecolor='orange', 
                              linewidth=1, linestyle='--', alpha=0.4,
                              label='Photon Sphere')
        ax.add_patch(photon_sphere)
        
        ax.plot(self.blackhole.position[0], self.blackhole.position[1], 
               'w*', markersize=15, zorder=101, label='Singularity')
        
        # Initialize scatter plot for particles with temperature-based colors
        scatter = ax.scatter([], [], s=2, c=[], 
                           alpha=0.8)
        
        # Light rays
        ray_lines = []
        if show_rays and self.light_rays:
            for ray in self.light_rays:
                if len(ray) > 1:
                    ray_array = np.array(ray)
                    line, = ax.plot(ray_array[:, 0], ray_array[:, 1], 
                                  color='#FFA500', linewidth=0.8, alpha=0.7)
                    ray_lines.append(line)
        
        # Info text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          color='white', verticalalignment='top',
                          fontsize=11, family='monospace',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Title
        ax.set_title('Active Black Hole Simulator - Continuous Evolution\n' + 
                    f'Mass: {self.blackhole.mass:.1f} M☉ | ' +
                    f'$r_s$: {self.blackhole.schwarzschild_radius:.2f}',
                    color='white', fontsize=14, pad=20)
        ax.set_xlabel('Distance', color='white', fontsize=12)
        ax.set_ylabel('Distance', color='white', fontsize=12)
        
        # Legend
        ax.legend(loc='upper right', facecolor='#000510', 
                 edgecolor='white', labelcolor='white', framealpha=0.8)
        
        # Styling
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        
        # Track time for FPS calculation
        frame_times = []
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
            return scatter, time_text
        
        def update_frame(frame):
            nonlocal frame_times
            start_time = time.time()
            
            # Update simulation
            self.update(dt=dt)
            
            # Update black hole size dynamically as it grows
            event_horizon.set_radius(self.blackhole.event_horizon)
            isco_circle.set_radius(self.blackhole.isco)
            photon_sphere.set_radius(1.5 * self.blackhole.schwarzschild_radius)
            
            # Update title with current mass
            ax.set_title('Active Black Hole Simulator - Continuous Evolution\n' + 
                        f'Mass: {self.blackhole.mass:.3f} M☉ | ' +
                        f'$r_s$: {self.blackhole.schwarzschild_radius:.3f} | ' +
                        f'Growth: {((self.blackhole.mass/self.initial_mass - 1)*100):.2f}%',
                        color='white', fontsize=14, pad=20)
            
            # Update particle positions with temperature-based colors
            if show_disk and self.accretion_disk:
                positions = self.accretion_disk.get_positions()
                
                if len(positions) > 0:
                    # Calculate accretion rate for color grading
                    accretion_rate = min(1.0, self.total_mass_accreted / (self.initial_mass * 0.1 + 0.01))
                    
                    # Get temperature-based colors
                    colors = self.accretion_disk.get_colors(accretion_rate)
                    
                    # Update scatter plot
                    scatter.set_offsets(positions)
                    scatter.set_color(colors)
            
            # Calculate FPS
            frame_times.append(time.time() - start_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_frame_time = np.mean(frame_times) if frame_times else 0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Update info display
            n_particles = len(self.accretion_disk.get_positions()) if self.accretion_disk else 0
            info_text = f'Frame: {frame:06d}\n'
            info_text += f'Active Particles: {n_particles}\n'
            info_text += f'BH Mass: {self.blackhole.mass:.4f} M☉\n'
            info_text += f'Mass Accreted: {self.total_mass_accreted:.4f} M☉\n'
            info_text += f'Accretion Events: {self.accretion_events}\n'
            info_text += f'Time Step: {dt:.3f}\n'
            if show_fps:
                info_text += f'FPS: {fps:.1f}\n'
            info_text += '\n[LIVE SIMULATION]'
            
            time_text.set_text(info_text)
            
            # Don't return the circles in blit mode since we're modifying their properties
            # matplotlib will handle the redraw
            return scatter, time_text
        
        # Create infinite animation (repeat=True means it loops forever)
        # Note: blit=False to allow dynamic circle size updates
        anim = FuncAnimation(fig, update_frame, init_func=init,
                           interval=interval, blit=False, repeat=True,
                           cache_frame_data=False)
        
        plt.show()
        
        return anim


def main():
    """
    Main function demonstrating the black hole simulation.
    
    By default, runs static visualization mode.
    Use '--active' flag or set ACTIVE_MODE=1 environment variable 
    to run the active simulator with continuous updates.
    """
    import os
    import sys
    
    # Check if active mode is requested
    active_mode = os.environ.get('ACTIVE_MODE', '0') == '1' or '--active' in sys.argv
    
    print("=" * 60)
    print("2D Black Hole Simulation Engine")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Schwarzschild metric implementation")
    print("  ✓ Gravitational lensing via ray tracing")
    print("  ✓ Accretion disk dynamics")
    print("  ✓ Time dilation visualization")
    print("  ✓ Event horizon and ISCO")
    if active_mode:
        print("  ✓ Active simulator mode (continuous running)")
    print("\n" + "=" * 60)
    
    # Create simulation
    print("\nInitializing simulation...")
    sim = BlackHoleSimulation(
        blackhole_mass=10.0,
        enable_accretion=True,
        enable_ray_tracing=True
    )
    
    print(f"Black hole mass: {sim.blackhole.mass} M☉")
    print(f"Schwarzschild radius: {sim.blackhole.schwarzschild_radius:.3f}")
    print(f"Event horizon: {sim.blackhole.event_horizon:.3f}")
    print(f"ISCO: {sim.blackhole.isco:.3f}")
    print(f"Photon sphere: {1.5 * sim.blackhole.schwarzschild_radius:.3f}")
    
    if active_mode:
        # Run active simulator
        print("\n" + "=" * 60)
        print("Starting ACTIVE SIMULATOR MODE")
        print("=" * 60)
        print("\nTip: Use active_simulator.py for more control options")
        print("     (e.g., --mass, --particles, --dt, etc.)\n")
        
        sim.run_active_simulator(
            dt=0.05,
            interval=50,
            show_rays=True,
            show_disk=True,
            show_fps=True
        )
    else:
        # Run static visualization mode (original behavior)
        # Create static visualization
        print("\nGenerating visualization...")
        fig, ax = sim.visualize(
            show_rays=True,
            show_disk=True,
            show_time_dilation=True
        )
        
        plt.savefig('blackhole_simulation.png', dpi=300, 
                    facecolor='#000510', edgecolor='none')
        print("Static visualization saved to: blackhole_simulation.png")
        
        # Run some simulation steps to evolve the accretion disk
        print("\nEvolving accretion disk...")
        for i in range(50):
            sim.update(dt=0.1)
            if i % 10 == 0:
                print(f"  Step {i}/50")
        
        # Create another visualization showing evolved state
        fig2, ax2 = sim.visualize(show_rays=True, show_disk=True, 
                                  show_time_dilation=False)
        plt.savefig('blackhole_evolved.png', dpi=300, 
                    facecolor='#000510', edgecolor='none')
        print("Evolved state saved to: blackhole_evolved.png")
        
        print("\n" + "=" * 60)
        print("Simulation complete!")
        print("=" * 60)
        print("\nTip: Run with --active flag or use active_simulator.py")
        print("     to launch the continuously running simulator!\n")
        
        # Show interactive plot
        plt.show()


if __name__ == "__main__":
    main()
