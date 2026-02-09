# Spaceing - 2D Black Hole Simulation Engine

A physically accurate 2D black hole simulation with ray tracing, gravitational lensing, and accretion disk dynamics.

## ✨ Visual Physics Engine Features

**Enhanced with advanced physics visualization:**
- 🔭 **Gravitational Lensing Mapper** - Calculates lensing distortion for background starfields
- 🌡️ **Temperature-Based Color Grading** - Planckian blackbody spectrum (blue/white for high temp, red/orange for low)
- 🌀 **Orbital Precession** - General relativistic precession near event horizon
- ⚫ **Event Horizon Shadow** - Captures particles/photons with periapsis < 1.5 × Rs

## Features

✨ **Core Physics**
- **Schwarzschild Metric**: Accurate spacetime curvature around a non-rotating black hole
- **Gravitational Lensing**: Ray tracing implementation showing light bending near massive objects
- **Time Dilation**: Visualization of time dilation effects near the event horizon
- **Event Horizon & ISCO**: Proper modeling of innermost stable circular orbit
- **Black Hole Growth**: Dynamic mass accretion with real-time event horizon expansion
- **Accretion Physics**: Viscous disk friction causing particles to spiral inward and be consumed
- **GR Precession**: Particles exhibit orbital precession as predicted by general relativity

🌟 **Visual Effects**
- **Ray Tracing**: Multiple light rays showing gravitational deflection
- **Accretion Disk**: 1000+ particles simulating matter spiraling into the black hole
- **Temperature Color Grading**: Realistic blackbody spectrum colors based on disk temperature
- **Time Dilation Field**: Contour visualization of spacetime warping
- **Growing Event Horizon**: Visual expansion as black hole consumes matter
- **Live Growth Statistics**: Real-time display of mass accreted and growth percentage
- **Event Horizon Shadow**: Dark region where light cannot escape

🎬 **Simulation Modes**
- **Static Visualization**: Generate high-quality PNG images
- **Active Simulator**: Continuously running real-time simulation window
- **Animation Export**: Create GIF animations of black hole dynamics
- **Interactive Exploration**: Real-time FPS monitoring and particle tracking

⚙️ **Technical Implementation**
- General relativistic corrections to Newtonian gravity
- Symplectic integration for stable particle dynamics
- Null geodesics for photon trajectories
- Efficient numpy-based computation
- Real-time rendering with matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/Master-yug/spaceing.git
cd spaceing

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Static Visualization Mode (Default)

Run the simulation with default parameters:

```bash
python blackhole_simulation.py
```

This will:
1. Create a black hole with 10 solar masses
2. Generate gravitational lensing ray traces
3. Simulate an accretion disk with 1000 particles
4. Save visualizations to PNG files
5. Display an interactive plot

### Active Simulator Mode (NEW!)

Run a continuously updating, real-time simulation:

```bash
# Quick start with active simulator
python active_simulator.py

# Or run the main script in active mode
python blackhole_simulation.py --active
```

The **Active Simulator** provides:
- ✨ **Continuous Real-time Updates** - Simulation runs indefinitely
- 📊 **Live Visualization** - Watch particles evolve in real-time
- 🚀 **Performance Metrics** - FPS counter and particle count
- 🎯 **Interactive Window** - Stays open until you close it
- ⚙️ **Customizable Parameters** - Mass, particle count, time step, etc.

**Active Simulator Options:**

```bash
# Custom black hole mass
python active_simulator.py --mass 50.0

# More particles for denser accretion disk
python active_simulator.py --particles 2000

# Faster simulation time step
python active_simulator.py --dt 0.1

# Show help for all options
python active_simulator.py --help
```

## Usage

### Basic Usage

```python
from blackhole_simulation import BlackHoleSimulation

# Create simulation
sim = BlackHoleSimulation(
    blackhole_mass=10.0,         # Mass in solar masses
    enable_accretion=True,        # Enable accretion disk
    enable_ray_tracing=True       # Enable ray tracing
)

# Visualize (static)
fig, ax = sim.visualize(
    show_rays=True,               # Show light ray trajectories
    show_disk=True,               # Show accretion disk
    show_time_dilation=True       # Show time dilation contours
)

# Evolve the system
for i in range(100):
    sim.update(dt=0.1)

# Create animation
anim = sim.animate(frames=200, interval=50, filename='blackhole.gif')
```

### Active Simulator (Continuous Running)

```python
from blackhole_simulation import BlackHoleSimulation

# Create simulation
sim = BlackHoleSimulation(
    blackhole_mass=10.0,
    enable_accretion=True,
    enable_ray_tracing=True
)

# Run continuously updating simulator
# Window stays open and updates in real-time until closed
sim.run_active_simulator(
    dt=0.05,           # Time step for each update
    interval=50,       # Update interval in milliseconds
    show_rays=True,    # Display gravitational lensing
    show_disk=True,    # Display accretion disk
    show_fps=True      # Show FPS counter
)
```

### Advanced: Custom Black Hole

```python
from blackhole_simulation import BlackHole, RayTracer, AccretionDisk

# Create a supermassive black hole
bh = BlackHole(mass=1000.0, position=(0.0, 0.0))

print(f"Event Horizon: {bh.event_horizon:.2f}")
print(f"ISCO: {bh.isco:.2f}")
print(f"Photon Sphere: {1.5 * bh.schwarzschild_radius:.2f}")

# Trace light rays
ray_tracer = RayTracer(bh)
start_pos = [30.0, 0.0]
direction = [-1.0, 0.3]
trajectory = ray_tracer.trace_ray(start_pos, direction)

# Create custom accretion disk
disk = AccretionDisk(bh, n_particles=2000, 
                     inner_radius=bh.isco, 
                     outer_radius=bh.isco * 20)
```

### Physics Parameters

The simulation implements several key physical concepts:

1. **Schwarzschild Radius**: 
   ```
   r_s = 2GM/c²
   ```
   The radius of the event horizon where escape velocity equals the speed of light.

2. **Time Dilation Factor**:
   ```
   τ = √(1 - r_s/r)
   ```
   Proper time passes slower near the black hole.

3. **Gravitational Deflection** (Einstein Formula):
   ```
   α ≈ 4GM/(c²b)
   ```
   Light bending angle for impact parameter b.

4. **ISCO** (Innermost Stable Circular Orbit):
   ```
   r_ISCO = 3r_s
   ```
   Closest stable orbit around a Schwarzschild black hole.

5. **Black Hole Growth**:
   ```
   M_final = M_initial + Σ m_accreted
   r_s_new = 2GM_final/c²
   ```
   As matter crosses the event horizon, it adds to the black hole's mass. The Schwarzschild radius grows proportionally, and all related properties (ISCO, photon sphere) scale accordingly. The event horizon expands in real-time as accretion occurs.

6. **Accretion Disk Friction**:
   Viscous drag removes angular momentum from particles, causing them to spiral inward:
   - Strong drag near ISCO (r < 3r_s) for rapid infall
   - Medium drag in transition zone (3r_s < r < 6r_s)
   - Light drag in outer disk (r > 6r_s)
   
   Particles spawn continuously to maintain disk density as matter is consumed.

## Output

The simulation generates:

1. **blackhole_simulation.png**: Static visualization showing all features
2. **blackhole_evolved.png**: System after temporal evolution
3. Interactive matplotlib window for real-time exploration

### Visualization Elements

- **Black Circle**: Event horizon (no escape possible)
- **White Star**: Singularity (center point)
- **Yellow Dashed Circle**: ISCO boundary
- **Orange Dashed Circle**: Photon sphere (1.5 r_s)
- **Orange Curves**: Light ray trajectories (gravitational lensing)
- **Colored Dots**: Accretion disk particles (color = velocity)
- **Cyan Contours**: Time dilation field

## Physics Accuracy

This simulation includes:

✅ **Schwarzschild Metric**: Non-rotating black hole spacetime
✅ **Geodesic Equations**: Proper light ray deflection
✅ **Time Dilation**: General relativistic time effects
✅ **ISCO**: Correct innermost stable orbit
✅ **Photon Sphere**: Unstable photon orbits at 1.5 r_s
✅ **Accretion Dynamics**: Particle spiraling with GR corrections
✅ **Mass Accretion**: Black hole grows as it consumes matter
✅ **Dynamic Event Horizon**: Schwarzschild radius expands with mass
✅ **Disk Replenishment**: Continuous particle spawning maintains disk

⚠️ **Limitations**:
- 2D projection of 3D physics
- Non-rotating (Schwarzschild) black hole only
- No electromagnetic effects
- Simplified accretion disk (no magnetohydrodynamics)
- No Hawking radiation

## Performance

- **Accretion Disk**: 1000 particles @ 60 FPS on modern CPU
- **Ray Tracing**: 12 rays with 2000 steps each
- **Time Step**: Adaptive integration with time dilation
- **Memory**: ~50 MB for full simulation

## Customization

### Adjust Black Hole Mass

```python
# Solar mass black hole (small)
sim = BlackHoleSimulation(blackhole_mass=1.0)

# Supermassive black hole (like Sagittarius A*)
sim = BlackHoleSimulation(blackhole_mass=4_000_000.0)
```

### Modify Accretion Disk

```python
sim = BlackHoleSimulation(blackhole_mass=10.0)
sim.accretion_disk = AccretionDisk(
    sim.blackhole, 
    n_particles=5000,           # More particles
    inner_radius=sim.blackhole.isco * 0.8,  # Closer to BH
    outer_radius=sim.blackhole.isco * 30    # Larger disk
)
```

### Custom Ray Tracing

```python
from blackhole_simulation import RayTracer
import numpy as np

ray_tracer = RayTracer(sim.blackhole)

# Trace ray from specific position
start = np.array([20.0, 10.0])
direction = np.array([-0.8, -0.5])
trajectory = ray_tracer.trace_ray(start, direction, max_steps=3000)

# Plot custom trajectory
import matplotlib.pyplot as plt
traj = np.array(trajectory)
plt.plot(traj[:, 0], traj[:, 1])
plt.show()
```

## Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Kerr metric (rotating black holes)
- [ ] 3D visualization
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Magnetohydrodynamics for accretion disk
- [ ] Interactive controls (zoom, pan, time control)
- [ ] VR/AR support
- [ ] Educational mode with annotations

## License

MIT License - feel free to use for educational, research, or commercial purposes.

## References

- Schwarzschild, K. (1916). "On the Gravitational Field of a Mass Point"
- Einstein, A. (1915). "The Field Equations of Gravitation"
- Chandrasekhar, S. (1983). "The Mathematical Theory of Black Holes"
- Misner, Thorne & Wheeler (1973). "Gravitation"

## Acknowledgments

Inspired by visualizations from:
- Interstellar (2014) - Kip Thorne's black hole rendering
- Event Horizon Telescope - First black hole image (M87*)
- NASA's Black Hole Visualization

---

**Made with ❤️ for space enthusiasts and physics lovers**