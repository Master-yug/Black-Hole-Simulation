import numpy as np
import matplotlib.pyplot as plt
from blackhole_simulation import (
    BlackHole, BlackHoleSimulation, RayTracer, 
    AccretionDisk, Particle
)


def example_1_basic_simulation():
    """
    Example 1: Basic black hole simulation with all features.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Simulation")
    print("="*60)
    
    sim = BlackHoleSimulation(
        blackhole_mass=10.0,
        enable_accretion=True,
        enable_ray_tracing=True
    )
    
    # Evolve the system
    print("Evolving system for 50 time steps...")
    for i in range(50):
        sim.update(dt=0.1)
    
    # Visualize
    fig, ax = sim.visualize()
    plt.savefig('example_1_basic.png', dpi=200, facecolor='#000510')
    print("Saved to: example_1_basic.png")
    plt.close()


def example_2_supermassive_blackhole():
    """
    Example 2: Supermassive black hole (like Sagittarius A*).
    """
    print("\n" + "="*60)
    print("Example 2: Supermassive Black Hole")
    print("="*60)
    
    sim = BlackHoleSimulation(
        blackhole_mass=1000.0,  # 1000 solar masses
        enable_accretion=True,
        enable_ray_tracing=True
    )
    
    print(f"Mass: {sim.blackhole.mass} M☉")
    print(f"Schwarzschild radius: {sim.blackhole.schwarzschild_radius:.2f}")
    
    # Create larger accretion disk
    sim.accretion_disk = AccretionDisk(
        sim.blackhole,
        n_particles=2000,
        inner_radius=sim.blackhole.isco,
        outer_radius=sim.blackhole.isco * 15
    )
    
    # Evolve
    for i in range(30):
        sim.update(dt=0.1)
    
    fig, ax = sim.visualize()
    plt.savefig('example_2_supermassive.png', dpi=200, facecolor='#000510')
    print("Saved to: example_2_supermassive.png")
    plt.close()


def example_3_ray_tracing_only():
    """
    Example 3: Focus on ray tracing without accretion disk.
    """
    print("\n" + "="*60)
    print("Example 3: Ray Tracing Focus")
    print("="*60)
    
    # Create black hole
    bh = BlackHole(mass=5.0)
    ray_tracer = RayTracer(bh)
    
    # Trace many rays from different angles
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_facecolor('#000510')
    fig.patch.set_facecolor('#000510')
    
    view_radius = 20 * bh.schwarzschild_radius
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    
    # Draw event horizon
    from matplotlib.patches import Circle
    event_horizon = Circle(bh.position, bh.event_horizon,
                          color='black', zorder=100)
    ax.add_patch(event_horizon)
    
    # Trace rays from a grid
    print("Tracing rays...")
    n_rays_per_side = 15
    for i in range(n_rays_per_side):
        y_start = -view_radius * 0.8 + i * (1.6 * view_radius / n_rays_per_side)
        start_pos = np.array([view_radius * 0.9, y_start])
        direction = np.array([-1.0, 0.0])
        
        trajectory = ray_tracer.trace_ray(start_pos, direction, 
                                         max_steps=3000, dt=0.005)
        
        if len(trajectory) > 1:
            traj_array = np.array(trajectory)
            # Color by proximity to black hole
            r_min = np.min(np.linalg.norm(traj_array - bh.position, axis=1))
            color_intensity = np.clip(1.0 - (r_min / bh.schwarzschild_radius - 1.0) / 10.0, 0, 1)
            color = plt.cm.plasma(color_intensity)
            
            ax.plot(traj_array[:, 0], traj_array[:, 1], 
                   color=color, linewidth=1, alpha=0.8)
    
    ax.plot(bh.position[0], bh.position[1], 'w*', markersize=15, zorder=101)
    ax.set_title('Gravitational Lensing - Ray Tracing', 
                color='white', fontsize=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.savefig('example_3_raytracing.png', dpi=200, facecolor='#000510')
    print("Saved to: example_3_raytracing.png")
    plt.close()


def example_4_time_dilation_analysis():
    """
    Example 4: Detailed time dilation analysis.
    """
    print("\n" + "="*60)
    print("Example 4: Time Dilation Analysis")
    print("="*60)
    
    bh = BlackHole(mass=10.0)
    
    # Calculate time dilation at various radii
    radii = np.linspace(bh.schwarzschild_radius * 1.01, 
                       bh.schwarzschild_radius * 20, 100)
    time_dilations = []
    
    for r in radii:
        pos = np.array([r, 0.0])
        td = bh.time_dilation_factor(pos)
        time_dilations.append(td)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Time dilation vs radius
    ax1.plot(radii / bh.schwarzschild_radius, time_dilations, 
            'cyan', linewidth=2)
    ax1.axvline(x=1, color='red', linestyle='--', label='Event Horizon')
    ax1.axvline(x=3, color='yellow', linestyle='--', label='ISCO')
    ax1.axvline(x=1.5, color='orange', linestyle='--', label='Photon Sphere')
    ax1.set_xlabel('Radius (in Schwarzschild radii)', fontsize=12)
    ax1.set_ylabel('Time Dilation Factor', fontsize=12)
    ax1.set_title('Time Dilation vs. Distance from Black Hole', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(1, 20)
    ax1.set_ylim(0, 1.1)
    
    # Proper time vs coordinate time
    coordinate_times = np.linspace(0, 10, 100)
    proper_times_at_different_radii = []
    radii_to_show = [1.5, 2, 3, 5, 10]
    
    for r_multiplier in radii_to_show:
        r = bh.schwarzschild_radius * r_multiplier
        pos = np.array([r, 0.0])
        td = bh.time_dilation_factor(pos)
        proper_time = coordinate_times * td
        ax2.plot(coordinate_times, proper_time, 
                label=f'r = {r_multiplier}$r_s$', linewidth=2)
    
    ax2.plot(coordinate_times, coordinate_times, 'k--', 
            label='Far from BH', linewidth=1)
    ax2.set_xlabel('Coordinate Time', fontsize=12)
    ax2.set_ylabel('Proper Time', fontsize=12)
    ax2.set_title('Proper Time vs Coordinate Time at Different Radii', 
                 fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('example_4_time_dilation.png', dpi=200)
    print("Saved to: example_4_time_dilation.png")
    plt.close()
    
    # Print some values
    print("\nTime dilation factors:")
    for r_mult in [1.5, 2.0, 3.0, 5.0, 10.0]:
        r = bh.schwarzschild_radius * r_mult
        pos = np.array([r, 0.0])
        td = bh.time_dilation_factor(pos)
        print(f"  At {r_mult}r_s: {td:.4f} (time passes {td*100:.1f}% as fast)")


def example_5_particle_trajectories():
    """
    Example 5: Individual particle trajectories around black hole.
    """
    print("\n" + "="*60)
    print("Example 5: Particle Trajectories")
    print("="*60)
    
    bh = BlackHole(mass=10.0)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_facecolor('#000510')
    fig.patch.set_facecolor('#000510')
    
    view_radius = 15 * bh.schwarzschild_radius
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    
    # Draw black hole
    from matplotlib.patches import Circle
    event_horizon = Circle(bh.position, bh.event_horizon,
                          color='black', zorder=100)
    ax.add_patch(event_horizon)
    
    isco_circle = Circle(bh.position, bh.isco,
                        fill=False, edgecolor='yellow',
                        linewidth=1, linestyle='--', alpha=0.5)
    ax.add_patch(isco_circle)
    
    # Create particles with different initial conditions
    print("Simulating particle trajectories...")
    initial_conditions = [
        # (radius, velocity_tangential_fraction, color, label)
        (bh.isco * 1.5, 0.95, '#FF6B6B', 'Decaying orbit'),
        (bh.isco * 2.0, 1.00, '#4ECDC4', 'Stable orbit'),
        (bh.isco * 3.0, 1.05, '#45B7D1', 'Expanding orbit'),
        (bh.isco * 2.5, 0.80, '#FFA07A', 'Spiral infall'),
    ]
    
    for r_initial, v_fraction, color, label in initial_conditions:
        # Create particle at distance r with tangential velocity
        theta = np.random.uniform(0, 2*np.pi)
        x = r_initial * np.cos(theta)
        y = r_initial * np.sin(theta)
        position = np.array([x, y])
        
        # Keplerian velocity for circular orbit
        v_circular = np.sqrt(bh.G * bh.mass / r_initial)
        v_tangential = v_circular * v_fraction
        
        vx = -v_tangential * np.sin(theta)
        vy = v_tangential * np.cos(theta)
        velocity = np.array([vx, vy])
        
        particle = Particle(position, velocity, mass=0.1)
        
        # Simulate for many steps
        for step in range(1000):
            r = np.linalg.norm(particle.position - bh.position)
            
            if r < bh.event_horizon or r > view_radius:
                break
            
            force = bh.gravitational_force(particle.position, particle.mass)
            acceleration = force / particle.mass
            time_dilation = bh.time_dilation_factor(particle.position)
            
            particle.update(acceleration, dt=0.05, time_dilation=time_dilation)
        
        # Plot trajectory
        traj = np.array(particle.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color=color, 
               linewidth=1.5, alpha=0.8, label=label)
        
        # Mark start and end
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)
        if particle.alive:
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=6)
    
    ax.plot(bh.position[0], bh.position[1], 'w*', markersize=15, zorder=101)
    ax.set_title('Particle Trajectories Around Black Hole', 
                color='white', fontsize=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.legend(loc='upper right', facecolor='#000510', 
             edgecolor='white', labelcolor='white')
    
    plt.savefig('example_5_trajectories.png', dpi=200, facecolor='#000510')
    print("Saved to: example_5_trajectories.png")
    plt.close()


def example_6_comparison_masses():
    """
    Example 6: Compare black holes of different masses.
    """
    print("\n" + "="*60)
    print("Example 6: Mass Comparison")
    print("="*60)
    
    masses = [1.0, 5.0, 10.0, 50.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, mass in enumerate(masses):
        ax = axes[idx]
        ax.set_aspect('equal')
        ax.set_facecolor('#000510')
        
        bh = BlackHole(mass=mass)
        ray_tracer = RayTracer(bh)
        
        view_radius = 15 * bh.schwarzschild_radius
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-view_radius, view_radius)
        
        # Draw structures
        from matplotlib.patches import Circle
        event_horizon = Circle(bh.position, bh.event_horizon,
                              color='black', zorder=100)
        ax.add_patch(event_horizon)
        
        isco_circle = Circle(bh.position, bh.isco,
                           fill=False, edgecolor='yellow',
                           linewidth=1, linestyle='--', alpha=0.5)
        ax.add_patch(isco_circle)
        
        # Trace a few rays
        n_rays = 8
        for i in range(n_rays):
            angle = 2 * np.pi * i / n_rays
            start_pos = np.array([
                view_radius * 0.8 * np.cos(angle),
                view_radius * 0.8 * np.sin(angle)
            ])
            direction = -start_pos / np.linalg.norm(start_pos)
            perp = np.array([-direction[1], direction[0]])
            direction = direction + 0.3 * perp
            
            trajectory = ray_tracer.trace_ray(start_pos, direction, 
                                            max_steps=1500, dt=0.01)
            if len(trajectory) > 1:
                traj_array = np.array(trajectory)
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                       color='#FFA500', linewidth=0.8, alpha=0.7)
        
        ax.plot(bh.position[0], bh.position[1], 'w*', 
               markersize=10, zorder=101)
        ax.set_title(f'Mass = {mass} M☉\n' + 
                    f'$r_s$ = {bh.schwarzschild_radius:.2f}',
                    color='white', fontsize=12)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    fig.patch.set_facecolor('#000510')
    plt.tight_layout()
    plt.savefig('example_6_mass_comparison.png', dpi=200, facecolor='#000510')
    print("Saved to: example_6_mass_comparison.png")
    plt.close()


def example_7_active_simulator_demo():
    """
    Example 7: Demonstrate active simulator (requires display).
    Note: This example only prints information in headless mode.
    """
    print("\n" + "="*60)
    print("Example 7: Active Simulator (Continuous Running)")
    print("="*60)
    
    print("\nThe active simulator provides a continuously running window")
    print("that updates in real-time until you close it.")
    print("\nTo use the active simulator, run:")
    print("  python active_simulator.py")
    print("\nOr programmatically:")
    print("  sim = BlackHoleSimulation(...)")
    print("  sim.run_active_simulator()")
    print("\nAvailable options:")
    print("  --mass MASS       : Black hole mass in solar masses")
    print("  --particles N     : Number of particles in accretion disk")
    print("  --dt DT          : Simulation time step")
    print("  --interval MS    : Update interval in milliseconds")
    print("  --no-rays        : Disable ray tracing")
    print("  --no-disk        : Disable accretion disk")
    print("  --no-fps         : Hide FPS counter")
    print("\nExample commands:")
    print("  python active_simulator.py --mass 50.0 --particles 2000")
    print("  python active_simulator.py --dt 0.1 --interval 30")
    print("  python blackhole_simulation.py --active")
    print("\n✓ Active simulator information displayed")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  BLACK HOLE SIMULATION - EXAMPLE DEMONSTRATIONS")
    print("="*70)
    
    try:
        example_1_basic_simulation()
        example_2_supermassive_blackhole()
        example_3_ray_tracing_only()
        example_4_time_dilation_analysis()
        example_5_particle_trajectories()
        example_6_comparison_masses()
        example_7_active_simulator_demo()
        
        print("\n" + "="*70)
        print("  ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  - example_1_basic.png")
        print("  - example_2_supermassive.png")
        print("  - example_3_raytracing.png")
        print("  - example_4_time_dilation.png")
        print("  - example_5_trajectories.png")
        print("  - example_6_mass_comparison.png")
        print("\nTo try the active simulator:")
        print("  python active_simulator.py")
        print("\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
