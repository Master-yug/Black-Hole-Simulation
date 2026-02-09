#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from blackhole_simulation import (
    BlackHole,
    BlackHoleSimulation,
    calculate_blackbody_color,
    calculate_disk_temperature
)

def example_lensing_map():
    """Demonstrate gravitational lensing map."""
    print("\n" + "="*60)
    print("Example 1: Gravitational Lensing Map")
    print("="*60)
    
    bh = BlackHole(mass=10.0, position=(0.0, 0.0))
    
    # Calculate lensing map
    distorted_x, distorted_y = bh.calculate_lensing_map(
        grid_size=(30, 30),
        view_radius=15 * bh.schwarzschild_radius
    )
    
    print(f"\nGenerated lensing map with {distorted_x.size} points")
    print(f"Schwarzschild radius: {bh.schwarzschild_radius:.2f}")
    print(f"View radius: {15 * bh.schwarzschild_radius:.2f}")
    print("\nThe lensing map shows how spacetime curvature distorts")
    print("the apparent position of background stars.")

def example_temperature_colors():
    """Demonstrate temperature-based color grading."""
    print("\n" + "="*60)
    print("Example 2: Temperature-Based Color Grading")
    print("="*60)
    
    # Show color spectrum
    print("\nBlackbody color spectrum:")
    temperatures = [1000, 3000, 5000, 10000, 20000, 40000]
    
    for temp in temperatures:
        rgb = calculate_blackbody_color(temp)
        print(f"  {temp:5d}K -> RGB({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})")
    
    print("\nDisk temperature profile:")
    bh_mass = 10.0
    radii = [6.0, 10.0, 20.0, 40.0, 80.0]
    
    for radius in radii:
        temp_low = calculate_disk_temperature(bh_mass, radius, accretion_rate=0.2)
        temp_high = calculate_disk_temperature(bh_mass, radius, accretion_rate=0.8)
        print(f"  r={radius:5.1f}: T={temp_low:.0f}K (low) -> {temp_high:.0f}K (high)")
    
    print("\nColors change based on:")
    print("  - Radius (inner disk is hotter)")
    print("  - Accretion rate (high rate = hotter)")

def example_precession():
    """Demonstrate general relativistic precession."""
    print("\n" + "="*60)
    print("Example 3: General Relativistic Precession")
    print("="*60)
    
    bh = BlackHole(mass=10.0, position=(0.0, 0.0))
    
    print(f"\nSchwarzschild radius: {bh.schwarzschild_radius:.2f}")
    print(f"ISCO: {bh.isco:.2f}")
    print(f"Photon sphere: {1.5 * bh.schwarzschild_radius:.2f}")
    
    print("\nPrecession rates at different radii:")
    
    radii_factors = [
        (1.5, "Photon sphere"),
        (3.0, "ISCO"),
        (5.0, "Inner disk"),
        (10.0, "Outer disk"),
    ]
    
    for factor, label in radii_factors:
        radius = factor * bh.schwarzschild_radius
        prec_rate = bh.calculate_precession_rate(radius)
        prec_deg = np.degrees(prec_rate)
        print(f"  {label:15s} ({factor}×Rs): {prec_rate:.4f} rad/orbit ({prec_deg:.1f}°)")
    
    print("\nPrecession is strongest near the event horizon,")
    print("causing orbits to rotate over time.")

def example_event_horizon_shadow():
    """Demonstrate event horizon shadow capture."""
    print("\n" + "="*60)
    print("Example 4: Event Horizon Shadow")
    print("="*60)
    
    bh = BlackHole(mass=10.0, position=(0.0, 0.0))
    
    print(f"\nCapture threshold: 1.5 × Rs = {1.5 * bh.schwarzschild_radius:.2f}")
    print("\nTesting different trajectories:")
    
    test_cases = [
        ("Safe orbit", np.array([50.0, 0.0]), np.array([0.0, 0.4])),
        ("Grazing", np.array([40.0, 0.0]), np.array([0.0, 0.6])),
        ("Capture", np.array([30.0, 0.0]), np.array([0.0, 0.8])),
        ("Direct hit", np.array([25.0, 0.0]), np.array([-0.5, 0.2])),
    ]
    
    for name, pos, vel in test_cases:
        periapsis = bh.calculate_periapsis(pos, vel)
        captured = periapsis < 1.5 * bh.schwarzschild_radius
        status = "CAPTURED ⚫" if captured else "Safe ✓"
        print(f"  {name:12s}: periapsis={periapsis:6.2f} -> {status}")
    
    print("\nParticles and photons with periapsis < 1.5×Rs are")
    print("captured and removed from the render buffer.")

def example_full_simulation():
    """Run a full simulation with all features."""
    print("\n" + "="*60)
    print("Example 5: Full Simulation with All Features")
    print("="*60)
    
    print("\nCreating simulation...")
    sim = BlackHoleSimulation(
        blackhole_mass=10.0,
        enable_accretion=True,
        enable_ray_tracing=True
    )
    
    print(f"Initial mass: {sim.blackhole.mass:.2f} M☉")
    print(f"Initial particles: {len(sim.accretion_disk.get_positions())}")
    
    print("\nRunning simulation...")
    for step in range(50):
        sim.update(dt=0.1)
        if step % 10 == 0:
            n_particles = len(sim.accretion_disk.get_positions())
            print(f"  Step {step:2d}: {n_particles} particles, mass={sim.blackhole.mass:.3f} M☉")
    
    # Get final statistics
    n_particles = len(sim.accretion_disk.get_positions())
    colors = sim.accretion_disk.get_colors(accretion_rate=0.5)
    temps = sim.accretion_disk.get_temperatures(accretion_rate=0.5)
    
    print(f"\nFinal state:")
    print(f"  Particles: {n_particles}")
    print(f"  Black hole mass: {sim.blackhole.mass:.3f} M☉")
    print(f"  Mass accreted: {sim.total_mass_accreted:.3f} M☉")
    print(f"  Growth: {((sim.blackhole.mass/sim.initial_mass - 1)*100):.1f}%")
    print(f"  Temperature range: {temps.min():.0f}K - {temps.max():.0f}K")
    
    print("\nAll features are active:")
    print("  ✓ Precession affects particle orbits")
    print("  ✓ Event horizon shadow captures particles")
    print("  ✓ Temperature-based colors show disk structure")
    print("  ✓ Ray tracing includes lensing effects")

def main():
    """Run all examples."""
    print("="*60)
    print("Visual Physics Engine - Feature Examples")
    print("="*60)
    print("\nThis script demonstrates the four new physics features")
    print("added to the black hole simulator.")
    
    example_lensing_map()
    example_temperature_colors()
    example_precession()
    example_event_horizon_shadow()
    example_full_simulation()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run 'python active_simulator.py' for live visualization")
    print("  2. Read VISUAL_PHYSICS_ENGINE.md for detailed documentation")
    print("  3. Experiment with different black hole masses and settings")
    print()

if __name__ == "__main__":
    main()
