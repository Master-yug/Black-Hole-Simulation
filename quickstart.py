#!/usr/bin/env python3
"""
Quick start script for the black hole simulation.
Generates a simple visualization to get started quickly.
"""

import sys
from blackhole_simulation import BlackHoleSimulation
import matplotlib.pyplot as plt


def main():
    """Run a quick demonstration of the black hole simulation."""
    
    print("\n" + "="*70)
    print(" Spaceing - 2D Black Hole Simulation")
    print("="*70)
    print("\nInitializing simulation...")
    print("  - Creating black hole (10 solar masses)")
    print("  - Generating ray traces (gravitational lensing)")
    print("  - Simulating accretion disk (1000 particles)")
    
    # Create simulation
    sim = BlackHoleSimulation(
        blackhole_mass=10.0,
        enable_accretion=True,
        enable_ray_tracing=True
    )
    
    print("\nSimulation parameters:")
    print(f"  - Schwarzschild radius: {sim.blackhole.schwarzschild_radius:.2f}")
    print(f"  - Event horizon: {sim.blackhole.event_horizon:.2f}")
    print(f"  - ISCO: {sim.blackhole.isco:.2f}")
    print(f"  - Photon sphere: {1.5 * sim.blackhole.schwarzschild_radius:.2f}")
    
    # Evolve the system
    print("\nEvolving system...")
    for i in range(30):
        sim.update(dt=0.1)
        if i % 10 == 0:
            print(f"  - Step {i}/30")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig, ax = sim.visualize(
        show_rays=True,
        show_disk=True,
        show_time_dilation=True
    )
    
    # Save and show
    filename = 'quickstart_demo.png'
    plt.savefig(filename, dpi=200, facecolor='#000510')
    print(f"\n✓ Visualization saved to: {filename}")
    
    print("\n" + "="*70)
    print(" Simulation Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. View the generated image: quickstart_demo.png")
    print("  2. Run examples.py for more demonstrations")
    print("  3. Check README.md for detailed documentation")
    print("  4. Modify blackhole_simulation.py to customize")
    print("\n")
    
    # Show interactive plot
    try:
        plt.show()
    except:
        print("(Interactive display not available in this environment)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
