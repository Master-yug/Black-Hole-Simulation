#!/usr/bin/env python3

import sys
import argparse
from blackhole_simulation import BlackHoleSimulation


def main():  
    parser = argparse.ArgumentParser(
        description='Active Black Hole Simulator - Continuously running visualization'
    )
    parser.add_argument('--mass', type=float, default=10.0,
                       help='Black hole mass in solar masses (default: 10.0)')
    parser.add_argument('--particles', type=int, default=1000,
                       help='Number of particles in accretion disk (default: 1000)')
    parser.add_argument('--dt', type=float, default=0.05,
                       help='Simulation time step (default: 0.05)')
    parser.add_argument('--interval', type=int, default=50,
                       help='Update interval in milliseconds (default: 50)')
    parser.add_argument('--no-rays', action='store_true',
                       help='Disable ray tracing visualization')
    parser.add_argument('--no-disk', action='store_true',
                       help='Disable accretion disk')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter')
    
    args = parser.parse_args()
    print("\n" + "="*70)
    print(" " * 15 + "SPACEING - ACTIVE SIMULATOR")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Black Hole Mass: {args.mass} M☉")
    print(f"  Particles: {args.particles}")
    print(f"  Time Step: {args.dt}")
    print(f"  Update Interval: {args.interval} ms")
    print(f"  Ray Tracing: {'Disabled' if args.no_rays else 'Enabled'}")
    print(f"  Accretion Disk: {'Disabled' if args.no_disk else 'Enabled'}")
    print(f"  FPS Display: {'Disabled' if args.no_fps else 'Enabled'}")
    print("\nInitializing simulation...")
    try:
        sim = BlackHoleSimulation(
            blackhole_mass=args.mass,
            enable_accretion=not args.no_disk,
            enable_ray_tracing=not args.no_rays
        )
        if not args.no_disk and args.particles != 1000:
            from blackhole_simulation import AccretionDisk
            print(f"Creating custom accretion disk with {args.particles} particles...")
            sim.accretion_disk = AccretionDisk(
                sim.blackhole,
                n_particles=args.particles,
                inner_radius=sim.blackhole.isco,
                outer_radius=sim.blackhole.isco * 10
            )        
        print("\nSimulation Properties:")
        print(f"  Schwarzschild Radius: {sim.blackhole.schwarzschild_radius:.3f}")
        print(f"  Event Horizon: {sim.blackhole.event_horizon:.3f}")
        print(f"  ISCO: {sim.blackhole.isco:.3f}")
        print(f"  Photon Sphere: {1.5 * sim.blackhole.schwarzschild_radius:.3f}")
        sim.run_active_simulator(
            dt=args.dt,
            interval=args.interval,
            show_rays=not args.no_rays,
            show_disk=not args.no_disk,
            show_fps=not args.no_fps
        )        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()
