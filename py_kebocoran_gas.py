from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start = MPI.Wtime()

# Parameters
num_particles = 1000000
num_steps = 10000
box_size = 10
dt = 0.01  # time step

# Divide particles among processes
particles_per_process = num_particles // size
remainder = num_particles % size

if rank < remainder:
    start = rank * (particles_per_process + 1)
    end = start + particles_per_process + 1
else:
    start = rank * particles_per_process + remainder
    end = start + particles_per_process

num_local_particles = end - start

# Initialize local particles' positions and velocities
np.random.seed(rank)  # Ensure different random seed for each process
positions = np.random.uniform(0, box_size, size=(num_local_particles, 2))
angles = np.random.uniform(0, 2 * np.pi, num_local_particles)
velocities = np.stack((np.cos(angles), np.sin(angles)), axis=1)

# Define the gap parameters
GAP_X_START = 9.9
GAP_X_END = 10.4
GAP_Y_START = 5
GAP_Y_END = 5.5

# Initialize variables for pressure calculation
total_momentum_change = 0

# Store pressures and number of remaining particles
pressures = []
num_particles_remaining = []

# Monte Carlo simulation
for step in range(num_steps):
    # Move particles
    new_positions = positions + velocities * dt
    
    # Reflection check and momentum change
    momentum_change = np.zeros_like(velocities)
    
    # Check x-boundaries
    mask_x_left = new_positions[:, 0] < 0
    mask_x_right = new_positions[:, 0] > box_size
    momentum_change[mask_x_left | mask_x_right, 0] = 2 * np.abs(velocities[mask_x_left | mask_x_right, 0])
    velocities[mask_x_left | mask_x_right, 0] *= -1
    
    # Check y-boundaries
    mask_y_bottom = new_positions[:, 1] < 0
    mask_y_top = new_positions[:, 1] > box_size
    momentum_change[mask_y_bottom | mask_y_top, 1] = 2 * np.abs(velocities[mask_y_bottom | mask_y_top, 1])
    velocities[mask_y_bottom | mask_y_top, 1] *= -1

    # Clip new positions
    new_positions = np.clip(new_positions, 0, box_size)
    
    # Check if particles hit the gap
    in_gap = (GAP_X_START <= new_positions[:, 0]) & (new_positions[:, 0] <= GAP_X_END) & (GAP_Y_START <= new_positions[:, 1]) & (new_positions[:, 1] <= GAP_Y_END)
    new_positions = new_positions[~in_gap]
    velocities = velocities[~in_gap]
    
    # Update positions of particles
    positions = new_positions
    
    # Calculate total momentum change for this step
    total_momentum_change += np.sum(momentum_change)
    
    # Gather positions and momentum changes from all processes
    all_positions = comm.gather(positions, root=0)
    all_momentum_change = comm.gather(total_momentum_change, root=0)
    
    if rank == 0:
        all_positions = np.vstack(all_positions)
        total_momentum_change = np.sum(all_momentum_change)
        
        # Calculate the pressure
        wall_area = 4 * box_size  # total length of all walls
        pressure = total_momentum_change / (wall_area * (step + 1) * dt)
        pressures.append(pressure)
        num_particles_remaining.append(len(all_positions))
        
        # Print status every 100 steps
        if step % 100 == 0:
            print(f"Step: {step+1}, Particles: {len(all_positions)}, Pressure: {pressure:.3f}")

# Finalize MPI
comm.Barrier()

if rank == 0:
    # Print final results
    print("Simulation complete.")
    print(f"Final number of particles: {num_particles_remaining[-1]}")
    print(f"Final pressure: {pressures[-1]:.3f}")
    print("waktu = ", MPI.Wtime()-start, "detik")

    # Optionally, save the results to a file
    #np.save('pressures.npy', pressures)
    #np.save('num_particles_remaining.npy', num_particles_remaining)

