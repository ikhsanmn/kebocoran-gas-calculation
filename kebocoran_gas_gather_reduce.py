from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
simulation_start_time = MPI.Wtime()

# Parameters
num_particles = 1000000
num_steps = 10000
box_size = 10
dt = 0.01  # time step

# Scatter particles among processes
particles_per_process = num_particles // size
remainder = num_particles % size

# Initialize local particles' positions and velocities
np.random.seed(rank)  # Ensure different random seed for each process
if rank < remainder:
    num_local_particles = particles_per_process + 1
else:
    num_local_particles = particles_per_process

# Initialize particle arrays
positions = np.random.uniform(0, box_size, size=(num_local_particles, 2))
angles = np.random.uniform(0, 2 * np.pi, num_local_particles)
velocities = np.stack((np.cos(angles), np.sin(angles)), axis=1)

# Define the gap parameters
GAP_X_START = 9.9
GAP_X_END = 10.4
GAP_Y_START = 5
GAP_Y_END = 5.5

# Initialize variables for pressure calculation
local_total_momentum_change = 0

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
    
    # Calculate local total momentum change for this step
    local_total_momentum_change += np.sum(momentum_change)

    # Print the number of particles processed by each rank at each step
    print(f"Rank {rank}: Step {step + 1}, Processing {len(positions)} particles")

# Reduce total momentum change to the root process
total_momentum_change = comm.reduce(local_total_momentum_change, op=MPI.SUM, root=0)

# Gather all positions to the root process
all_positions = comm.gather(positions, root=0)

if rank == 0:
    all_positions = np.vstack(all_positions)
    
    # Calculate the pressure
    wall_area = 4 * box_size  # total length of all walls
    pressure = total_momentum_change / (wall_area * num_steps * dt)
    print(f"Final Pressure: {pressure:.3f}")
    print(f"Final number of particles: {len(all_positions)}")
    print("Execution time: ", MPI.Wtime() - simulation_start_time, "seconds")

    # Optionally, save the results to a file
    # np.save('pressures.npy', [pressure])
    # np.save('num_particles_remaining.npy', [len(all_positions)])