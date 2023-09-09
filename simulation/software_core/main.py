import numpy as np
import random
import threading
import multiprocessing
import numba as nb


class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.zeros(3)


class GridCell:
    def __init__(self, position, size):
        self.position = np.array(position)
        self.size = size
        self.particles = []
        self.total_mass = 0.0

    def add_particle(self, particle):
        self.particles.append(particle)
        self.total_mass += particle.mass


class SimulationCore:
    def __init__(self, G=6.67430e-11, theta=0.5, n_threads=4):
        self.G = G  # Gravitational constant
        self.theta = theta  # Barnes-Hut opening angle
        self.particles = []
        self.grid = None
        self.n_threads = n_threads  # Number of threads for parallel computation

    def add_particle(self, mass, position, velocity):
        particle = Particle(mass, position, velocity)
        self.particles.append(particle)

    def initialize_grid(self, grid_origin, grid_size, num_cells):
        # Create a uniform grid to efficiently represent particles in hydrodynamics simulations
        cell_size = grid_size / num_cells
        self.grid = np.empty((num_cells, num_cells, num_cells), dtype=object)

        for i in range(num_cells):
            for j in range(num_cells):
                for k in range(num_cells):
                    cell_position = grid_origin + np.array([i, j, k]) * cell_size
                    self.grid[i, j, k] = GridCell(cell_position, cell_size)

        for particle in self.particles:
            i, j, k = np.floor((particle.position - grid_origin) / cell_size).astype(int)
            self.grid[i, j, k].add_particle(particle)

    def calculate_acceleration(self, particle):
        # Calculate acceleration on a particle due to other particles using the Barnes-Hut approximation
        direction = self.grid_position - particle.position
        distance_sq = np.sum(direction ** 2)
        distance = np.sqrt(distance_sq)

        if distance_sq == 0:
            return np.zeros(3)

        if distance_sq < self.theta * self.grid_size ** 2:
            return self.G * self.grid_total_mass * direction / distance ** 3
        else:
            acceleration = np.zeros(3)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        child = self.grid_child_nodes[i, j, k]
                        if child is not None:
                            acceleration += child.calculate_acceleration(particle)
            return acceleration

    def evolve(self, dt):
        # Evolve the simulation over time using multi-threading
        num_particles = len(self.particles)
        threads = []
        for i in range(self.n_threads):
            start_idx = i * num_particles // self.n_threads
            end_idx = (i + 1) * num_particles // self.n_threads
            thread = threading.Thread(target=self._calculate_accelerations_range, args=(start_idx, end_idx))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Calculate new positions and velocities for each particle
        for particle in self.particles:
            particle.acceleration = np.zeros(3)
            i, j, k = np.floor((particle.position - self.grid_origin) / self.cell_size).astype(int)
            self.grid_position = self.grid[i, j, k].position
            self.grid_size = self.grid[i, j, k].size
            self.grid_total_mass = self.grid[i, j, k].total_mass
            self.grid_child_nodes = self.get_child_nodes(i, j, k)

            particle.acceleration = self.calculate_acceleration(particle)
            particle.velocity += particle.acceleration * dt
            particle.position += particle.velocity * dt

    @nb.jit(nopython=True, parallel=True)
    def _calculate_accelerations_range(self, start_idx, end_idx):
        for i in nb.prange(start_idx, end_idx):
            particle = self.particles[i]
            particle.acceleration = np.zeros(3)
            i, j, k = np.floor((particle.position - self.grid_origin) / self.cell_size).astype(int)
            self.grid_position = self.grid[i, j, k].position
            self.grid_size = self.grid[i, j, k].size
            self.grid_total_mass = self.grid[i, j, k].total_mass
            self.grid_child_nodes = self.get_child_nodes(i, j, k)


class RadiativeTransfer:
    def __init__(self, scattering_probability=0.5):
        self.scattering_probability = scattering_probability

    def scatter_photon(self, position, direction):
        # Simulate photon scattering event
        new_direction = np.random.normal(size=3)
        new_direction /= np.linalg.norm(new_direction)
        return position, new_direction

    def perform_radiative_transfer(self, grid_cells, n_photons=1000):
        # Perform radiative transfer for each grid cell in the simulation
        for cell in grid_cells:
            for _ in range(n_photons):
                position = cell.position
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)

                while True:
                    distance = random.expovariate(1.0)  # Assume mean free path of 1.0
                    position += distance * direction

                    # Check if the photon interacts with the particles in the cell
                    for particle in cell.particles:
                        if np.linalg.norm(position - particle.position) < particle.radius:
                            break
                    else:
                        # No interaction with particles, check if the photon exits the simulation domain
                        if any(position < 0.0) or any(position > 1.0):
                            break

                        # Check for scattering event
                        if random.random() < self.scattering_probability:
                            position, direction = self.scatter_photon(position, direction)


if __name__ == "__main__":
    G = 6.67430e-11
    num_simulations = int(input("Enter the number of simulations: "))
    dt = float(input("Enter the time step (dt): "))
    theta = float(input("Enter the Barnes-Hut opening angle (e.g., 0.5): "))
    n_threads = int(input("Enter the number of threads for parallel computation (e.g., 4): "))
    n_photons = int(input("Enter the number of photons for radiative transfer: "))

    simulation = SimulationCore(G=G, theta=theta, n_threads=n_threads)

    # Add particles with mass, position, and velocity
    num_particles = int(input("Enter the number of particles: "))
    for i in range(num_particles):
        mass = float(input(f"Enter the mass of particle {i + 1}: "))
        position = list(map(float, input(f"Enter the position (x, y, z) of particle {i + 1}: ").split()))
        velocity = list(map(float, input(f"Enter the velocity (vx, vy, vz) of particle {i + 1}: ").split()))
        simulation.add_particle(mass=mass, position=position, velocity=velocity)

    # Initialize the grid for hydrodynamics simulation
    grid_origin_input = input("Enter the grid origin (x, y, z): ")
    grid_origin = np.array(list(map(float, grid_origin_input.split())))

    grid_size = float(input("Enter the grid size: "))
    num_cells = int(input("Enter the number of cells: "))
    simulation.initialize_grid(grid_origin, grid_size, num_cells)

    # Create grid cells for radiative transfer
    grid_cells = np.ravel(simulation.grid)

    num_processes = multiprocessing.cpu_count()
    chunk_size = len(grid_cells) // num_processes
    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else len(grid_cells)
        process = multiprocessing.Process(
            target=RadiativeTransfer(scattering_probability=0.5).perform_radiative_transfer,
            args=(grid_cells[start_idx:end_idx], n_photons),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    # Evolve the simulation
    for step in range(num_simulations):
        simulation.evolve(dt=dt)