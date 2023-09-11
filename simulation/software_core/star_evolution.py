import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
Msun = 1.989e30  # Solar mass (kg)
Rsun = 6.96342e8   # Solar radius (m)
Lsun = 3.828e26   # Solar luminosity (W)

massun = int(input("how many times the mass of the sun? ex: 2 output: 3.978e+33: "))
radun = int(input("how many times the radius of the sun? ex: 2 output: 1392680000: "))
lasun = int(input("how many times the luminosity of the sun? (W) ex: 2 output:7.656e+29 : "))

# Initial conditions
mass_star = massun * Msun  # Initial mass of the star
radius_star = radun * Rsun  # Initial radius of the star
luminosity_star = lasun * Lsun  # Initial luminosity of the star
temperature_star = int(input("surface temperature of the star (K): "))  # Initial surface temperature of the star (K)

# Lists to store evolution data
masses = [mass_star]
radii = [radius_star]
luminosities = [luminosity_star]
temperatures = [temperature_star]

# Simulation parameters
total_time = int(input("Total simulation time (years): "))  # Total simulation time (years)
time_step = int(input("Time step (years)"))   # Time step (years)

# Main Sequence evolution simulation
current_time = 0

while current_time < total_time:
    # Calculate the star's core density (assuming constant density)
    core_density = mass_star / ((4/3) * np.pi * (radius_star ** 3))
    
    # Calculate the star's core temperature using the ideal gas law
    core_temperature = (luminosity_star / (16 * np.pi * G * core_density * radius_star ** 2)) ** 0.25
    
    # Calculate the rate of energy generation (nuclear fusion)
    energy_generation_rate = (mass_star / Msun) ** 3.5 * (core_temperature / 5778) ** 16
    
    # Update the star's luminosity
    luminosity_star = energy_generation_rate * Msun * (current_time + time_step - current_time)
    
    # Update the star's radius using the Stefan-Boltzmann law
    radius_star = np.sqrt(luminosity_star / (4 * np.pi * G * core_temperature ** 4))
    
    # Update the star's mass (loss due to nuclear fusion)
    mass_loss_rate = energy_generation_rate * Msun / (core_temperature ** 2)
    mass_star -= mass_loss_rate * time_step
    
    # Update the surface temperature (assuming constant surface luminosity)
    temperature_star = (luminosity_star / (4 * np.pi * radius_star ** 2 * 5.67e-8)) ** 0.25
    
    # Append data to lists
    masses.append(mass_star)
    radii.append(radius_star)
    luminosities.append(luminosity_star)
    temperatures.append(temperature_star)
    
    # Update time
    current_time += time_step

# Plot the evolution
time_axis = np.arange(0, total_time + time_step, time_step)
plt.figure(figsize=(10, 6))
plt.plot(time_axis, masses, label='Mass')
plt.plot(time_axis, radii, label='Radius')
plt.plot(time_axis, luminosities, label='Luminosity')
plt.plot(time_axis, temperatures, label='Surface Temperature')
plt.xlabel('Time (years)')
plt.ylabel('Stellar Parameters')
plt.yscale('log')
plt.legend()
plt.title('Main Sequence Star Evolution')
plt.show()